"""Compare frozen MMCR backbones using a fixed CIFAR-10 ridge probe.

This is a representation diagnostic, not the MMCR paper's full linear-evaluation
protocol. CIFAR-10 must already exist locally; this script never downloads it.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def sha256_file(path: Path) -> str:
    """Hash a file without loading an entire snapshot into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(indices: np.ndarray) -> str:
    """Hash ordered indices as little-endian signed 64-bit integers."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def feature_summary(features: np.ndarray) -> dict:
    """Summarize finite features and variation relevant to collapse checks."""
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected a finite [samples, features] matrix")
    spread = features.std(axis=0, dtype=np.float64)
    return {
        "shape": list(features.shape),
        "all_finite": True,
        "feature_std_min": float(spread.min()),
        "feature_std_median": float(np.median(spread)),
        "feature_std_max": float(spread.max()),
        "feature_std_below_1e-8_count": int((spread < 1e-8).sum()),
        "mean_feature_l2_norm": float(np.linalg.norm(features, axis=1).mean()),
    }


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[dict, Pipeline]:
    """Fit scaling and ridge on training examples only.

    Args:
        train_features: Frozen encoder outputs for the probe's training split.
        train_labels: Labels used only by the probe fit.
        val_features: Held-out outputs; never used to fit preprocessing.
        val_labels: Held-out labels used only to score predictions.

    Returns:
        Evaluation metrics and the fitted scaler/classifier pipeline.
    """
    train_summary = feature_summary(train_features)
    val_summary = feature_summary(val_features)
    if len(train_labels) != len(train_features) or len(val_labels) != len(val_features):
        raise ValueError("Feature and label counts differ")
    if len(np.unique(train_labels)) < 2:
        raise ValueError("The probe requires at least two training classes")
    pipeline = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=1.0, solver="cholesky")
    )
    pipeline.fit(train_features, train_labels)
    predictions = pipeline.predict(val_features)
    correct = int(np.count_nonzero(predictions == val_labels))
    scaler = pipeline.named_steps["standardscaler"]
    return {
        "top1_percent": 100.0 * correct / len(val_labels),
        "correct": correct,
        "evaluated": len(val_labels),
        "train_features": train_summary,
        "val_features": val_summary,
        "scaler_fit_samples": int(scaler.n_samples_seen_),
        "scaler_mean_sha256": hashlib.sha256(
            np.asarray(scaler.mean_, dtype="<f8").tobytes()
        ).hexdigest(),
    }, pipeline


def self_test() -> None:
    """Check separability and that held-out features cannot affect scaling."""
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 40)
    centers = np.eye(3) * 4
    train = centers[labels] + rng.normal(0, 0.05, (len(labels), 3))
    val_labels = np.repeat(np.arange(3), 10)
    val = centers[val_labels] + rng.normal(0, 0.05, (len(val_labels), 3))
    result, fitted = fit_probe(train, labels, val, val_labels)
    assert result["top1_percent"] == 100.0
    scaler = fitted.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
    _, shifted_fit = fit_probe(train, labels, val + 1000, val_labels)
    np.testing.assert_array_equal(
        scaler.mean_, shifted_fit.named_steps["standardscaler"].mean_
    )
    assert int(scaler.n_samples_seen_) == len(train)
    try:
        feature_summary(np.array([[np.nan, 1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite features were accepted")
    print("Self-test passed: separability, train-only scaling, nonfinite rejection.")


def extract_features(
    model: Any,
    dataset: Any,
    indices: np.ndarray,
    device: str,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deterministic frozen features; labels never enter the encoder."""
    import torch
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    feature_batches, label_batches = [], []
    with torch.inference_mode():
        for images, labels in loader:
            embeddings = model(images.to(device))
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                raise ValueError("Backbone must return [batch, features] embeddings")
            feature_batches.append(embeddings.cpu().numpy())
            label_batches.append(labels.numpy())
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def environment() -> dict:
    """Collect runtime and source provenance without altering the checkout."""
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scikit-learn", "stable-pretraining"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    git_errors = {}

    def git_output(name: str, arguments: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            git_errors[name] = "Timed out after 10 seconds; provenance unavailable."
            return None
        if result.returncode != 0:
            git_errors[name] = result.stderr.strip()
            return None
        return result.stdout

    revision = git_output("revision", ["rev-parse", "HEAD"])
    dirty = git_output("status", ["status", "--porcelain"])
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_sha": revision.strip() if revision is not None else None,
        "git_status": dirty.splitlines() if dirty is not None else None,
        "git_errors": git_errors,
        "script_sha256": sha256_file(Path(__file__)),
        "command_argv": sys.argv,
    }


def write_report(path: Path, report: dict) -> None:
    """Persist completed snapshots, including a difference when both exist."""
    snapshots = report["snapshots"]
    if "initial" in snapshots and "final" in snapshots:
        report["final_minus_initial_top1_percentage_points"] = (
            snapshots["final"]["top1_percent"] - snapshots["initial"]["top1_percent"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Evaluate named backbone snapshots with identical fixed probe settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("frozen-probe-results.json")
    )
    parser.add_argument(
        "--merge", type=Path, help="Merge prior results with identical protocol"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Optional cache of extracted features"
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.snapshot:
        parser.error("Supply at least one --snapshot NAME=PATH")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    if not 1 <= args.train_samples <= 45000 or not 1 <= args.val_samples <= 5000:
        parser.error("train-samples must be 1..45000 and val-samples 1..5000")
    snapshots = {}
    for item in args.snapshot:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in snapshots:
            parser.error("Snapshots must have unique nonempty NAME=PATH entries")
        snapshots[name] = Path(path).resolve(strict=True)

    import torch
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    import stable_pretraining as spt

    torch.set_num_threads(4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    dataset = CIFAR10(
        args.data_dir,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        ),
    )
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    ).numpy()
    train_indices = order[5000:][: args.train_samples]
    val_indices = order[:5000][: args.val_samples]
    protocol = {
        "name": "frozen ridge probe; not the paper's full linear evaluation",
        "backbone": "spt.backbone.from_torchvision(resnet18, low_resolution=True, weights=None)",
        "dataset": "CIFAR10 train=True; official test set unused",
        "seed": args.seed,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_indices_sha256_le_int64": index_hash(train_indices),
        "val_indices_sha256_le_int64": index_hash(val_indices),
        "transform": {"to_tensor": True, "mean": list(MEAN), "std": list(STD)},
        "scaler": "StandardScaler fit on training features only",
        "classifier": {"name": "RidgeClassifier", "alpha": 1.0, "solver": "cholesky"},
        "hyperparameter_search": False,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    report = {"protocol": protocol, "snapshots": {}}
    if args.merge:
        report = json.loads(args.merge.read_text())
        if report["protocol"] != protocol:
            raise ValueError("Cannot merge results with different evaluation protocols")
    provenance = environment()
    factory_source = inspect.getsourcefile(spt.backbone.from_torchvision)
    if factory_source is None:
        raise RuntimeError("Cannot identify backbone factory source for provenance")
    provenance["backbone_factory_source_sha256"] = sha256_file(Path(factory_source))
    for previous in report["snapshots"].values():
        for field in ("script_sha256", "packages", "backbone_factory_source_sha256"):
            if previous["environment"].get(field) != provenance[field]:
                raise ValueError(f"Cannot merge results with different {field}")
    for name, path in snapshots.items():
        started = time.perf_counter()
        snapshot_hash = sha256_file(path)
        key_data = {
            "snapshot_sha256": snapshot_hash,
            "protocol": protocol,
            "script_sha256": provenance["script_sha256"],
            "packages": provenance["packages"],
            "git_sha": provenance["git_sha"],
            "backbone_factory_source_sha256": provenance[
                "backbone_factory_source_sha256"
            ],
        }
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        cache_path = args.cache_dir / f"{key}.npz" if args.cache_dir else None
        cache_hit = cache_path is not None and cache_path.exists()
        if cache_hit:
            with np.load(cache_path, allow_pickle=False) as cached:
                train_x, train_y = cached["train_x"], cached["train_y"]
                val_x, val_y = cached["val_x"], cached["val_y"]
        else:
            backbone = spt.backbone.from_torchvision(
                "resnet18", low_resolution=True, weights=None
            )
            backbone.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
            backbone.requires_grad_(False).eval().to(args.device)
            train_x, train_y = extract_features(
                backbone,
                dataset,
                train_indices,
                args.device,
                args.batch_size,
                args.workers,
            )
            val_x, val_y = extract_features(
                backbone,
                dataset,
                val_indices,
                args.device,
                args.batch_size,
                args.workers,
            )
            del backbone
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                )
        expected_labels = np.asarray(dataset.targets)
        np.testing.assert_array_equal(train_y, expected_labels[train_indices])
        np.testing.assert_array_equal(val_y, expected_labels[val_indices])
        result, _ = fit_probe(train_x, train_y, val_x, val_y)
        result.update(
            {
                "snapshot_path": str(path),
                "snapshot_sha256": snapshot_hash,
                "environment": provenance,
                "seconds": time.perf_counter() - started,
                "feature_cache_hit": cache_hit,
                "feature_cache_path": str(cache_path.resolve()) if cache_path else None,
            }
        )
        report["snapshots"][name] = result
        write_report(args.output, report)
        print(
            json.dumps({"snapshot": name, "top1_percent": result["top1_percent"]}),
            flush=True,
        )
    print(f"Frozen ridge probe report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
