"""Summarize the completed MMCR run without selecting favorable epochs."""

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    """Export per-epoch evidence and a figure from the original registry CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = list(args.run_dir.rglob("experiment-summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one completed experiment, found {summaries}")
    summary = json.loads(summaries[0].read_text())
    metrics_path = summaries[0].with_name("metrics.csv")
    rows = list(csv.DictReader(metrics_path.open()))
    validation = []
    losses = {}
    for row in rows:
        if not row.get("epoch"):
            continue
        epoch = int(float(row["epoch"])) + 1
        accuracy = row.get("eval/probe_accuracy_epoch") or row.get(
            "eval/probe_accuracy"
        )
        if accuracy:
            value = float(accuracy)
            if not math.isfinite(value):
                raise ValueError("Nonfinite validation accuracy")
            validation.append(
                {"epoch": epoch, "accuracy": value, "step": int(float(row["step"]))}
            )
        value = row.get("fit/loss") or row.get("fit/loss_step")
        if value:
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError("Nonfinite training loss")
            losses.setdefault(epoch, []).append(scalar)
    expected_epochs = summary["config"]["epochs"]
    if [row["epoch"] for row in validation] != list(range(1, expected_epochs + 1)):
        raise RuntimeError("Missing or repeated epoch validation records")
    expected_steps = expected_epochs * (
        summary["train_size"] // summary["config"]["batch_size"]
    )
    if summary["global_step"] != expected_steps:
        raise RuntimeError("The run did not complete its planned number of steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "epoch-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "step", "accuracy", "mean_ssl_loss"]
        )
        writer.writeheader()
        for row in validation:
            values = losses.get(row["epoch"], [])
            writer.writerow(
                {**row, "mean_ssl_loss": sum(values) / len(values) if values else ""}
            )
    result = {
        "experiment_summary": str(summaries[0]),
        "raw_metrics": str(metrics_path),
        "completed_epochs": expected_epochs,
        "completed_steps": summary["global_step"],
        "first_epoch_online_top1": validation[0]["accuracy"],
        "final_epoch_online_top1": validation[-1]["accuracy"],
        "online_gain_percentage_points": 100
        * (validation[-1]["accuracy"] - validation[0]["accuracy"]),
        "mean_first_five_online_top1": sum(r["accuracy"] for r in validation[:5])
        / min(5, len(validation)),
        "mean_last_five_online_top1": sum(r["accuracy"] for r in validation[-5:])
        / min(5, len(validation)),
        "seconds": summary["seconds"],
        "evaluation": summary["evaluation"],
        "fake_data": summary["config"]["fake_data"],
        "selection": "Report the final planned epoch and all epoch measurements, not the best validation checkpoint.",
    }
    (args.output_dir / "training-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(
        [r["epoch"] for r in validation],
        [100 * r["accuracy"] for r in validation],
        marker="o",
        color="#146c94",
    )
    dataset_name = (
        "Synthetic preflight" if summary["config"]["fake_data"] else "CIFAR-10"
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Held-out online probe top-1 (%)",
        title=f"{dataset_name}: {summary['train_size']} train / {summary['val_size']} validation",
    )
    ordered = sorted(losses)
    axes[1].plot(
        ordered,
        [sum(losses[e]) / len(losses[e]) for e in ordered],
        marker="o",
        color="#b35a23",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Mean training MMCR loss",
        title="Batch 64, two views, seed 7",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(args.output_dir / "learning-curves.png", dpi=180)
    fig.savefig(args.output_dir / "learning-curves.svg")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
