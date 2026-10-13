"""Tabulate and plot same-protocol runs: Mac MMCR, Linux CPU MMCR and SimCLR.

Usage: python compare.py RESULTS_DIR
RESULTS_DIR holds one subdirectory per Linux run (``mmcr``, ``simclr``) with
``epoch-metrics.csv``, ``frozen-probe.json`` and ``feature-spectrum.json``.
"""

import csv
import json
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1]


def load(label: str, metrics: Path, probe: Path, spectrum: Path | None) -> dict:
    rows = list(csv.DictReader(metrics.open()))
    acc = [100 * float(r["accuracy"]) for r in rows]
    snaps = json.loads(probe.read_text())["snapshots"]
    run = {
        "label": label,
        "online": acc,
        "online_first": acc[0],
        "online_final": acc[-1],
        "online_last5": sum(acc[-5:]) / 5,
        "frozen_initial": snaps["initial"]["top1_percent"],
        "frozen_final": snaps["final"]["top1_percent"],
    }
    if spectrum and spectrum.exists():
        spec = json.loads(spectrum.read_text())
        run["erank_initial"] = spec["initial"]["effective_rank"]
        run["erank_final"] = spec["final"]["effective_rank"]
    return run


def main(results: str) -> None:
    root = Path(results)
    runs = [
        load(
            "MMCR, Mac MPS (original run)",
            EVIDENCE / "mac-run/results/epoch-metrics.csv",
            EVIDENCE / "mac-run/frozen-probe.json",
            None,
        )
    ]
    for name, label in (("mmcr", "MMCR, Linux CPU"), ("simclr", "SimCLR, Linux CPU")):
        if (root / name / "frozen-probe.json").exists():
            runs.append(
                load(
                    label,
                    root / name / "epoch-metrics.csv",
                    root / name / "frozen-probe.json",
                    root / name / "feature-spectrum.json",
                )
            )

    def fmt(run, key):
        return f"{run[key]:.2f}" if key in run else "n/a"

    lines = [
        "| Run | Online probe, epoch 1 → 20 (%) | Online, last-5 mean (%) "
        "| Frozen ridge, initial → final (%) | Δ (pp) | Effective rank, initial → final |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['label']} | {r['online_first']:.2f} → {r['online_final']:.2f} "
            f"| {r['online_last5']:.2f} | {r['frozen_initial']:.2f} → {r['frozen_final']:.2f} "
            f"| {r['frozen_final'] - r['frozen_initial']:+.2f} "
            f"| {fmt(r, 'erank_initial')} → {fmt(r, 'erank_final')} |"
        )
    table = "\n".join(lines) + "\n"
    (root / "comparison.md").write_text(table)
    (root / "comparison.json").write_text(json.dumps(runs, indent=2) + "\n")
    print(table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4), constrained_layout=True)
    for r, style in zip(runs, ("-o", "-s", "--^")):
        ax.plot(
            range(1, len(r["online"]) + 1), r["online"], style, ms=3, label=r["label"]
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Held-out online probe top-1 (%)",
        title="CIFAR-10, 10k train / 5k held-out, batch 64, seed 7",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(root / "comparison.png", dpi=180)


if __name__ == "__main__":
    main(sys.argv[1])
