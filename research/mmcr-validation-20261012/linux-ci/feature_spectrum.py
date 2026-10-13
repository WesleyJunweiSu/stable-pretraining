"""Spectral collapse diagnostics for cached frozen-probe features.

Per-dimension standard deviations cannot rule out low-rank collapse. This reads
the feature caches written by ``frozen_probe.py --cache-dir`` and reports the
effective rank (Roy & Vetterli, 2007) of the centered validation features.

Usage: python feature_spectrum.py FROZEN_PROBE_JSON OUTPUT_JSON
"""

import json
import sys
from pathlib import Path

import numpy as np


def spectrum(features: np.ndarray) -> dict:
    """Summarize the singular-value spectrum of centered features."""
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    p = singular / singular.sum()
    variance = singular**2 / (singular**2).sum()
    cumulative = np.cumsum(variance)
    return {
        "dimensions": int(features.shape[1]),
        "effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
        "components_for_90pct_variance": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_for_99pct_variance": int(np.searchsorted(cumulative, 0.99) + 1),
        "top_component_variance_fraction": float(variance[0]),
    }


def main(probe_json: str, output: str) -> None:
    report = json.loads(Path(probe_json).read_text())
    result = {}
    for name, snapshot in report["snapshots"].items():
        with np.load(snapshot["feature_cache_path"], allow_pickle=False) as cached:
            result[name] = spectrum(cached["val_x"])
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
