"""
run_pipeline.py
===============
CLI orchestrator: runs the full DeepWhale pipeline end-to-end by importing
and calling each src module's run() function.

Usage:
    python run_pipeline.py [--blocks N] [--steps all|1,2,3,4,5,6]

Steps:
    1  data_collection
    2  feature_engineering
    3  labeling
    4  clustering
    5  classification
    6  anomaly_detection
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "data"

RAW_CSV       = DATA_DIR / "raw_whale_transactions.csv"
FEATURES_CSV  = DATA_DIR / "address_features.csv"
LABELED_CSV   = DATA_DIR / "labeled_addresses.csv"
CLUSTERED_CSV = DATA_DIR / "clustered_addresses.csv"
ANOMALY_CSV   = DATA_DIR / "anomaly_scores.csv"
MODEL_PKL     = MODELS_DIR / "whale_classifier.pkl"


def _step(name: str, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="DeepWhale end-to-end pipeline")
    parser.add_argument(
        "--blocks", type=int, default=500,
        help="Blocks to scan in data_collection (default: 500)",
    )
    parser.add_argument(
        "--steps", type=str, default="all",
        help="Comma-separated step numbers to run, e.g. '2,3,4', or 'all'",
    )
    args = parser.parse_args()

    if args.steps == "all":
        active = {1, 2, 3, 4, 5, 6}
    else:
        active = {int(s.strip()) for s in args.steps.split(",")}

    timings: dict[str, float] = {}

    if 1 in active:
        from src import data_collection
        t0 = time.time()
        data_collection.run(output_csv=RAW_CSV, num_blocks=args.blocks)
        timings["data_collection"] = time.time() - t0

    if 2 in active:
        from src import feature_engineering
        t0 = time.time()
        feature_engineering.run(raw_csv=RAW_CSV, features_csv=FEATURES_CSV)
        timings["feature_engineering"] = time.time() - t0

    if 3 in active:
        from src import labeling
        t0 = time.time()
        labeling.run(features_csv=FEATURES_CSV, labeled_csv=LABELED_CSV)
        timings["labeling"] = time.time() - t0

    if 4 in active:
        from src import clustering
        t0 = time.time()
        clustering.run(
            features_csv=FEATURES_CSV,
            output_csv=CLUSTERED_CSV,
            figures_dir=FIGURES_DIR,
            labeled_csv=LABELED_CSV,
        )
        timings["clustering"] = time.time() - t0

    if 5 in active:
        from src import classification
        t0 = time.time()
        classification.run(
            clustered_csv=CLUSTERED_CSV,
            model_pkl=MODEL_PKL,
            figures_dir=FIGURES_DIR,
        )
        timings["classification"] = time.time() - t0

    if 6 in active:
        from src import anomaly_detection
        t0 = time.time()
        anomaly_detection.run(
            features_csv=FEATURES_CSV,
            anomaly_csv=ANOMALY_CSV,
            figures_dir=FIGURES_DIR,
            raw_csv=RAW_CSV,
            clustered_csv=CLUSTERED_CSV,
            labeled_csv=LABELED_CSV,
        )
        timings["anomaly_detection"] = time.time() - t0

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    for step, elapsed in timings.items():
        print(f"  {step:<25} {elapsed:6.1f}s")

    print("\nOutput files:")
    for label, path in [
        ("raw transactions", RAW_CSV),
        ("address features", FEATURES_CSV),
        ("labeled addresses", LABELED_CSV),
        ("clustered addresses", CLUSTERED_CSV),
        ("anomaly scores", ANOMALY_CSV),
        ("model bundle", MODEL_PKL),
    ]:
        if path.exists():
            print(f"  [OK] {label:<22} {path.stat().st_size/1024:.1f} KB  {path}")
        else:
            print(f"  [--] {label:<22} (not generated)")


if __name__ == "__main__":
    main()
