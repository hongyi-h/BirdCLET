"""Rank-blend BirdCLEF submission CSV files.

This is intentionally small and strict: each input must have the same row ids
and class columns. Scores are converted to per-class percentile ranks before
weighted averaging because ROC-AUC depends on ordering, and heterogeneous
branches are poorly calibrated against each other.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Submission CSV paths to blend.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Blend weights, one per input CSV.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Optional sample_submission.csv used only to enforce final column order.",
    )
    return parser.parse_args()


def validate_args(inputs: list[str], weights: list[float]) -> np.ndarray:
    if len(inputs) < 2:
        raise ValueError("At least two input submissions are required for blending.")
    if len(inputs) != len(weights):
        raise ValueError(f"Got {len(inputs)} inputs but {len(weights)} weights.")
    weights_arr = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_arr).all():
        raise ValueError("Weights contain NaN or Inf.")
    if np.any(weights_arr < 0):
        raise ValueError("Weights must be non-negative.")
    total = float(weights_arr.sum())
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    return weights_arr / total


def read_submission(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "row_id" not in df.columns:
        raise ValueError(f"{csv_path} has no row_id column.")
    if df["row_id"].duplicated().any():
        dup = df.loc[df["row_id"].duplicated(), "row_id"].iloc[0]
        raise ValueError(f"{csv_path} has duplicated row_id: {dup}")
    if df.shape[1] <= 1:
        raise ValueError(f"{csv_path} has no class columns.")
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{csv_path} contains NaN or Inf scores.")
    return df


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return per-column percentile ranks in [1/n, 1]."""
    return pd.DataFrame(values).rank(axis=0, method="average", pct=True).to_numpy(
        dtype=np.float32
    )


def align_to_reference(df: pd.DataFrame, ref: pd.DataFrame, path: str) -> pd.DataFrame:
    ref_rows = ref["row_id"].tolist()
    ref_cols = ref.columns.tolist()

    if set(df["row_id"]) != set(ref["row_id"]):
        missing = sorted(set(ref["row_id"]) - set(df["row_id"]))
        extra = sorted(set(df["row_id"]) - set(ref["row_id"]))
        raise ValueError(
            f"{path} row_id mismatch. Missing first={missing[:3]}, extra first={extra[:3]}"
        )

    if set(df.columns) != set(ref.columns):
        missing = sorted(set(ref.columns) - set(df.columns))
        extra = sorted(set(df.columns) - set(ref.columns))
        raise ValueError(
            f"{path} column mismatch. Missing first={missing[:3]}, extra first={extra[:3]}"
        )

    return df.set_index("row_id").loc[ref_rows, ref_cols[1:]].reset_index()


def enforce_sample_order(out: pd.DataFrame, sample_path: str | None) -> pd.DataFrame:
    if sample_path is None:
        return out
    sample = pd.read_csv(sample_path, nrows=1)
    sample_cols = sample.columns.tolist()
    if set(out.columns) != set(sample_cols):
        missing = sorted(set(sample_cols) - set(out.columns))
        extra = sorted(set(out.columns) - set(sample_cols))
        raise ValueError(
            f"Output/sample column mismatch. Missing first={missing[:3]}, extra first={extra[:3]}"
        )
    return out[sample_cols]


def main() -> None:
    args = parse_args()
    weights = validate_args(args.inputs, args.weights)

    dfs = [read_submission(path) for path in args.inputs]
    ref = dfs[0]
    aligned = [ref] + [
        align_to_reference(df, ref, path) for df, path in zip(dfs[1:], args.inputs[1:])
    ]

    blend = np.zeros((len(ref), ref.shape[1] - 1), dtype=np.float64)
    for weight, df, path in zip(weights, aligned, args.inputs):
        ranks = percentile_rank(df.iloc[:, 1:].to_numpy(dtype=np.float32))
        blend += float(weight) * ranks
        print(f"Added {path}: weight={weight:.4f}, rank_std={float(ranks.std()):.6f}")

    out = pd.DataFrame(blend.astype(np.float32), columns=ref.columns[1:])
    out.insert(0, "row_id", ref["row_id"].to_numpy())
    out = enforce_sample_order(out, args.sample)

    output_path = Path(args.output)
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {output_path}: rows={out.shape[0]}, cols={out.shape[1]}, "
        f"score_std={float(blend.std()):.6f}"
    )


if __name__ == "__main__":
    main()
