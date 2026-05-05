"""Score BirdCLEF submission-like CSVs against labeled train soundscape rows.

This is a submission-budget gate, not a hidden-LB replacement. It is meant for
Kaggle notebook or cloud-server runs where full data exists. For any model that
uses train_soundscapes_labels during training, pass out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except ImportError:  # pragma: no cover - used in lightweight local envs
    _sklearn_roc_auc_score = None

import src.config as CFG
from src.blend_submissions import percentile_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Prediction CSVs with row_id plus 234 class columns.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional rank-blend weights, one per input.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to train_soundscapes_labels.csv. Defaults to competition data dir.",
    )
    parser.add_argument(
        "--taxonomy",
        default=None,
        help="Path to taxonomy.csv. Defaults to competition data dir.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Optional sample_submission.csv to define class column order.",
    )
    parser.add_argument(
        "--out_json",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--out_class_csv",
        default=None,
        help="Optional per-class score report path.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Score only labeled rows present in predictions instead of failing.",
    )
    return parser.parse_args()


def competition_data_dir() -> Path:
    local = Path(CFG.DATA_DIR)
    if local.exists():
        return local
    for candidate in [
        Path("/kaggle/input/birdclef-2026"),
        Path("/kaggle/input/competitions/birdclef-2026"),
    ]:
        if (candidate / "taxonomy.csv").exists():
            return candidate
    return local


def default_comp_path(name: str) -> Path:
    return competition_data_dir() / name


def load_class_columns(sample_path: str | None, taxonomy_path: str | None) -> list[str]:
    if sample_path and Path(sample_path).exists():
        return pd.read_csv(sample_path, nrows=0).columns[1:].astype(str).tolist()

    default_sample = default_comp_path("sample_submission.csv")
    if default_sample.exists():
        return pd.read_csv(default_sample, nrows=0).columns[1:].astype(str).tolist()

    path = Path(taxonomy_path) if taxonomy_path else default_comp_path("taxonomy.csv")
    taxonomy = pd.read_csv(path)
    return taxonomy["primary_label"].astype(str).tolist()


def row_id_from_label_row(row: pd.Series) -> str:
    filename = str(row["filename"])
    stem = filename[:-4] if filename.endswith(".ogg") else Path(filename).stem
    end = row["end"]
    if isinstance(end, str) and ":" in end:
        end_time = int(pd.to_timedelta(end).total_seconds())
    else:
        end_time = int(float(end))
    return f"{stem}_{end_time}"


def split_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    labels = []
    for part in str(value).split(";"):
        label = part.strip()
        if label:
            labels.append(label)
    return labels


def load_targets(labels_path: str | None, class_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(labels_path) if labels_path else default_comp_path("train_soundscapes_labels.csv")
    labels_df = pd.read_csv(path)
    required = {"filename", "end", "primary_label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    labels_df = labels_df.copy()
    labels_df["row_id"] = labels_df.apply(row_id_from_label_row, axis=1)
    row_ids = labels_df["row_id"].drop_duplicates().tolist()
    row_to_idx = {row_id: i for i, row_id in enumerate(row_ids)}
    col_to_idx = {label: i for i, label in enumerate(class_cols)}

    y = np.zeros((len(row_ids), len(class_cols)), dtype=np.uint8)
    unknown_labels: set[str] = set()
    for _, row in labels_df.iterrows():
        r = row_to_idx[row["row_id"]]
        for label in split_labels(row["primary_label"]):
            c = col_to_idx.get(label)
            if c is None:
                unknown_labels.add(label)
                continue
            y[r, c] = 1

    if unknown_labels:
        preview = sorted(unknown_labels)[:10]
        raise ValueError(f"Labels not found in class columns: {preview}")

    target_rows = pd.DataFrame({"row_id": row_ids})
    return target_rows, y


def read_prediction(path: str, class_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"{path} has no row_id column")
    if df["row_id"].duplicated().any():
        dup = df.loc[df["row_id"].duplicated(), "row_id"].iloc[0]
        raise ValueError(f"{path} has duplicated row_id: {dup}")
    missing = sorted(set(class_cols) - set(df.columns))
    extra = sorted(set(df.columns) - (set(class_cols) | {"row_id"}))
    if missing or extra:
        raise ValueError(
            f"{path} column mismatch. Missing first={missing[:5]}, extra first={extra[:5]}"
        )
    values = df[class_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN or Inf predictions")
    return df[["row_id"] + class_cols]


def align_predictions(
    dfs: list[pd.DataFrame],
    target_rows: pd.DataFrame,
    class_cols: list[str],
    allow_missing: bool,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    row_ids = target_rows["row_id"].tolist()
    common = set(row_ids)
    for df in dfs:
        common &= set(df["row_id"])

    if allow_missing:
        kept_rows = [row_id for row_id in row_ids if row_id in common]
    else:
        kept_rows = row_ids
        for i, df in enumerate(dfs):
            missing = sorted(set(row_ids) - set(df["row_id"]))
            if missing:
                raise ValueError(
                    f"Input {i} is missing labeled rows. First missing={missing[:5]}. "
                    "Use --allow_missing only for partial smoke tests."
                )

    aligned = []
    for df in dfs:
        arr = df.set_index("row_id").loc[kept_rows, class_cols].to_numpy(dtype=np.float32)
        aligned.append(arr)

    target_idx = {row_id: i for i, row_id in enumerate(row_ids)}
    y_idx = [target_idx[row_id] for row_id in kept_rows]
    return aligned, np.asarray(y_idx, dtype=np.int64), kept_rows


def validate_weights(inputs: list[str], weights: list[float] | None) -> np.ndarray:
    if weights is None:
        weights_arr = np.ones(len(inputs), dtype=np.float64)
    else:
        if len(weights) != len(inputs):
            raise ValueError(f"Got {len(inputs)} inputs but {len(weights)} weights")
        weights_arr = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_arr).all() or np.any(weights_arr < 0):
        raise ValueError(f"Invalid weights: {weights_arr.tolist()}")
    total = float(weights_arr.sum())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return weights_arr / total


def blend_predictions(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if len(preds) == 1:
        return preds[0]
    out = np.zeros_like(preds[0], dtype=np.float64)
    for pred, weight in zip(preds, weights):
        out += float(weight) * percentile_rank(pred)
    return out.astype(np.float32)


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(y_true, y_score))

    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("ROC-AUC requires at least one positive and one negative")
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def score_macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_cols: list[str],
    taxonomy_path: str | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    aucs = []
    for i, label in enumerate(class_cols):
        positives = int(y_true[:, i].sum())
        negatives = int(len(y_true) - positives)
        auc = np.nan
        if positives > 0 and negatives > 0:
            auc = roc_auc_score_binary(y_true[:, i], y_score[:, i])
            aucs.append(auc)
        rows.append(
            {
                "primary_label": label,
                "positives": positives,
                "negatives": negatives,
                "auc": auc,
                "evaluable": bool(np.isfinite(auc)),
            }
        )

    class_report = pd.DataFrame(rows)

    taxonomy_file = Path(taxonomy_path) if taxonomy_path else default_comp_path("taxonomy.csv")
    if taxonomy_file.exists():
        taxonomy = pd.read_csv(taxonomy_file)
        taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
        class_report = class_report.merge(
            taxonomy[["primary_label", "class_name"]],
            on="primary_label",
            how="left",
        )

    summary: dict[str, object] = {
        "macro_auc": float(np.nanmean(class_report["auc"].to_numpy(dtype=np.float64))),
        "rows": int(len(y_true)),
        "classes": int(len(class_cols)),
        "evaluable_classes": int(class_report["evaluable"].sum()),
        "positive_labels": int(y_true.sum()),
    }

    if "class_name" in class_report.columns:
        taxon_auc = (
            class_report[class_report["evaluable"]]
            .groupby("class_name")["auc"]
            .mean()
            .sort_index()
            .to_dict()
        )
        summary["taxon_auc"] = {str(k): float(v) for k, v in taxon_auc.items()}

    return summary, class_report


def main() -> None:
    args = parse_args()
    class_cols = load_class_columns(args.sample, args.taxonomy)
    target_rows, y_all = load_targets(args.labels, class_cols)
    dfs = [read_prediction(path, class_cols) for path in args.inputs]
    preds, y_idx, kept_rows = align_predictions(
        dfs=dfs,
        target_rows=target_rows,
        class_cols=class_cols,
        allow_missing=args.allow_missing,
    )
    weights = validate_weights(args.inputs, args.weights)
    y_true = y_all[y_idx]
    y_score = blend_predictions(preds, weights)
    summary, class_report = score_macro_auc(
        y_true=y_true,
        y_score=y_score,
        class_cols=class_cols,
        taxonomy_path=args.taxonomy,
    )
    summary.update(
        {
            "inputs": args.inputs,
            "weights": weights.tolist(),
            "scored_rows": len(kept_rows),
            "warning": (
                "This is a train-soundscape proxy score. Use OOF predictions for any "
                "component trained on train_soundscapes_labels."
            ),
        }
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.out_class_csv:
        out_path = Path(args.out_class_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        class_report.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
