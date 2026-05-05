"""Train an out-of-fold custom Perch head from cached soundscape features.

Run this only on Kaggle or the cloud server with full data. It consumes features
saved by `src.perch_sed_head_experiment --save_features` and writes:

- `oof_head_v2.csv`: group-OFF predictions for offline scoring;
- `head_v2_weights.npz`: full-data linear weights for later inference integration;
- `head_v2_report.json`: training metadata.

The model is intentionally linear. The goal is to add a cheap, controlled,
orthogonal rank signal over Perch embeddings before spending submissions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import src.config as CFG
from src.offline_score import load_class_columns, load_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_dir",
        required=True,
        help="Directory containing perch_meta.csv, perch_embs.npy, and primary_labels.csv.",
    )
    parser.add_argument("--output_dir", default="outputs/perch_head_v2")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--min_pos", type=int, default=2)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument(
        "--use_scores",
        action="store_true",
        help="Concatenate mapped Perch logits from perch_scores.npy to embeddings.",
    )
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))).astype(np.float32)


def load_features(features_dir: Path, use_scores: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    meta_path = features_dir / "perch_meta.csv"
    emb_path = features_dir / "perch_embs.npy"
    labels_path = features_dir / "primary_labels.csv"
    if not meta_path.exists() or not emb_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"{features_dir} must contain perch_meta.csv, perch_embs.npy, primary_labels.csv"
        )

    meta = pd.read_csv(meta_path)
    embs = np.load(emb_path).astype(np.float32)
    primary_labels = pd.read_csv(labels_path)["primary_label"].astype(str).tolist()
    if len(meta) != len(embs):
        raise ValueError(f"Feature row mismatch: meta={len(meta)}, embs={len(embs)}")

    x = embs
    if use_scores:
        scores_path = features_dir / "perch_scores.npy"
        if not scores_path.exists():
            raise FileNotFoundError(scores_path)
        scores = np.load(scores_path).astype(np.float32)
        if len(scores) != len(embs):
            raise ValueError(f"Score row mismatch: scores={len(scores)}, embs={len(embs)}")
        x = np.concatenate([embs, scores], axis=1).astype(np.float32)

    return meta, x, primary_labels


def align_targets(
    meta: pd.DataFrame,
    class_cols: list[str],
    labels_path: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    target_rows, y_all = load_targets(labels_path, class_cols)
    row_to_target = {row_id: i for i, row_id in enumerate(target_rows["row_id"])}
    keep = []
    target_idx = []
    for i, row_id in enumerate(meta["row_id"].astype(str)):
        j = row_to_target.get(row_id)
        if j is not None:
            keep.append(i)
            target_idx.append(j)
    if not keep:
        raise RuntimeError("No cached feature rows match train_soundscapes_labels row_id values")
    return np.asarray(keep, dtype=np.int64), y_all[np.asarray(target_idx, dtype=np.int64)]


def fit_one_class(
    x_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    max_iter: int,
) -> LogisticRegression | None:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives <= 0 or negatives <= 0:
        return None
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        solver="liblinear",
        random_state=CFG.SEED,
    )
    model.fit(x_train, y_train)
    return model


def scaled_weights_to_raw(
    scaler: StandardScaler,
    coef_scaled: np.ndarray,
    intercept_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = scaler.scale_.astype(np.float64)
    mean = scaler.mean_.astype(np.float64)
    coef_raw = coef_scaled.astype(np.float64) / scale.reshape(1, -1)
    intercept_raw = intercept_scaled.astype(np.float64) - coef_raw @ mean
    return coef_raw.astype(np.float32), intercept_raw.astype(np.float32)


def train_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_cols: list[str],
    n_splits: int,
    min_pos: int,
    C: float,
    max_iter: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    unique_groups = np.unique(groups)
    splits = min(n_splits, len(unique_groups))
    if splits < 2:
        raise RuntimeError("Need at least two soundscape files for OOF training")

    oof = np.full(y.shape, 0.5, dtype=np.float32)
    fold_reports: list[dict[str, object]] = []
    gkf = GroupKFold(n_splits=splits)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), 1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr_idx]).astype(np.float32)
        x_va = scaler.transform(x[va_idx]).astype(np.float32)
        trained = 0

        for c, label in enumerate(class_cols):
            positives = int(y[tr_idx, c].sum())
            negatives = int(len(tr_idx) - positives)
            if positives < min_pos or negatives <= 0:
                continue
            model = fit_one_class(x_tr, y[tr_idx, c], C=C, max_iter=max_iter)
            if model is None:
                continue
            logits = model.decision_function(x_va).astype(np.float32)
            oof[va_idx, c] = sigmoid(logits)
            trained += 1

        report = {
            "fold": fold,
            "train_rows": int(len(tr_idx)),
            "valid_rows": int(len(va_idx)),
            "trained_classes": trained,
        }
        print(json.dumps(report, sort_keys=True))
        fold_reports.append(report)

    return oof, fold_reports


def train_full_weights(
    x: np.ndarray,
    y: np.ndarray,
    class_cols: list[str],
    min_pos: int,
    C: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    weights = np.zeros((len(class_cols), x.shape[1]), dtype=np.float32)
    bias = np.zeros(len(class_cols), dtype=np.float32)
    trained_mask = np.zeros(len(class_cols), dtype=bool)

    for c, label in enumerate(class_cols):
        positives = int(y[:, c].sum())
        negatives = int(len(y) - positives)
        if positives < min_pos or negatives <= 0:
            continue
        model = fit_one_class(x_scaled, y[:, c], C=C, max_iter=max_iter)
        if model is None:
            continue
        coef_raw, intercept_raw = scaled_weights_to_raw(
            scaler,
            model.coef_.astype(np.float32),
            model.intercept_.astype(np.float32),
        )
        weights[c] = coef_raw[0]
        bias[c] = intercept_raw[0]
        trained_mask[c] = True

    return weights, bias, trained_mask, scaler


def write_prediction_csv(
    path: Path,
    row_ids: pd.Series,
    preds: np.ndarray,
    class_cols: list[str],
) -> None:
    out = pd.DataFrame(preds.astype(np.float32), columns=class_cols)
    out.insert(0, "row_id", row_ids.astype(str).to_numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, x_all, feature_labels = load_features(features_dir, args.use_scores)
    class_cols = load_class_columns(args.sample, args.taxonomy)
    if class_cols != feature_labels:
        raise ValueError("Feature primary_labels.csv does not match competition class columns")

    keep_idx, y = align_targets(meta, class_cols, args.labels)
    x = x_all[keep_idx]
    meta_labeled = meta.iloc[keep_idx].reset_index(drop=True)
    groups = meta_labeled["filename"].astype(str).to_numpy()
    row_ids = meta_labeled["row_id"].astype(str)

    print(
        f"Training Perch head v2: rows={len(x)}, dim={x.shape[1]}, "
        f"positive_labels={int(y.sum())}, files={len(np.unique(groups))}"
    )
    oof, fold_reports = train_oof(
        x=x,
        y=y,
        groups=groups,
        class_cols=class_cols,
        n_splits=args.n_splits,
        min_pos=args.min_pos,
        C=args.C,
        max_iter=args.max_iter,
    )
    write_prediction_csv(output_dir / "oof_head_v2.csv", row_ids, oof, class_cols)

    weights, bias, trained_mask, scaler = train_full_weights(
        x=x,
        y=y,
        class_cols=class_cols,
        min_pos=args.min_pos,
        C=args.C,
        max_iter=args.max_iter,
    )
    np.savez_compressed(
        output_dir / "head_v2_weights.npz",
        W=weights,
        b=bias,
        trained_mask=trained_mask,
        class_cols=np.asarray(class_cols),
        use_scores=np.asarray(bool(args.use_scores)),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )

    report = {
        "features_dir": str(features_dir),
        "rows": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "files": int(len(np.unique(groups))),
        "positive_labels": int(y.sum()),
        "trained_classes_full": int(trained_mask.sum()),
        "min_pos": int(args.min_pos),
        "C": float(args.C),
        "use_scores": bool(args.use_scores),
        "folds": fold_reports,
        "warning": "Score oof_head_v2.csv with src.offline_score before any submission.",
    }
    (output_dir / "head_v2_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
