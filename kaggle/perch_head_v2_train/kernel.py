import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = SR * 60
N_WINDOWS = 12
MIN_POS = 2
N_SPLITS = 5
C = 0.1
MAX_ITER = 500


def find_one(pattern):
    hits = sorted(INPUT.rglob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import onnxruntime as ort
except ImportError:
    wheel_hits = sorted(INPUT.rglob("onnxruntime-*.whl"))
    if wheel_hits:
        install_target = str(wheel_hits[0])
    else:
        install_target = "onnxruntime"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", install_target], check=True)
    import onnxruntime as ort


def competition_dir():
    candidates = [
        Path("/kaggle/input/birdclef-2026"),
        Path("/kaggle/input/competitions/birdclef-2026"),
    ]
    candidates.extend(path.parent for path in INPUT.rglob("train_soundscapes_labels.csv"))
    print("Input roots:", [str(p) for p in sorted(INPUT.iterdir())[:30]])
    print("taxonomy hits:", [str(p) for p in sorted(INPUT.rglob("taxonomy.csv"))[:20]])
    for path in candidates:
        if (path / "taxonomy.csv").exists() and (path / "train_soundscapes_labels.csv").exists():
            return path
    raise FileNotFoundError("Could not find BirdCLEF competition data under /kaggle/input")


BASE = competition_dir()
TRAIN_SCAPE_DIR = BASE / "train_soundscapes"
TAXONOMY_PATH = BASE / "taxonomy.csv"
LABELS_PATH = BASE / "train_soundscapes_labels.csv"
SAMPLE_PATH = BASE / "sample_submission.csv"
PERCH_ONNX = find_one("perch_v2.onnx")
PERCH_LABELS = find_one("labels.csv")

print("BASE", BASE)
print("PERCH_ONNX", PERCH_ONNX)
print("PERCH_LABELS", PERCH_LABELS)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def end_seconds(end):
    if isinstance(end, str) and ":" in end:
        return int(pd.to_timedelta(end).total_seconds())
    return int(float(end))


def row_id(stem, end):
    return f"{stem}_{end_seconds(end)}"


def read_60s(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if len(audio) < FILE_SAMPLES:
        audio = np.pad(audio, (0, FILE_SAMPLES - len(audio)))
    else:
        audio = audio[:FILE_SAMPLES]
    return audio.astype(np.float32, copy=False)


def load_class_cols():
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH, nrows=0).columns[1:].astype(str).tolist()
    tax = pd.read_csv(TAXONOMY_PATH)
    return tax["primary_label"].astype(str).tolist()


def load_targets(class_cols):
    labels = pd.read_csv(LABELS_PATH)
    row_ids = []
    for _, r in labels.iterrows():
        stem = str(r["filename"])
        if stem.endswith(".ogg"):
            stem = stem[:-4]
        row_ids.append(row_id(stem, r["end"]))
    labels = labels.copy()
    labels["row_id"] = row_ids
    target_rows = labels["row_id"].drop_duplicates().tolist()
    row_to_i = {r: i for i, r in enumerate(target_rows)}
    col_to_i = {c: i for i, c in enumerate(class_cols)}
    y = np.zeros((len(target_rows), len(class_cols)), dtype=np.uint8)
    for _, r in labels.iterrows():
        rr = row_to_i[r["row_id"]]
        for label in str(r["primary_label"]).split(";"):
            label = label.strip()
            if label:
                y[rr, col_to_i[label]] = 1
    return labels, target_rows, y


def load_perch_label_df():
    df = pd.read_csv(PERCH_LABELS).reset_index().rename(columns={"index": "bc_index"})
    if "scientific_name" not in df.columns:
        for candidate in ["inat2024_fsd50k", "label", "name"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "scientific_name"})
                break
    if "scientific_name" not in df.columns:
        raise ValueError(f"Cannot find scientific_name in {PERCH_LABELS}: {df.columns.tolist()}")
    return df


def build_perch_mapping(class_cols):
    tax = pd.read_csv(TAXONOMY_PATH)
    tax["primary_label"] = tax["primary_label"].astype(str)
    bc = load_perch_label_df()
    no_label = len(bc)
    merged = tax.merge(bc[["bc_index", "scientific_name"]], on="scientific_name", how="left")
    merged["bc_index"] = merged["bc_index"].fillna(no_label).astype(int)
    lookup = merged.set_index("primary_label")["bc_index"]
    idx = np.array([int(lookup.loc[c]) for c in class_cols], dtype=np.int32)
    mapped_pos = np.where(idx != no_label)[0].astype(np.int32)
    mapped_bc = idx[mapped_pos].astype(np.int32)
    print(f"Perch mapped classes: {len(mapped_pos)}/{len(class_cols)}")
    return mapped_pos, mapped_bc


def make_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def extract_features(paths, class_cols, batch_rows=192):
    mapped_pos, mapped_bc = build_perch_mapping(class_cols)
    sess = make_session(PERCH_ONNX)
    inp = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    out_map = {name: i for i, name in enumerate(out_names)}
    if "label" not in out_map or "embedding" not in out_map:
        raise RuntimeError(out_names)

    all_rows, all_files, audio_rows = [], [], []
    t0 = time.time()
    for n, path in enumerate(paths, 1):
        audio = read_60s(path)
        chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
        audio_rows.append(chunks)
        all_rows.extend([row_id(path.stem, t) for t in range(5, 65, 5)])
        all_files.extend([path.name] * N_WINDOWS)
        if n == 1 or n % 10 == 0 or n == len(paths):
            print(f"read {n}/{len(paths)} files | {time.time() - t0:.1f}s")

    x_audio = np.concatenate(audio_rows, axis=0).astype(np.float32)
    scores = np.zeros((len(x_audio), len(class_cols)), dtype=np.float32)
    embs = np.zeros((len(x_audio), 1536), dtype=np.float32)
    for start in range(0, len(x_audio), batch_rows):
        end = min(start + batch_rows, len(x_audio))
        outs = sess.run(None, {inp: x_audio[start:end]})
        labels = outs[out_map["label"]].astype(np.float32)
        emb = outs[out_map["embedding"]].astype(np.float32)
        scores[start:end, mapped_pos] = labels[:, mapped_bc]
        embs[start:end] = emb
        print(f"perch {end}/{len(x_audio)} rows | {time.time() - t0:.1f}s")

    meta = pd.DataFrame({"row_id": all_rows, "filename": all_files})
    return meta, scores, embs


def percentile_rank(values):
    return pd.DataFrame(values).rank(axis=0, method="average", pct=True).to_numpy(np.float32)


def macro_auc(y_true, y_score, class_cols):
    rows, aucs = [], []
    for i, label in enumerate(class_cols):
        pos = int(y_true[:, i].sum())
        neg = int(len(y_true) - pos)
        auc = np.nan
        if pos > 0 and neg > 0:
            auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
            aucs.append(auc)
        rows.append({"primary_label": label, "positives": pos, "negatives": neg, "auc": auc})
    report = pd.DataFrame(rows)
    return float(np.nanmean(report["auc"].to_numpy(np.float64))), report


def fit_model(x_train, y_train):
    if y_train.sum() <= 0 or y_train.sum() >= len(y_train):
        return None
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=MAX_ITER,
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def raw_weights(scaler, coef, intercept):
    coef_raw = coef.astype(np.float64) / scaler.scale_.reshape(1, -1)
    intercept_raw = intercept.astype(np.float64) - coef_raw @ scaler.mean_
    return coef_raw.astype(np.float32), intercept_raw.astype(np.float32)


class_cols = load_class_cols()
labels, target_rows, y_all = load_targets(class_cols)
files = sorted(set(labels["filename"].astype(str)))
paths = [TRAIN_SCAPE_DIR / f for f in files]
paths = [p for p in paths if p.exists()]
print(f"labeled files requested={len(files)} found={len(paths)}")

meta, perch_scores, embs = extract_features(paths, class_cols)
meta.to_csv(WORK / "perch_meta.csv", index=False)
np.save(WORK / "perch_scores.npy", perch_scores.astype(np.float32))
np.save(WORK / "perch_embs.npy", embs.astype(np.float32))
pd.Series(class_cols, name="primary_label").to_csv(WORK / "primary_labels.csv", index=False)

row_to_feature = {r: i for i, r in enumerate(meta["row_id"].astype(str))}
row_to_target = {r: i for i, r in enumerate(target_rows)}
keep_feat, keep_tgt = [], []
for r in target_rows:
    if r in row_to_feature:
        keep_feat.append(row_to_feature[r])
        keep_tgt.append(row_to_target[r])
if not keep_feat:
    raise RuntimeError("No labeled rows matched extracted features")

base_probs = sigmoid(perch_scores[keep_feat])
x = np.concatenate([embs[keep_feat], perch_scores[keep_feat]], axis=1).astype(np.float32)
y = y_all[keep_tgt]
row_ids = [target_rows[i] for i in keep_tgt]
groups = meta.iloc[keep_feat]["filename"].astype(str).to_numpy()
print(f"train rows={len(x)} dim={x.shape[1]} files={len(np.unique(groups))} positives={int(y.sum())}")

oof = base_probs.copy()
gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
fold_reports = []
for fold, (tr, va) in enumerate(gkf.split(x, y, groups), 1):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x[tr]).astype(np.float32)
    x_va = scaler.transform(x[va]).astype(np.float32)
    trained = 0
    for c, label in enumerate(class_cols):
        if int(y[tr, c].sum()) < MIN_POS:
            continue
        model = fit_model(x_tr, y[tr, c])
        if model is None:
            continue
        oof[va, c] = sigmoid(model.decision_function(x_va).astype(np.float32))
        trained += 1
    rep = {"fold": fold, "train_rows": int(len(tr)), "valid_rows": int(len(va)), "trained_classes": trained}
    print(json.dumps(rep, sort_keys=True))
    fold_reports.append(rep)

oof_auc, class_report = macro_auc(y, oof, class_cols)
print("OOF macro AUC", oof_auc)
class_report.to_csv(WORK / "oof_class_auc.csv", index=False)
oof_df = pd.DataFrame(oof.astype(np.float32), columns=class_cols)
oof_df.insert(0, "row_id", row_ids)
oof_df.to_csv(WORK / "oof_head_v2.csv", index=False)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x).astype(np.float32)
W = np.zeros((len(class_cols), x.shape[1]), dtype=np.float32)
b = np.zeros(len(class_cols), dtype=np.float32)
trained_mask = np.zeros(len(class_cols), dtype=bool)
for c, label in enumerate(class_cols):
    if int(y[:, c].sum()) < MIN_POS:
        continue
    model = fit_model(x_scaled, y[:, c])
    if model is None:
        continue
    coef_raw, intercept_raw = raw_weights(scaler, model.coef_, model.intercept_)
    W[c] = coef_raw[0]
    b[c] = intercept_raw[0]
    trained_mask[c] = True

np.savez_compressed(
    WORK / "head_v2_weights.npz",
    W=W,
    b=b,
    trained_mask=trained_mask,
    class_cols=np.asarray(class_cols),
    use_scores=np.asarray(True),
    input_dim=np.asarray(x.shape[1]),
)
report = {
    "oof_macro_auc": oof_auc,
    "rows": int(len(x)),
    "files": int(len(np.unique(groups))),
    "positive_labels": int(y.sum()),
    "trained_classes_full": int(trained_mask.sum()),
    "folds": fold_reports,
    "note": "Training notebook output. Attach head_v2_weights.npz to a separate submission notebook before using daily submissions.",
}
(WORK / "head_v2_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
