"""First controlled Perch/SED/head experiment.

This script is intentionally narrower than the public 0.94+ ProtoSSM notebooks:
it runs the three strong public signal sources we already have locally and writes
branch CSVs with strict row/column alignment.

Branches:
- Perch direct target logits with genus proxies for unmapped classes.
- Tucker Arrants distilled SED ONNX folds.
- Konbu17 train-audio Perch linear head.

The Perch-direct branch is not a replacement for ProtoSSM. It is the first
smoke/probe experiment that validates model IO, paths, row ids, and rank blending
before we port the sequence model.

Local note: set `NUMBA_CACHE_DIR` before importing librosa because some conda
environments cannot cache numba-decorated librosa functions from site-packages.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import glob
import os
import re
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/birdclef_numba_cache")

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

import src.config as CFG
from src.blend_submissions import percentile_rank


N_WINDOWS = 12
WINDOW_SEC = 5
WINDOW_SAMPLES = CFG.SR * WINDOW_SEC
FILE_SAMPLES = CFG.SR * 60
PERCH_EMB_DIM = 1536

SED_N_MELS = 256
SED_N_FFT = 2048
SED_HOP = 512
SED_TOP_DB = 80

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--fallback_train_count", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/perch_sed_head_direct")
    parser.add_argument("--batch_files", type=int, default=16)
    parser.add_argument("--io_workers", type=int, default=4)
    parser.add_argument("--ort_threads", type=int, default=4)
    parser.add_argument("--skip_sed", action="store_true")
    parser.add_argument("--skip_head", action="store_true")
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Save Perch metadata, mapped logits, and embeddings for offline head training.",
    )
    parser.add_argument("--cnn_submission", default=None)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.50, 0.35, 0.15],
        help=(
            "Blend weights for perch_direct sed head. If --cnn_submission is set, "
            "provide four weights: perch_direct sed head cnn."
        ),
    )
    return parser.parse_args()


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/input")


def find_first_existing(candidates: list[Path], required_file: str | None = None) -> Path | None:
    for path in candidates:
        if required_file is None and path.exists():
            return path
        if required_file is not None and (path / required_file).exists():
            return path
    return None


def competition_data_dir() -> Path:
    env_path = os.environ.get("BIRDCLEF_COMPETITION_DATA_DIR")
    if env_path:
        return Path(env_path)
    if is_kaggle():
        found = find_first_existing(
            [
                Path("/kaggle/input/birdclef-2026"),
                Path("/kaggle/input/competitions/birdclef-2026"),
            ],
            required_file="taxonomy.csv",
        )
        if found is not None:
            return found
    return Path(CFG.DATA_DIR)


def find_kaggle_file(pattern: str) -> Path | None:
    if not is_kaggle():
        return None
    hits = sorted(Path("/kaggle/input").rglob(pattern))
    return hits[0] if hits else None


def perch_onnx_path() -> Path:
    local = Path(CFG.PERCH_ONNX_PATH)
    if local.exists():
        return local
    found = find_kaggle_file("perch_v2.onnx")
    if found is None:
        raise FileNotFoundError("perch_v2.onnx not found locally or under /kaggle/input")
    return found


def perch_labels_path() -> Path:
    for local in [Path(CFG.PERCH_LABELS_PATH), Path(CFG.PERCH_TF_LABELS_PATH)]:
        if local.exists():
            return local
    found = find_kaggle_file("labels.csv")
    if found is None:
        raise FileNotFoundError("Perch labels.csv not found locally or under /kaggle/input")
    return found


def sed_model_dir() -> Path:
    local = Path(CFG.SED_MODEL_DIR)
    if (local / "sed_fold0.onnx").exists():
        return local
    found = find_kaggle_file("sed_fold0.onnx")
    if found is None:
        raise FileNotFoundError("sed_fold0.onnx not found locally or under /kaggle/input")
    return found.parent


def train_audio_head_path() -> Path:
    local = Path(CFG.TRAIN_AUDIO_HEAD_PATH)
    if local.exists():
        return local
    found = find_kaggle_file("head_weights_train_audio.npz")
    if found is None:
        raise FileNotFoundError("head_weights_train_audio.npz not found locally or under /kaggle/input")
    return found


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))).astype(np.float32)


def parse_fname(name: str) -> dict[str, int | str]:
    match = FNAME_RE.match(name)
    if not match:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = match.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def read_60s(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != CFG.SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CFG.SR)
    if len(audio) < FILE_SAMPLES:
        audio = np.pad(audio, (0, FILE_SAMPLES - len(audio)))
    else:
        audio = audio[:FILE_SAMPLES]
    return audio.astype(np.float32, copy=False)


def get_soundscape_paths(input_dir: str | None, comp_dir: Path, fallback_train_count: int) -> list[Path]:
    soundscape_dir = Path(input_dir) if input_dir else comp_dir / "test_soundscapes"
    paths = sorted(soundscape_dir.glob("*.ogg"))
    if paths:
        print(f"Using input soundscapes: {soundscape_dir} ({len(paths)} files)")
        return paths

    fallback = sorted((comp_dir / "train_soundscapes").glob("*.ogg"))[:fallback_train_count]
    if not fallback:
        raise FileNotFoundError(
            f"No .ogg files found in {soundscape_dir} and no fallback train soundscapes found."
        )
    print(
        f"No test soundscapes in {soundscape_dir}; dry-run on "
        f"{len(fallback)} train soundscape files."
    )
    return fallback


def load_primary_labels(comp_dir: Path) -> list[str]:
    sample_path = comp_dir / "sample_submission.csv"
    if sample_path.exists():
        cols = pd.read_csv(sample_path, nrows=0).columns.tolist()
        return [str(c) for c in cols[1:]]

    taxonomy = pd.read_csv(comp_dir / "taxonomy.csv")
    return taxonomy["primary_label"].astype(str).tolist()


def load_perch_labels() -> pd.DataFrame:
    label_path = perch_labels_path()
    labels = pd.read_csv(label_path).reset_index()
    return labels.rename(
        columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}
    )


def build_perch_mapping(
    comp_dir: Path, primary_labels: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    taxonomy = pd.read_csv(comp_dir / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    bc_labels = load_perch_labels()
    no_label = len(bc_labels)

    mapping = taxonomy.merge(bc_labels, on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label).astype(int)
    lbl2bc = mapping.set_index("primary_label")["bc_index"]

    bc_indices = np.array([int(lbl2bc.loc[c]) for c in primary_labels], dtype=np.int32)
    mapped_mask = bc_indices != no_label
    mapped_pos = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc_idx = bc_indices[mapped_mask].astype(np.int32)

    label_to_idx = {label: i for i, label in enumerate(primary_labels)}
    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    unmapped_pos = np.where(~mapped_mask)[0].astype(np.int32)
    proxy_map: dict[int, list[int]] = {}
    proxy_taxa = {"Amphibia", "Insecta", "Aves"}

    unmapped_labels = [primary_labels[i] for i in unmapped_pos]
    unmapped_df = taxonomy[taxonomy["primary_label"].isin(unmapped_labels)].copy()
    for _, row in unmapped_df.iterrows():
        target = str(row["primary_label"])
        genus = str(row["scientific_name"]).split()[0]
        if class_name_map.get(target) not in proxy_taxa:
            continue
        hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(
                rf"^{re.escape(genus)}\s", na=False
            )
        ]
        if len(hits) > 0:
            proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

    print(f"Perch direct mapped classes: {len(mapped_pos)}/{len(primary_labels)}")
    print(f"Perch genus proxy classes: {len(proxy_map)}")
    print(f"Perch no-signal classes: {len(primary_labels) - len(mapped_pos) - len(proxy_map)}")
    return mapped_pos, mapped_bc_idx, proxy_map


def make_ort_session(path: str | Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_perch(
    paths: list[Path],
    comp_dir: Path,
    primary_labels: list[str],
    batch_files: int,
    io_workers: int,
    ort_threads: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    onnx_path = perch_onnx_path()
    mapped_pos, mapped_bc_idx, proxy_map = build_perch_mapping(comp_dir, primary_labels)
    session = make_ort_session(onnx_path, ort_threads)
    input_name = session.get_inputs()[0].name
    out_map = {out.name: i for i, out in enumerate(session.get_outputs())}
    if "label" not in out_map or "embedding" not in out_map:
        raise RuntimeError(f"Unexpected Perch outputs: {list(out_map)}")

    n_rows = len(paths) * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.zeros(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, len(primary_labels)), dtype=np.float32)
    embs = np.zeros((n_rows, PERCH_EMB_DIM), dtype=np.float32)

    print(f"Perch ONNX: {onnx_path}")
    t0 = time.time()
    wr = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as pool:
        next_paths = paths[:batch_files]
        future_audio = [pool.submit(read_60s, path) for path in next_paths]

        for start in range(0, len(paths), batch_files):
            batch_paths = next_paths
            batch_audio = [future.result() for future in future_audio]

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [pool.submit(read_60s, path) for path in next_paths]

            x = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            batch_row_start = wr
            for bi, path in enumerate(batch_paths):
                audio = batch_audio[bi]
                meta = parse_fname(path.name)
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = audio.reshape(
                    N_WINDOWS, WINDOW_SAMPLES
                )
                row_ids[wr:wr + N_WINDOWS] = [f"{path.stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites[wr:wr + N_WINDOWS] = meta["site"]
                hours[wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            outputs = session.run(None, {input_name: x})
            perch_logits = outputs[out_map["label"]].astype(np.float32)
            perch_emb = outputs[out_map["embedding"]].astype(np.float32)

            scores[batch_row_start:wr, mapped_pos] = perch_logits[:, mapped_bc_idx]
            for label_idx, bc_idxs in proxy_map.items():
                scores[batch_row_start:wr, label_idx] = perch_logits[:, bc_idxs].max(axis=1)
            embs[batch_row_start:wr] = perch_emb

            done_files = min(start + batch_files, len(paths))
            print(f"Perch: {done_files}/{len(paths)} files | {time.time() - t0:.1f}s")

            del x, outputs, perch_logits, perch_emb, batch_audio
            gc.collect()

    meta = pd.DataFrame(
        {"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours}
    )
    return meta, scores, embs


def run_sed(paths: list[Path], primary_labels: list[str], ort_threads: int) -> pd.DataFrame:
    sed_dir = sed_model_dir()
    fold_paths = sorted(
        glob.glob(os.path.join(sed_dir, "sed_fold*.onnx")),
        key=lambda p: int(re.search(r"sed_fold(\d+)", os.path.basename(p)).group(1)),
    )
    if len(fold_paths) != 5:
        raise FileNotFoundError(f"Expected 5 sed_fold*.onnx files in {sed_dir}")

    sessions = [make_ort_session(path, ort_threads) for path in fold_paths]
    print(f"SED folds: {[Path(path).name for path in fold_paths]}")

    rows: list[str] = []
    preds: list[np.ndarray] = []
    t0 = time.time()

    for file_i, path in enumerate(paths, 1):
        audio = read_60s(path)
        chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
        mel = audio_to_sed_mel(chunks)
        pred_sum = np.zeros((N_WINDOWS, len(primary_labels)), dtype=np.float32)

        for session in sessions:
            outputs = session.run(None, {session.get_inputs()[0].name: mel})
            clip_logits = outputs[0]
            frame_max = outputs[1].max(axis=1)
            pred_sum += 0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)

        pred = pred_sum / len(sessions)
        pred = gaussian_filter1d(pred, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        rows.extend([f"{path.stem}_{t}" for t in range(5, 65, 5)])
        preds.append(pred)

        if file_i == 1 or file_i % 25 == 0 or file_i == len(paths):
            print(f"SED: {file_i}/{len(paths)} files | {time.time() - t0:.1f}s")

    out = pd.DataFrame(np.concatenate(preds, axis=0), columns=primary_labels)
    out.insert(0, "row_id", rows)
    return out


def audio_to_sed_mel(chunks: np.ndarray) -> np.ndarray:
    mels = []
    for audio in chunks:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=CFG.SR,
            n_fft=SED_N_FFT,
            hop_length=SED_HOP,
            n_mels=SED_N_MELS,
            fmin=CFG.FMIN,
            fmax=CFG.FMAX,
            power=2.0,
        )
        mel = librosa.power_to_db(mel, top_db=SED_TOP_DB)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mels.append(mel)
    return np.stack(mels)[:, None].astype(np.float32)


def run_head(meta: pd.DataFrame, embs: np.ndarray, primary_labels: list[str]) -> pd.DataFrame:
    head_path = train_audio_head_path()
    weights = np.load(head_path, allow_pickle=True)
    w = weights["W"].astype(np.float32)
    b = weights["b"].astype(np.float32)
    trained_mask = weights["trained_mask"].astype(bool)
    if embs.shape[1] != w.shape[1] or w.shape[0] != len(primary_labels):
        raise RuntimeError(f"Head shape mismatch: embs={embs.shape}, W={w.shape}")

    logits = embs.astype(np.float32) @ w.T + b
    logits *= trained_mask.reshape(1, -1).astype(np.float32)
    probs = sigmoid(logits)
    out = pd.DataFrame(probs, columns=primary_labels)
    out.insert(0, "row_id", meta["row_id"].to_numpy())
    print(f"Head weights: {head_path}")
    print(f"Head trained classes: {int(trained_mask.sum())}/{len(primary_labels)}")
    return out


def write_submission(df: pd.DataFrame, output_path: Path, primary_labels: list[str]) -> None:
    expected_cols = ["row_id"] + primary_labels
    if list(df.columns) != expected_cols:
        raise RuntimeError("Submission columns are not in expected order.")
    values = df[primary_labels].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{output_path} contains NaN or Inf.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}: {df.shape}")


def align_branch(df: pd.DataFrame, row_ids: pd.Series, primary_labels: list[str], name: str) -> pd.DataFrame:
    if set(df["row_id"]) != set(row_ids):
        missing = sorted(set(row_ids) - set(df["row_id"]))
        extra = sorted(set(df["row_id"]) - set(row_ids))
        raise RuntimeError(f"{name} row mismatch: missing={missing[:3]}, extra={extra[:3]}")
    return df.set_index("row_id").loc[row_ids, primary_labels].reset_index()


def rank_blend(
    branches: list[tuple[str, pd.DataFrame]],
    weights: list[float],
    primary_labels: list[str],
) -> pd.DataFrame:
    if len(branches) != len(weights):
        raise ValueError(f"Got {len(branches)} branches and {len(weights)} weights.")
    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr < 0) or weights_arr.sum() <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    weights_arr = weights_arr / weights_arr.sum()

    row_ids = branches[0][1]["row_id"]
    blended = np.zeros((len(row_ids), len(primary_labels)), dtype=np.float64)
    for weight, (name, df) in zip(weights_arr, branches):
        aligned = align_branch(df, row_ids, primary_labels, name)
        ranks = percentile_rank(aligned[primary_labels].to_numpy(dtype=np.float32))
        blended += float(weight) * ranks
        print(f"Blend branch {name}: normalized_weight={weight:.4f}, rank_std={ranks.std():.6f}")

    out = pd.DataFrame(blended.astype(np.float32), columns=primary_labels)
    out.insert(0, "row_id", row_ids.to_numpy())
    return out


def load_cnn_submission(path: str, row_ids: pd.Series, primary_labels: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    return align_branch(df, row_ids, primary_labels, "cnn_submission")


def main() -> None:
    args = parse_args()
    start_time = time.time()

    comp_dir = competition_data_dir()
    print(f"Competition data: {comp_dir}")
    primary_labels = load_primary_labels(comp_dir)
    if len(primary_labels) != CFG.NUM_CLASSES:
        raise RuntimeError(f"Expected {CFG.NUM_CLASSES} labels, got {len(primary_labels)}")
    paths = get_soundscape_paths(args.input_dir, comp_dir, args.fallback_train_count)
    out_dir = Path(args.output_dir)

    meta, perch_scores, embs = run_perch(
        paths=paths,
        comp_dir=comp_dir,
        primary_labels=primary_labels,
        batch_files=args.batch_files,
        io_workers=args.io_workers,
        ort_threads=args.ort_threads,
    )
    if args.save_features:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta.to_csv(out_dir / "perch_meta.csv", index=False)
        np.save(out_dir / "perch_scores.npy", perch_scores.astype(np.float32))
        np.save(out_dir / "perch_embs.npy", embs.astype(np.float32))
        pd.Series(primary_labels, name="primary_label").to_csv(
            out_dir / "primary_labels.csv", index=False
        )
        print(f"Saved Perch features under {out_dir}")

    perch_df = pd.DataFrame(sigmoid(perch_scores), columns=primary_labels)
    perch_df.insert(0, "row_id", meta["row_id"].to_numpy())
    write_submission(perch_df, out_dir / "submission_perch_direct.csv", primary_labels)

    branches: list[tuple[str, pd.DataFrame]] = [("perch_direct", perch_df)]

    if not args.skip_sed:
        sed_df = run_sed(paths, primary_labels, args.ort_threads)
        sed_df = align_branch(sed_df, meta["row_id"], primary_labels, "sed")
        write_submission(sed_df, out_dir / "submission_sed.csv", primary_labels)
        branches.append(("sed", sed_df))

    if not args.skip_head:
        head_df = run_head(meta, embs, primary_labels)
        write_submission(head_df, out_dir / "submission_head.csv", primary_labels)
        branches.append(("head", head_df))

    if args.cnn_submission:
        cnn_df = load_cnn_submission(args.cnn_submission, meta["row_id"], primary_labels)
        write_submission(cnn_df, out_dir / "submission_cnn_aligned.csv", primary_labels)
        branches.append(("cnn", cnn_df))

    expected_weight_count = 4 if args.cnn_submission else len(branches)
    if len(args.weights) != expected_weight_count:
        raise ValueError(
            f"Expected {expected_weight_count} weights for branches {[name for name, _ in branches]}, "
            f"got {len(args.weights)}."
        )

    blend_df = rank_blend(branches, args.weights, primary_labels)
    write_submission(blend_df, out_dir / "submission.csv", primary_labels)

    stats = blend_df[primary_labels].to_numpy(dtype=np.float32)
    print(
        f"Done. rows={len(blend_df)}, files={len(paths)}, "
        f"blend_min={stats.min():.6f}, blend_max={stats.max():.6f}, "
        f"blend_std={stats.std():.6f}, time={(time.time() - start_time) / 60:.1f}min"
    )


if __name__ == "__main__":
    main()
