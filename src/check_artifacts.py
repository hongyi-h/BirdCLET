"""Lightweight integrity checks for BirdCLEF 2026 artifacts.

Examples:
    python -m src.check_artifacts
    python -m src.check_artifacts --check_precomputed
    python -m src.check_artifacts --check_models
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import src.config as CFG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_precomputed", action="store_true")
    parser.add_argument("--check_models", action="store_true")
    parser.add_argument("--check_external", action="store_true",
                        help="Check local Perch/SED/reference model datasets")
    parser.add_argument("--model_dir", default=CFG.CHECKPOINT_DIR)
    return parser.parse_args()


def build_label_map():
    taxonomy = pd.read_csv(os.path.join(CFG.DATA_DIR, "taxonomy.csv"))
    labels = sorted(taxonomy["primary_label"].astype(str).tolist())
    return {label: i for i, label in enumerate(labels)}


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def check_taxonomy(errors):
    tax_path = os.path.join(CFG.DATA_DIR, "taxonomy.csv")
    require(os.path.exists(tax_path), f"Missing {tax_path}", errors)
    if not os.path.exists(tax_path):
        return

    taxonomy = pd.read_csv(tax_path)
    labels = taxonomy["primary_label"].astype(str).tolist()
    label_map = build_label_map()
    require(len(labels) == CFG.NUM_CLASSES, f"taxonomy rows != {CFG.NUM_CLASSES}: {len(labels)}", errors)
    require(len(set(labels)) == len(labels), "taxonomy primary_label has duplicates", errors)
    require(sorted(labels) == list(label_map.keys()), "label map order mismatch", errors)

    sample_path = os.path.join(CFG.DATA_DIR, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample_cols = pd.read_csv(sample_path, nrows=0).columns.tolist()
        require(sample_cols[0] == "row_id", "sample_submission first column is not row_id", errors)
        require(set(sample_cols[1:]) == set(labels), "sample_submission columns differ from taxonomy", errors)

    print(f"taxonomy: {len(labels)} classes")
    print(taxonomy["class_name"].value_counts().to_string())


def check_csv_data():
    train_path = os.path.join(CFG.DATA_DIR, "train.csv")
    sc_path = os.path.join(CFG.DATA_DIR, "train_soundscapes_labels.csv")
    tax_path = os.path.join(CFG.DATA_DIR, "taxonomy.csv")
    if not all(os.path.exists(path) for path in [train_path, sc_path, tax_path]):
        return

    train = pd.read_csv(train_path)
    soundscape = pd.read_csv(sc_path)
    taxonomy = pd.read_csv(tax_path)
    focal_species = set(train["primary_label"].astype(str))
    all_species = set(taxonomy["primary_label"].astype(str))
    sc_species = set()
    for labels in soundscape["primary_label"].astype(str):
        sc_species.update(label.strip() for label in labels.split(";") if label.strip())

    print(f"train.csv rows: {len(train)}, focal species: {len(focal_species)}")
    print(
        f"train_soundscapes_labels rows: {len(soundscape)}, "
        f"files: {soundscape['filename'].nunique()}, species: {len(sc_species)}"
    )
    print(f"taxonomy classes without focal audio: {len(all_species - focal_species)}")


def check_precomputed_subset(name, errors, required=True):
    subset_dir = os.path.join(CFG.PRECOMPUTED_DIR, name)
    if not os.path.isdir(subset_dir):
        print(f"precomputed/{name}: missing")
        if required:
            errors.append(f"precomputed/{name} is missing")
        return

    manifest_path = os.path.join(subset_dir, "manifest.csv")
    require(os.path.exists(manifest_path), f"precomputed/{name} missing manifest.csv", errors)
    if not os.path.exists(manifest_path):
        return

    manifest = pd.read_csv(manifest_path)
    require({"mel_path", "target_path", "labels", "domain"}.issubset(manifest.columns),
            f"precomputed/{name} manifest missing required columns", errors)
    missing = 0
    bad_mels = 0
    bad_targets = 0
    for _, row in manifest.head(2000).iterrows():
        mel_path = str(row["mel_path"])
        target_path = str(row["target_path"])
        if not os.path.isabs(mel_path):
            mel_path = os.path.join(subset_dir, mel_path)
        if not os.path.isabs(target_path):
            target_path = os.path.join(subset_dir, target_path)
        if not os.path.exists(mel_path) or not os.path.exists(target_path):
            missing += 1
            continue
        mel = np.load(mel_path)
        target = np.load(target_path)
        if len(mel.shape) != 2 or mel.shape[0] != CFG.N_MELS:
            bad_mels += 1
        if target.shape != (CFG.NUM_CLASSES,):
            bad_targets += 1

    require(missing == 0, f"precomputed/{name}: missing files in manifest sample: {missing}", errors)
    require(bad_mels == 0, f"precomputed/{name}: bad mel shape in manifest sample: {bad_mels}", errors)
    require(bad_targets == 0, f"precomputed/{name}: bad target shape in manifest sample: {bad_targets}", errors)
    print(f"precomputed/{name}: {len(manifest)} rows")


def check_unlabeled_precomputed(errors):
    subset_dir = os.path.join(CFG.PRECOMPUTED_DIR, "soundscape_unlabeled")
    if not os.path.isdir(subset_dir):
        errors.append("precomputed/soundscape_unlabeled is missing")
        print("precomputed/soundscape_unlabeled: missing")
        return

    manifest_path = os.path.join(subset_dir, "manifest.csv")
    require(os.path.exists(manifest_path), "precomputed/soundscape_unlabeled missing manifest.csv", errors)
    if not os.path.exists(manifest_path):
        return

    manifest = pd.read_csv(manifest_path)
    require({"mel_path", "filename", "start", "site"}.issubset(manifest.columns),
            "precomputed/soundscape_unlabeled manifest missing required columns", errors)
    missing = 0
    bad_mels = 0
    for _, row in manifest.head(2000).iterrows():
        mel_path = str(row["mel_path"])
        if not os.path.isabs(mel_path):
            mel_path = os.path.join(subset_dir, mel_path)
        if not os.path.exists(mel_path):
            missing += 1
            continue
        mel = np.load(mel_path)
        if len(mel.shape) != 2 or mel.shape[0] != CFG.N_MELS:
            bad_mels += 1

    require(missing == 0, f"precomputed/soundscape_unlabeled: missing mel files in manifest sample: {missing}", errors)
    require(bad_mels == 0, f"precomputed/soundscape_unlabeled: bad mel shape in manifest sample: {bad_mels}", errors)
    print(f"precomputed/soundscape_unlabeled: {len(manifest)} rows")


def check_specialist_mapping(model_dir, errors):
    mapping_path = os.path.join(model_dir, "specialist_mapping.npy")
    if not os.path.exists(mapping_path):
        print("specialist_mapping.npy: missing")
        return

    taxonomy = pd.read_csv(os.path.join(CFG.DATA_DIR, "taxonomy.csv"))
    tax_map = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))
    label_map = build_label_map()
    species_by_index = {idx: sp for sp, idx in label_map.items()}
    mapping = np.load(mapping_path, allow_pickle=True).item()
    species = [str(sp) for sp in mapping["nonbird_species"]]
    indices = [int(i) for i in mapping["nonbird_indices"]]

    require(len(species) == len(indices), "specialist mapping species/indices length mismatch", errors)
    for sp, idx in zip(species, indices):
        require(0 <= idx < CFG.NUM_CLASSES, f"specialist index out of range: {idx}", errors)
        require(species_by_index.get(idx) == sp, f"specialist index mismatch: {sp} -> {idx}", errors)
        require(tax_map.get(sp) in CFG.NON_BIRD_CLASSES, f"specialist species is not non-bird: {sp}", errors)

    print(f"specialist mapping: {len(species)} non-bird classes")


def check_models(model_dir, errors):
    paths = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
    print(f"onnx models in {model_dir}: {len(paths)}")
    if not paths:
        return

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"onnxruntime unavailable, skipping load checks: {exc}")
        return

    for path in paths:
        try:
            sess = ort.InferenceSession(path)
            print(f"  {os.path.basename(path)} -> output {sess.get_outputs()[0].shape}")
        except Exception as exc:
            errors.append(f"failed to load ONNX {path}: {exc}")


def check_external_artifacts(errors):
    print("external data/model paths:")
    print(f"  competition data: {CFG.DATA_DIR}")
    print(f"  external data: {CFG.EXTERNAL_DATA_DIR}")
    print(f"  pretrained models: {CFG.PRETRAINED_MODEL_DIR}")

    require(os.path.exists(CFG.PERCH_ONNX_PATH), f"Missing Perch ONNX: {CFG.PERCH_ONNX_PATH}", errors)
    require(
        os.path.exists(CFG.PERCH_LABELS_PATH) or os.path.exists(CFG.PERCH_TF_LABELS_PATH),
        f"Missing Perch labels: {CFG.PERCH_LABELS_PATH} or {CFG.PERCH_TF_LABELS_PATH}",
        errors,
    )

    sed_folds = sorted(glob.glob(os.path.join(CFG.SED_MODEL_DIR, "sed_fold*.onnx")))
    require(len(sed_folds) == 5, f"Expected 5 SED folds in {CFG.SED_MODEL_DIR}, found {len(sed_folds)}", errors)

    perch_meta_arrays = os.path.join(CFG.PERCH_META_DIR, "full_perch_arrays.npz")
    perch_meta_table = os.path.join(CFG.PERCH_META_DIR, "full_perch_meta.parquet")
    require(os.path.exists(perch_meta_arrays), f"Missing Perch meta arrays: {perch_meta_arrays}", errors)
    require(os.path.exists(perch_meta_table), f"Missing Perch meta table: {perch_meta_table}", errors)

    tf_saved_model = os.path.join(CFG.PERCH_TF_MODEL_DIR, "saved_model.pb")
    require(os.path.exists(tf_saved_model), f"Missing Perch TF SavedModel: {tf_saved_model}", errors)

    if os.path.exists(CFG.TRAIN_AUDIO_HEAD_PATH):
        print(f"train-audio head: {CFG.TRAIN_AUDIO_HEAD_PATH}")
    else:
        print(f"train-audio head: missing optional {CFG.TRAIN_AUDIO_HEAD_PATH}")

    print(f"perch onnx: {CFG.PERCH_ONNX_PATH}")
    print(f"sed folds: {len(sed_folds)}")


def main():
    args = parse_args()
    errors = []
    check_taxonomy(errors)
    check_csv_data()
    if args.check_precomputed:
        check_precomputed_subset("focal", errors, required=True)
        check_precomputed_subset("soundscape_labeled", errors, required=True)
        check_unlabeled_precomputed(errors)
        check_precomputed_subset("pseudo", errors, required=False)
    if args.check_models:
        check_specialist_mapping(args.model_dir, errors)
        check_models(args.model_dir, errors)
    if args.check_external:
        check_external_artifacts(errors)

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
