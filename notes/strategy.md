# Competition Strategy

## Winning Patterns from BirdCLEF 2023-2025
- Log-mel spectrogram + CNN/SED backbone is the dominant paradigm
- Pseudo-labeling unlabeled soundscapes is the single biggest domain-shift gain
- Background mixing (focal calls onto soundscape backgrounds) bridges domain gap
- Class-balanced sampling essential under macro AUC
- Small diverse ensembles (2-4 models) with rank averaging
- ONNX/OpenVINO FP16 export for CPU inference

## MVP Pipeline (Phase 1)
Goal: submit a working notebook with a real model, get a baseline score.

### Training (local GPU)
- Input: 32kHz → 128 mel bins, fmin=20Hz, fmax=16kHz, 5s crops
- Data: train_audio (primary_label) + labeled train_soundscapes (multi-label)
- Model: EfficientNet-B0 (ImageNet pretrained) + SED attention head → 234 sigmoid
- Loss: BCE
- Basic augmentations: gain jitter, time shift, SpecAugment
- Export: ONNX FP16

### Inference (Kaggle notebook, CPU)
- Load ONNX model via onnxruntime
- Decode ogg → mel spectrogram → 12 x 5s segments → batch predict
- Output submission.csv

## Iteration Plan (Phase 2+)
1. Add background mixing augmentation (mix focal onto soundscape backgrounds)
2. Pseudo-label unlabeled train_soundscapes (1-2 rounds)
3. Upgrade backbone to EfficientNetV2-S or NFNet-L0
4. Train non-bird specialist model
5. Ensemble 2-3 models with rank averaging
6. Post-processing: temporal smoothing, class-wise temperature scaling

## Key Design Decisions
- Why EfficientNet-B0 for MVP: fast training, fast CPU inference, proven in BirdCLEF
- Why SED attention head: calls are temporally sparse within 5s, attention focuses on active regions
- Why ONNX: fastest CPU inference, well-supported, easy export
- Why not Transformer/AST: CNN consistently wins on mel spectrograms for this task, and much faster on CPU
