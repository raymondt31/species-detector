# Species Detector

*A PyTorch implementation of YOLOv1 to be used in building a custom wildlife species detector.*

## Motivation

I built this project to better understand the architecture and training process behind YOLO object detectors.

## Status

**Phase 1 (complete):** A YOLOv1-style detector trained and evaluated end-to-end on Pascal VOC, covering the model, loss function, data augmentation, training loop, and a VOC-protocol evaluation pipeline.

**Phase 2 (planned):** Swap in a species-specific dataset and class set to turn this into an actual wildlife/species detector.

The model, loss function, and training loop are implemented from scratch in pytorch, using the original YOLOv1 paper and educational implementations as references. The only pretrained component is the ImageNet ResNet50 backbone weights from `torchvision`.

## Results

**44.3 mAP@0.5 on Pascal VOC 2007 test** (4,952 images), trained on VOC 2007 + 2012 trainval (16,551 images) for 30 epochs with an ImageNet-pretrained ResNet50 backbone.

| Epochs trained | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| mAP@0.5 (VOC07 test) | 0.13 | 0.29 | 0.36 | 0.43 | 0.44 | 0.443 |

## Architecture

- **Backbone:** ResNet50 trunk (`torchvision`, `IMAGENET1K_V2` weights) with the average pool and classifier removed. A 448×448 input becomes a 2048×14×14 feature map.
- **Neck:** three conv blocks (conv → batch norm → LeakyReLU 0.1) that map the features onto the 7×7 grid: 1×1 conv 2048→512, 3×3 stride-2 conv 512→1024 (14×14 → 7×7), then a 3×3 conv 1024→1024.
- **Head:** flatten → Linear(1024·7·7 → 2048) → Dropout(0.5) → LeakyReLU → Linear(2048 → 7·7·30), reshaped to `S × S × (C + B·5)` with `S=7`, `B=2`, `C=20`.
- **Loss:** the YOLOv1 sum-squared-error loss (box coordinates, objectness, no-object, class) with `λ_coord = 5` and `λ_noobj = 0.5`.

## Training

- **Data:** VOC 2007 trainval + VOC 2012 trainval, resized to 448×448.
- **Augmentation:** box-aware horizontal flip, random scale and translate (up to 20%, boxes less than 25% visible are dropped), and color jitter.
- **Fine-tuning schedule:** the backbone is frozen for the first 3 epochs, then unfrozen.
- **Optimizer:** Adam (weight decay 1e-4) with differential learning rates: 1e-5 for the backbone and 1e-4 for the neck and head.
- **LR schedule:** 1,000-iteration linear warmup, then cosine decay to 1% of the base rate over the planned 45 epochs.
- **Other:** batch size 16, seed 123. Checkpoints are saved every 5 epochs, and training resumes automatically from the latest checkpoint. Runs are tracked in Weights & Biases.

## Evaluation
 
`evaluate.py` follows the VOC detection protocol:
 
- Ground truth comes from the raw VOC annotations, not from the 7×7 training targets. The training targets keep only one object per grid cell, so scoring against them silently drops overlapping objects.
- Objects marked `difficult` are ignored: they do not count as missed detections and detections matching them are not penalized.
- Every cell's two boxes are decoded, with score = box confidence × class probability for all 20 classes. Candidates above 0.005 are kept (top 100 per image), then per-class non-maximum suppression (IoU 0.5) is applied.
- AP per class is computed at IoU 0.5 from the precision/recall curve with a monotonic precision envelope, then averaged over the 20 classes.
The IoU, NMS, and mAP functions are in `utils.py` and covered by unit tests in `test_utils.py`.

## Files
 
| File | Purpose |
|---|---|
| `model.py` | `Yolov1`: ResNet50 trunk, conv neck, and fully connected detection head, with backbone freeze/unfreeze |
| `augment.py` | Box-aware transforms (flip, scale/translate, color jitter) and the train/val transform pipelines |
| `dataset.py` | `VOCdataset`: loads Pascal VOC via `torchvision` and converts boxes into the S×S×(C+B·5) grid target tensor |
| `loss.py` | `YOLOLoss`: the multi-part YOLOv1 loss |
| `train.py` | Training loop: freeze/unfreeze, warmup + cosine LR, periodic mAP evaluation, checkpointing, W&B logging |
| `evaluate.py` | VOC-protocol evaluation (`evaluate_model`) and a script that scores a saved checkpoint |
| `utils.py` | IoU, non-max suppression, mean average precision, checkpoint save/load |
| `test_utils.py` | Unit tests for IoU, NMS, mAP, and prediction decoding |
| `log_to_csv.py`, `training_logs/` | Raw logs and CSVs from the earlier from-scratch runs |

## Setup

```bash
git clone https://github.com/raymondt31/species-detector.git
cd species-detector
pip install torch torchvision tqdm
```

The data is loaded from `./data` through `torchvision.datasets.VOCDetection`: VOC 2007 trainval and test, and VOC 2012 trainval. On a fresh environment, set `download=True` in **both** `dataset.py` and `evaluate.py` for the first run, then set it back to `False`.
 
`CHECKPOINT_DIR` in `train.py` and `CKPT` in `evaluate.py` point to a Google Drive folder (I train on Colab).

## Usage

```bash
python train.py                    # train; resumes from the latest checkpoint if one exists
python evaluate.py                 # score the checkpoint at CKPT on VOC2007 test
```

## Reference

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)