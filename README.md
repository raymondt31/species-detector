# Species Detector

*A PyTorch implementation of YOLOv1 to be used in building a custom wildlife species detector.*

## Motivation

I built this project to better understand the architecture and training process behind YOLO object detectors.

## Status

**Phase 1 (current):** Building and tuning a raw YOLOv1 object detector on Pascal VOC 2012 to validate the architecture, loss function, and training pipeline end-to-end.

**Phase 2 (next):** Swap in a species-specific dataset and class set to turn this into an actual wildlife/species detector.

The model, loss function, and training loop are implemented from scratch in pytorch, using the original YOLOv1 paper and educational implementations as references.

| File | Purpose |
|---|---|
| `model.py` | Darknet-style CNN backbone + fully connected detection head (`Yolov1`) |
| `dataset.py` | `VOCdataset` — loads Pascal VOC 2012 via `torchvision`, converts raw boxes into the S×S×(C+B·5) grid target tensor YOLO expects |
| `loss.py` | `YOLOLoss` — the multi-part YOLOv1 loss (box coordinates, objectness, no-object, and class loss) |
| `train.py` | Training loop: Adam optimizer, per-epoch mAP tracking, checkpointing |
| `utils.py` / `test_utils.py` | IOU, non-max suppression, mean average precision, converting grid cells back to bounding boxes, plotting, checkpoint save/load |

## Results

Currently without pretrained backbone: 

- Best validation mAP: 0.047
- Best epoch: 126
- Training loss @ best epoch: 2.86

Pending results with pretrained backbone...

## Architecture

The backbone follows the original 24-conv-layer Darknet config (kernel size, filters, stride, padding per layer), ending in two fully connected layers that predict an `S × S × (C + B·5)` tensor. Defaults: `S=7` grid cells, `B=2` boxes per cell, `C=20` classes (VOC).

## Setup

```bash
git clone https://github.com/raymondt31/species-detector.git
cd species-detector
pip install torch torchvision tqdm
```

Pascal VOC 2012 needs to be available locally — set `download=True` in `dataset.py` on first run, or fetch it manually into `./data`.

## Training

```bash
python train.py
```

## Reference

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)