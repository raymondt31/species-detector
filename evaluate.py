import torch
from dataset import VOCdataset
from torch.utils.data import DataLoader
from model import Yolov1
from utils import (
    plot_image,
    mean_average_precision,
    get_bboxes,
    load_checkpoint, 
)

seed = 123
torch.manual_seed(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # increase if Collab/Kaggle can handle it
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False # only relevant for GPU usage
CHECKPOINT_DIR = "/content/drive/MyDrive/Projects/species-detector/Checkpoints" 
BEST_CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/best_checkpoint.pth.tar"
S, B, C = 7, 2, 20

if __name__ == "__main__":

    # Intialize YOLO Model
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)

    # Load Model Weights
    best_checkpoint = torch.load(BEST_CHECKPOINT_FILE, map_location=DEVICE)
    load_checkpoint(best_checkpoint, model)

    # Set model to eval mode
    model.eval()

    # Build dataset and loader
    eval_dataset = VOCdataset(mode="val")

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Call get_bboxes and mean_average_precision
    pred_boxes, target_boxes = get_bboxes( 
        loader=eval_loader,
        model=model,
        iou_threshold=0.5,
        threshold=0.4,
        device=DEVICE,
    )

    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5)

    print(f"Eval mAP: {mean_avg_prec:.4f}")