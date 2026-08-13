# Ready for testing!

import torch
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from model import Yolov1
from dataset import VOCdataset
from utils import(
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint
)

from loss import YOLOLoss

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # increase if Collab/Kaggle can handle it
SMALL_BATCH_SIZE = 4 # for small subset testing
WEIGHT_DECAY = 5e-4
EPOCHS = 135 
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False # only relevant for GPU usage
CHECKPOINT_DIR = "/content/drive/MyDrive/Projects/species-detector/Checkpoints" 
CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/checkpoint.pth.tar"
BEST_CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/best_checkpoint.pth.tar"
SAVE_MODEL = True
CHECKPOINT_FREQ = 5
MAP_FREQ = 5
USE_FULL_DATASET = True

# Allows us to input both the img and bounding boxes
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    # Note that this will fail if data augmentation is implemented
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

train_transform = Compose([
    transforms.Resize((448, 448)), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), 
])

val_transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True) # progress bar
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        
        # move data to GPU/CPU
        x, y = x.to(DEVICE), y.to(DEVICE)

        # forward pass
        out = model(x)

        # loss calcs
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        # Pytorch natively accumulates gradients; need to manually zero it out after each pass
        optimizer.zero_grad()

        # back prop
        loss.backward()

        # update weights
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2, # only drop LR after two mAP checks with no improvment
    )

    loss_fn = YOLOLoss()

    curr_epoch = 0
    best_map = 0.0

    # safer alternative to manually flipping SAVE_MODEL on and off
    if os.path.exists(CHECKPOINT_FILE):
        l_checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        load_checkpoint(l_checkpoint, model, optimizer)
        curr_epoch = l_checkpoint.get("epoch", -1) + 1
        best_map = l_checkpoint.get("best_map", 0.0)
        print(f"Starting from epoch {curr_epoch}, best mAP so far: {best_map:.4f}")
    else:
        print("No checkpoint found -- starting from epoch 0")

    train_dataset = VOCdataset(mode="train", transform=train_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    val_dataset = VOCdataset(mode="val", transform=val_transform)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False, # unecesary
        drop_last=False # unecesary
    )
        
    # Small Dataset for Initial Testing:
    # small_dataset = Subset(train_dataset, indices=list(range(8)))
    
    # small_loader = DataLoader(
    #     dataset=small_dataset,
    #     batch_size=SMALL_BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     shuffle=True,
    #     drop_last=True
    # )

    active_loader = train_loader # if USE_FULL_DATASET else small_loader

    for epoch in range(curr_epoch, EPOCHS):
        print(f"\nEpoch {epoch}/{EPOCHS-1}")

        # Reduce frequency that mAP is recorded to reduce overhead
        if epoch % MAP_FREQ == 0 or epoch == EPOCHS-1:
            pred_boxes, target_boxes = get_bboxes(
                val_loader, model, iou_threshold=0.5, threshold=0.01, device=DEVICE
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5 # box_format="midpoint"
            )
            mean_avg_prec = float(mean_avg_prec)

            print(f"Validation mAP: {mean_avg_prec:.4f}")

            # Update best
            if mean_avg_prec > best_map:
                best_map = mean_avg_prec
                print(f"New best mAP: {best_map:.4f}")
                best_checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_map": best_map,
                }
                save_checkpoint(best_checkpoint, filename=BEST_CHECKPOINT_FILE)

            scheduler.step(mean_avg_prec)

        train_fn(active_loader, model, optimizer, loss_fn) 

        if SAVE_MODEL and (epoch % CHECKPOINT_FREQ == 0 or epoch == EPOCHS-1):
            s_checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
            }
            save_checkpoint(s_checkpoint, filename=CHECKPOINT_FILE)

if __name__ == "__main__":
    main()