import torch
import os
import wandb
import math
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from model import Yolov1
from dataset import VOCdataset
from evaluate import VOCEvalSet, evaluate_model
from utils import(
    save_checkpoint,
    load_checkpoint
)
from augment import (
    train_transform,
    val_transform,
)

from loss import YOLOLoss

seed = 123
torch.manual_seed(seed)

# --- optim parameters --- 
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4 # for the neck and FC head
MIN_LR_FRAC = 0.01 # LR decays to this fraction of starting
LR_WARMUP_ITERS = 1000 # linear ramp for ~1 epoch
FREEZE_EPOCHS = 3 
WEIGHT_DECAY = 1e-4
EPOCHS = 45 # '07+'12 is ~3x the old dataset, so train 1/3 of the time

BATCH_SIZE = 16 # increase if Collab/Kaggle can handle it
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False # only relevant for GPU usage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "/content/drive/MyDrive/Projects/species-detector/Checkpoints" 
CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/checkpoint.pth.tar"
BEST_CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/best_checkpoint.pth.tar"
SAVE_MODEL = True
CHECKPOINT_FREQ = 5
MAP_FREQ = 5
# --------------------------

def lr_factor(it, total_iters):
    """NEW: Calc multiplier to the LR at iteration 'it', then cosine decays instead of step"""

    # Decide how many iters to spend on warmup
    warmup = min(LR_WARMUP_ITERS, max(1, total_iters // 10))
   
    if it < warmup:
        return (it + 1) / warmup

    # How far through cosine decay
    t = min(
        1.0,
        (it - warmup) / max(1, total_iters - warmup)
    )

    # return cosine decay via formula
    return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (
        1 + math.cos(math.pi * t)
    )

def train_fn(train_loader, model, optimizer, loss_fn, base_lrs, start_iter, total_iters):
    model.train() 
    loop = tqdm(train_loader, leave=True) # progress bar
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):

        # set learning rate of this iteration
        factor = lr_factor(start_iter + batch_idx, total_iters)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        
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

    avg_loss = sum(mean_loss)/len(mean_loss)
    print(f"Mean loss was {avg_loss}")
    return avg_loss

def main():

    wandb.init(
        project="yolov1-species-detector",
        name="[insert_name]",
        # id= [run_ID], UNCOMMENT when resuming a run
        # resume="allow",
    )

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    # Fix one optimizer for the whole run
    head_params = list(model.neck.parameters()) + list(model.fcs.parameters())
    base_lrs = [BACKBONE_LR, HEAD_LR]

    optimizer = optim.Adam([
        {"params": list(model.darknet.parameters()), "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    loss_fn = YOLOLoss()

    curr_epoch = 0
    best_map = 0.0

    # safer alternative to manually flipping SAVE_MODEL on and off
    if os.path.exists(CHECKPOINT_FILE):
        l_checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        curr_epoch = l_checkpoint.get("epoch", -1) + 1
        best_map = l_checkpoint.get("best_map", 0.0)
        if curr_epoch >= FREEZE_EPOCHS:
            model.unfreeze_backbone()
        load_checkpoint(l_checkpoint, model, optimizer)
        print(f"Starting from epoch {curr_epoch}, best mAP so far: {best_map:.4f}")
    else:
        print("No checkpoint found -- starting from epoch 0")

    # Concate '07 and '12
    train_dataset = ConcatDataset([
        VOCdataset(year="2007", mode="trainval", transform=train_transform),
        VOCdataset(year="2012", mode="trainval", transform=train_transform),
    ])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    val_dataset = VOCEvalSet(root="./data", year="2007", split="test", transform=val_transform)

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

    iters_per_epoch = len(train_loader)
    total_iters = EPOCHS * iters_per_epoch

    for epoch in range(curr_epoch, EPOCHS):
        print(f"\nEpoch {epoch}/{EPOCHS-1}")

        # Unfreeze weights and adjust hyperparams after warmup
        if epoch == FREEZE_EPOCHS:
            print("Warmup complete. Unfreezing backbone and adjusting learning rate.")
            model.unfreeze_backbone()

        train_loss = train_fn(
            train_loader, 
            model, 
            optimizer, 
            loss_fn,
            base_lrs,
            epoch * iters_per_epoch,
            total_iters,
        ) 

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }

        # Train epoch then score, so the last epoch isn't neglected
        if (epoch + 1) % MAP_FREQ == 0 or epoch == EPOCHS-1:
            current_map = evaluate_model(model, val_loader, val_dataset, DEVICE)

            print(f"Validation mAP: {current_map:.4f}")
            log["val_mAP"] = current_map

            # Change: best_map is just kept for monitoring purposes, we will report the final epoch instead
            if current_map > best_map:
                best_map = current_map
                print(f"New best mAP: {best_map:.4f}")

                if SAVE_MODEL:
                    save_checkpoint({
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_map": best_map
                    }, filename=BEST_CHECKPOINT_FILE)

        wandb.log(log)

        if SAVE_MODEL and ((epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == EPOCHS - 1):
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_map": best_map
            }, filename=CHECKPOINT_FILE)

if __name__ == "__main__":
    main()