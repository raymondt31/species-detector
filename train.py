# Ready for testing!

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCdataset
from utils import(
    intersection_over_union,
    non_max_suppression, 
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint, # worry about this after the model is able to overfit a small dataset
    load_checkpoint
)

from loss import YOLOLoss

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # increase if Collab/Kaggle can handle it; for small set testing, this must be less than the size of the data
WEIGHT_DECAY = 0 # for testing that overfitting works
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False # only relevant for GPU usage
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar" # for testing purposes

# Allows us to input both the img and bounding boxes
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    # Note that this will fail if data augmentation is implemented
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

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

    loss_fn = YOLOLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # TODO: test on a small dataset first before committing to a large one

    train_dataset = VOCdataset(transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5 # box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()



    