import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
 
from augment import val_transform
from model import Yolov1
from utils import non_max_suppression, mean_average_precision

CKPT = "/content/drive/MyDrive/Projects/species-detector/Checkpoints/checkpoint.pth.tar"  # same file train.py writes
DATA_ROOT = "./data"
YEAR, SPLIT = "2007", "test"
BATCH_SIZE = 16
NUM_WORKERS = 2
SCORE_THRESH = 0.005    
MAX_CANDIDATES = 100    
S, B, C = 7, 2, 20
 
CLASS_TO_IDX = {
    'horse': 0, 'person': 1, 'bottle': 2, 'dog': 3, 'tvmonitor': 4, 'car': 5, 'aeroplane': 6, 'bicycle': 7,
    'boat': 8, 'chair': 9, 'diningtable': 10, 'pottedplant': 11, 'train': 12, 'cat': 13, 'sofa': 14,
    'bird': 15, 'sheep': 16, 'motorbike': 17, 'bus': 18, 'cow': 19,
}

class VOCEvalSet(Dataset):
    def __init__(self, root=DATA_ROOT, year="2007", split="test", transform=val_transform, limit=None):
        self.raw = torchvision.datasets.VOCDetection(
            root=root, year=year, image_set=split, download=False # Change this if running eval on a fresh session
        )
        self.transform = transform

        # Holds every ground truth object
        self.true_boxes = []
        for idx in range(len(self.raw)):    
            img, annotation = self.raw[idx]
            W, H = img.size
            
            for obj in annotation["annotation"]["object"]:
                bb = obj["bndbox"]
                x1, y1, x2, y2 = float(bb["xmin"]), float(bb["ymin"]), float(bb["xmax"]), float(bb["ymax"])
                self.true_boxes.append([
                    idx, 
                    CLASS_TO_IDX[obj["name"]], 
                    1.0, 
                    (x1 + x2) / 2 / W, 
                    (y1 + y2) / 2 / H,
                    (x2 - x1) / W, 
                    (y2 - y1) / H, 
                    int(obj["difficult"])
                ])

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        img, _ = self.raw[idx]
        return self.transform(img, [])[0], idx

@torch.no_grad()
def evaluate_model(model, loader, dataset, device):
    was_training = model.training
    model.eval()

    # to help with cell-relative to image-relative conversions 
    rows, cols = torch.meshgrid(
        torch.arange(S), 
        torch.arange(S), 
        indexing="ij"
    )
 
    pred_boxes = []  # [img_idx, class, score, x_center, y_center, w, h]

    # Iterate over all images in batches
    for x, idxs in tqdm(loader):
        preds = model(x.to(device)).reshape(
            -1, S, S, C + 5 * B
        ).cpu()

        # Iterate over images per batch
        for n, img_idx in enumerate(idxs.tolist()):
            candidates = []

            # Iterate over bounding box predictors
            for b in range(B):  

                # Track the beginning of each bounding boxes's 5 values                                                 
                o = C + 5 * b

                # Box confidence * class probability
                scores = preds[n, :, :, o:o + 1] * preds[n, :, :, :C]

                # cell relative to image relative             
                boxes = torch.stack([(preds[n, :, :, o + 1] + cols) / S,          
                                     (preds[n, :, :, o + 2] + rows) / S,
                                     preds[n, :, :, o + 3].clamp(min=0),
                                     preds[n, :, :, o + 4].clamp(min=0)], dim=-1)

                # Grid coords and class index of valid predictions above set threshold
                for r, c, k in (scores > SCORE_THRESH).nonzero().tolist():
                    candidates.append([k, scores[r, c, k].item(), *boxes[r, c].tolist()])

            candidates = sorted(candidates, key=lambda cand: cand[1], reverse=True)[:MAX_CANDIDATES]
            
            for box in non_max_suppression(
                candidates, 
                iou_threshold=0.5, 
                prob_threshold=SCORE_THRESH
            ):
                pred_boxes.append([img_idx] + box)
 
    model.train(was_training)

    return float(mean_average_precision(
        pred_boxes, 
        dataset.true_boxes, 
        iou_threshold=0.5,
        num_classes=C
    ))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VOCEvalSet()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
 
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
 
    mAP = evaluate_model(model, loader, dataset, device)
    print(f"mAP@0.5 on VOC{YEAR} {SPLIT} (checkpoint from epoch {ckpt.get('epoch', '?')}): {mAP:.4f}")
  
if __name__ == "__main__":
    main()
