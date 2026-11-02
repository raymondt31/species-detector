import random
import torchvision.transforms as T
import torchvision.transforms.functional as F

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
FILL = tuple(int(round(m * 255)) for m in MEAN)

class Compose: 
    """Similar to torchvision's native Compose, but also hands boxes to the transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            if getattr(t, "take_boxes", False):
                img, boxes = t(img, boxes)
            else:
                img = t(img)
        return img, boxes

class RandomHorizontalFlips:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() < self.p:
            img = F.hflip(img)
            boxes = [[c, 1.0 - x, y, w, h] for c, x, y, w, h in boxes]
        return img, boxes

class RandomScaleTranslates:
    def __init__(self, max_frac=0.2, min_visible=0.25, fill=FILL):
        self.max_frac = max_frac
        self.min_visible = min_visible
        self.fill = fill

    def __call__(self, img, boxes):

        # Generate random scaling factor
        s = random.uniform(1 - self.max_frac, 1 + self.max_frac)
        w_img, h_img = img.size

        # Generate random horizontal + vertical translations
        tx_px = round(random.uniform(-self.max_frac, self.max_frac) * w_img)
        ty_px = round(random.uniform(-self.max_frac, self.max_frac) * h_img)

        img = F.affine(
            img, 
            angle=0.0,
            translate=(tx_px, ty_px),
            scale=s,
            shear=0.0,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=self.fill,
        )

        tx, ty = tx_px / w_img, ty_px / h_img # Compute applied shift

        kept = []
        for c, x, y, w, h in boxes:

            # Apply transformation to each pixel, 0.5 to scale about image center not top left corner
            x, y = 0.5 + s * (x - 0.5) + tx, 0.5 + s * (y - 0.5) + ty
            
            # Scale bounding box dims too
            w, h = s * w, s * h

            # Convert center back to corner coords
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2

            # Clip boundaries of new bounding box
            cx1, cy1, cx2, cy2 = (
                max(x1, 0.0),
                max(y1, 0.0),
                min(x2, 1.0),
                min(y2, 1.0)
            )

            # Discard images that are mostly or fully out of frame
            if cx2 <= cx1 or cy2 <= cy1:
                continue                              
            if (cx2 - cx1) * (cy2 - cy1) < self.min_visible * (w * h):
                continue   

        return img, kept

# Move transforms here instead of keeping them in train
train_transform = Compose([
    T.Resize((448, 448)),
    RandomHorizontalFlips(0.5),
    RandomScaleTranslates(0.2),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])                   

val_transform = Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])