# ----------
# !!!FIX!!!
# ----------

import torch

def IOU(boxes_preds, boxes_labels):
    """ 
    Given two vectors Box 1 = [x1, y1, x2, y2] and Box 2 = [x1, y1, x2, y2] find:
    1) Corners Coords:
        - Top Left: (x1,y1) = (max(b1[x1], b2[x1]), (max(b1[y1], b2[y1]))
        - Bottom right: (x2,y2) = (min(b1[x2], b2[x2]), min(b1[y2]), b2[y2])
    2) Calc areas
    3) Return IOU
    """
    
    # Both boxes are of shape (N, 4), where N is the number of boxes
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]

    # TODO: Figure out why box2 is based off boxes_preds rather than boxes_labels
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y2)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp 0 to prevent negative values when boxes don't intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # 1e-6 padding to prevent NaN w/ div by 0 and exploding denominators
    return intersection / (box1_area + box2_area - intersection + 1e-6)

    #TODO: Implement the following
    # def NMS():
    # def mean_average_precision():
    # def cellboxes_to_boxes():
    # def get_bboxes():
    # def plot_image():
    # def save_checkpoint():
    # def load_checkpoint():
