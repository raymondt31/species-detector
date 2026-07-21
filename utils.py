# Working!

# TODO: consider adding a box_format option later

import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def intersection_over_union(boxes_preds, boxes_labels):
    """ 
    Given two vectors Box 1 = [x_center, y_center, width, height] and Box 2 = [x_center, y_center, width, height] find:
    1) Corners Coords:
        - Top Left: (x1,y1) = (max(b1[x1], b2[x1]), max(b1[y1], b2[y1]))
        - Bottom right: (x2,y2) = (min(b1[x2], b2[x2]), min(b1[y2], b2[y2]))
    2) Calc areas
    3) Return IOU
    * note: this only accepts boxes w/ midpoint coords; consider adding a box format option
    """

    epsilon = 1e-6

    # Both boxes are of shape (N, 4), where N is the number of boxes
    
    # Get corner coordinates from the passed in midpoint coordinates
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp 0 to prevent negative values when boxes don't intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # 1e-6 padding to prevent NaN w/ div by 0 and exploding denominators
    return intersection / (box1_area + box2_area - intersection + epsilon)

#TODO: Implement the following
def non_max_suppression(
        bboxes,
        iou_threshold,
        prob_threshold
):
    """
    The point of non max supression is to eliminate redundant/duplicate boxes
    """

    # predictions = [class, probability, x1, y1, x2, y2], [...], [...], ...]
    assert type(bboxes) == list
        
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
        
    # highest prob at beginning
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] # make sure diff class
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
                # consider adding box_format as well
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        num_classes=20
):
    """
    A standard metric for evaluating object detection models:
    1) For each bounding box of a certain class across all images, determine if we have a TP (True positive) or FP (False positive)
    2) Sort ALL by descending order in terms of confidence
    3) Calculate a cumulative precision and recall for all outputs starting from the box w/ highest confidence
        a) Precision: TP / total # detections (TP + FP)
        b) Recall: TP / total # ground truths
    4) Plot prec and rec for each bbox on a Precision (y-axis) - Recall (x-axis) graph
    5) Take the area under the graph to get the AP for a single class
    6) Repeat 1-5 for ALL classes
    7) mAP = sum(AP) / total classes
    8) Note that this is all for a single IOU threshold (what this function does). Repeat steps 1-7 for various thresholds
    9) mAP@thresh1:thresh2:thresh3... = sum(mAP_thresh_i) / total thresholds
    """

    # pred_boxes: [
    #   [train_idx (img that this bounding box comes from), class_pred, prob_score, x1, y1, x2, y2], 
    #   [], 
    #   ...
    # ]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # get all relevant detections
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        # get all relevant gts
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # amount_bboxes = {0: torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        # this is to keep track of which gt bounding boxes we have covered so far
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort all detections by descending probability
        detections.sort(key=lambda x: x[2], reverse=True)
        
        # True positives
        TP = torch.zeros(len(detections))

        # False positives
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        # Early exit -- may be holding mAP back for small datasets
        if total_true_bboxes == 0:
            continue
        
        # loop through every prediction the model made for specific class, c,
        # starting from the most confident prediction
        for detection_idx, detection in enumerate(detections):
            
            # filter gts that have the same img idx as given detection
            ground_truth_img = [
                bbox
                for bbox in ground_truths
                if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0
            best_gt_idx = -1

            # determine which gt best corresponds to this current detection, "detection"
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:])
                    # consider adding box format
                )
            
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # second condition is incase no gts
            if best_iou > iou_threshold and best_gt_idx != -1:

                # check that this gt box has not been marked
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1 # bbox marked
                else: # false positive; bounding box already marked
                    FP[detection_idx] = 1

            # if IOU is below threshold
            else:
                FP[detection_idx] = 1
        
        # ie [1, 1, 0, 1, 0] => [1, 2, 0, 3, 0]
        # this is for precision and recall purposes
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        
        # start at point (0,1) for integration purposes
        precisions = torch.cat((torch.tensor([1]), precisions)) # y-axis
        recalls = torch.cat((torch.tensor([0]), recalls)) # x-axis

        # trapezoidal integration over precisions and recalls
        average_precisions.append(torch.trapz(precisions, recalls))

    # remember, this is mAP for only one threshold
    return sum(average_precisions) / len(average_precisions) 

# The following QOL functions were all mostly copied directly from Aladdin Persson
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold
            )

            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])