import unittest
import numpy as np
import torch
import torch.nn as nn
from utils import (
    intersection_over_union, 
    non_max_suppression, 
    mean_average_precision
)

import evaluate as E
 
class IoUTest(unittest.TestCase):
    # boxes are [x_center, y_center, width, height]
    def test_identical(self):
        b = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        self.assertAlmostEqual(intersection_over_union(b, b).item(), 1.0, places=3)   # 1e-6 epsilon in the denominator
 
    def test_disjoint(self):
        a, b = torch.tensor([[0.2, 0.2, 0.1, 0.1]]), torch.tensor([[0.8, 0.8, 0.1, 0.1]])
        self.assertAlmostEqual(intersection_over_union(a, b).item(), 0.0, places=5)
 
    def test_partial_overlap(self):
        a, b = torch.tensor([[0.5, 0.5, 0.2, 0.2]]), torch.tensor([[0.6, 0.5, 0.2, 0.2]])
        self.assertAlmostEqual(intersection_over_union(a, b).item(), 1 / 3, places=4)
 
class NMSTest(unittest.TestCase):
    def test_simple_nms(self):
        boxes = [
            [0, 0.9, 0, 0, 100, 100],
            [0, 0.8, 10, 10, 100, 100],   # same class, heavy overlap -> suppressed
            [1, 0.7, 0, 0, 100, 100],     # different class -> kept
        ]
        result = non_max_suppression(boxes, iou_threshold=0.5, prob_threshold=0.5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], 0.9)
 
    def test_matches_torchvision_batched_nms(self):
        """Reference check against torchvision. Trials with a same-class pair within 1e-4 of the IoU threshold
        are skipped: utils.intersection_over_union adds 1e-6 to its denominator, so right at the threshold the
        two can legitimately disagree."""
        from torchvision.ops import batched_nms
        rng = np.random.default_rng(0)
        compared = 0
        for _ in range(100):
            K = int(rng.integers(10, 80))
            boxes = [[int(rng.integers(0, 4)), float(rng.random()), *rng.uniform(0.35, 0.65, 2), *rng.uniform(0.15, 0.4, 2)]
                     for _ in range(K)]
            b = np.array(boxes)
            x1, y1, x2, y2 = b[:, 2] - b[:, 4] / 2, b[:, 3] - b[:, 5] / 2, b[:, 2] + b[:, 4] / 2, b[:, 3] + b[:, 5] / 2
            iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
            ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
            inter = iw * ih
            area = (x2 - x1) * (y2 - y1)
            iou = inter / (area[:, None] + area[None] - inter)
            same = (b[:, 0][:, None] == b[:, 0][None]) & ~np.eye(K, dtype=bool)
            if (np.abs(iou - 0.5)[same] < 1e-4).any():
                continue
            mine = {round(x[1], 9) for x in non_max_suppression([list(r) for r in boxes], 0.5, 0.0)}
            xyxy = torch.tensor(np.stack([x1, y1, x2, y2], 1))
            keep = batched_nms(xyxy, torch.tensor(b[:, 1]), torch.tensor(b[:, 0]).long(), 0.5)
            self.assertEqual(mine, {round(boxes[i][1], 9) for i in keep.tolist()})
            compared += 1
        self.assertGreater(compared, 80)
 
BOX = [60, 60, 100, 100]        # [x_center, y_center, w, h]
FAR = [250, 250, 100, 100]
 
def gt(img, cls, box, difficult=0):
    return [img, cls, 1.0, *box, difficult]
 
def det(img, cls, score, box):
    return [img, cls, score, *box]
 
def mAP(preds, trues, classes=1):
    return float(mean_average_precision(preds, trues, iou_threshold=0.5, num_classes=classes))
 
 
class MeanAveragePrecisionTest(unittest.TestCase):
    def test_perfect_detections(self):
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, BOX)], [gt(0, 0, BOX)]), 1.0, places=4)
 
    def test_missing_half_the_objects(self):
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, BOX)], [gt(0, 0, BOX), gt(1, 0, BOX)]), 0.5, places=4)
 
    def test_false_positive_ranked_first_hurts(self):
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, FAR), det(0, 0, 0.8, BOX)], [gt(0, 0, BOX)]), 0.5, places=4)
 
    def test_duplicate_detection_is_a_false_positive(self):
        preds = [det(0, 0, 0.9, BOX), det(0, 0, 0.8, BOX), det(0, 0, 0.7, BOX)]
        self.assertAlmostEqual(mAP(preds, [gt(0, 0, BOX)]), 1.0, places=4)     # recall 1 was reached at precision 1
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, FAR)] + preds, [gt(0, 0, BOX)]), 0.5, places=4)  # a leading FP costs
 
    def test_difficult_objects_are_ignored(self):
        trues = [gt(0, 0, BOX), gt(0, 0, FAR, difficult=1)]
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, BOX), det(0, 0, 0.8, FAR)], trues), 1.0, places=4)  # hit on difficult: ignored
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, BOX)], trues), 1.0, places=4)                       # missing it: no penalty
 
    def test_iou_threshold_is_strict(self):
        self.assertEqual(mAP([det(0, 0, 0.9, [100, 50, 100, 100])], [gt(0, 0, [50, 50, 100, 100])]), 0.0)   # IoU = 1/3
 
    def test_wrong_class_is_not_a_hit(self):
        self.assertEqual(mAP([det(0, 1, 0.9, BOX)], [gt(0, 0, BOX)], classes=2), 0.0)
 
    def test_classes_without_ground_truth_are_skipped(self):
        self.assertAlmostEqual(mAP([det(0, 0, 0.9, BOX)], [gt(0, 0, BOX)], classes=3), 1.0, places=4)
 
    def test_original_seven_value_format_still_works(self):
        trues = [[0, 0, 1.0, 0, 0, 100, 100]]
        preds = [[0, 0, 0.9, 0, 0, 10, 10], [0, 0, 0.8, 0, 0, 100, 100]]
        self.assertAlmostEqual(mAP(preds, trues), 0.5, places=4)
 
class StubModel(nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred
    def forward(self, x):
        return self.pred.repeat(len(x), 1)
 
class StubDataset:
    def __init__(self, true_boxes):
        self.true_boxes = true_boxes
 
def run_stub(pred, true_boxes):
    loader = [(torch.zeros(1, 3, 448, 448), torch.tensor([0]))]
    return E.evaluate_model(StubModel(pred.reshape(1, -1)), loader, StubDataset(true_boxes), "cpu")
 
 
class DecodeTest(unittest.TestCase):
    def make_pred(self):
        p = torch.zeros(7, 7, 30)
        c = p[2, 4]                                            # row 2, col 4
        c[5], c[3] = 0.9, 0.6                                  # class probabilities
        c[20], c[21:25] = 0.8, torch.tensor([0.5, 0.5, 0.2, 0.4])   # box 1: conf, x, y, w, h
        return p
 
    def test_every_class_is_scored_as_confidence_times_class_prob(self):
        cell_box = [(4 + 0.5) / 7, (2 + 0.5) / 7, 0.2, 0.4]
        # class 5 (score .72) and class 3 (score .48) both come from the same box
        self.assertAlmostEqual(run_stub(self.make_pred(), [gt(0, 5, cell_box), gt(0, 3, cell_box)]), 1.0, places=3)
 
    def test_box_coordinates_are_decoded_into_the_right_cell(self):
        wrong_cell = [(1 + 0.5) / 7, (2 + 0.5) / 7, 0.2, 0.4]
        self.assertEqual(run_stub(self.make_pred(), [gt(0, 5, wrong_cell)]), 0.0)
 
 
if __name__ == '__main__':
    unittest.main()
