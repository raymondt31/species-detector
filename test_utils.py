import unittest
import torch
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision
)

class NMSTest(unittest.TestCase):
    
    # --- IOU Tests ---
    def test_iou_dup_box(self):
        box1 = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        box2 = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        self.assertAlmostEqual(intersection_over_union(box1, box2).item(), 1.0, places=5)

    def test_iou_no_overlap(self):
        box1 = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
        box2 = torch.tensor([[60.0, 60.0, 100.0, 100.0]])
        self.assertAlmostEqual(intersection_over_union(box1, box2).item(), 0.0, places=5)

    # --- NMS Tests ---
    def test_simple_nms(self):
        boxes = [
            [0, 0.9, 0, 0, 100, 100], 
            [0, 0.8, 10, 10, 110, 110], # duplicate
            [1, 0.7, 0, 0, 100, 100] # diff class
        ]

        result = non_max_supression(boxes, iou_threshold=0.5, prob_threshold=0.5)

        self.assertEqual(len(result), 2) # only want to keep 2 boxes
        self.assertEqual(result[0][1], 0.9) # make sure highest confidence is still 0.9

    # --- mAP Tests ---
    def test_mAP_simple(self):
        true_boxes = [[0, 0, 1.0, 0, 0, 100, 100]]

        pred_boxes = [
            [0, 0, 0.9, 0, 0, 10, 10], # Expected FP => wrong box, high confidence
            [0, 0, 0.8, 0, 0, 100, 100] # Expected TP => right box, low confidence
        ]

        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=1)

        self.assertLess(mAP, 1.0)
        self.assertGreater(mAP, 0.0)

if __name__ == '__main__':
    unittest.main()