import torch
import torch.nn as nn
from util import IOU

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # calculate IOUs
        iou1 = IOU(predictions[..., 21:25], target[..., 21:25])
        iou2 = IOU(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat((iou1.unsqueeze(0), iou2.unsqueeze(0)), dim=0)
        iou_max, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # I_obji

        # ------------------------------------
        #       1) Box Coord Loss
        # ------------------------------------

        box_pred = exists_box * (
            (
                 best_box * (predictions[..., 26:30]) + 
                 (1 - best_box) * (predictions[..., 21:25])
            )
        )
        box_target = exists_box * target[..., 21:25]

        # Prevent 0 and (-) outputs
        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(
            torch.abs(box_pred[..., 2:4] + 1e-6) 
        )
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_pred, end_dim=2),
            torch.flatten(box_target, end_dim=2)
        )

        # ------------------------------------
        #             2) Obj Loss
        # ------------------------------------
        
        pred_box = ( 
            best_box * (predictions[..., 25:26]) + 
            (1 - best_box) * (predictions[..., 20:21])
        )

        # (N, S, S, 1) -> (N*S*S)
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box), # start_dim = -1 default
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ------------------------------------
        #            3) No Obj Loss
        # ------------------------------------
        
        # (N,S,S,1) -> (N, S*S)
        no_obj_loss = self.mse( 
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ------------------------------------
        #            4) Class Loss
        # ------------------------------------

        # (N,S,S,20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=2),
            torch.flatten(exists_box * target[..., :20], end_dim=2)
        )

        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )

        return loss
    

if __name__ == "__main__":
    print("Testing loss function...")
    loss_fn = YOLOLoss()
    test_preds = torch.rand(2, 7, 7, 30)
    test_target = torch.rand(2, 7, 7, 30)
    output = loss_fn(test_preds, test_target)
    print(output)
