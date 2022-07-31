import torch
import torch.nn as nn
from functions import intersetion_over_union

class YoloLoss (nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

        # predictions and target is (N,S,S,25) tensor, example(this case is 2), SxS cells, and 25 output
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
# 0-19 from classes, 20 from prob_score, 21:25 4 of the bbox coordinate
        iou_b1 = intersetion_over_union(predictions[...,21:25], target[...,21:25])
        iou_b2 = intersetion_over_union(predictions[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqeeze(0), iou_b2.unsqeeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[...,20].unsqueeze(3) # identity_obj_i # torch need a 2d tensor
        # probability that it is a target box # output will be one or zero

   #  For Box Coordinates #

        box_prediction = exists_box * (
            (
                bestbox * predictions[...,26:30]
                +(1 - bestbox) * predictions[...,21:25]
            )
        )
        box_targets = exists_box * target[...,21:25]

        # Since output of width is representing to the camera output, not pixels.
        # The range is from -1000 to 1000, it can be a negative value so we have to abs(width)
        # sign output 1,0,-1
        box_prediction[...,2:4] = torch.sign(box_prediction[...,2:4]) *\
                                  torch.sqrt(torch.abs(box_prediction[...,2:4] + 1e-6))

        # target are from our training set, those images will not be negative.
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # (N,S,S,4 -> (N*S*S,4) #2d array for mse function
        box_loss = self.mse(torch.flatten(box_prediction, end_dim = -2),
                            torch.flatten(box_targets, end_dim = -2)
                            )
# http://jevois.org/doc/group__coordhelpers.html

   #   For Object Loss   #
        # probability of that classes
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[...,20:21]
        )
        # (N*S*S,1)
        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[...,20:21]))

   #   For No Object Loss #

        # flatten: (N,S,S,1) 4 tensor -> (N,S*S) # start_dim = 1 is two tensor
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
                torch.flatten((1 - exists_box) * target[...,20:21], start_dim=1)
        )

        no_object_loss +=  self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                torch.flatten((1 - exists_box) * target[...,20:21], start_dim=1)
        )

  #    For Class Loss     #

        # (N,S,S,20) -> (N*S*S,20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[...,20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss # First two rows of loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss