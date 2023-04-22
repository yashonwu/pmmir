import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, average=True):
        dist = torch.sum(
                (anchor - positive) ** 2 - (anchor - negative) ** 2 ,
                dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)
        if average:
            return torch.mean(dist_hinge)
        else:
            return dist_hinge
        
def infomax(anchor, positive, negative):
    loss = torch.log(torch.sigmoid(torch.sum(anchor*positive, dim=1))) + torch.log(1- torch.sigmoid(torch.sum(anchor*negative, dim=1)))
    return loss