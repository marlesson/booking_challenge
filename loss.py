from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss
import torchbearer
from torchbearer import metrics, Metric
from torchbearer.metrics import default_for_key, running_mean, mean
import torch.nn.functional as F


@metrics.default_for_key("top_k_acc")
@running_mean
@mean
@metrics.lambda_metric("top_k_acc", on_epoch=False)
def top_k_acc(input:torch.Tensor, targets:torch.Tensor, k:int=4):
    "Computes the Top-k accuracy (target is in the top k predictions)."
    targets, _, _ = targets
    input,  _, _  = input

    input = input.topk(k=k, dim=-1)[1]
    #targets, _ = targets
    targets = targets.unsqueeze(dim=-1).expand_as(input)
    return (input == targets).max(dim=-1)[0].float().mean()

@metrics.default_for_key("top_k_acc2")
@running_mean
@mean
@metrics.lambda_metric("top_k_acc2", on_epoch=False)
def top_k_acc(input:torch.Tensor, targets:torch.Tensor, k:int=4):
    "Computes the Top-k accuracy (target is in the top k predictions)."
    _, targets, _ = targets
    _,  input, _  = input

    input = input.topk(k=k, dim=-1)[1]
    #targets, _ = targets
    targets = targets.unsqueeze(dim=-1).expand_as(input)
    return (input == targets).max(dim=-1)[0].float().mean()


class FocalLoss(_Loss):
    def __init__(self, alpha=1, gamma=2, c=0.8, l2=0, logits=False, 
                        size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.alpha  = alpha
        self.gamma  = gamma
        self.logits = logits
        self.reduce = reduce
        self.c      = c
        self.l2     = l2
        self.loss   = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, inputs2, emb, targets, targets2, neighbors):
        self.fix_inputs_by_neighbors(inputs, neighbors)
        ce_loss   = self.loss(inputs, targets)
        loss2     = self.loss(inputs2, targets2)

        pt        = torch.exp(-ce_loss)
        loss1     = self.alpha * (1-pt)**self.gamma * ce_loss

        _loss     = loss1*self.c + loss2*(1-self.c)

        if self.reduction == "mean":
            return _loss.mean() + self.penality(emb)
        elif self.reduction == "sum":
            return _loss.sum()
        else:
            return _loss

    def fix_inputs_by_neighbors(self, inputs, neighbors):
        for i, x in enumerate(neighbors):
            inputs[i][x] = torch.zeros(inputs[i][x].shape).to(inputs.device)

    def penality(self, emb):
        return self.l2*torch.norm(emb)

from topk.svm import SmoothTopkSVM as TopKSmoothTopkSVM

class CustomLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean", clip=None):
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.clip = clip
        self.loss = nn.CrossEntropyLoss(reduction='none')


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights) -> torch.Tensor:
        loss = self.loss(inputs, targets)
        loss = loss * 1/torch.log(1+sample_weights)
        loss[torch.isnan(loss)] = 0

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class SmoothTopkSVM(TopKSmoothTopkSVM):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights) -> torch.Tensor:
        return super().forward(inputs, targets)
