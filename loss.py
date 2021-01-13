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
    input = input.topk(k=k, dim=-1)[1]
    #targets, _ = targets
    targets = targets.unsqueeze(dim=-1).expand_as(input)
    return (input == targets).max(dim=-1)[0].float().mean()

class FocalLoss(_Loss):
  def __init__(self, alpha=1, gamma=2, logits=False, size_average=None, reduce=None, reduction="mean"):
    super().__init__(size_average, reduce, reduction)
    self.reduction = reduction
    self.alpha = alpha
    self.gamma = gamma
    self.logits = logits
    self.reduce = reduce
    self.loss = nn.CrossEntropyLoss(reduction='none')

  def forward(self, inputs, targets):
    ce_loss = self.loss(inputs, targets)

    pt      = torch.exp(-ce_loss)
    _loss   = self.alpha * (1-pt)**self.gamma * ce_loss

    if self.reduction == "mean":
        return _loss.mean()
    elif self.reduction == "sum":
        return _loss.sum()
    else:
        return _loss

class CustomLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean", clip=None):
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.clip = clip
        self.loss = nn.CrossEntropyLoss(reduction='none')


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights) -> torch.Tensor:
        loss = self.loss(inputs, targets)
        loss = loss * sample_weights
        loss[torch.isnan(loss)] = 0

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss