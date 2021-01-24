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
    targets, _ = targets
    input,  _  = input

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
    _, targets = targets
    _,  input  = input

    input = input.topk(k=k, dim=-1)[1]
    #targets, _ = targets
    targets = targets.unsqueeze(dim=-1).expand_as(input)
    return (input == targets).max(dim=-1)[0].float().mean()

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
        https://medium.com/towards-artificial-intelligence/how-to-use-label-smoothing-for-regularization-aa349f7f1dbb
    """

    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

class FocalLoss(_Loss):
  def __init__(self, alpha=1, gamma=2, c=0.8, epsilon=0.1, logits=False, size_average=None, reduce=None, reduction="mean"):
    super().__init__(size_average, reduce, reduction)
    self.reduction = reduction
    self.alpha  = alpha
    self.gamma  = gamma
    self.logits = logits
    self.reduce = reduce
    self.c      = c
    self.loss   = LabelSmoothingCrossEntropy(reduction='none', epsilon=epsilon)

  def focal(self, inputs, targets):
    ce_loss   = self.loss(inputs, targets)
    pt        = torch.exp(-ce_loss)
    loss     = self.alpha * (1-pt)**self.gamma * ce_loss
    return loss

  def forward(self, inputs1, inputs2, targets1, targets2):
    # ce_loss1   = self.loss(inputs1, targets1)
    # ce_loss2   = self.loss(inputs2, targets2)

    # pt        = torch.exp(-ce_loss)
    # _loss     = self.alpha * (1-pt)**self.gamma * ce_loss

    _loss1    = self.focal(inputs1, targets1)
    _loss2    = self.focal(inputs2, targets2)
    _loss     = _loss1*self.c + _loss2*(1-self.c)

    if self.reduction == "mean":
        return _loss.mean()
    elif self.reduction == "sum":
        return _loss.sum()
    else:
        return _loss

# from topk.svm import SmoothTopkSVM as TopKSmoothTopkSVM

# class CustomLoss(_Loss):
#     def __init__(self, size_average=None, reduce=None, reduction="mean", clip=None):
#         super().__init__(size_average, reduce, reduction)
#         self.reduction = reduction
#         self.clip = clip
#         self.loss = nn.CrossEntropyLoss(reduction='none')


#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights) -> torch.Tensor:
#         loss = self.loss(inputs, targets)
#         loss = loss * 1/torch.log(1+sample_weights)
#         loss[torch.isnan(loss)] = 0

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss

# class SmoothTopkSVM(TopKSmoothTopkSVM):
#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights) -> torch.Tensor:
#         return super().forward(inputs, targets)
