import numpy as np
import torch
from torch import jit as jit
from torch import nn as nn
from torch.nn import functional as F

from setquence.base.base import BaseJITModule, BaseModule


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def cross_entropy_loss_smooth(logits, y, weight):
    return LabelSmoothingLoss(smoothing=0.5, classes=33,)(logits, y)


@jit.script_method
def cross_entropy_loss(logits, y, weight):
    return F.cross_entropy(logits, y, weight=weight)


@jit.script_method
def binary_cross_entropy_loss(logits, y, weight):
    return F.binary_cross_entropy_with_logits(logits.view(-1), y.float())


class DownTaskClassification(BaseJITModule):
    def __init__(self, config, env=None, *args, **kwargs):
        super(DownTaskClassification, self).__init__(config, env)
        self.config = config

        self.DropoutDownTaskClassification = nn.Dropout(self.config.p_dropout)
        self.ActivationDownTaskClassification = nn.ReLU()
        self.LinearDownTaskClassification = nn.Linear(self.config.hidden_size, self.config.n_classes)
        self.SoftmaxDownTaskClassification = nn.Softmax(dim=1)
        self.weights = self.config.loss.weights
        self.n_classes = self.config.n_classes

        if self.weights is not None:
            self.weights = self.weights.to(self.env.device)

    @jit.script_method
    def forward(self, x):
        x_embed = self.DropoutDownTaskClassification(torch.mean(x, dim=1))
        x_embed = self.ActivationDownTaskClassification(x_embed)
        logits = self.LinearDownTaskClassification(x_embed)

        return logits

    def prediction(self, y=None):
        return {
            "predicted": y.detach().cpu().numpy(),
            "label": np.argmax(self.SoftmaxDownTaskClassification(y).detach().cpu().numpy(), axis=1),
        }

    def loss(self, output, objective):
        if isinstance(objective, list):
            objective = torch.cat(objective)

        objective = objective.view(-1)

        if self.weights is not None and self.weights.device != self.env.device:
            self.weights = self.weights.to(self.env.device)

        if self.n_classes > 2:
            return F.cross_entropy(output.view(-1, self.n_classes), objective, weight=self.weights)
        else:
            return F.binary_cross_entropy_with_logits(output.view(-1), objective.float())


class FFCBlock(nn.Module):
    def __init__(self, hidden_size, out_size, activation=True, dropout=0.3):
        super().__init__()
        self.LinearDownTaskClassification = nn.Linear(hidden_size, out_size)
        self.DropoutDownTaskClassification = nn.Dropout(dropout)
        self.activation = activation
        if activation:
            self.ActivationDownTaskClassification = nn.ReLU()

    def forward(self, x):
        x = self.LinearDownTaskClassification(x)
        x = self.DropoutDownTaskClassification(x)
        if self.activation:
            x = self.ActivationDownTaskClassification(x)

        return x


class DownTaskRegression(BaseModule):
    def __init__(self, config, env=None, *args, **kwargs):
        super(DownTaskRegression, self).__init__(config, env)
        self.config = config

        self.DropoutDownTaskRegression = nn.Dropout(self.config.p_dropout)
        self.ActivationDownTaskRegression = nn.ReLU()
        self.LinearDownTaskRegression = nn.Linear(self.config.hidden_size, self.config.dim_out)
        self.dim_out = self.config.dim_out

    def forward(self, x):
        x_embed = self.DropoutDownTaskRegression(torch.mean(x, dim=1))
        x_embed = self.ActivationDownTaskRegression(x_embed)
        predict = self.LinearDownTaskRegression(x_embed)

        return predict

    def loss(self, output, objective):
        if isinstance(objective, list):
            objective = torch.cat(objective)

        objective = objective.view(-1, self.dim_out)
        output = output.view(-1, self.dim_out)
        idx_nonzero = torch.where(objective != 0)
        objective = objective[idx_nonzero]
        output = output[idx_nonzero]

        output = F.log_softmax(output)
        objective = F.softmax(objective)

        return F.kl_div(output, objective.float())
