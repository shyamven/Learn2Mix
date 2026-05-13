import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Class-balanced focal loss used in CBL baselines."""

    def __init__(self, alpha=None, gamma: float = 2, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

