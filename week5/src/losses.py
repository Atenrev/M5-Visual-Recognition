import torch

from torch import nn


class SymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
            self,
            logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: torch.Tensor of shape (batch_size, batch_size)
        Returns:
            torch.Tensor of shape (1,)
        """
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = self.cross_entropy(logits, labels)
        loss_j = self.cross_entropy(logits.T, labels)
        return (loss_i + loss_j) / 2