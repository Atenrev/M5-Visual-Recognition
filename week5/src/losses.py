import torch
import torch.nn.functional as F


class SymmetricCrossEntropyLoss(torch.nn.Module):
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
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_j) / 2