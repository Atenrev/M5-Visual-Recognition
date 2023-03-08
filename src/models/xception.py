import torch
import timm

from typing import Any

from src.models.base_model import BaseModel


class Xception(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(Xception, self).__init__(config, device)
        self.model = timm.create_model('xception', pretrained=True, num_classes=config.num_classes)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor, target: torch.Tensor, **kwargs) -> dict:
        outputs = self.model(image)

        loss = None

        if target is not None:
            loss = self.loss_function(outputs, target)

        return {
            'loss': loss,
            'logits': outputs,
        }