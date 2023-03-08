import torch

from typing import Any
from torch import nn
# Importing the model from torchvision
from torchvision import models

from src.models.base_model import BaseModel


class BaseClassifier(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(BaseClassifier, self).__init__(config, device)
        self.loss_function = nn.CrossEntropyLoss()
        
        num_classes = 8
        model_ft = models.inception_v3(pretrained=True)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        self.model = model_ft

    def forward(self, image: torch.Tensor, target: torch.Tensor, **kwargs) -> dict:
        outputs = self.model(image)
        
        if isinstance(outputs, tuple):
            outputs, aux_outputs = outputs
        else:
            aux_outputs = None

        loss = None

        if target is not None:
            if aux_outputs is not None:
                loss1 = self.loss_function(outputs, target)
                loss2 = self.loss_function(aux_outputs, target)
                loss = loss1 + 0.4*loss2
            else:
                loss = self.loss_function(outputs, target)

        return {
            'loss': loss,
            'logits': outputs,
        }