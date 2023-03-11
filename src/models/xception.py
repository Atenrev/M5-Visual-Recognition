import torch
import timm

from typing import Any

from src.models.base_model import BaseModel


class Xception(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(Xception, self).__init__(config, device)
        self.model = timm.create_model('xception', pretrained=True, num_classes=config.num_classes)
        self.loss_function = torch.nn.CrossEntropyLoss()

        ##### Freeze/Unfreeze Layers
        param_list = self.model.parameters()
        assert 0 <= config.p_freeze <= 1
        n_to_freeze = int(len(param_list) * config.p_freeze)
        for param in self.model.parameters()[:n_to_freeze]:
            param.requires_grad = False

        for param in self.model.parameters()[n_to_freeze:]:
            param.requires_grad = True
        #####

    def forward(self, image: torch.Tensor, **kwargs) -> dict:
        outputs = self.model(image)

        loss = None
        target = kwargs.get('target', None)

        if target is not None:
            loss = self.loss_function(outputs, target)

        return {
            'loss': loss,
            'logits': outputs,
        }
