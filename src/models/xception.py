import torch
import timm

from typing import Any

from src.models.base_model import BaseModel


def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(loopthrough(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(loopthrough(n))
        except:
            ret.append(modules)
    return flatten_list(ret)


class Xception(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(Xception, self).__init__(config, device)
        self.model = timm.create_model('xception', pretrained=True, num_classes=config.num_classes)
        self.loss_function = torch.nn.CrossEntropyLoss()

        ##### Freeze layers
        target_layers = []
        module_list = [module for module in self.model.modules()]
        flatted_list = flatten_model(module_list)

        assert 0 < config.p_freeze <= 1
        n_to_freeze = int(len(flatted_list) * config.p_freeze)
        for count, value in enumerate(flatted_list[:n_to_freeze]):
            print(count, value)
            target_layers.append(value)
            value.requires_grad = False
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
