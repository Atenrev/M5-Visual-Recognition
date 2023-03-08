
from src.models.base_model import BaseModel

from typing import Any
import torch

import torch.nn as nn
import torch.nn.functional as F


class SmallSqueezeNetCNN(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(SmallSqueezeNetCNN, self).__init__(config, device)
        self.loss_function = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire2_squeeze = nn.Conv2d(96, 16, kernel_size=1)
        self.fire2_expand1 = nn.Conv2d(16, 64, kernel_size=1)
        self.batch_norm_fire21 = nn.BatchNorm2d(64)
        self.fire2_expand2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.batch_norm_fire22 = nn.BatchNorm2d(64)
        self.fire2_residual = nn.Conv2d(96, 128, kernel_size=1)

        self.fire3_squeeze = nn.Conv2d(128, 16, kernel_size=1)
        self.fire3_expand1 = nn.Conv2d(16, 64, kernel_size=1)
        self.batch_norm_fire31 = nn.BatchNorm2d(64)
        self.fire3_expand2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.batch_norm_fire32 = nn.BatchNorm2d(64)

        self.fire4_squeeze = nn.Conv2d(128, 32, kernel_size=1)
        self.fire4_expand1 = nn.Conv2d(32, 128, kernel_size=1)
        self.batch_norm_fire41 = nn.BatchNorm2d(128)
        self.fire4_expand2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.batch_norm_fire42 = nn.BatchNorm2d(128)
        self.fire4_residual = nn.Conv2d(128, 256, kernel_size=1)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire5_squeeze = nn.Conv2d(256, 32, kernel_size=1)
        self.fire5_expand1 = nn.Conv2d(32, 128, kernel_size=1)
        self.batch_norm_fire51 = nn.BatchNorm2d(128)
        self.fire5_expand2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.batch_norm_fire52 = nn.BatchNorm2d(128)

        self.dropout_fire5 = nn.Dropout2d(0.5)
        self.conv10 = nn.Conv2d(256, self.config.num_classes, kernel_size=1)

        self.activation = getattr(F, 'relu')

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, image: torch.Tensor, target: torch.Tensor, **kwargs) -> dict:
        x = self.conv1(image)
        x = self.activation(x)
        x = self.maxpool1(x)

        x2 = self.fire2_squeeze(x)
        x21 = self.fire2_expand1(x2)
        x21 = self.batch_norm_fire21(x21)
        x21 = self.activation(x21)

        x22 = self.fire2_expand2(x2)
        x22 = self.batch_norm_fire22(x22)
        x22 = self.activation(x22)

        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.fire2_residual(x) + x2

        x3 = self.fire3_squeeze(x2)
        x31 = self.fire3_expand1(x3)
        x31 = self.batch_norm_fire31(x31)
        x31 = self.activation(x31)

        x32 = self.fire3_expand2(x3)
        x32 = self.batch_norm_fire32(x32)
        x32 = self.activation(x32)

        x3 = torch.cat([x31, x32], dim=1)
        x3 = x2 + x3

        x4 = self.fire4_squeeze(x3)
        x41 = self.fire4_expand1(x4)
        x41 = self.batch_norm_fire41(x41)
        x41 = self.activation(x41)

        x42 = self.fire4_expand2(x4)
        x42 = self.batch_norm_fire42(x42)
        x42 = self.activation(x42)

        x4 = torch.cat([x41, x42], dim=1)
        x4 = self.fire4_residual(x3) + x4
        x4 = self.maxpool4(x4)

        x5 = self.fire5_squeeze(x4)
        x51 = self.fire5_expand1(x5)
        x51 = self.batch_norm_fire51(x51)
        x51 = self.activation(x51)

        x52 = self.fire5_expand2(x5)
        x52 = self.batch_norm_fire52(x52)
        x52 = self.activation(x52)

        x5 = torch.cat([x51, x52], dim=1)
        x5 = x5 + x4

        dropout_fire5 = self.dropout_fire5(x5)
        x_conv10 = self.conv10(dropout_fire5)
        outputs = x_conv10.mean(dim=(-2, -1))

        loss = None

        if target is not None:
            target = F.one_hot(target, num_classes=self.config.num_classes).float()
            loss = self.loss_function(outputs, target)

        return {
            'loss': loss,
            'logits': outputs,
        }
