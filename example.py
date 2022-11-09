from lib2to3.pytree import convert
import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        normalize,
    ])


#print(transform)

#transform_ts = torch.jit.script(transform)
#transform_ts = torch.jit.trace(transform(), torch.rand(1, 3, 224, 224))
#print(transform_ts.graph)


class TorchTransformWrapper(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = torch.nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transform(x)
            return x

def script_transform(transform):
    if isinstance(transform, transforms.Compose):
        print("YALA")
        transform = torch.nn.Sequential(*transform.transforms)
    return torch.jit.script(transform)

#transform_ts = script_transform(transform)
#print(transform_ts)


from transform_utils import PreprocessorConvertor

convertor = PreprocessorConvertor()

convertor.from_torchvision(transform)
