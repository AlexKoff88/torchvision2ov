import tempfile
import unittest
from functools import partial

import numpy as np

from tv2ov.converter import PreprocessorConvertor
import pytest

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.transforms as transforms

from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core
import openvino.runtime as ov


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

@pytest.mark.parametrize("transform", (transforms.Compose(transform), transforms.Compose(transform))) #nn.Sequential(*transform)))
def test_transform(transform):
    core = Core()
    model = core.read_model(model="/home/alex/work/experimental/mobilenetv2_food101/mobilenetv2_food101/mobilenet_v2_food101.onnx")
    dst_path = "test.xml"

    convertor = PreprocessorConvertor(model)

    model = convertor.from_torchvision(0, transform, [1,3,-1,-1])

    ov.serialize(model, dst_path, dst_path.replace(".xml", ".bin"))

