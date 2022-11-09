from lib2to3.pytree import convert
import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from openvino.runtime import Core
import openvino.runtime as ov

from tv2ov.converter import PreprocessorConvertor

MODEL_LOCAL_PATH="mobilenet_v2.onnx"
OUTPUT_MODEL="mobilenet_v2_preprocess.xml"

def prepare_model():
    model = models.mobilenet_v2(pretrained=True)  
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, MODEL_LOCAL_PATH, verbose=True, input_names=input_names, output_names=output_names)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

model = prepare_model()
core = Core()
model = core.read_model(model=MODEL_LOCAL_PATH)

convertor = PreprocessorConvertor(model)
model = convertor.from_torchvision(0, transform, [1,3,-1,-1])   

ov.serialize(model, OUTPUT_MODEL, OUTPUT_MODEL.replace(".xml", ".bin"))