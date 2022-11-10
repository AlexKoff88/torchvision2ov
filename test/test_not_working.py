import numpy as np
from PIL import Image
import copy
import tempfile
import pytest

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from openvino.runtime import Core
import openvino.runtime as ov

from tv2ov.converter import PreprocessorConvertor


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

transform1 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

transform2 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
transform3 = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

def get_onnx_model(model, output_path):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=input_names, output_names=output_names)

def verify_pipelines(torch_model, compiled_model, transform, test_input):
    ## Test inference

    ## Torch results
    torch_input = copy.deepcopy(test_input)
    test_image = Image.fromarray(torch_input.astype('uint8'), 'RGB')
    transformed_input = transform(test_image)
    transformed_input = torch.unsqueeze(transformed_input, dim=0)
    with torch.no_grad():
        torch_result = torch_model(transformed_input).numpy()

    ## OpenVINO results
    ov_input = copy.deepcopy(test_input)
    ov_input = np.expand_dims(ov_input, axis=0)
    output = compiled_model.output(0)
    ov_result = compiled_model(ov_input)[output]

    result = np.max(np.absolute(torch_result - ov_result))
    return result

@pytest.mark.parametrize(("transform", "test_input"), 
            [(transform1, np.random.randint(255, size=(300, 300, 3), dtype=np.uint8)), 
            (transform2, np.random.randint(255, size=(300, 300, 3), dtype=np.uint8)),
            (transform2, np.random.randint(255, size=(224, 224, 3), dtype=np.uint8))]) #nn.Sequential(*transform)))
def test_transform(transform, test_input):
    torch_model = models.mobilenet_v2(pretrained=True) 

    core = Core()
    with tempfile.NamedTemporaryFile() as tmp:
        get_onnx_model(torch_model, tmp.name)
        model = core.read_model(model=tmp.name)

    convertor = PreprocessorConvertor(model)
    model = convertor.from_torchvision(0, transform)
    OUTPUT_MODEL = "tmp.xml"
    ov.serialize(model, OUTPUT_MODEL, OUTPUT_MODEL.replace(".xml", ".bin"))

    compiled_model = core.compile_model(model, "CPU")
    result = verify_pipelines(torch_model, compiled_model, transform, test_input)
    assert result < 1e-4

