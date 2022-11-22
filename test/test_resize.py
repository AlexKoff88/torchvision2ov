import numpy as np
from PIL import Image
import copy

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from openvino.runtime import Core
import openvino.runtime as ov
from openvino.runtime import Type
from openvino.runtime import opset10 as opset

from tv2ov.converter import PreprocessorConvertor



def test_transform():
    INPUT_SIZE=4
    TARGET_SIZE=2

    transform = transforms.Compose([
            transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    test_input = np.random.randint(255, size=(INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

    ## Torch results
    torch_input = copy.deepcopy(test_input)
    test_image = Image.fromarray(torch_input.astype('uint8'), 'RGB')
    transformed_input = transform(test_image)
    torch_result = np.asarray(transformed_input)
    

    ## create OpenVINO model
    input = opset.parameter([1,INPUT_SIZE,INPUT_SIZE,3], name="input", dtype=Type.u8)
    axes = opset.constant([1,2], dtype=Type.i64)
    sizes = opset.constant([TARGET_SIZE,TARGET_SIZE], dtype=Type.i64)
    scales = opset.constant([1,1], dtype=np.float32)

    resize = opset.interpolate(input, output_shape=sizes, scales=scales, axes=axes,
                                shape_calculation_mode="sizes",
                                antialias=False, coordinate_transformation_mode="half_pixel", 
                                mode="linear", pads_begin=[0, 0, 0, 0], pads_end=[0, 0, 0, 0],
                                name="resize")
    result = opset.result(resize, name="result")

    model = ov.Model([result], [input])
    ov.serialize(model, "resize.xml")
    core = ov.Core()
    compiled = core.compile_model(model, "CPU")
    ov_input = np.expand_dims(test_input, axis=0)
    print(f"OV input shape: {ov_input.shape}")
    ov_result = compiled(ov_input)[compiled.output()]
    #print(f"OV result: {}")

    print(f"test_image:\n{np.asarray(test_image)}")
    print(f"ov_input:\n{ov_input}")

    res = np.max(np.absolute(torch_result - np.squeeze(ov_result, axis=0)))

    print(f"torch_result:\n{torch_result}")
    print(f"ov_result:\n{ov_result}")

    assert res < 1e-4
    