from typing import List
from abc import ABCMeta, abstractmethod
from typing import Callable, Any
from enum import Enum
import numbers
from collections.abc import Sequence
import logging
import copy
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino.runtime import Core, Layout, Type

class Status(Enum):
    SUCCEEDED = 0
    FAILED = 1
    SKIPPED = 2

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)
    
    return size

def _change_layout_shape(input_shape):
    new_shape = copy.deepcopy(input_shape)
    new_shape[1] = input_shape[3]
    new_shape[2] = input_shape[1]
    new_shape[3] = input_shape[2]
    return new_shape


class TransformConverterBase(metaclass=ABCMeta):
    """ Base class for an executor """
 
    def __init__(self, **kwargs):
        """ Constructor """
        pass
 
    @abstractmethod
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform) -> Status:
        """ Abstract method to run a command """
        return False
 
class TransformConverterFactory:
    """ The factory class for creating executors"""
 
    registry = {}
    """ Internal registry for available executors """
 
    @classmethod
    def register(cls, target_type=None) -> Callable:
        def inner_wrapper(wrapped_class: TransformConverterBase) -> Callable:
            registered_name = wrapped_class.__name__ if target_type == None else target_type.__name__
            if registered_name in cls.registry:
                logging.warning('Executor %s already exists. Will replace it', registered_name)
            cls.registry[registered_name] = wrapped_class
            return wrapped_class
 
        return inner_wrapper
 
    @classmethod
    def convert(cls, converter_type, *args, **kwargs) -> Status:
        name = converter_type.__name__
        if name not in cls.registry:
            return Status.FAILED, None
 
        converter = cls.registry[name]()
        return converter.convert(*args, **kwargs)

### Converters definition 
@TransformConverterFactory.register(transforms.Normalize)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None) -> Status:
        mean = transform.mean
        scale = transform.std #[1/std for std in transform.std]
        ppp.input(input_idx).preprocess().mean(mean).scale(scale)
        return Status.SUCCEEDED, None 

@TransformConverterFactory.register(transforms.ToTensor)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None) -> Status:
        input_shape = meta["input_shape"]
        layout = meta["layout"]

        ppp.input(input_idx).tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.RGB)

        if layout == Layout("NHWC"):
            input_shape = _change_layout_shape(input_shape)
            layout = Layout("NCHW")
            ppp.input(input_idx).preprocess().convert_layout(layout)
        ppp.input(input_idx).preprocess().convert_element_type(Type.f32)
        ppp.input(input_idx).preprocess().scale(255.0)

        meta["input_shape"] = input_shape
        meta["layout"] = layout
        return Status.SUCCEEDED, meta

@TransformConverterFactory.register(transforms.CenterCrop)
class NormalizeConverter(TransformConverterBase):
    def _compute_corners(self, source_size, target_size):
        if target_size[0] > source_size[0] or \
            target_size[1] > source_size[1]:
            ValueError(f"CenterCrop size={target_size} is greater than source_size={source_size}")
        
        bottom_left = [0,0]
        top_right = [0,0]

        bottom_left[1] = int((source_size[1] - target_size[1]) / 2) # Compute x
        bottom_left[0] = int((source_size[0] - target_size[0]) / 2) # Compute y

        top_right[1] = min(bottom_left[1] + target_size[1], source_size[1] - 1)
        top_right[0] = min(bottom_left[0] + target_size[0], source_size[0] - 1)

        return bottom_left, top_right


    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None) -> Status:
        if "shape" not in meta:
            logging.warning('Shape is not defined')
            return Status.FAILED, None

        input_shape = meta["input_shape"]
        layout = meta["layout"]
        
        source_size= meta["shape"]
        target_size = _setup_size(transform.size, "Incorrect size type for CenterCrop operation")        
        bl, tr = self._compute_corners(source_size, target_size)
        
        bl = [0]*len(input_shape[:-2]) + bl if layout == Layout("NCHW") else [0] + bl + [0]
        tr = input_shape[:-2] + tr if layout == Layout("NCHW") else input_shape[:1] + tr + input_shape[-1:]

        print(f"CenterCrop: {bl} {tr}")
        print(f"CenterCrop: {input_shape}, layout: {meta['layout']}")

        # Change corners layout in case of ToTensor (NHWC)
        '''if meta["has_totensor"] and layout == Layout("NHWC"):
            print("Change layout")
            bl = _change_layout_shape(bl)
            tr = _change_layout_shape(tr)'''

        ppp.input(input_idx).preprocess().crop(bl, tr)
        meta["shape"] = (target_size[-2],target_size[-1])
        return Status.SUCCEEDED, meta

@TransformConverterFactory.register(transforms.Resize)
class NormalizeConverter(TransformConverterBase):
    RESIZE_MODE_MAP = {
        InterpolationMode.BILINEAR: ResizeAlgorithm.RESIZE_LINEAR,
        InterpolationMode.BICUBIC: ResizeAlgorithm.RESIZE_CUBIC,
        InterpolationMode.NEAREST: ResizeAlgorithm.RESIZE_NEAREST,
    }
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None) -> Status:
        if transform.max_size != None:
            logging.warning('Resize with max_size if not supported')
            return Status.FAILED, None

        mode = transform.interpolation
        size = transform.size
        h, w = _setup_size(size, "Incorrect size type for Resize operation")

        ppp.input(input_idx).tensor().set_layout(Layout('NCHW'))

        layout = meta["layout"] 
        input_shape = meta["input_shape"]

        if layout == Layout("NHWC"):
            input_shape[1] = -1
            input_shape[2] = -1
        else:
            input_shape[2] = -1
            input_shape[3] = -1

        print(f"Resize: {input_shape}, layout: {layout}")

        ppp.input(input_idx).tensor().set_shape(input_shape)
        ppp.input(input_idx).preprocess().resize(NormalizeConverter.RESIZE_MODE_MAP[mode], h, w)
        meta["input_shape"] = input_shape
        meta["shape"] = (h,w)

        return Status.SUCCEEDED, meta

def _to_list(transform) -> List:
        if isinstance(transform, torch.nn.Sequential):
            return [t for t in transform]
        elif isinstance(transform, transforms.Compose):
            return transform.transforms
        else:
            raise TypeError(f"Unsupported transform type: {type(transform)}")

def _get_shape_layout_from_data(input_example):
    """
    Disregards rank of shape and return 
    """
    shape = None
    layout = None
    if isinstance(input_example, torch.Tensor): # PyTorch
        shape = list(input_example.shape)
        layout = Layout("NCHW")
    elif isinstance(input_example, np.ndarray): # OpenCV, numpy
        shape = list(input_example.shape)
        layout = Layout("NHWC")
    elif isinstance(input_example, Image.Image): # PILLOW
        shape = list(np.array(input_example).shape)
        layout = Layout("NHWC")
    else:
        raise TypeError(f"Unsupported input type: {type(input_example)}")

    if len(shape) == 3:
        shape = [1] + shape

    print(f"Shape: {shape}, layout: {layout}")
    return shape, layout

def from_torchvision(model: ov.Model, input_name: str, transform: Callable, input_example: Any) -> ov.Model:
        transform_list = _to_list(transform)
        input_idx = next((i for i, p in enumerate(model.get_parameters()) if p.get_friendly_name() == input_name), None)
        if input_idx is None:
            raise ValueError(f"Input with name {input_name} is not found")

        input_shape, layout = _get_shape_layout_from_data(input_example)

        ppp = PrePostProcessor(model)
        ppp.input(input_idx).tensor().set_layout(layout) 
        ppp.input(input_idx).tensor().set_shape(input_shape)

        results = []
        global_meta = {"input_shape": input_shape}
        global_meta["layout"] = layout
        global_meta["has_totensor"] = any(isinstance(item, transforms.ToTensor) for item in transform_list)

        for t in transform_list:
            status, _ = TransformConverterFactory.convert(type(t), input_idx, ppp, t, global_meta)
            results.append(status)

        updated_model = ppp.build()
        return updated_model