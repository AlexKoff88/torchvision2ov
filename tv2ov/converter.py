from typing import List
from abc import ABCMeta, abstractmethod
from typing import Callable
from enum import Enum
import numbers
from collections.abc import Sequence
import logging
import copy

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
    new_shape[-1] = input_shape[-3]
    new_shape[-3] = input_shape[-2]
    new_shape[-2] = input_shape[-1]
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
        new_shape = _change_layout_shape(input_shape)

        ppp.input(input_idx).tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.RGB) \
        .set_shape(new_shape)

        ppp.input(input_idx).preprocess().convert_layout(Layout('NCHW'))
        #ppp.input(input_idx).preprocess().convert_color(ColorFormat.BGR)
        ppp.input(input_idx).preprocess().convert_element_type(Type.f32)
        ppp.input(input_idx).preprocess().scale(255.0)

        meta["input_shape"] = new_shape

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
        
        source_size= meta["shape"]
        target_size = _setup_size(transform.size, "Incorrect size type for CenterCrop operation")        
        bl, tr = self._compute_corners(source_size, target_size)
        input_shape = meta["input_shape"]
        bl = [0]*len(input_shape[:-2]) + bl
        tr = input_shape[:-2] + tr

        # Change corners layout in case of ToTensor (NHWC)
        if meta["has_totensor"] and meta["layout"] == Layout("NCHW"):
            bl = _change_layout_shape(bl)
            tr = _change_layout_shape(tr)

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

        if layout == Layout('NHWC'):
            input_shape[1] = -1
            input_shape[2] = -1
        else:
            input_shape[2] = -1
            input_shape[3] = -1

        ppp.input(input_idx).tensor().set_shape(input_shape)
        ppp.input(input_idx).preprocess().resize(NormalizeConverter.RESIZE_MODE_MAP[mode], h, w)
        meta["input_shape"] = input_shape
        meta["shape"] = (h,w)

        return Status.SUCCEEDED, meta

class PreprocessorConvertor():
    def __init__(self, model: ov.Model):
        self._model = model

    @staticmethod
    def to_list(transform) -> List:
        if isinstance(transform, torch.nn.Sequential):
            return [t for t in transform]
        elif isinstance(transform, transforms.Compose):
            return transform.transforms
        else:
            raise TypeError(f"Unsupported transform type: {type(transform)}")

    def from_torchvision(self, transform, input_name: str, input_shape:list=None) -> ov.Model:
        transform_list = PreprocessorConvertor.to_list(transform)
        input_idx = next((i for i, p in enumerate(self._model.get_parameters()) if p.get_friendly_name() == input_name), None)
        if input_idx is None:
            raise ValueError(f"Input with name {input_name} is not found")

        ppp = PrePostProcessor(self._model)
        ppp.input(input_idx).tensor().set_layout(Layout('NCHW')) 
        input_shape = input_shape = list(self._model.input(input_idx).shape) if \
                        input_shape == None else input_shape

        input_shape = list(self._model.input(input_idx).get_shape())
        results = []
        global_meta = {"input_shape": input_shape}
        global_meta["layout"] = Layout('NCHW')
        global_meta["has_totensor"] = any(isinstance(item, transforms.ToTensor) for item in transform_list)

        for t in transform_list:
            status, _ = TransformConverterFactory.convert(type(t), input_idx, ppp, t, global_meta)
            results.append(status)

        updated_model = ppp.build()
        return updated_model
