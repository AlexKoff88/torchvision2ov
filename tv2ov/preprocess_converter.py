from typing import Callable, Any
import logging
import numpy as np

import openvino.runtime as ov

class PreprocessorConvertor():
    def __init__(self, model: ov.Model):
        self._model = model

    @staticmethod
    def from_torchvision(model: ov.Model, input_name: str, transform: Callable, input_example: Any) -> ov.Model:
        """
        Convert torchvision transform to OpenVINO preprocessing
        Arguments:
            model (ov.Model):
                Result name
            input_name (str):
                Name of the result node to connect with preprocessing
            transform (Callable):
                torchvision transform to convert
            input_example (torch.Tensor or np.ndarray or PIL.Image):
                Example of input data
        Returns:
            ov.Mode: OpenVINO Model object with preprocessing
        Example:
            >>> model = PreprocessorConvertor.from_torchvision(model, "input", transform, input_example)
        """
        try:
            from torchvision import transforms
            from tv2ov.torchvision import from_torchvision
            return from_torchvision(model, input_name, transform, input_example)
        except ImportError:
            raise ImportError("Please install torchvision")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise e
