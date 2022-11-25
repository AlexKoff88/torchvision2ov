# Torchvision to OpenVINO (tv2ov)
Converts and embeds preprocessing into OpenVINO model

## Installation
```
python setup.py install
```

## Usage
- Embed Torchvision preprocessing into the model
```python
from tv2ov import PreprocessConverter
model = PreprocessConverter.from_torchvision(
    model=model, 
    input_name="input",
    transform=transform,
    input_example=image)
```

## Example
```
python example.py
```
