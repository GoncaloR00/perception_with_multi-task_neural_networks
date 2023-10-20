import numpy as np
import torch
import cv2

import yaml
from yaml.loader import SafeLoader
from pathlib import Path

mod_path = Path(__file__).parent
det_classes = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['object detection']:
        det_classes.append(data['object detection'][name])

model_img_size = (640, 360)

# image dimensions in the format (height, width)

def output_organizer(original_output, original_img_size, model_img_size):
    """This function receives the output from the inference and organizes the
     data to a specific format.
     
     Args:
        original_output: Output from the inference
        original_img_size: Image size before transformations
        model_img_size: Image size used in the model

    Outputs:
        Two variables:
            2D_Detections: A list of two elements: A list of ordered classes and 
    a list of bounding boxes corners positions.
            Segmentations: A list of two elements: A list of ordered classes and
    a list of segmentation masks
    
    Notes:
        -The output from the inference is a group of PyTorch tensors. To use 
        other framework, each tensor should be converted to Numpy and then 
        converted to the desired framework.
        -To convert to Numpy, it is as simple as adding .numpy() to the end of
    the variable
        -To convert from numpy to the desired framework, check the framework 
    documentation"""


    shape = (original_img_size[1], original_img_size[0])
    x0=original_output[0]

    _,da_predict=torch.max(x0, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    DA = cv2.resize(DA, shape)

    mask_DA = np.zeros(original_img_size, dtype=np.uint8)
    mask_DA[DA>100]=255
    segmentations = (["road"], [mask_DA], 'semantic')

    detections = None
    # print('TwinLiteNet-Da')
    return detections, segmentations

def transforms(image, cuda:bool, device, half):

    original_img_size = (image.shape[0],image.shape[1])

    img = cv2.resize(image, model_img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.float() / 255.0
    img = img.to(device)
    if half:
        img = img.half()
    """This function transforms the input image into a format compatible with
    the model.
    
    Args:
        image: Image in a numpy array
        cuda: Boolean value of available cuda - handled by inference_class
        device: Device name (cpu/cuda/cuda1, etc) - handled by inference_class"""

    return img, original_img_size, model_img_size
