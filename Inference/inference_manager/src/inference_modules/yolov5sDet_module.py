# YOLOv5
# Adapted from https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py

from .yolov5det_utils import non_max_suppression, \
                          pred2bbox
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

model_img_size = (384, 640)
model_loader_name = "torchscript_cuda_half"

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
    
    

    # Separate variables in the output of the inference
    pred = original_output
    pred = non_max_suppression(pred)
    det2d_class_list, det2d_list = pred2bbox(pred, original_img_size, model_img_size, det_classes)
    segmentations = None
    # Returns of variables; If more outputs are needed, it is required to adapt 
    # the inference_class script. If less, the unused variables should be =None
    print(det2d_list)
    return (det2d_class_list, det2d_list), segmentations

def transforms(image, cuda:bool, device, half):
    """This function transforms the input image into a format compatible with
    the model.
    
    Args:
        image: Image in a numpy array
        cuda: Boolean value of available cuda - handled by inference_class
        device: Device name (cpu/cuda/cuda1, etc) - handled by inference_class"""
    
    original_img_size = (image.shape[0],image.shape[1])
   
    # model_img_size = (640, 640)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_img_size[1], model_img_size[0]))
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = np.transpose(img / 255.0, [2, 0, 1])
    img = torch.from_numpy(img).to(device)
    if len(img.shape) == 3:
        img = img[None]
    img.permute(0, 2, 3, 1) 
    img = img.to(device)
    if half:
        img = img.half()

    return img, original_img_size, model_img_size