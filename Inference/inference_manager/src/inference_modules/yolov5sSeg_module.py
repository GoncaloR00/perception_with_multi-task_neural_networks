# YOLOv5
# Adapted from https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py

from .yolov5seg_utils import non_max_suppression
import numpy as np
import torch
import cv2
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from .coco2bdd100k import coco2bdd100k

mod_path = Path(__file__).parent

det_classes = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['semantic segmentation']:
        det_classes.append(data['semantic segmentation'][name])

dataset_converter = coco2bdd100k("semantic segmentation")

model_loader_name = "torchscript_cuda_half"
model_img_size = (384, 640)

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
    pred, protos = original_output[:2]
    pred = non_max_suppression(pred, nm=32)
    for i, det in enumerate(pred):
        if len(det):
            c, mh, mw = protos[i].shape
            masks = (det[:, 6:] @ protos[i].float().view(c, -1)).sigmoid().view(-1, mh, mw)
            masks = masks.cpu().numpy()
            seg_list = []
            seg_classes = []
            for idx, x in enumerate(det[:, 5]):
                seg_classes.append(det_classes[dataset_converter.convert(int(x))])
                mask = (masks[idx] * 255).astype('uint8')
                mask = cv2.resize(mask, (original_img_size[1], original_img_size[0]))
                seg_list.append(mask)
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "instance")
    detections = None
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    """This function transforms the input image into a format compatible with
    the model.
    
    Args:
        image: Image in a numpy array
        cuda: Boolean value of available cuda - handled by inference_class
        device: Device name (cpu/cuda/cuda1, etc) - handled by inference_class"""
    
    original_img_size = (image.shape[0],image.shape[1])
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