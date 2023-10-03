# YOLOPv2
# Adapted from https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py

from .yolopv2_utils import split_for_trace_model, \
                          non_max_suppression, \
                          driving_area_mask, \
                          lane_line_mask, \
                          pred2bbox, \
                          letterbox
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
    
    # Classes lists
    seg_classes = ["road", "lane divider"]

    # Separate variables in the output of the inference
    [pred,anchor_grid],seg,ll = original_output

    # Based in https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py, perform all
    # operations needed to get the desired variables format
    pred = split_for_trace_model(pred,anchor_grid)
    pred = non_max_suppression(pred)
    da_seg_mask = driving_area_mask(original_img_size, seg)
    da_seg_mask = cv2.cvtColor(da_seg_mask, cv2.COLOR_BGR2GRAY)
    ll_seg_mask = lane_line_mask(original_img_size, ll)
    ll_seg_mask = cv2.cvtColor(ll_seg_mask, cv2.COLOR_BGR2GRAY)
    det2d_class_list, det2d_list = pred2bbox(pred, original_img_size, model_img_size, det_classes)
    seg_list = [da_seg_mask, ll_seg_mask]

    # Returns of variables; If more outputs are needed, it is required to adapt 
    # the inference_class script. If less, the unused variables should be =None

    return (det2d_class_list, det2d_list), (seg_classes, seg_list, "panoptic")
    # seg_classes = ["lane divider"]
    # seg_list = [ll_seg_mask]
    # return None, (seg_classes, seg_list, "panoptic")

def transforms(image, cuda:bool, device, half):
    """This function transforms the input image into a format compatible with
    the model.
    
    Args:
        image: Image in a numpy array
        cuda: Boolean value of available cuda - handled by inference_class
        device: Device name (cpu/cuda/cuda1, etc) - handled by inference_class"""
    
    img_size = 640
    stride = 32
    img0 = image
    img = letterbox(img0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img # uint8 to fp16/32
    img = img/255  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    original_img_size = (img0.shape[0],img0.shape[1])
    # model_img_size = (384, 640)
    return img, original_img_size, model_img_size