#!/usr/bin/python3

import cv2
import numpy as np
import math
# image = cv2.imread('./bus.jpg')
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsv[:,:,0]=100
# hsv[:,:,1]=255
# final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow("Image", final)
# cv2.waitKey(0)
# teste1 = np.zeros((1,1,3), np.uint8)
# teste2 = np.array([[[100,255,255]]], np.uint8)
# bgr = cv2.cvtColor(np.array([[[100,255,255]]], np.uint8), cv2.COLOR_HSV2BGR)
# print(bgr.squeeze())
# # teste2 = np.asarray(teste2, np.uint8)

# print(teste1)
# # # teste=np.asarray(teste1)
# hsv = cv2.cvtColor(teste2, cv2.COLOR_BGR2HSV)
# print(hsv)
# cv2.imshow('teste', bgr)
# cv2.waitKey(0)

name = 1
# bgr = cv2.cvtColor(np.array([[[int(255/(math.sqrt(name))),255,255]]], np.uint8), cv2.COLOR_HSV2BGR).squeeze()
# frac, intNum = math.modf()
# bgr = 255//10
# print(bgr)
# print(5/3)
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import math
mod_path = Path(__file__).parent
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

# frac, intNum = math.modf(256/(len(data['semantic segmentation'])))
# intNum = int(intNum)
# frac = int(math.prod((frac, (len(data['semantic segmentation']))))) 
# semantic_cls = {}
# first_val = -1
# second_val = 0
# for name in data['semantic segmentation']:
#     print(name)
#     if frac > 0:
#         second_val = first_val + intNum + 1
#         semantic_cls[data['semantic segmentation'][name]] = [first_val+1, second_val]
#         first_val = second_val
#         frac -=1
#     else:
#         second_val = first_val + intNum
#         semantic_cls[data['semantic segmentation'][name]] = [first_val+1, second_val]
#         first_val = second_val
# print(semantic_cls)

def get_color_range(data):
    # Define a different color for each object class or instance
    frac, intNum = math.modf(256/(len(data)))
    intNum = int(intNum)
    frac = int(round(math.prod((frac, (len(data)))))) 
    output_cls = {}
    first_val = -1
    second_val = 0
    for name in data:
        if frac > 0:
            second_val = first_val + intNum + 1
            output_cls[data[name]] = [first_val+1, second_val]
            first_val = second_val
            frac -=1
        else:
            second_val = first_val + intNum
            output_cls[data[name]] = [first_val+1, second_val]
            first_val = second_val
    return output_cls

mod_path = Path(__file__).parent
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

# Define a different color for each object class
objDect_cls = {}
for name in data['object detection']:
    objDect_cls[data['object detection'][name]] = cv2.cvtColor(np.array([[[int(255/(math.sqrt(name))),255,255]]], np.uint8), cv2.COLOR_HSV2BGR).squeeze().tolist()

semantic_cls = get_color_range(data['semantic segmentation'])
panoptic_cls = get_color_range(data['panoptic segmentation'])