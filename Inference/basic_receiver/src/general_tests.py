#!/usr/bin/python3

import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import math
import copy
import cv2
import numpy as np
from collections import Counter
import timeit

mod_path = Path(__file__).parent
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

def count_none_values(dictionary):
    value_counts = Counter(dictionary.values())
    return value_counts[None]

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

void_semantic = copy.deepcopy(semantic_cls)
void_semantic = {key: [np.zeros((10000,10000), dtype=np.int32)] for key in void_semantic}

void_instance = copy.deepcopy(void_semantic)

void_panoptic = copy.deepcopy(panoptic_cls)
void_panoptic = {key: None for key in void_panoptic}

# print(len(void_semantic != None))
# first_element_of_non_empty = [l[0] for l in void_semantic.values() if l]
# print(first_element_of_non_empty)

# print(count_none_values(void_semantic))


tic=timeit.default_timer()
print(sum([isinstance(void_semantic[i], list) for i in void_semantic]))
toc=timeit.default_timer()
print(toc-tic)
tic=timeit.default_timer()
print(sum([isinstance(void_panoptic[i], list) for i in void_panoptic]))
toc=timeit.default_timer()
print(toc-tic)