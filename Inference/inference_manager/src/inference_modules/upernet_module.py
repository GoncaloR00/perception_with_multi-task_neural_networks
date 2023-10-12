import cv2
import copy
import torch
from .cityscapes2bdd100k import cityscapes2bdd100k
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import numpy as np
import time

dataset_converter = cityscapes2bdd100k("semantic segmentation")
mod_path = Path(__file__).parent
seg_classes_name = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['semantic segmentation']:
        seg_classes_name.append(data['semantic segmentation'][name])

model_img_size = (512, 512)

def output_organizer(original_output, original_img_size, model_img_size):
    # print('here')
    detections = None
    seg_classes = []
    seg_list = []
    logits = original_output[0]
    logits = logits.to('cpu')
    predicted_label = logits.argmax(1)
    predicted_label = predicted_label.squeeze()
    predicted_label = cv2.resize(predicted_label.numpy().astype(np.uint8), (original_img_size[1], (original_img_size[0])))
    for i in range(int(predicted_label.min()),int(predicted_label.max())+1):
        if i==6:
            seg_list.append(((predicted_label == i)*255).astype(np.uint8))
            # seg_classes.append(seg_classes_name[dataset_converter.convert(int(i))])
            seg_classes.append('road')
    if not('road' in seg_classes):
        seg_classes.append('road')
        seg_list.append(np.zeros(original_img_size, dtype=np.uint8))
    print(seg_classes)
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "semantic")
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    original_img_size = (image.shape[0],image.shape[1])
    img = copy.deepcopy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img/255
    # mean = np.mean(img, axis=(0, 1))
    # std = np.std(img, axis=(0, 1))
    mean = [
        0.48500001430511475,
        0.4560000002384186,
        0.4059999883174896
    ]
    std = [
        0.2290000021457672,
        0.2239999920129776,
        0.22499999403953552
    ]
    img = (img - mean) / std
    img = torch.tensor(img)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.float()
    img = img.to(device)
    if half:
        img = img.half()
    return img, original_img_size, model_img_size