import cv2
import copy
import torch
from .ade20k2bdd100k import ade20k2bdd100k
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import numpy as np
import time

dataset_converter = ade20k2bdd100k("semantic segmentation")
mod_path = Path(__file__).parent
seg_classes_name = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['semantic segmentation']:
        seg_classes_name.append(data['semantic segmentation'][name])

model_img_size = (512, 1024)
model_loader_name = "torchscript_cuda"

def output_organizer(original_output, original_img_size, model_img_size):
    time_a = time.time()
    _, predictions = torch.max(original_output.data, 1)
    predictions = predictions.to("cpu")
    seg_classes = []
    seg_list = []
    range_i = np.unique(predictions)
    if not(range_i is None):
        for i in range_i:
            elements = predictions == i
            mask = (elements*255).squeeze(0).numpy().astype(np.uint8)
            mask = cv2.resize(mask, (original_img_size[1], (original_img_size[0])))
            seg_classes.append(seg_classes_name[dataset_converter.convert(int(i))])
            seg_list.append(mask)
    time_b = time.time()
    print(f"Tempo de organização: {time_b-time_a}")
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "semantic")

    # Temporary solution for evaluation problem
    if segmentations is None or not("road" in seg_list):
        mask = np.zeros(original_img_size, dtype=np.uint8)
        seg_classes.append("road")
        seg_list.append(mask)
        segmentations = (seg_classes, seg_list, "semantic")

    detections = None
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    original_img_size = (image.shape[0],image.shape[1])
    img_0 = copy.deepcopy(image)
    img_0 = cv2.resize(img_0, (model_img_size[1], model_img_size[0]))
    img = torch.Tensor(img_0).permute(2, 0, 1)
    img = img/255
    img = img.unsqueeze(0)
    img = img.to(device)
    if half:
        img = img.half()
    return img, original_img_size, model_img_size