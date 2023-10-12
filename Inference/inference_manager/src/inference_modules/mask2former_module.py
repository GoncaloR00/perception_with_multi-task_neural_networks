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

model_img_size = (384, 384)

def output_organizer(original_output, original_img_size, model_img_size):
    detections = None
    class_queries_logits = original_output[0].to('cpu')
    masks_queries_logits = original_output[1].to('cpu')
    masks_queries_logits = torch.nn.functional.interpolate(masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False)
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    semantic_segmentation = segmentation.argmax(dim=1)
    semantic_segmentation = semantic_segmentation.squeeze(0).numpy()
    predicted_label = cv2.resize(semantic_segmentation.astype(np.uint8), (original_img_size[1], (original_img_size[0])))
    seg_list = []
    seg_classes = []
    for i in range(int(predicted_label.min()),int(predicted_label.max())+1):
        seg_list.append(((predicted_label == i)*255).astype(np.uint8))
        seg_classes.append(seg_classes_name[dataset_converter.convert(int(i))])
    if not('road' in seg_classes):
        seg_classes.append('road')
        seg_list.append(np.zeros(original_img_size, dtype=np.uint8))
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "semantic")
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    original_img_size = (image.shape[0],image.shape[1])
    img = copy.deepcopy(image)
    img = cv2.resize(img, model_img_size)
    img = img/255
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