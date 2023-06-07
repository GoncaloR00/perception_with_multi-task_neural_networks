import cv2
import copy
import torch
from .cityscapes2bdd100k import cityscapes2bdd100k
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import numpy as np

dataset_converter = cityscapes2bdd100k("semantic segmentation")
mod_path = Path(__file__).parent
seg_classes_name = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['semantic segmentation']:
        seg_classes_name.append(data['semantic segmentation'][name])


def output_organizer(original_output, original_img_size, model_img_size):
    _, predictions = torch.max(original_output.data, 1)
    predictions = predictions.to("cpu")
    seg_classes = []
    seg_list = []
    for i in range(20):
        if i == 1 or i == 2 or i == 14:
            elements = predictions == i
            # if not any(elements):
            mask = (elements*255).squeeze(0).numpy().astype(np.uint8)
            mask = cv2.resize(mask, (original_img_size[1], (original_img_size[0])))
            seg_classes.append(seg_classes_name[dataset_converter.convert(int(i))])
            seg_list.append(mask)

    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "semantic")
    detections = None
    return detections, segmentations

def transforms(image, cuda:bool, device):
    original_img_size = (image.shape[0],image.shape[1])
    model_img_size = (512, 1024)
    img_0 = copy.deepcopy(image)
    img_0 = cv2.resize(img_0, (model_img_size[1], model_img_size[0]))
    cv2.imshow("original", img_0)
    cv2.waitKey(1)
    img = torch.Tensor(img_0).permute(2, 0, 1)
    img = img/255
    img = img.unsqueeze(0)
    img = img.to(device)
    # if cuda:
    #     img = img.half()

    return img, original_img_size, model_img_size