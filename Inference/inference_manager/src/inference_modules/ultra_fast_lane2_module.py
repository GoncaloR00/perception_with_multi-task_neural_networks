import cv2
import copy
import torch
from .ade20k2bdd100k import ade20k2bdd100k
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import numpy as np
import time
from torchvision import transforms as trf

dataset_converter = ade20k2bdd100k("semantic segmentation")
mod_path = Path(__file__).parent
seg_classes_name = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['semantic segmentation']:
        seg_classes_name.append(data['semantic segmentation'][name])

model_img_size = (512, 1024)
crop_ratio = 0.8
num_row= 56
num_col= 41
row_anchor = np.linspace(160,710, num_row)/720
col_anchor = np.linspace(0,1, num_col)

def output_organizer(original_output, original_img_size, model_img_size):

    coords = pred2coords(original_output, row_anchor, col_anchor, original_image_width = original_img_size[1], original_image_height = original_img_size[0])
    mask = np.zeros(original_img_size, dtype=np.uint8)
    seg_classes = []
    seg_list = []
    for lane in coords:
        for i in range(1,len(lane)):
            if np.linalg.norm(np.array(lane[i - 1]) - np.array(lane[i])) < 50:
                val = int(4*lane[i][1]/100)-7
                if val < 4 :
                    val = 4
                cv2.line(mask, lane[i - 1], lane[i], 255, thickness=val)
    seg_list.append(mask)
    seg_classes.append('lane divider')
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "panoptic")

    # Temporary solution for evaluation problem
    # if segmentations is None or not("road" in seg_list):
    #     mask = np.zeros(original_img_size, dtype=np.uint8)
    #     seg_classes.append("road")
    #     seg_list.append(mask)
    #     segmentations = (seg_classes, seg_list, "semantic")

    detections = None
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    original_img_size = (image.shape[0],image.shape[1])
    img_transforms = trf.Compose([
                    trf.ToPILImage(),
                    trf.Resize((int(320 / crop_ratio), 800)),
                    trf.ToTensor(),
                    trf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img_transforms(img)
    img = img.unsqueeze(0)
    img = img[:, :, -320:, :]
    img = img.to(device)
    if half:
        img = img.half()
    print(img.shape)
    return img, original_img_size, model_img_size








def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred[0].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred[1].shape

    max_indices_row = pred[0].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred[2].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred[1].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred[3].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred[0] = pred[0].cpu()
    pred[1] = pred[1].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred[0][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred[1][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords