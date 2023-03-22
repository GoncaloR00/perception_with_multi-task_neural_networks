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


def output_organizer(original_output, original_img_size, model_img_size):
    # Enviar string ou índice?
    det_classes = ["car"]
    seg_classes = ["drivable_area", "lane_lines"]

    [pred,anchor_grid],seg,ll = original_output
    pred = split_for_trace_model(pred,anchor_grid)
    pred = non_max_suppression(pred)
    da_seg_mask = driving_area_mask(original_img_size, seg)
    da_seg_mask = cv2.cvtColor(da_seg_mask, cv2.COLOR_BGR2GRAY)
    ll_seg_mask = lane_line_mask(original_img_size, ll)
    ll_seg_mask = cv2.cvtColor(ll_seg_mask, cv2.COLOR_BGR2GRAY)
    det2d_class_list, det2d_list = pred2bbox(pred, original_img_size, model_img_size, det_classes)
    seg_list = [da_seg_mask, ll_seg_mask]
    return (det2d_class_list, det2d_list), (seg_classes, seg_list)

def transforms(image, cuda, device):
    img_size = 640
    stride = 32
    img0 = image
    # img0 = cv2.resize(img0, (1280,720), interpolation=cv2.INTER_LINEAR)
    img = letterbox(img0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if cuda else img.float()  # uint8 to fp16/32
    # img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # original_img_size = (720,1280)
    original_img_size = (img0.shape[0],img0.shape[1])
    print(img0.shape)
    model_img_size = (384, 640)
    return img, original_img_size, model_img_size