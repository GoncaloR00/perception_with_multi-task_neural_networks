# YOLOPv2
# Adapted from https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py

import torchvision.transforms as tf


# from .yolopv2_utils import split_for_trace_model, \
#                           non_max_suppression, \
#                           driving_area_mask, \
#                           lane_line_mask, \
#                           pred2bbox, \
#                           letterbox
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
    # pred,seg,ll = original_output

    # Based in https://github.com/CAIC-AD/YOLOPv2/blob/main/demo.py, perform all
    # operations needed to get the desired variables format
    # pred = split_for_trace_model(pred,anchor_grid)

    # pred = non_max_suppression(pred)
    # da_seg_mask = driving_area_mask(original_img_size, seg)
    # da_seg_mask = cv2.cvtColor(da_seg_mask, cv2.COLOR_BGR2GRAY)
    # ll_seg_mask = lane_line_mask(original_img_size, ll)
    # ll_seg_mask = cv2.cvtColor(ll_seg_mask, cv2.COLOR_BGR2GRAY)
    # det2d_class_list, det2d_list = pred2bbox(pred, original_img_size, model_img_size, det_classes)
    # seg_list = [da_seg_mask, ll_seg_mask]

    # Returns of variables; If more outputs are needed, it is required to adapt 
    # the inference_class script. If less, the unused variables should be =None

    det_out, da_seg_out,ll_seg_out = original_output['det_out'], original_output['drive_area_seg'], original_output['lane_line_seg']
    det_pred = non_max_suppression(det_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    det = det_pred[0].to('cpu').numpy()
    o_img_h, o_img_w = original_img_size
    img_h, img_w = model_img_size
    det[:,0] = det[:,0]/img_w*o_img_w
    det[:,1] = det[:,1]/img_h*o_img_h
    det[:,2] = det[:,2]/img_w*o_img_w
    det[:,3] = det[:,3]/img_h*o_img_h
    det2d_class_list = []
    det2d_list = []

    for box in det:
        x1,y1,x2,y2,prob,class_id = box
        det2d_list.append([[int(x1),int(y1)],[int(x2),int(y2)]])
        if class_id > 10:
            class_id = 10
        det2d_class_list.append(det_classes[2])

    height, width = model_img_size
    h,w=original_img_size
    pad = (0,12)
    shapes = (height, width), ((h / height, w / width), pad)
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
    da_seg_mask = da_predict
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (original_img_size[1], original_img_size[0]))


    ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
    ll_seg_mask = ll_predict
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (original_img_size[1], original_img_size[0]))

    seg_list = [(da_seg_mask*255), (ll_seg_mask*255)]


    return (det2d_class_list, det2d_list), (seg_classes, seg_list, "panoptic")

def transforms(image, cuda:bool, device, half):
    """This function transforms the input image into a format compatible with
    the model.
    
    Args:
        image: Image in a numpy array
        cuda: Boolean value of available cuda - handled by inference_class
        device: Device name (cpu/cuda/cuda1, etc) - handled by inference_class"""
    normalize = tf.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
    transform=tf.Compose([
                        tf.ToTensor(),
                        normalize,
                    ])
    original_img_size = (image.shape[0],image.shape[1])
    h0, w0 = original_img_size

    img, ratio, pad = letterbox_for_img(image)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    img = transform(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img, original_img_size, model_img_size
















def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))


    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

import time
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output