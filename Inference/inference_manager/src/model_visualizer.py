#!/usr/bin/python3
from torch.utils.tensorboard import SummaryWriter
import torch
import cv2
import numpy as np

# def transforms(image, cuda:bool, device):
#     """This function transforms the input image into a format compatible with
#     the model.
    
#     Args:
#         image: Image in a numpy array
#         cuda: Boolean value of available cuda - handled by inference_class
#         device: Device name (cpu/cuda/cuda1, etc) - handled by inference_class"""
    
#     original_img_size = (image.shape[0],image.shape[1])
#     # model_img_size = (384, 640)
#     model_img_size = (640, 640)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (model_img_size[1], model_img_size[0]))
#     img = np.ascontiguousarray(img, dtype=np.float32)
#     img = np.transpose(img / 255.0, [2, 0, 1])
#     img = torch.from_numpy(img).to(device)
#     if len(img.shape) == 3:
#         img = img[None]
#     img.permute(0, 2, 3, 1) 
#     img = img.to(device)
#     img = img.half()
#     return img, original_img_size, model_img_size
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    #print(sem_img.shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
     
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)

def transforms(image, cuda:bool, device):
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
    img = img.half() if cuda else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    original_img_size = (img0.shape[0],img0.shape[1])
    model_img_size = (384, 640)
    return img, original_img_size, model_img_size




im0 = cv2.imread('bus.jpg')
img = transforms(im0, 1, 'cuda')
# model_path = '../../../models/yolov5s-seg.torchscript'
model_path = '../../../models/yolopv2.pt'
net = torch.jit.load(model_path)
net.to('cuda')
writer = SummaryWriter('./Tensorboard/experiment_3')
writer.add_graph(net, img)
writer.close()