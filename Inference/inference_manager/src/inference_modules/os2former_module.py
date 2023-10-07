import cv2
import copy
import torch
from .ade20k2bdd100k import ade20k2bdd100k
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import numpy as np
import time
from scipy.interpolate import InterpolatedUnivariateSpline

dataset_converter = ade20k2bdd100k("semantic segmentation")
mod_path = Path(__file__).parent
seg_classes_name = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['semantic segmentation']:
        seg_classes_name.append(data['semantic segmentation'][name])

model_img_size = (320, 800)
num_points = 72
n_offsets = num_points
n_strips = num_points - 1
num_classes = 4
max_lanes = num_classes
ori_img_w = 1280
ori_img_h = 720
img_w = 800
img_h = 320
cut_height = 270
sample_y = range(589, 270, -8)
conf_threshold=0.5

def output_organizer(original_output, original_img_size, model_img_size):
    seg_classes, seg_list = print_result(original_img_size, get_lanes(original_output))
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "panoptic")

    detections = None
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    original_img_size = (image.shape[0],image.shape[1])
    img = np.asarray(image)
    img = img[cut_height:, :, :].astype(np.float32)
    img = cv2.resize(img, (img_w, img_h),interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)/255
    img = torch.tensor(img)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.float()
    img = img.to(device)
    if half:
        img = img.half()
    print(img.shape)
    # modelo = torch.jit.load('')
    # modelo.eval()
    # modelo.to('cuda')
    # with torch.no_grad():
    #     preds = modelo(img.cuda())
    # print(preds)

    
    
    return img, original_img_size, model_img_size

















import torch
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1],
                                                     points[:, 0],
                                                     k=min(3,
                                                           len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) |
                (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self, sample_y,img_w, img_h ):
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                              axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration

def predictions_to_pred(predictions):
    '''
    Convert predictions to internal Lane structure for evaluation.
    '''
    prior_ys =torch.linspace(1, 0, n_offsets, dtype=torch.float32)
    prior_ys = prior_ys.to(predictions.device)
    prior_ys = prior_ys.double()
    lanes = []
    for lane in predictions:
        lane_xs = lane[6:]  # normalized value
        start = min(max(0, int(round(lane[2].item() * n_strips))),
                    n_strips)
        length = int(round(lane[5].item()))
        end = start + length - 1
        end = min(end, len(prior_ys) - 1)
        # end = label_end
        # if the prediction does not start at the bottom of the image,
        # extend its prediction until the x is outside the image
        mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                   ).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        lane_ys = prior_ys[lane_xs >= 0]
        lane_xs = lane_xs[lane_xs >= 0]
        lane_xs = lane_xs.flip(0).double()
        lane_ys = lane_ys.flip(0)

        lane_ys = (lane_ys * (ori_img_h - cut_height) +
                   cut_height) / ori_img_h
        if len(lane_xs) <= 1:
            continue
        points = torch.stack(
            (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
            dim=1).squeeze(2)
        lane = Lane(points=points.cpu().numpy(),
                    metadata={
                        'start_x': lane[3],
                        'start_y': lane[2],
                        'conf': lane[1]
                    })
        lanes.append(lane)
    return lanes
    
def get_lanes(predictions, as_lanes=True):
    '''
    Convert model output to lanes.
    '''

#        print("pred shape:",predictions.shape)
    predictions = predictions[0,...]
    decoded = []
    threshold = conf_threshold
    # scores = torch.nn.functional.softmax(predictions[..., :2],dim=-1)[..., 1]
    scores = torch.nn.functional.sigmoid(predictions[..., :2])[..., 1]
    score_topk,score_indice = torch.topk(scores,max_lanes,dim=-1)
    predictions = predictions[score_indice]
    keep_inds = score_topk >= threshold # use thres to filter false preditction
    predictions = predictions[keep_inds]
#        scores = scores[keep_inds]

    if predictions.shape[0] == 0:
        decoded.append([])
        return decoded

    predictions[..., 5] = torch.round(predictions[..., 5] * n_strips)
    if as_lanes:
        pred = predictions_to_pred(predictions)
    else:
        pred = predictions
    decoded.append(pred)
    return decoded

import copy

def print_result(original_img_size, lanes, width=4):
        lanes = lanes[0]
        lanes = [lane.to_array(sample_y,ori_img_w,ori_img_h) for lane in lanes]
        lanes_xys = []
        mask = np.zeros(original_img_size).astype(np.uint8)
        seg_classes = []
        seg_list = []
        for _, lane in enumerate(lanes):
            xys = []
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                xys.append((x, y))
            lanes_xys.append(xys)
        lanes_xys = [xys for xys in lanes_xys if xys!=[]]
        for idx, xys in enumerate(lanes_xys):
            for i in range(1, len(xys)):
                # cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
                cv2.line(mask, xys[i - 1], xys[i], 255, thickness=width)
        seg_classes.append('lane divider')
        seg_list.append(mask)
        return seg_classes, seg_list