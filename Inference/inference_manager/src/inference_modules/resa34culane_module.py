
import time
import cv2
import copy
import torch

model_img_size = (288, 800)
model_loader_name = "resa_34_culane_cuda_half"

img_height= 288
img_width = 800
divider = 3

def output_organizer(original_output, original_img_size, model_img_size):
    ori_img_h, ori_img_w = original_img_size
    val = ori_img_h - int(ori_img_w / divider) # For video
    val = 0 # For evaluation
    ori_img_h = ori_img_h - val
    # ori_img_h= 540 - 240
    # ori_img_w = 960
    sample_y=range(ori_img_h-2, 0, -10)
    sample_y2=range(ori_img_h-2, 0, -1)
    time_a = time.time()
    lanes_a=get_lanes(original_output, ori_img_h, ori_img_w, sample_y)
    lanes = [lane.to_array(sample_y2, ori_img_h, ori_img_w) for lane in lanes_a[0]]
    # print(lanes)
    mask = np.zeros((original_img_size[0], original_img_size[1],3), dtype=np.uint8)
    if len(lanes) >0:
        for lane in lanes:
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                # 700 e 15 -> transformar em relação a tamanho
                tik = int(y**1.5/500)
                if tik > 25:
                    tik = 25
                if tik < 0:
                    tik = 0
                mask[val:] = cv2.circle(mask[val:], (x, y), 0, (255, 255, 255), tik)
    seg_list =[mask[:,:,0]]
    seg_classes = ["lane divider"]
    # else:
    #     seg_list = []
    #     seg_classes = []
    time_b = time.time()
    print(f"Tempo de organização: {time_b-time_a}")
    if len(seg_classes) == 0:
        segmentations = None
    else:
        segmentations = (seg_classes, seg_list, "panoptic")
    detections = None
    return detections, segmentations

def transforms(image, cuda:bool, device, half):
    original_img_size = (image.shape[0],image.shape[1])
    ori_img_h, ori_img_w = original_img_size
    val = ori_img_h - int(ori_img_w / divider) # For video
    val = 0 # For evaluation
    img_0 = copy.deepcopy(image)
    img_0 = cv2.resize(img_0[val:], (model_img_size[1], model_img_size[0]))
    img = torch.Tensor(img_0)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.to(device)
    if half:
        img = img.half()
    return img, original_img_size, model_img_size




import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import torch.nn.functional as F

class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self, sample_y, ori_img_h, ori_img_w):
        img_w, img_h = ori_img_w, ori_img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration

def get_lanes(output, ori_img_h, ori_img_w, sample_y):
    segs = output['seg']
    segs = F.softmax(segs, dim=1)
    segs = segs.detach().cpu().numpy()
    if 'exist' in output:
        exists = output['exist']
        exists = exists.detach().cpu().numpy()
        exists = exists > 0.5
    else:
        exists = [None for _ in segs]

    ret = []
    for seg, exist in zip(segs, exists):
        lanes = probmap2lane(seg, ori_img_h, ori_img_w, sample_y, exist)
        ret.append(lanes)
    return ret

# def probmap2lane(probmaps, exists=None, cut_height=0, ori_img_h=720, ori_img_w = 1280, img_height=288, img_width = 800, sample_y=range(588, 230, -20), thr=0.6):
def probmap2lane(probmaps, ori_img_h, ori_img_w, sample_y, exists=None, cut_height=0, img_height=img_height, img_width = img_width, thr=0.6):
# def probmap2lane(probmaps, exists=None, cut_height=240, ori_img_h=300, ori_img_w = 800, img_height=288, img_width = 800, sample_y=range(589, 230, -20), thr=0.6):
    lanes = []
    probmaps = probmaps[1:, ...]
    if exists is None:
        exists = [True for _ in probmaps]
    for probmap, exist in zip(probmaps, exists):
        if exist == 0:
            continue
        probmap = cv2.blur(probmap.astype(np.float32), (9, 9), borderType=cv2.BORDER_REPLICATE)
        ori_h = ori_img_h - cut_height
        coord = []
        for y in sample_y:
            proj_y = round((y - cut_height) * img_height/ori_h)
            line = probmap[proj_y]
            if np.max(line) < thr:
                continue
            value = np.argmax(line)
            x = value*ori_img_w/img_width#-1.
            if x > 0:
                coord.append([x, y])
        if len(coord) < 5:
            continue

        coord = np.array(coord)
        coord = np.flip(coord, axis=0)
        coord[:, 0] /= ori_img_w
        coord[:, 1] /= ori_img_h
        lanes.append(Lane(coord))
    return lanes

