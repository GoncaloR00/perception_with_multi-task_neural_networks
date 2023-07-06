#!/usr/bin/python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pathlib import Path
import time
from tqdm import tqdm
from inference_manager.msg import detect2d, segmentation
import copy
import yaml
from yaml import SafeLoader
import numpy as np
import math
import os
import json

n_images = 5
topic = '/cameras/evaluation'
curr_path = Path(__file__).parent

mode_obj_dect = 1
mode_drivable = 1
mode_lane = 1
save_path = 'YolopV2/labels/'
save_path = str(curr_path / save_path) + '/'
with open(curr_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

def get_color_range(data):
    # Define a different color for each object class or instance
    frac, intNum = math.modf(180/(len(data)))
    intNum = int(intNum)
    frac = int(round(math.prod((frac, (len(data)))))) 
    output_cls = {}
    first_val = -1
    second_val = 0
    for name in data:
        if frac > 0:
            second_val = first_val + intNum + 1
            output_cls[data[name]] = [first_val+1, second_val]
            first_val = second_val
            frac -=1
        else:
            second_val = first_val + intNum
            output_cls[data[name]] = [first_val+1, second_val]
            first_val = second_val
    return output_cls

def isAllEmpty(dictionary):
    counter = 0
    for key in dictionary:
        counter += len(dictionary[key])
    return counter == 0

# Define a different color for each object class
objDect_cls = {}
for name in data['object detection']:
    objDect_cls[data['object detection'][name]] = cv2.cvtColor(np.array([[[int(255/(math.sqrt(name))),255,255]]], np.uint8), cv2.COLOR_HSV2BGR).squeeze().tolist()


semantic_cls = get_color_range(data['semantic segmentation'])
panoptic_cls = get_color_range(data['panoptic segmentation'])

void_semantic = copy.deepcopy(semantic_cls)
void_semantic = {key: [] for key in void_semantic}

void_instance = copy.deepcopy(void_semantic)

void_panoptic = copy.deepcopy(panoptic_cls)
void_panoptic = {key: [] for key in void_panoptic}

class Receiver:
    def __init__(self):
        topic_detection2d = 'detection2d'
        topic_segmentation = 'segmentation'
        self.bridge = CvBridge()
        self.received_det = 0
        self.received_lane = 0
        self.received_drivable = 0
        self.seg_frameId = "None"
        self.det2d_frameId = "None"
        self.panoptic = void_panoptic
        self.subscriber_detection2d = rospy.Subscriber(topic_detection2d, detect2d, self.detection2dCallback)
        self.subscriber_segmentation = rospy.Subscriber(topic_segmentation, segmentation, self.segmentationCallback)
    def reset_all(self):
        self.received_det = 0
        self.received_lane = 0
        self.received_drivable = 0
        print('Reset!')
    def detection2dCallback(self, msg):
        self.BBoxes = msg
        self.det2d_frameId = msg.frame_id
        self.received_det = 1
    def segmentationCallback(self, msg):
        if msg.Category.data == "panoptic":
            self.seg_frameId = msg.frame_id
            # Clear previous masks
            self.panoptic = copy.deepcopy(void_panoptic)
            for idx, seg_class in enumerate(msg.ClassList):
                if len(self.panoptic[seg_class.data]) == 0:
                    self.panoptic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                else:
                    self.panoptic[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))
                if seg_class.data == "lane divider":
                    self.received_lane = 1
                if seg_class.data == "road":
                    self.received_drivable = 1



image_pub = rospy.Publisher(topic,Image, queue_size=10)
bridge = CvBridge()
rospy.init_node('sender', anonymous=False)
image_list = sorted(os.listdir(curr_path / 'bdd100k/images/100k/val/'))

# Warmup
frame = cv2.imread(str(curr_path / 'bdd100k/images/100k/val/b1c9c847-3bda4659.jpg'))
counter = 0
# Wait for the other node to start
time.sleep(2)
print(f"Warming up!")
while counter<10:
    counter +=1
    image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
    image_message.header.stamp = rospy.Time.now()
    image_message.header.frame_id = "None"
    image_pub.publish(image_message)
    time.sleep(1.5)

print(f"Ready for inference!")
counter = -1
color_red = '\033[1;31;48m'
reset = '\033[1;37;0m'
receiver = Receiver()
first_run = 1
b = ""
last_time = time.time()
boxes = {}
pbar = tqdm(total=n_images)
while not(rospy.is_shutdown()) and counter < n_images:
    new_time = time.time()
    if ((receiver.received_det or not(mode_obj_dect)) and (receiver.received_drivable or not(mode_drivable)) and (receiver.received_lane or not(mode_lane))) or first_run:
        last_time = new_time
        if not(first_run):
            if mode_obj_dect:
                boxes[image_list[counter]] = {}
                boxes[image_list[counter]]['Bboxes'] = {}
                bbox_list = receiver.BBoxes.BBoxList
                bbox_classes = receiver.BBoxes.ClassList
                bbox_start_time = receiver.BBoxes.start_stamp
                bbox_end_time = receiver.BBoxes.end_stamp
                boxes[image_list[counter]]['Start'] = bbox_start_time.to_sec()
                boxes[image_list[counter]]['End'] = bbox_end_time.to_sec()
                for idx_cls, classe in enumerate(bbox_classes):
                    if classe.data in boxes[image_list[counter]]['Bboxes']:
                        boxes[image_list[counter]]['Bboxes'][classe.data].append((bbox_list[idx_cls].Px1, bbox_list[idx_cls].Py1, bbox_list[idx_cls].Px2, bbox_list[idx_cls].Py2))
                    else:
                        boxes[image_list[counter]]['Bboxes'][classe.data] = [(bbox_list[idx_cls].Px1, bbox_list[idx_cls].Py1, bbox_list[idx_cls].Px2, bbox_list[idx_cls].Py2)]
                # E se não detetar??
            if mode_drivable:
                mask_drivable = receiver.panoptic["road"][0]
                #TODO mudar isto para incluir todas as mascaras!!!!

                # print('mask drivable')
                # print(mask_drivable)
                # cv2.imshow('teste', mask_drivable)
                path = save_path + 'drivable/masks/' + image_list[counter]
                print(path)
                cv2.imwrite(path, mask_drivable)
                # E se não detetar??
            if mode_lane:
                mask_lane = receiver.panoptic["lane divider"][0]
                # print('mask lane')
                # print(mask_lane)
                cv2.imwrite(save_path + 'lane/masks/' + image_list[counter], mask_lane)
                # E se não detetar??
            pbar.update(1)
        first_run = 0
        counter += 1
        image_path = str(curr_path / 'bdd100k/images/100k/val' / image_list[counter])
        frame = cv2.imread(image_path)
        image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        image_message.header.stamp = rospy.Time.now()
        image_message.header.frame_id = image_list[counter]
        image_pub.publish(image_message)
        receiver.reset_all()
    elif new_time - last_time > 2:
        print('Retrying!')
        last_time = new_time
        frame = cv2.imread(image_path)
        image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        image_message.header.stamp = rospy.Time.now()
        image_message.header.frame_id = image_list[counter]
        image_pub.publish(image_message)

    panoptic_state = isAllEmpty(receiver.panoptic)
    a = f"Sended: {color_red}{image_list[counter]}{reset}\nSegmentation: {color_red}{receiver.seg_frameId}{reset}\nDetection: {color_red}{receiver.det2d_frameId}{reset}"
    if a != b:
        b = a
        print(a)
    # print(f"Detection: {receiver.received_det}  |  Drivable: {receiver.received_drivable}  |  Lanes: {receiver.received_lane}")
    time.sleep(0.1)
    # print(f"Sended: {color_red}{image_list[counter]}{reset}\nSegmentation: {color_red}{receiver.seg_frameId}{reset}\nDetection: {color_red}{receiver.det2d_frameId}{reset}")
    # if not(panoptic_state):
    #     for key in receiver.panoptic:
# json_object = json.dump(boxes)
pbar.close()
json_object = json.dumps(boxes, indent = 4) 
if mode_obj_dect:
    with open(str(curr_path / "sample.json"), "w") as outfile:
        outfile.write(json_object)
    print('saved')





# Determinar numero de imagens -argparse
# Enviar imagem
# Receber e dividir por det2d, drivable e lane

# Warmup
