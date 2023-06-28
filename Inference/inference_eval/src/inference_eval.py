#!/usr/bin/python3
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from inference_manager.msg import detect2d, segmentation
import copy
import numpy as np
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import math
import time

threshold_semantic = 200
threshold_instance = 235
threshold_panoptic = 200
fps = 0.0005
max_time = rospy.Duration.from_sec(1/fps)

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

mod_path = Path(__file__).parent
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

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

class BasicReceiver:
    def __init__(self):
        topic_input = '/cameras/frontcamera'
        topic_detection2d = 'detection2d'
        topic_segmentation = 'segmentation'
        self.bridge = CvBridge()
        self.original_image = None
        self.BBoxes = None
        self.semantic = void_semantic
        self.instance = void_instance
        self.panoptic = void_panoptic
        self.subscriber_input = rospy.Subscriber(topic_input, Image, self.inputCallback)
        self.subscriber_detection2d = rospy.Subscriber(topic_detection2d, detect2d, self.detection2dCallback)
        self.subscriber_segmentation = rospy.Subscriber(topic_segmentation, segmentation, self.segmentationCallback)
    def inputCallback(self, msg):
        self.original_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.origin_stamp = msg.header.stamp
    def detection2dCallback(self, msg):
        if self.origin_stamp - msg.stamp < max_time:
            self.BBoxes = msg
        else:
            self.BBoxes = None
    def segmentationCallback(self, msg):
        if msg.Category.data == "semantic":
            # Clear previous masks
            self.semantic = copy.deepcopy(void_semantic)
            print(f"Atraso: {(self.origin_stamp - msg.stamp).to_sec()}")
            if self.origin_stamp - msg.stamp < max_time:
                for idx, seg_class in enumerate(msg.ClassList):
                    self.semantic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                    # # Temporary union of instances of the same classe -> Instance to semantic segmentation
                    # if len(self.semantic[seg_class.data]) == 0:
                    #     self.semantic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                    # else:
                    #     self.semantic[seg_class.data] = [np.maximum(self.semantic[seg_class.data][0], self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))]
        if msg.Category.data == "instance":
            # Clear previous masks
            self.instance = copy.deepcopy(void_instance)
            if self.origin_stamp - msg.stamp < max_time:
                for idx, seg_class in enumerate(msg.ClassList):
                    if len(self.instance[seg_class.data]) == 0:
                        self.instance[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                    else:
                        self.instance[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))
        if msg.Category.data == "panoptic":
            # Clear previous masks
            self.panoptic = copy.deepcopy(void_panoptic)
            if self.origin_stamp - msg.stamp < max_time:
                for idx, seg_class in enumerate(msg.ClassList):
                    if len(self.panoptic[seg_class.data]) == 0:
                        self.panoptic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                    else:
                        self.panoptic[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))