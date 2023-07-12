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
import argparse
import sys

topic = '/cameras/evaluation'
curr_path = Path(__file__).parent



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

class Receiver:
    def __init__(self):
        topic_detection2d = 'detection2d'
        topic_segmentation = 'segmentation'
        self.bridge = CvBridge()
        self.received_det = 0
        self.received_lane = 0
        self.received_drivable = 0
        self.semseg_frameId = "None"
        self.panseg_frameId = "None"
        self.det2d_frameId = "None"
        self.panoptic = void_panoptic
        self.semantic = void_semantic
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
            self.panseg_frameId = msg.frame_id
            # Clear previous masks
            self.panoptic = copy.deepcopy(void_panoptic)
            for idx, seg_class in enumerate(msg.ClassList):
                if len(self.panoptic[seg_class.data]) == 0:
                    self.panoptic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                else:
                    self.panoptic[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))
                if seg_class.data == "lane divider":
                    self.received_lane = 1
                    self.lane_start = msg.start_stamp
                    self.lane_end = msg.end_stamp
                if seg_class.data == "road":
                    self.received_drivable = 1
                    self.drivable_start = msg.start_stamp
                    self.drivable_end = msg.end_stamp
        if msg.Category.data == "semantic":
            self.semseg_frameId = msg.frame_id
            # Clear previous masks
            self.semantic = copy.deepcopy(void_semantic)
            for idx, seg_class in enumerate(msg.ClassList):
                if len(self.semantic[seg_class.data]) == 0:
                    self.semantic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                else:
                    self.semantic[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))
                if seg_class.data == "road":
                    self.received_drivable = 1
                    self.drivable_start = msg.start_stamp
                    self.drivable_end = msg.end_stamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                            prog = 'inference_eval',
                            description='This node send an image and receives and\
                                saves the inference results into a folder')
    
    parser.add_argument('-fn', '--folder_name', type=str, 
                        dest='folder_name', required=True, 
                        help='Name of the folder to save results')

    parser.add_argument('-nimg', '--number_images', type=int, 
                        dest='number_images', required=True, 
                        help='Quantity of images for evaluation')
    
    parser.add_argument('-obj', '--obj_dect', type=int, 
                        dest='obj_dect', required=True, 
                        help='True if object detection evaluation')
    parser.add_argument('-l', '--lane', type=int, 
                        dest='lane', required=True, 
                        help='True if lane detection evaluation')
    parser.add_argument('-da', '--drivable', type=int, 
                        dest='drivable', required=True, 
                        help='True if drivable area evaluation')
    
    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    n_images = args['number_images']
    mode_obj_dect = args['obj_dect']
    mode_drivable = args['drivable']
    mode_lane = args['lane']
    save_path = args['folder_name'] + '/labels/'
    save_path = str(curr_path / save_path) + '/'
    with open(curr_path / 'bdd100k.yaml') as f:
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
    lanes = {}
    drivable = {}
    pbar = tqdm(total=n_images)
    while not(rospy.is_shutdown()) and counter < n_images:
        new_time = time.time()
        if ((receiver.received_det or not(mode_obj_dect)) and (receiver.received_drivable or not(mode_drivable)) and (receiver.received_lane or not(mode_lane))) or first_run:# or (image_list[counter] == receiver.det2d_frameId and image_list[counter] == receiver.semseg_frameId and image_list[counter] == receiver.panseg_frameId and image_list[counter] != "None"):
            print(f'Saving {image_list[counter]}')
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
                    if len(receiver.panoptic["road"])>0:
                        mask_drivable = receiver.panoptic["road"][0]
                    else:
                        mask_drivable = receiver.semantic["road"][0]
                    # print(mask_drivable)
                    drivable[image_list[counter]] = {}
                    drivable[image_list[counter]]['Start'] = receiver.drivable_start.to_sec()
                    drivable[image_list[counter]]['End'] = receiver.drivable_end.to_sec()
                    #TODO mudar isto para incluir todas as mascaras!!!!

                    # print('mask drivable')
                    # print(mask_drivable)
                    # cv2.imshow('teste', mask_drivable)
                    path = save_path + 'drivable/masks/' + image_list[counter].split('.')[0] + '.png'
                    cv2.imwrite(path, mask_drivable)
                    # E se não detetar??
                if mode_lane:
                    mask_lane = receiver.panoptic["lane divider"][0]
                    lanes[image_list[counter]] = {}
                    lanes[image_list[counter]]['Start'] = receiver.lane_start.to_sec()
                    lanes[image_list[counter]]['End'] = receiver.lane_end.to_sec()
                    # print('mask lane')
                    # print(mask_lane)
                    cv2.imwrite(save_path + 'lane/masks/' + image_list[counter].split('.')[0] + '.png', mask_lane)
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
            receiver.reset_all()
        panoptic_state = isAllEmpty(receiver.panoptic)
        a = f"Sended: {color_red}{image_list[counter]}{reset}\nSemantic: {color_red}{receiver.semseg_frameId}{reset}\nPanoptic: {color_red}{receiver.panseg_frameId}{reset}\nDetection: {color_red}{receiver.det2d_frameId}{reset}"
        if a != b:
            b = a
            print(a)
            # print(receiver.received_det)
            # print(receiver.received_drivable)
            # print(receiver.received_lane)
        time.sleep(0.1)
    pbar.close()
    if mode_obj_dect:
        json_bboxes = json.dumps(boxes, indent = 4) 
        with open(str(curr_path / args['folder_name'] / 'labels/det_20'/ "bboxes.json"), "w") as outfile:
            outfile.write(json_bboxes)
    if mode_drivable:
        json_drivable = json.dumps(drivable, indent = 4) 
        with open(str(curr_path / args['folder_name'] / 'labels/drivable'/  "drivable.json"), "w") as outfile:
            outfile.write(json_drivable)

    if mode_lane:
        json_lane = json.dumps(lanes, indent = 4) 
        with open(str(curr_path / args['folder_name'] / 'labels/lane'/  "lane.json"), "w") as outfile:
            outfile.write(json_lane)





    # Determinar numero de imagens -argparse

    # Adicionar mensagens de drivable area e lane marking