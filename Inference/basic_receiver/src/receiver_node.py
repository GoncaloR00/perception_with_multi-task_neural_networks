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
import timeit

threshold_semantic = 200
threshold_instance = 235
threshold_panoptic = 200

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
    def detection2dCallback(self, msg):
        self.BBoxes = msg
    def segmentationCallback(self, msg):
        if msg.Category.data == "semantic":
            # Clear previous masks
            self.semantic = copy.deepcopy(void_semantic)
            for idx, seg_class in enumerate(msg.ClassList):
                self.semantic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                # # União temporária de instâncias da mesma classe -> Instâncias para semântica
                # if len(self.semantic[seg_class.data]) == 0:
                #     self.semantic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                # else:
                #     self.semantic[seg_class.data] = [np.maximum(self.semantic[seg_class.data][0], self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))]
        if msg.Category.data == "instance":
            # Clear previous masks
            self.instance = copy.deepcopy(void_instance)
            for idx, seg_class in enumerate(msg.ClassList):
                if len(self.instance[seg_class.data]) == 0:
                    self.instance[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                else:
                    self.instance[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))
        if msg.Category.data == "panoptic":
            # Clear previous masks
            self.panoptic = copy.deepcopy(void_panoptic)
            for idx, seg_class in enumerate(msg.ClassList):
                if len(self.panoptic[seg_class.data]) == 0:
                    self.panoptic[seg_class.data] = [self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')]
                else:
                    self.panoptic[seg_class.data].append(self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))

if __name__ == '__main__':
    teste = BasicReceiver()
    rospy.init_node('image_plotter', anonymous=True)
    namespace = rospy.get_namespace()
    try:
        window_name = f"Camera: {namespace.split('/')[-3]}  |  Model: {namespace.split('/')[-2]}"
    except:
        window_name = f"Camera: NO DATA  |  Model: NO DATA"
    while not rospy.is_shutdown():
        if not(teste.original_image is None):
            image = teste.original_image
            image = copy.copy(image)
            semantic_state = isAllEmpty(teste.semantic)
            instance_state = isAllEmpty(teste.instance)
            panoptic_state = isAllEmpty(teste.panoptic)
            if not(semantic_state and instance_state and panoptic_state):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                if not(semantic_state):
                    for key in teste.semantic:
                        if len(teste.semantic[key])>0:
                            for mask in teste.semantic[key]:
                                image[:,:,0][mask>threshold_semantic] = int((semantic_cls[key][0] + semantic_cls[key][1])/2)
                                image[:,:,1][mask>threshold_semantic] = 255

                if not(instance_state):
                    # Colors randomly distributed
                    for key in teste.instance:
                            if len(teste.instance[key])>0:
                                counter = 0
                                for mask in teste.instance[key]:
                                    color = 0 + counter
                                    if color > 179:
                                        color = 0
                                        counter = 10
                                    else:
                                        counter += 10
                                    image[:,:,0][mask>threshold_instance] = color
                                    image[:,:,1][mask>threshold_instance] = 255
                        
                        # # Colors distributed by classes
                        # if len(teste.instance[key])>0:
                        #     color_range = semantic_cls[key][1] - semantic_cls[key][0]
                        #     counter = 0
                        #     for mask in teste.instance[key]:
                        #         color = semantic_cls[key][0] + counter
                        #         if color > semantic_cls[key][1]:
                        #             color = semantic_cls[key][0]
                        #             counter = 1
                        #         else:
                        #             counter += 1
                        #         image[:,:,0][mask>threshold_instance] = color
                        #         image[:,:,1][mask>threshold_instance] = 255
                if not(panoptic_state):
                    for key in teste.panoptic:
                        # Colors distributed by classes
                        if len(teste.panoptic[key])>0:
                            color_range = panoptic_cls[key][1] - panoptic_cls[key][0]
                            counter = 0
                            for mask in teste.panoptic[key]:
                                color = panoptic_cls[key][0] + counter
                                if color > panoptic_cls[key][1]:
                                    color = panoptic_cls[key][0]
                                    counter = 1
                                else:
                                    counter += 1
                                image[:,:,0][mask>threshold_instance] = color
                                image[:,:,1][mask>threshold_instance] = 120

                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            # toc = timeit.default_timer()
            # print(f"Time = {toc-tic}")
            # if not(teste.drivable_area is None):
            #     image[:,:,2][teste.drivable_area >200] = 255
            # if not(teste.lanes is None):
            #     image[:,:,0][teste.lanes >200] = 255

            # Draw bounding boxes
            if not(teste.BBoxes is None):
                bboxes = teste.BBoxes
                bbox_list = bboxes.BBoxList
                bbox_classes = bboxes.ClassList
                fontFace=cv2.FONT_HERSHEY_COMPLEX
                thickness= 1
                fontScale=0.5
                for idx, bbox in enumerate(bbox_list):
                    c1 = (bbox.Px1, bbox.Py1)
                    c2 = (bbox.Px2, bbox.Py2)
                    text = bbox_classes[idx].data
                    color = objDect_cls[text]
                    image = cv2.rectangle(image, c1, c2, color = color, thickness=2, lineType=cv2.LINE_AA)
                    top_center = [int((c1[0]+c2[0])/2), c1[1]]
                    label_size = cv2.getTextSize(text=text, fontFace=fontFace, thickness=thickness, fontScale=fontScale)
                    org = (top_center[0]-int(label_size[0][0]/2),top_center[1]-int(label_size[0][1]/2))
                    image = cv2.putText(image, text=text, org=org, fontFace=fontFace, thickness=thickness, fontScale=fontScale, color=color)
            # if len(teste.instance["car"]) > 0:
            #     cv2.imshow('carro1', teste.instance["car"][0])
            #     if len(teste.instance["car"]) > 1:
            #         cv2.imshow('carro2', teste.instance["car"][1])
            #         if len(teste.instance["car"]) > 2:
            #             cv2.imshow('carro3', teste.instance["car"][2])
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
    cv2.destroyAllWindows()