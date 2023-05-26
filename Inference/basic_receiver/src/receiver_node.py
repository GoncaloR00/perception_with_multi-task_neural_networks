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

def get_color_range(data):
    # Define a different color for each object class or instance
    frac, intNum = math.modf(256/(len(data)))
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

mod_path = Path(__file__).parent
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

# Define a different color for each object class
objDect_cls = {}
for name in data['object detection']:
    objDect_cls[data['object detection'][name]] = cv2.cvtColor(np.array([[[int(255/(math.sqrt(name))),255,255]]], np.uint8), cv2.COLOR_HSV2BGR).squeeze().tolist()

semantic_cls = get_color_range(data['semantic segmentation'])
panoptic_cls = get_color_range(data['panoptic segmentation'])

class BasicReceiver:
    def __init__(self):
        topic_input = '/cameras/frontcamera'
        topic_detection2d = 'detection2d'
        topic_segmentation = 'segmentation'
        self.subscriber_input = rospy.Subscriber(topic_input, Image, self.inputCallback)
        self.subscriber_detection2d = rospy.Subscriber(topic_detection2d, detect2d, self.detection2dCallback)
        self.subscriber_segmentation = rospy.Subscriber(topic_segmentation, segmentation, self.segmentationCallback)
        self.bridge = CvBridge()
        self.original_image = None
        self.BBoxes = None
        self.drivable_area = None
        self.lanes = None
    def inputCallback(self, msg):
        self.original_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    def detection2dCallback(self, msg):
        self.BBoxes = msg
        # self.BBox_classes = msg.ClassList
    def segmentationCallback(self, msg):
        # if msg.Category = "semantic"
        pedestrian = None
        car = None
        for idx, seg_class in enumerate(msg.ClassList):
            if seg_class.data == "pedestrian" or seg_class.data == "person":
                if pedestrian is None:
                    pedestrian = self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')
                else:
                    pedestrian = np.maximum(pedestrian, self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))
            if seg_class.data == "car":
                if car is None:
                    car = self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough')
                else:
                    car = np.maximum(car, self.bridge.imgmsg_to_cv2(msg.MaskList[idx], desired_encoding='passthrough'))

        self.drivable_area = car
        self.lanes = pedestrian
        # self.drivable_area = self.bridge.imgmsg_to_cv2(msg.MaskList[0], desired_encoding='passthrough')
        # self.lanes = self.bridge.imgmsg_to_cv2(msg.MaskList[1], desired_encoding='passthrough')
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
            if not(teste.drivable_area is None):
                image[:,:,2][teste.drivable_area >200] = 255
            if not(teste.lanes is None):
                image[:,:,0][teste.lanes >200] = 255
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
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
    cv2.destroyAllWindows()