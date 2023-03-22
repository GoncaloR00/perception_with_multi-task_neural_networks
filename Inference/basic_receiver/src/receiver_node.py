#!/usr/bin/python3
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from inference_manager.msg import detect2d, segmentation
import copy


class BasicReceiver:
    def __init__(self):
        topic_input = 'inference/stream'
        topic_detection2d = 'inference/detection2d'
        topic_segmentation = 'inference/segmentation'
        self.subscriber_input = rospy.Subscriber(topic_input, Image, self.inputCallback)
        self.subscriber_detection2d = rospy.Subscriber(topic_detection2d, detect2d, self.detection2dCallback)
        self.subscriber_segmentation = rospy.Subscriber(topic_segmentation, segmentation, self.segmentationCallback)
        self.bridge = CvBridge()
        self.original_image = None
        self.BBox_list = None
        self.drivable_area = None
        self.lanes = None
    def inputCallback(self, msg):
        self.original_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    def detection2dCallback(self, msg):
        self.BBox_list = msg.BBoxList
    def segmentationCallback(self, msg):
        self.drivable_area = self.bridge.imgmsg_to_cv2(msg.MaskList[0], desired_encoding='passthrough')
        self.lanes = self.bridge.imgmsg_to_cv2(msg.MaskList[1], desired_encoding='passthrough')
if __name__ == '__main__':
    teste = BasicReceiver()
    rospy.init_node('image_plotter', anonymous=True)
    while True:
        if not(teste.original_image is None):
            image = teste.original_image
            image = copy.copy(image)
            if not(teste.drivable_area is None):
                image[:,:,2][teste.drivable_area !=0] = 255
            if not(teste.lanes is None):
                image[:,:,0][teste.lanes !=0] = 255
            if not(teste.BBox_list is None):
                for bbox in teste.BBox_list:
                    c1 = (bbox.Px1, bbox.Py1)
                    c2 = (bbox.Px2, bbox.Py2)
                    image = cv2.rectangle(image, c1, c2, [0,255,255], thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow('teste', image)
            cv2.waitKey(1)