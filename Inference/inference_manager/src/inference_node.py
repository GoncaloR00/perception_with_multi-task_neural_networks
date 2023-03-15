#!/usr/bin/python3
from inference_class import Inference
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# import argparse


# TODO ARGPARSE

class InferenceNode:
    def __init__(self):
        # ---------------------------------------------------
        #   Model and inference module
        # ---------------------------------------------------

        # The inference module must have a output_organizer and a transforms 
        # function and be in the inference_modules folder
        infer_function_name = 'yolopv2_module'
        model_path = '../../../models/yolopv2.pt'
        self.inference = Inference(model_path, infer_function_name)

        # ---------------------------------------------------
        #   ROS
        # ---------------------------------------------------
        topic = 'inference/stream'
        
        subscriber_stream = rospy.Subscriber(topic, Image, self.InferenceCallback)
        self.bridge = CvBridge()

    def InferenceCallback(self,msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.inference.load_image(image)
        (det2d_class_list, det2d_list), (seg_classes, seg_list) = self.inference.infer()
        cv2.imshow('teste', seg_list[0])
        cv2.waitKey(1)


if __name__ == '__main__':
    teste = InferenceNode()
    rospy.init_node('receiver', anonymous=True)
    rospy.spin()
