#!/usr/bin/python3
from inference_class import Inference
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from inference_manager.msg import detect2d, segmentation, BBox
# from geometry_msgs.msg import Quaternion
from std_msgs.msg import String

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
        topic_input = '/inference/stream'
        topic_detection2d = 'inference/detection2d'
        topic_segmentation = 'inference/segmentation'
        subscriber_stream = rospy.Subscriber(topic_input, Image, self.InferenceCallback)
        self.detection2d_pub = rospy.Publisher(topic_detection2d,detect2d, queue_size=10)
        self.segmentation_pub = rospy.Publisher(topic_segmentation,segmentation, queue_size=10)
        self.bridge = CvBridge()

    def InferenceCallback(self,msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.inference.load_image(image)
        (det2d_class_list, det2d_list), (seg_classes, seg_list) = self.inference.infer()
        detect2d_msg = detect2d()
        coords = []
        strings = []
        string = String()
        coord = BBox()
        for k, i in enumerate(det2d_list):
            string.data = det2d_class_list[k]
            coord.Px1 = i[0][0]
            coord.Py1 = i[0][1]
            coord.Px2 = i[1][0]
            coord.Py2 = i[1][1]
            coords.append(coord)
            strings.append(string)
        detect2d_msg.BBoxList = coords
        detect2d_msg.ClassList = strings
        segmentation_msg = segmentation()
        mask_msg = []
        strings = []
        for k, mask in enumerate(seg_list):
            image_message = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.append(image_message)
            string.data = seg_classes[k]
            strings.append(string)

        segmentation_msg.ClassList = strings
        segmentation_msg.MaskList = mask_msg
        self.detection2d_pub.publish(detect2d_msg)

        self.segmentation_pub.publish(segmentation_msg)
        # cv2.imshow('teste', seg_list[0])
        # cv2.waitKey(1)


if __name__ == '__main__':
    teste = InferenceNode()
    rospy.init_node('receiver', anonymous=True)
    rospy.spin()
