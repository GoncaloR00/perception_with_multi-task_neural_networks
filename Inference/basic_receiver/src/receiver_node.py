#!/usr/bin/python3
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from inference_manager.msg import detect2d, segmentation
import numpy as np
# from geometry_msgs.msg import Quaternion
# from std_msgs.msg import String

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
        # print(not(teste.original_image is None))
        if not(teste.original_image is None) and not(teste.BBox_list is None) and not(teste.drivable_area is None) and not(teste.lanes is None):
            image = teste.original_image
            image = cv2.resize(image, (1280,720), interpolation=cv2.INTER_LINEAR)
            # Get a mask for each 
            # drivable_area_R = teste.drivable_area[:,:,2] == 255
            # drivable_area_G = teste.drivable_area[:,:,1] == 255
            # drivable_area_B = teste.drivable_area[:,:,0] == 255
            # drivable_area_R = teste.lanes[:,:,2] == 255
            # drivable_area_G = teste.lanes[:,:,1] == 255
            # drivable_area_B = teste.lanes[:,:,0] == 255


            image[:,:,2][teste.drivable_area == 255] = 255
            # mask2 = teste.drivable_area[:,:,1] == 255
            # image[:,:,1][mask2] = np.around((image[:,:,1][mask2])*2 ,0)
            # image[:,:,1][mask2] = 255
            # mask_idx = teste.drivable_area != 0
            # red = np.zeros_like(image)
            # red[:,:,2]=255
            # image[mask_idx] = red[mask_idx]
            # print(teste.drivable_area.shape)
            # print(image.shape)
            # image[:][0][teste.drivable_area[0] == 255] = 255
            # image[teste.drivable_area == 0][1] = 0
            # image[teste.drivable_area == 0][2] = 0
            cv2.imshow('teste', image)
            cv2.waitKey(1)
    # rospy.spin()