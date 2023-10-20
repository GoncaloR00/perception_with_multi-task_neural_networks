#!/usr/bin/python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time

fps = 24
# topic = 'cameras/frontcamera'
# topic = 'inference/stream'
topic = '/cameras/frontcamera'
image_pub = rospy.Publisher(topic,Image, queue_size=10)
bridge = CvBridge()
# Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture('/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/sample_qHD.mp4')
# cap = cv2.VideoCapture('/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/video2.mov')
# cap = cv2.VideoCapture('/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/b2de6f59-9f74dea1.mov')
# cap = cv2.VideoCapture('/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/b2de6f59-9f74dea1.mp4')
cap = cv2.VideoCapture("/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/portugal/Video_output0.mp4")


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
rospy.init_node('sender', anonymous=False)

# Read until video is completed
while(cap.isOpened() and not rospy.is_shutdown()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (853, 480))
    # cv2.imwrite('/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/frame.png', frame)
    if ret == True:
      image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
      image_message.header.stamp = rospy.Time.now()
      image_pub.publish(image_message)
    # Break the loop
    else: 
      break
    time.sleep(1/fps)
cv2.destroyAllWindows()


# # TEMPORARIO
# rospy.init_node('sender', anonymous=False)
# # frame = cv2.imread("/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/b2de6f59-9f74dea1.jpg")
# frame = cv2.imread("/home/gribeiro/catkin_ws/src/perception_with_multi-task_neural_networks/Inference/basic_sender/src/data/portugal/Video_output6.png")
# while (not rospy.is_shutdown()):
#   image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
#   image_message.header.stamp = rospy.Time.now()
#   image_pub.publish(image_message)
#   time.sleep(1/fps)