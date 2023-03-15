#!/usr/bin/python3
from inference_class import Inference
import cv2
import time
# import argparse
# import rospy


infer_function_name = 'yolopv2_module'
model_path = '../../../models/yolopv2.pt'

inference = Inference(model_path, infer_function_name)

fps = 24
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('../../../data/sample_qHD.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    inference.load_image(frame)
    (det2d_class_list, det2d_list), (seg_classes, seg_list) = inference.infer()
    cv2.imshow('teste', seg_list[0])
    cv2.waitKey(int(1/fps * 1000))
  else:
    break
#   time.sleep(1/fps)





# inference.load_image(cv2.imread('../../../data/example.jpg'))

# (det2d_class_list, det2d_list), (seg_classes, seg_list) = inference.infer()

# print(seg_classes)
# cv2.imshow('teste', seg_list[0])
# cv2.waitKey(0)
