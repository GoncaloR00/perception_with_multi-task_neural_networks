#!/usr/bin/python3
from inference_class import Inference
import cv2
# import argparse
# import rospy

infer_function_name = 'yolopv2_module'
model_path = '../../../models/yolopv2.pt'

inference = Inference(model_path, infer_function_name)

inference.load_image(cv2.imread('../../../data/example.jpg'))

(det2d_class_list, det2d_list), (seg_classes, seg_list) = inference.infer()

# print(seg_classes)
cv2.imshow('teste', seg_list[0])
cv2.waitKey(0)
