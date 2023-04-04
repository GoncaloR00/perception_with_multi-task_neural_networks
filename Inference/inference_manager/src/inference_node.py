#!/usr/bin/python3
from inference_class import Inference
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from inference_manager.msg import detect2d, segmentation, BBox
from std_msgs.msg import String
import argparse
import sys


class InferenceNode:
    def __init__(self, infer_function_name:str, model_path:str, source:str):
        # ---------------------------------------------------
        #   Model and inference module
        # ---------------------------------------------------

        # The inference module must have a output_organizer and a transforms 
        # function and be in the inference_modules folder
        # infer_function_name = 'yolopv2_module'
        # model_path = '../../../models/yolopv2.pt'
        self.inference = Inference(model_path, infer_function_name)

        # topic_input = '/cameras/frontcamera'
        topic_detection2d = 'detection2d'
        topic_segmentation = 'segmentation'
        subscriber_stream = rospy.Subscriber(source, Image, self.InferenceCallback)
        self.detection2d_pub = rospy.Publisher(topic_detection2d,detect2d, queue_size=10)
        self.segmentation_pub = rospy.Publisher(topic_segmentation,segmentation, queue_size=10)
        self.bridge = CvBridge()

    def InferenceCallback(self,msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.inference.load_image(image)
        detections_2d, segmentations = self.inference.infer()
        if not(detections_2d is None):
            (det2d_class_list, det2d_list) = detections_2d
            detect2d_msg = detect2d()
            coords = []
            strings = []
            for k, i in enumerate(det2d_list):
                string = String()
                string.data = det2d_class_list[k]
                coord = BBox()
                coord.Px1 = i[0][0]
                coord.Py1 = i[0][1]
                coord.Px2 = i[1][0]
                coord.Py2 = i[1][1]
                coords.append(coord)
                strings.append(string)
            detect2d_msg.BBoxList = coords
            detect2d_msg.ClassList = strings
            self.detection2d_pub.publish(detect2d_msg)
        if not(segmentations is None):
            (seg_classes, seg_list) = segmentations
            segmentation_msg = segmentation()
            mask_msg = []
            strings = []
            for k, mask in enumerate(seg_list):
                image_message = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
                mask_msg.append(image_message)
                string = String()
                string.data = seg_classes[k]
                strings.append(string)

            segmentation_msg.ClassList = strings
            segmentation_msg.MaskList = mask_msg
            self.segmentation_pub.publish(segmentation_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                            prog = 'inference_node',
                            description='This node receives an image as input and\
                            outputs the result of the inference')
    
    parser.add_argument('-fn', '--function_name', type=str, 
                        dest='infer_function', required=True, 
                        help='Name of the module with the output_organizer and \
                            transforms functions')
    
    parser.add_argument('-mp', '--model_path', type=str, 
                        dest='model_path', required=True, 
                        help='Model directory')
    
    parser.add_argument('-sr', '--source', type=str, 
                        dest='source', required=True, 
                        help='Topic with the image messages to process')
    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))
    # model_name = args['model_path'].split('/')[-1].split('.')[0]
    # source_name = args['source'].split('/')[-1]
    teste = InferenceNode(infer_function_name = args['infer_function'], 
                          model_path = args['model_path'], 
                          source = args['source']
                          )
    rospy.init_node('inference_node', anonymous=False)
    rospy.spin()