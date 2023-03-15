import argparse
import rospy

# ---------------------------------------------------
#   Model
# ---------------------------------------------------

# model_folder = pathlib.Path('../../../models/YOLOPv2')
# info_folder = pathlib.Path('../../../models/YOLOPv2')
# model_name = 'YOLOPv2.pth'
# model_info = 'YOLOPv2.txt'

# model_dir = model_folder / model_name
# info_dir = info_folder / model_info


# ---------------------------------------------------
#   Ros topics
# ---------------------------------------------------
inference_inputs_topic = '/inference/inputs'
inference_outputs_topic = '/inference/outputs'

def get_args():
    """This function..."""
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-md', '--model_dir', 
                        type=str, 
                        required=True, 
                        help="Model's file directory. \
                            The model must be in '.pth' or '.pt' format")
    parser.add_argument('-id', '--info_dir', 
                        type=str, 
                        required=True, 
                        help="Model's info directory. \
                            The file must be in '.txt' format")

def main():
    # ---------------------------------------------------
    #   Initialization
    # ---------------------------------------------------
    rospy.init_node('basic_inference', anonymous=False)
    args = get_args()
    basic_inference = BasicInference(model_dir, info_dir)
    # ---------------------------------------------------
    #   Execution
    # ---------------------------------------------------
    rospy.spin()


if __name__ == "__main__":
    main()