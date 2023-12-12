<br/>
<p align="center">
  <h3 align="center">Perception with Multi-task Neural Networks</h3>
</p>

![Issues](https://img.shields.io/github/issues/GoncaloR00/perception_with_multi-task_neural_networks) 

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

![Demo Video](YOLOPv2.gif)

This GitHub repository contains the developed software in the context of my dissertation.
The developed software was designed to interface with ROS topics for data streaming, route the data through the model for inference, and send the results through other topics.
The software accommodates diverse image transformations and model formats and standardizes the output for ease of handling on receiver devices.
The software adopts a modular architecture for robustness and simplicity, utilizing different components based on the model's features. Two versions of the software were created: one for illustration and software testing and another for model evaluation.

## Built With

This project uses mainly the following frameworks:

* [ROS](https://www.ros.org/)
* [PyTorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/)
* [Torch-TensorRT](https://github.com/pytorch/TensorRT)
* [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)

## Getting Started

There are some prerequisites that must be fulfilled before installing this software. To get a local copy up and running, follow these steps.

### Prerequisites

If x64:
* Get the Nvidia drivers up to date
* Install CUDA (v12.1), CUdnn(v8.8.1) and TensorRT(v8.6) (Detailed instructions here)

If Jetson:
* Install JetPack(v5.2.2)

For both:
* PyTorch v2.0
* TorchVision v0.15 (Jetson - from source)
* Torch-TensorRT v1.4.0 (Jetson - See [this](https://github.com/pytorch/TensorRT/issues/2352))
* Polygraphy
* Numpy
* Scipy

### Installation

1. Go to the src folder of your ROS Workspace directory
```sh
cd ~/catkin_ws/src
```
2. Clone the repo
```sh
git clone https://github.com/GoncaloR00/perception_with_multi-task_neural_networks
```

3. Run catkin_make at your ROS Workspace
```sh
cd ~/catkin_ws/src && catkin_make
```
4. Create a new folder with the name "models" inside the main repository folder
```sh
mkdir ~/catkin_ws/src/perception_with_multi-task_neural_networks/models
```
5. Copy the models files into the models folder (download [here]())
## Setup for multiple devices
The first step is to connect all the devices into the same network. After that take note of the IP of each device.

To use more than one device, it is easier to change the hosts file on each device, and add the IP and hostname of the other devices:
```sh
sudo nano /etc/hosts
```
Then select a device to run the ROSCore, and, in all the remaining devices, execute the following command in the same terminal that will be used to launch the lauchfiles.
```sh
export ROS_HOSTNAME=hostname_roscoredevice
```
## Usage

This project contains some launch files for both evaluation and normal usage.

* For normal usage in a single computer:
  ```sh
  roslaunch inference_manager sync_inference.launch
  ```
* For evaluation in a single computer:
  ```sh
  roslaunch inference_eval evaluation.launch
  ```
* For multiple devices:

  * For normal use in the inference device:
  ```sh
  roslaunch inference_manager inference-jetson.launch
  ```
  * For normal use in the sender/receiver device:
  ```sh
  roslaunch inference_manager sender_receiver.launch
  ```
  * For evaluation in the inference device:
  ```sh
  roslaunch inference_manager inference-jetson.launch
  ```
  * For evaluation in the sender/receiver device:
  ```sh
  roslaunch inference_manager evaluation-jetson.launch
  ```




## Authors

* **[Gonçalo Ribeiro](https://github.com/GoncaloR00)** - *Mech. Eng. student*

## Acknowledgements
* Professor Vítor Santos - University of Aveiro
* [YOLOP](https://github.com/hustvl/YOLOP)
* [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)
* [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [Mask2Former](https://huggingface.co/facebook/mask2former-swin-tiny-cityscapes-semantic)
* [SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
* [UperNet + ConvxNet](https://huggingface.co/openmmlab/upernet-convnext-tiny)
* [RESA](https://github.com/Turoad/lanedet)
* [O2SFormer](https://github.com/zkyseu/O2SFormer)
* [UFLDv2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)