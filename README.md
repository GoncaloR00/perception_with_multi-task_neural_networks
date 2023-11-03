<br/>
<p align="center">
  <a href="https://github.com/GoncaloR00/perception_with_multi-task_neural_networks">
    <img src="docs/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Perception with Multi-task Neural Networks</h3>

  <p align="center">
    <a href="https://github.com/GoncaloR00/perception_with_multi-task_neural_networks">View Demo</a>
    .
  </p>
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

![Screen Shot](docs/screenshot.png)

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
cd ~\catkin_ws\src
```
2. Clone the repo
```sh
git clone https://github.com/GoncaloR00/perception_with_multi-task_neural_networks
```

3. Run catkin_make at your ROS Workspace
```sh
cd ~\catkin_ws\src && catkin_make
```

## Usage

This project contains some launch files for both evaluation and normal usage:




## Authors

* **[Gonçalo Ribeiro](https://github.com/GoncaloR00)** - *Mech. Eng. student*

## Acknowledgements

* [YOLOv7](https://github.com/WongKinYiu/yolov7)
