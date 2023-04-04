#!/usr/bin/python3
import onnx
import torch

model_path = "../../models/yolov5s.onnx"
model = onnx.load(model_path)

model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted.pt')

