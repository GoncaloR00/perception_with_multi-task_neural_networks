import torch
import torch_tensorrt

def load(original_img_size, model_img_size, model_path):
    model = torch.jit.load(model_path)
    model.to('cuda')
    cuda = 1
    half = 1
    engine = 0
    framework = 'torch'
    return model, cuda, half, engine, framework
