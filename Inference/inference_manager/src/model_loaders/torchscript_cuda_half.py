import torch

def load(original_img_size, model_img_size, model_path):
    model = torch.jit.load(model_path)
    model.to('cuda')
    model.half()
    cuda = 1
    half = 1
    engine = 0
    return model, cuda, half, engine
