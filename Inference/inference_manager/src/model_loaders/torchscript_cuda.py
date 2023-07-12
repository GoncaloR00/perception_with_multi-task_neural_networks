import torch

def load(original_img_size, model_img_size, model_path):
    model = torch.jit.load(model_path)
    model.to('cuda')
    cuda = 1
    half = 0
    engine = 0
    return model, cuda, half, engine
