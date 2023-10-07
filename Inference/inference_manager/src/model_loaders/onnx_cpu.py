import onnxruntime

def load(original_img_size, model_img_size, model_path):
    model = onnxruntime.InferenceSession(model_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    cuda = 0
    half = 0
    engine = 0
    framework = 'onnx'
    return model, cuda, half, engine, framework