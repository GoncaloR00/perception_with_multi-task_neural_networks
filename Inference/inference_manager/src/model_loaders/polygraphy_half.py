from polygraphy.util import load_file as load_engine
from polygraphy.backend.trt import EngineFromBytes

def load(original_img_size, model_img_size, model_path):
    serialized_engine = load_engine(model_path);
    model = EngineFromBytes(serialized_engine)
    cuda = 1
    half = 1
    engine = 0
    framework = 'polygraphy'
    return model, cuda, half, engine, framework