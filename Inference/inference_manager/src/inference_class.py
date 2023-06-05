import torch
import importlib
import torch_tensorrt

# def convert_model(model):
#     # traced_model = torch.jit.trace(model, [torch.randn((1, 3, 640, 384)).to("cuda")])
#     trt_model = torch_tensorrt.compile(model,
#                                 inputs = [torch_tensorrt.Input((1, 3, 384, 640), dtype=torch.float32)],
#                                 enabled_precisions = {torch.float32},
#                                 truncate_long_and_double = True)
#     return trt_model

class Inference:
    def __init__(self, model_path:str, infer_function_name:str):
        # Get output reorganizer class name and import
        class_name = 'inference_modules.' + infer_function_name
        # try:
        self.output_function = importlib.import_module(class_name)
        # except:
        #     print(f"{infer_function_name} does not existe in inference_modules folder!")
        #     exit()
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == 'cpu':
            print('\033[1;31;48m' + "CUDA NOT DETECTED! Aborting..." + '\033[1;37;0m')
            exit()
        self.cuda = self.device == "cuda"
        print(f"Using device: {self.device}")
        # Load model - TODO allow more formats
        self.model = torch.jit.load(model_path)
        # with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        #     self.model = runtime.deserialize_cuda_engine(f.read())
        # load = torch.jit.load(model_path)
        # self.model = convert_model(load)
        # self.model = torch.hub.load('./', 'custom', path=model_path, source='local')
        self.model.to(self.device)

        if self.cuda:
            self.model.half()
            
        self.model.eval()
    
    def load_image(self, image):

        self.transformed_image, self.original_img_size, self.model_img_size = self.output_function.transforms(image, self.cuda, self.device)
        # print(self.transformed_image)


    def infer(self):
        with torch.no_grad():
            outputs = self.model(self.transformed_image)
        organized_outputs = self.output_function.output_organizer(outputs, self.original_img_size, self.model_img_size)
        return organized_outputs