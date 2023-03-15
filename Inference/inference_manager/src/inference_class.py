import torch
import importlib

class Inference:
    def __init__(self, model_path:str, infer_function_name:str):
        # Get output reorganizer class name and import
        class_name = 'inference_modules.' + infer_function_name
        try:
            self.output_function = importlib.import_module(class_name)
        except:
            print(f"{infer_function_name} does not existe in inference_modules folder!")
            exit()
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda = self.device == "cuda"
        print(f"Using device: {self.device}")
        # Load model file (pt or pth file)
        # self.model = torch.load(model_path)
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)
        if self.cuda:
            self.model.half()
            
        self.model.eval()
    
    def load_image(self, image):

        self.transformed_image, self.original_img_size, self.model_img_size = self.output_function.transforms(image, self.cuda, self.device)


    def infer(self):
        with torch.no_grad():
            outputs = self.model(self.transformed_image)
        organized_outputs = self.output_function.output_organizer(outputs, self.original_img_size, self.model_img_size)
        return organized_outputs
