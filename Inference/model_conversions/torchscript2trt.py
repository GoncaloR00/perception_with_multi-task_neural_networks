#!/usr/bin/python3
import torch_tensorrt
import torch
import pickle


# traced_model = torch.jit.trace(model, [torch.randn((1, 3, 640, 384)).to("cuda")])
model_path = "../../models/yolov5s.torchscript"
model = torch.jit.load(model_path).to('cuda')
trt_model = torch_tensorrt.compile(model,
                            inputs = [torch_tensorrt.Input((1, 3, 384, 640))],
                            enabled_precisions = {torch.float32}, truncate_long_and_double = True, workspace_size=4194304)

torch.jit.save(trt_model, "teste2.trt")
# try:
#     torch.save(trt_model, "teste.trt")
# except:
#     print("Torch.save nao dá")

# try:
#     torch.jit.save(trt_model, "teste2.trt")
# except:
#     print("Torch.jit.save nao dá")

# try:
#     with open('filename.pickle', 'wb') as handle:
#         pickle.dump(trt_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
# except:
#     print("pickle nao dá")