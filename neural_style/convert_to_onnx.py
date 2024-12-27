import torch
import onnx

from transformer_net import TransformerNet
# from neural_style.neural_style import IMG_SIZE

IMG_SIZE = 1024

# Suppose content_image is already a tensor on CUDA with shape (1,3,H,W).
# Or create a dummy tensor if needed:
H, W = IMG_SIZE, IMG_SIZE  # Example resolution
dummy_input = torch.randn(1, 3, H, W).cuda()

style_model = TransformerNet()
style_model.load_state_dict(torch.load("/home/ubuntu/fast-neural-style-fs/fast-neural-style/saved-models/mosaic.pth"))

# Put model on GPU if not already
style_model.cuda()
style_model.eval()

torch.onnx.export(
    style_model,            # PyTorch model
    dummy_input,            # Dummy input to trace the model
    "style_model.onnx",     # Output ONNX file
    export_params=True,     # Store trained parameters
    opset_version=11,       # ONNX opset version
    do_constant_folding=True, 
    input_names=["input"],  
    output_names=["output"],
)

onnx_model = onnx.load("style_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

