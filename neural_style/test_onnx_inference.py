import torch
import onnxruntime as ort
import numpy as np
from PIL import Image

def load_pytorch_model(pth_path):
    """
    Load your PyTorch model from a `.pth` or `.pt` file.
    (Or if you have it in code, just instantiate and load_state_dict.)
    """
    # from .transformer_net import TransformerNet  # Adjust import if needed
    from transformer_net import TransformerNet

    model = TransformerNet()
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image_path, size=(256, 256)):
    """
    Basic preprocess: resize, convert to float32, scale to [0,1], convert to NCHW for PyTorch/ONNX.
    Adjust if your model doesn't expect normalization.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0  # if your model expects [0,1]
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0)   # Add batch dimension -> NCHW
    return img_np  # shape: (1,3,H,W)

def postprocess_image(output_tensor, filename):
    """
    Convert output tensor in [0,1] range back to uint8 image.
    """
    # output_tensor: shape (1,3,H,W) or (3,H,W)
    output_tensor = np.squeeze(output_tensor, axis=0)  # remove batch dim -> (3,H,W)
    output_tensor = np.transpose(output_tensor, (1,2,0))  # (3,H,W) -> (H,W,3)
    output_tensor = np.clip(output_tensor * 255.0, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(output_tensor)
    out_img.save(filename)
    print(f"Saved {filename}")

def main():
    # Paths
    pytorch_model_path = "saved-models/mosaic.pth"   # Adjust to your model
    onnx_model_path = "style_model.onnx"            # Adjust if needed
    image_path = "images/content-images/croissant.jpg"
    
    # 1. Load PyTorch model
    pytorch_model = load_pytorch_model(pytorch_model_path)
    
    # 2. Preprocess the same input for both models
    input_np = preprocess_image(image_path, size=(256, 256))  # or 1024 if you want
    input_torch = torch.from_numpy(input_np)  # shape: (1,3,H,W)
    
    # 3. Run PyTorch inference
    with torch.no_grad():
        output_torch = pytorch_model(input_torch)  # shape (1,3,H,W)
    output_torch_np = output_torch.cpu().numpy()  # convert to numpy for comparison
    
    # 4. Load ONNX model and create session
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    
    # The ONNX model input name might differ from "input"; 
    # find the actual input name from session.get_inputs()
    onnx_input_name = session.get_inputs()[0].name
    
    # 5. Run ONNX inference
    output_onnx = session.run(None, {onnx_input_name: input_np})  # returns a list of outputs
    output_onnx_np = output_onnx[0]  # assume the model has a single output
    
    # 6. Compare outputs numerically
    #    e.g. Mean Absolute Error, or just min/max
    mae = np.mean(np.abs(output_torch_np - output_onnx_np))
    print(f"Mean Absolute Error between PyTorch and ONNX: {mae:.6f}")
    print("PyTorch output stats:", 
          "min:", output_torch_np.min(), 
          "max:", output_torch_np.max(), 
          "mean:", output_torch_np.mean())
    print("ONNX output stats:", 
          "min:", output_onnx_np.min(), 
          "max:", output_onnx_np.max(), 
          "mean:", output_onnx_np.mean())
    
    # 7. Save both images for visual comparison
    postprocess_image(output_torch_np, "output_pytorch.jpg")
    postprocess_image(output_onnx_np, "output_onnx.jpg")

if __name__ == "__main__":
    main()

