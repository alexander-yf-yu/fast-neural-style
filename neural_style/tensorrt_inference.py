import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import argparse
import time

def preprocess_image(image_path, target_size=(1024, 1024)):
    """
    Preprocess the input image for TensorRT inference.
    - Resizes and normalizes the image.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32)
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return np.ascontiguousarray(image)  # Ensure the array is contiguous

# def preprocess_image(image_path, target_size=(1024, 1024)):
#     """
#     Preprocess the input image for TensorRT inference.
#     - Resizes the image to the target size.
#     """
#     image = Image.open(image_path).convert("RGB")
#     image = image.resize(target_size)  # Resize to the target size
#     image = np.array(image).astype(np.float32)  # Convert to float32
#     image = image.transpose(2, 0, 1)  # Convert from HWC to CHW
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return np.ascontiguousarray(image)  # Ensure the array is contiguous

# def postprocess_image(output_tensor, output_path, target_size=(1024, 1024)):
#     """
#     Postprocess the output tensor and save it as an image.
#     - Converts the output tensor to an image format without scaling.
#     """
#     output_tensor = output_tensor.squeeze()  # Remove batch dimension
#     output_tensor = output_tensor.transpose(1, 2, 0)  # Convert from CHW to HWC
#     output_tensor = output_tensor.clip(0, 255).astype(np.uint8)  # Clip and convert to uint8
#     output_image = Image.fromarray(output_tensor)  # Create image from array
#     output_image = output_image.resize(target_size)  # Resize to target size
#     output_image.save(output_path)  # Save the output image
#     print(f"Output image saved to {output_path}")

def postprocess_image(output_tensor, output_path, target_size=(1024, 1024)):
    output_tensor = output_tensor.squeeze()  # Remove batch dimension
    output_tensor = output_tensor.transpose(1, 2, 0)  # CHW to HWC
    output_tensor = (output_tensor * 255.0).clip(0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_tensor)
    output_image = output_image.resize(target_size)
    output_image.save(output_path)
    print(f"Output image saved to {output_path}")

def main(args):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Print bindings to debug tensor names
    for binding in engine:
        print(f"Binding: {binding}, Shape: {engine.get_binding_shape(engine.get_binding_index(binding))}, Is Input: {engine.binding_is_input(binding)}")

    input_binding_idx = engine.get_binding_index(engine.get_tensor_name(0))
    output_binding_idx = engine.get_binding_index(engine.get_tensor_name(1))

    input_shape = (1, 3, 1024, 1024)  # Fixed input shape
    input_size = np.product(input_shape)
    output_shape = (1, 3, 1024, 1024)
    output_size = np.product(output_shape)

    d_input = cuda.mem_alloc(int(input_size * np.float32().nbytes))
    d_output = cuda.mem_alloc(int(output_size * np.float32().nbytes))

    input_image = preprocess_image(args.input_image, target_size=(1024, 1024))
    print("Raw input min:", input_image.min(),
      "max:", input_image.max(),
      "mean:", input_image.mean())
    
    cuda.memcpy_htod(d_input, input_image)

    start_time = time.time()
    context.execute_v2([int(d_input), int(d_output)])
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    host_output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_output, d_output)
    print("Raw output min:", host_output.min(),
      "max:", host_output.max(),
      "mean:", host_output.mean())

    postprocess_image(host_output, args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Inference Script")
    parser.add_argument("--engine-path", type=str, required=True, help="Path to the TensorRT engine file")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-image", type=str, required=True, help="Path to save the output image")
    args = parser.parse_args()

    main(args)

