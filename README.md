# fast-neural-style :city_sunrise: :rocket:


This repository contains a pytorch implementation of an algorithm for artistic style transfer. This repository was forked from [here](https://github.com/abhiskk/fast-neural-style), and is based on the pytorch examples repository available at [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style).The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Inference Profiling

The model inference process was profiled with the help of the pytorch profiler, and additional profiling was added to the post processing and pre processing steps.

```
ubuntu@fast-neural-style: docker run --rm --gpus all --volume "$(pwd)/:/data"   fast-neural-style-distillation python3 /data/neural_style/neural_style.py eval --content-image images/content-images/IMG_1168_imported.jpeg --model /data/student_models/student_epoch_2_Mon_Dec_30_18_38_18_202415000.model --output-image /data/output_student_la.jpg --cuda 1 --model-type 1

STAGE:2024-12-31 15:24:32 1:1 ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2024-12-31 15:24:34 1:1 ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2024-12-31 15:24:34 1:1 ActivityProfilerController.cpp:321] Completed Stage: Post Processing
stylize tensor_load_rgbimage took 0.4112 seconds
stylize unsqueeze took 0.0004 seconds
stylize cuda transfer took 0.8284 seconds
stylize preprocess_batch took 0.0010 seconds
stylize StudentTransformerNet initialization took 0.0052 seconds
stylize load_state_dict took 0.0276 seconds
stylize style_model.cuda() took 0.0030 seconds
Profiling the style model forward pass...
stylize forward pass took 2.3616 seconds
Profiling completed. Check the logs in './log'
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                               style_model_forward_pass         0.19%       3.528ms        89.33%        1.695s        1.695s       0.000us         0.00%     300.899ms     300.899ms             1
                                           aten::conv2d         0.00%      65.000us       174.50%        3.311s     137.978ms       0.000us         0.00%     218.428ms       9.101ms            24
                                      aten::convolution         0.00%      60.000us        86.95%        1.650s     137.514ms       0.000us         0.00%      88.358ms       7.363ms            12
                                     aten::_convolution         0.01%     252.000us        86.95%        1.650s     137.509ms       0.000us         0.00%      88.358ms       7.363ms            12
                                aten::cudnn_convolution         3.99%      75.715ms        86.87%        1.648s     137.375ms      70.170ms        25.27%      76.211ms       6.351ms            12
                                         aten::_to_copy         0.01%     223.000us         0.57%      10.754ms     298.722us       0.000us         0.00%      41.712ms       1.159ms            36
                                            aten::copy_         0.16%       3.117ms         0.40%       7.524ms     209.000us      24.498ms         8.82%      41.712ms       1.159ms            36
                                               aten::to         0.01%     248.000us         0.57%      10.811ms     300.306us       0.000us         0.00%      41.709ms       1.159ms            36
                                              aten::add         0.17%       3.184ms         0.22%       4.149ms     165.960us      37.718ms        13.59%      37.718ms       1.509ms            25
                                              aten::pad         0.00%      63.000us         0.35%       6.583ms     548.583us       0.000us         0.00%      33.952ms       2.829ms            12
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.898s
Self CUDA time total: 277.644ms

stylize tensor_save_bgrimage took 1.1939 seconds
Total stylize function execution took 4.9189 seconds
```

## Distillation Improvements

To improve inference time, we defined a smaller model with the same architecture [StudentTransformerNet](https://github.com/alexander-yf-yu/fast-neural-style/blob/cuda/neural_style/transformer_net.py) with 276k parameters, around 75% less than the original TransformerNet. 

We then setup a distillation loop where we ran both models on a set of 15000 images and calculated the loss of between the image produced by the original teacher model and untrained student model. Backpropagation was applied to the student model and the loss decreased.

```
ubuntu@fast-neural-style: docker run -it --gpus all   --volume "$(pwd)/:/data"   fast-neural-style-distillation   /bin/bash -c "python3 /app/neural_style/neural_style.py train_student \
    --dataset /data/train \
    --teacher-model /app/saved-models/mosaic.pth \
    --vgg-model-dir /app/vgg_dir \
    --save-model-dir /app/student_models \
    --epochs 2 \
    --max-images 15000 \
    --cuda 1; \
  exec bash"



CUDA Version 12.2.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/~ntainer-license


Tue Dec 31 17:56:06 2024    Epoch 1:    [4/15000]   Distillation Loss: 32123.609375
Tue Dec 31 17:56:49 2024    Epoch 1:    [2004/15000]    Distillation Loss: 8182.595690
Tue Dec 31 17:57:30 2024    Epoch 1:    [4004/15000]    Distillation Loss: 5142.127423
Tue Dec 31 17:58:13 2024    Epoch 1:    [3004/15000]    Distillation Loss: 4039.312359
Tue Dec 31 17:58:56 2024    Epoch 1:    [8004/15000]    Distillation Loss: 3422.285089
Tue Dec 31 17:59:40 2024    Epoch 1:    [10004/15000]   Distillation Loss: 3014.730820
Tue Dec 31 18:00:22 2024    Epoch 1:    [12004/15000]   Distillation Loss: 2720.566788
Tue Dec 31 18:01:04 2024    Epoch 1:    [14004/15000]   Distillation Loss: 2487.271722
Tue Dec 31 18:01:26 2024    Epoch 2:    [4/15000]   Distillation Loss: 1003.572632
Tue Dec 31 18:02:11 2024    Epoch 2:    [2004/15000]    Distillation Loss: 894.277443
Tue Dec 31 18:02:54 2024    Epoch 2:    [4004/15000]    Distillation Loss: 858.662606
Tue Dec 31 18:03:38 2024    Epoch 2:    [3004/15000]    Distillation Loss: 828.555800
Tue Dec 31 18:04:21 2024    Epoch 2:    [8004/15000]    Distillation Loss: 802.533305
Tue Dec 31 18:05:02 2024    Epoch 2:    [10004/15000]   Distillation Loss: 782.942599
Tue Dec 31 18:05:47 2024    Epoch 2:    [12004/15000]   Distillation Loss: 766.535369
Tue Dec 31 18:06:30 2024    Epoch 2:    [14004/15000]   Distillation Loss: 749.956614

Done, distilled student model saved at /data/student_models/StudentTransformerNet_epoch_2_Tue_Dec_31_18_06_49_2024_15000.model
```

## Results

Original images:

<img src="https://github.com/user-attachments/assets/9165404b-d6f4-4990-a28c-869e15b938ca" alt="croissant" width="300">

<img src="https://github.com/user-attachments/assets/b1acdca3-6e80-4479-b33d-fcb8b1325e88" alt="IMG_1168_imported" width="300">

Original teacher model image output:

<img src="https://github.com/user-attachments/assets/f07e70f8-7039-4636-b7d9-5820aef384f9" alt="output_teacher_croissant" width="300">

<img src="https://github.com/user-attachments/assets/9001c297-4d92-4c4c-bea0-b3dfdedbc09e" alt="output_teacher_la" width="300">

Student model image output:

<img src="https://github.com/user-attachments/assets/1c664ad6-f836-4436-89e9-7b516ec6c196" alt="output_student_croissant" width="300">

<img src="https://github.com/user-attachments/assets/b530a1f5-6f7a-4ddf-87b4-3218df45293c" alt="output_student_la" width="300">


GPU inference time decreased from 2.7s to 2.3s (excluding preprocessing and postprocessing time):

```
ubuntu@fast-neural-style: docker run --rm --gpus all --volume "$(pwd)/:/data"   fast-neural-style-distillation python3 /data/neural_style/neural_style.py eval --content-image images/content-images/IMG_1168_imported.jpeg --model /app/saved-models/mosaic.pth --output-image /data/output_teacher_la.jpg --cuda 1 --model-type 0

stylize forward pass took 2.7773 seconds
Profiling completed. Check the logs in './log'
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                               style_model_forward_pass         0.20%       4.439ms        75.13%        1.693s        1.693s       0.000us         0.00%     697.330ms     697.330ms             1
                                           aten::conv2d        -0.17%   -3849.000us       145.88%        3.288s     102.754ms       0.000us         0.00%     509.001ms      15.906ms            32
                                      aten::convolution         0.00%      76.000us        72.64%        1.637s     102.339ms       0.000us         0.00%     216.206ms      13.513ms            16
                                     aten::_convolution         0.01%     285.000us        72.64%        1.637s     102.334ms       0.000us         0.00%     216.206ms      13.513ms            16
                                aten::cudnn_convolution         3.38%      76.236ms        72.55%        1.635s     102.210ms     161.567ms        24.86%     187.519ms      11.720ms            16
                                              aten::add         0.18%       4.167ms         0.25%       5.611ms     160.314us      95.927ms        14.76%      95.927ms       2.741ms            35
                                         aten::_to_copy         0.01%     263.000us         0.57%      12.855ms     267.812us       0.000us         0.00%      76.589ms       1.596ms            48
                                            aten::copy_         0.18%       4.003ms         0.38%       8.599ms     179.146us      55.109ms         8.48%      76.589ms       1.596ms            48
                                              aten::pad         0.00%      72.000us         0.36%       8.078ms     504.875us       0.000us         0.00%      76.216ms       4.763ms            16
                                 aten::reflection_pad2d         0.17%       3.766ms         0.36%       8.006ms     500.375us      76.216ms        11.73%      76.216ms       4.763ms            16
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.254s
Self CUDA time total: 649.898ms

ubuntu@fast-neural-style: docker run --rm --gpus all --volume "$(pwd)/:/data"   fast-neural-style-distillation python3 /data/neural_style/neural_style.py eval --content-image images/content-images/IMG_1168_imported.jpeg --model /data/student_models/student_epoch_2_Mon_Dec_30_18_38_18_202415000.model --output-image /data/output_student_la.jpg --cuda 1 --model-type 1

stylize forward pass took 2.3616 seconds
Profiling completed. Check the logs in './log'
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                               style_model_forward_pass         0.19%       3.528ms        89.33%        1.695s        1.695s       0.000us         0.00%     300.899ms     300.899ms             1
                                           aten::conv2d         0.00%      65.000us       174.50%        3.311s     137.978ms       0.000us         0.00%     218.428ms       9.101ms            24
                                      aten::convolution         0.00%      60.000us        86.95%        1.650s     137.514ms       0.000us         0.00%      88.358ms       7.363ms            12
                                     aten::_convolution         0.01%     252.000us        86.95%        1.650s     137.509ms       0.000us         0.00%      88.358ms       7.363ms            12
                                aten::cudnn_convolution         3.99%      75.715ms        86.87%        1.648s     137.375ms      70.170ms        25.27%      76.211ms       6.351ms            12
                                         aten::_to_copy         0.01%     223.000us         0.57%      10.754ms     298.722us       0.000us         0.00%      41.712ms       1.159ms            36
                                            aten::copy_         0.16%       3.117ms         0.40%       7.524ms     209.000us      24.498ms         8.82%      41.712ms       1.159ms            36
                                               aten::to         0.01%     248.000us         0.57%      10.811ms     300.306us       0.000us         0.00%      41.709ms       1.159ms            36
                                              aten::add         0.17%       3.184ms         0.22%       4.149ms     165.960us      37.718ms        13.59%      37.718ms       1.509ms            25
                                              aten::pad         0.00%      63.000us         0.35%       6.583ms     548.583us       0.000us         0.00%      33.952ms       2.829ms            12
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.898s
Self CUDA time total: 277.644ms
```

CPU inference time was reduced from 64 to 27s:

```
ubuntu@fast-neural-style: docker run --rm --volume "$(pwd)/:/data"   fast-neural-style-distillation python3 /data/neural_style/neural_style.py eval --content-image images/content-images/IMG_1168_imported.jpeg --model /app/saved-models/mosaic.pth --output-image /data/output_teacher_la_cpu.jpg --model-type 0 --cuda 0

stylize forward pass took 64.2303 seconds
Profiling completed. Check the logs in './log'
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                        Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    style_model_forward_pass        37.60%       24.104s       100.00%       64.106s       64.106s             1
                   aten::pad         0.00%     796.000us         5.53%        3.542s     221.400ms            16
      aten::reflection_pad2d         5.52%        3.541s         5.52%        3.542s     221.351ms            16
                 aten::empty         0.02%      15.780ms         0.02%      15.780ms     222.254us            71
               aten::resize_         0.00%     641.000us         0.00%     641.000us      40.062us            16
                aten::conv2d         0.00%     178.000us        31.34%       20.089s        1.256s            16
           aten::convolution         0.00%     278.000us        31.34%       20.089s        1.256s            16
          aten::_convolution         0.00%     376.000us        31.34%       20.088s        1.256s            16
    aten::mkldnn_convolution        31.31%       20.072s        31.34%       20.088s        1.255s            16
           aten::as_strided_         0.00%     196.000us         0.00%     196.000us      12.250us            16
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 64.106s


ubuntu@fast-neural-style: docker run --rm --volume "$(pwd)/:/data"   fast-neural-style-distillation python3 /data/neural_style/neural_style.py eval --content-image images/content-images/IMG_1168_imported.jpeg --model /data/student_models/student_epoch_2_Mon_Dec_30_18_38_18_202415000.model --output-image
/data/output_student_la_cpu.jpg --cuda 0 --model-type 1

stylize forward pass took 27.6407 seconds
Profiling completed. Check the logs in './log'
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                        Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    style_model_forward_pass        38.85%       10.700s       100.00%       27.543s       27.543s             1
                   aten::pad         0.00%     561.000us         6.77%        1.865s     155.456ms            12
      aten::reflection_pad2d         6.77%        1.864s         6.77%        1.865s     155.410ms            12
                 aten::empty         0.00%     896.000us         0.00%     896.000us      16.291us            55
               aten::resize_         0.00%     495.000us         0.00%     495.000us      41.250us            12
                aten::conv2d         0.00%     136.000us        27.63%        7.610s     634.126ms            12
           aten::convolution         0.00%     240.000us        27.63%        7.609s     634.114ms            12
          aten::_convolution         0.00%     275.000us        27.63%        7.609s     634.094ms            12
    aten::mkldnn_convolution        27.62%        7.608s        27.63%        7.609s     634.071ms            12
           aten::as_strided_         0.00%     142.000us         0.00%     142.000us      11.833us            12
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 27.543s
```

# Usage

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop, desktop using saved models.

## Setup the environnment

### Run with virtualenv

Create a virtualenv with python3.5 or python3.6. Older versions are not supported due to a lack of compatibilty with pytorch.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run with Docker

Build the image:
```bash
docker build . -t fast-neural-style
```

Run the container:
```bash
docker run --rm --volume "$(pwd)/:/data" style eval --content-image /data/image.jpg --model /app/saved-models/mosaic.pth --output-image /data/output.jpg --cuda 0
```

## Usage
Stylize image
```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Train model
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --vgg-model-dir </path/to/vgg/folder> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).
* `--style-image`: path to style-image.
* `--vgg-model-dir`: path to folder where the vgg model will be downloaded.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments.

## Models

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/gtwnyp9n49lqs7t/saved-models.zip?dl=0) or by running the script ``download_styling_models.sh``.

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">
</div>

<div align='center'>
  <img src='images/style-images/mosaic.jpg' height="174px">
  <img src='images/output-images/amber-mosaic.jpg' height="174px">
  <img src='images/output-images/amber-candy.jpg' height="174px">
  <img src='images/style-images/candy.jpg' height="174px">
  <br>
  <img src='images/style-images/starry-night-cropped.jpg' height="174px">
  <img src='images/output-images/amber-starry-night.jpg' height="174px">
  <img src='images/output-images/amber-udnie.jpg' height="174px">
  <img src='images/style-images/udnie.jpg' height="174px">
</div>
