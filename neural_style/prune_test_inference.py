import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import Variable

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


def stylize(args):
    total_start = time.time()  # Start total timing

    start = time.time()
    content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.content_scale)
    print(f"stylize tensor_load_rgbimage took {time.time() - start:.4f} seconds")

    start = time.time()
    content_image = content_image.unsqueeze(0)
    print(f"stylize unsqueeze took {time.time() - start:.4f} seconds")

    if args.cuda:
        start = time.time()
        content_image = content_image.cuda()
        print(f"stylize cuda transfer took {time.time() - start:.4f} seconds")

    start = time.time()
    with torch.no_grad():
        content_image = Variable(utils.preprocess_batch(content_image))
        print(f"stylize preprocess_batch took {time.time() - start:.4f} seconds")

    start = time.time()
    style_model = TransformerNet()
    print(f"stylize TransformerNet initialization took {time.time() - start:.4f} seconds")

    start = time.time()
    style_model.load_state_dict(torch.load(args.model))
    print(f"stylize load_state_dict took {time.time() - start:.4f} seconds")

    if args.cuda:
        start = time.time()
        style_model.cuda()
        print(f"stylize style_model.cuda() took {time.time() - start:.4f} seconds")

    style_model.eval()

    if args.prune_percent:
        print("Pruning the conv3 layer...")
        conv_layer = style_model.conv3.conv2d
        # Prune 30% of the filters by L1 norm, removing entire filters along dim=0
        amount_to_prune = args.prune_percent
        prune.ln_structured(
            conv_layer,
            name="weight",   # parameter to prune
            amount=amount_to_prune,
            n=1,             # L1 norm
            dim=0            # prune entire output filters
        )
        prune.remove(conv_layer, 'weight')

        # If the user provided a path to save the pruned model:
        if args.save_pruned_model:
            torch.save(style_model.state_dict(), args.save_pruned_model)
            print(f"Pruned model saved to {args.save_pruned_model}")


    print("Profiling the style model forward pass...")
    start = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("style_model_forward_pass"):
            with torch.no_grad():
                output = style_model(content_image)
    forward_pass_time = time.time() - start
    print(f"stylize forward pass took {forward_pass_time:.4f} seconds")
    print("Profiling completed. Check the logs in './log'")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("stylize tensor_save_bgrimage")
    start = time.time()
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)
    print(f"stylize tensor_save_bgrimage took {time.time() - start:.4f} seconds")

    total_time = time.time() - total_start
    print(f"Total stylize function execution took {total_time:.4f} seconds")


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
                                  help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--style-weight", type=float, default=5.0,
                                  help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--prune-percent", type=float, default=None,
                                 help="Percentage by which to prune a specified layer")
    eval_arg_parser.add_argument("--save-pruned-model", type=str, default=None,
                             help="Path to save the pruned model (optional). If not provided, model isn't saved.")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        # check_paths(args)
        # train(args)
        pass
    else:
        stylize(args)


if __name__ == "__main__":
    main()
