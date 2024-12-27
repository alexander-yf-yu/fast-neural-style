import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


def perceptual_loss(student_output, teacher_output):
    vgg = Vgg16().eval().cuda()
    student_features = vgg(student_output)
    teacher_features = vgg(teacher_output)
    loss = 0
    for sf, tf in zip(student_features, teacher_features):
        loss += torch.nn.functional.mse_loss(sf, tf)
    return loss

def train_student(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    # Dataset setup
    transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    # Load teacher model
    teacher_model = TransformerNet()
    teacher_model.load_state_dict(torch.load(args.teacher_model))
    teacher_model.eval()

    # Create student model
    student_model = StudentTransformerNet()
    optimizer = Adam(student_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    # Load VGG for perceptual loss (optional)
    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
    vgg.eval()

    if args.cuda:
        teacher_model.cuda()
        student_model.cuda()
        vgg.cuda()

    for e in range(args.epochs):
        student_model.train()
        agg_distillation_loss = 0.0
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()

            # Teacher's prediction (frozen)
            with torch.no_grad():
                teacher_output = teacher_model(x)

            # Student's prediction
            student_output = student_model(x)

            # Distillation loss: match student output to teacher output
            pixel_loss = mse_loss(student_output, teacher_output)

            # Optional: Perceptual loss via VGG
            teacher_features = vgg(teacher_output)
            student_features = vgg(student_output)
            perceptual_loss = sum(mse_loss(sf, tf) for sf, tf in zip(student_features, teacher_features))

            # Total loss
            total_loss = pixel_loss + args.perceptual_weight * perceptual_loss
            total_loss.backward()
            optimizer.step()

            agg_distillation_loss += total_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tDistillation Loss: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                    agg_distillation_loss / (batch_id + 1),
                )
                print(mesg)

    # Save the student model
    student_model.eval()
    student_model.cpu()
    save_model_filename = "student_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(student_model.state_dict(), save_model_path)

    print("\nDone, distilled student model saved at", save_model_path)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.repeat(args.batch_size, 1, 1, 1)
    style = utils.preprocess_batch(style)
    if args.cuda:
        style = style.cuda()
    style_v = Variable(style, volatile=True)
    style_v = utils.subtract_imagenet_mean_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            xc = Variable(x.data.clone(), volatile=True)

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)



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
                with torch.cuda.amp.autocast():
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
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
