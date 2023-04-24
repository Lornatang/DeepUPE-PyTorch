# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF_torch ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.utils.data import distributed, DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import CUDAPrefetcher, ImageDataset, PairedImageDataset
from imgproc import random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from test import test
from utils import build_iqa_model, load_pretrained_state_dict, load_resume_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter

# Default to start training from scratch
start_epoch = 0

# Initialize the image clarity evaluation index
best_psnr = 0.0
best_ssim = 0.0


def main():
    # Initialize global variables
    global start_epoch, best_psnr, best_ssim

    # Read YAML configuration file
    with open("configs/train/DEEPUPE_FIVEK.YAML", "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Define the dataset
    train_prefetcher, test_prefetcher = load_dataset(config, device)
    cr_model = build_model(config, device)
    pixel_criterion, color_criterion, tv_criterion = define_loss(config, device)
    optimizer = define_optimizer(cr_model, config)

    # Load the pre-trained model weights and fine-tune the model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_MODEL"]:
        cr_model = load_pretrained_state_dict(cr_model,
                                              config["MODEL"]["COMPILED"],
                                              config["TRAIN"]["CHECKPOINT"]["PRETRAINED_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption node
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_MODEL"]:
        cr_model, start_epoch, best_psnr, best_ssim, optimizer = load_resume_state_dict(
            cr_model,
            optimizer,
            config["MODEL"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Initialize image sharpness evaluation method
    psnr_model, ssim_model = build_iqa_model(0, config["TEST"]["ONLY_TEST_Y_CHANNEL"], device)

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(cr_model,
              train_prefetcher,
              pixel_criterion,
              color_criterion,
              tv_criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)
        psnr, ssim = test(cr_model,
                          test_prefetcher,
                          psnr_model,
                          ssim_model,
                          device,
                          config["TEST"]["PRINT_FREQ"],
                          False,
                          None)
        print("\n")

        # Write the evaluation indicators of each round of Epoch to the log
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)

        # Automatically save model weights
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "psnr": psnr,
                         "ssim": ssim,
                         "state_dict": cr_model.state_dict(),
                         "optimizer": optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "best.pth.tar",
                        "last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        config: Any,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher, distributed.DistributedSampler, distributed.DistributedSampler]:
    train_datasets = ImageDataset(config["TRAIN"]["DATASET"]["GT_IMAGES_DIR"],
                                  config["TRAIN"]["DATASET"]["INPUT_IMAGES_DIR"],
                                  config["TRAIN"]["DATASET"]["LOW_RESOLUTION_SIZE"],
                                  config["TRAIN"]["DATASET"]["HIGH_RESOLUTION_SIZE"])

    test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["GT_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["INPUT_IMAGES_DIR"])

    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                  shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                  num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                  pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                  drop_last=True,
                                  persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])

    # Replace the dataset iterator with CUDA to speed up
    degenerated_train_prefetcher = CUDAPrefetcher(train_dataloader, device)
    paired_test_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return degenerated_train_prefetcher, paired_test_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module or Any]:
    cr_model = model.__dict__[config["MODEL"]["NAME"]](in_channels=config["MODEL"]["IN_CHANNELS"],
                                                       out_channels=config["MODEL"]["OUT_CHANNELS"],
                                                       luma_bins=config["MODEL"]["LUMA_BINS"],
                                                       channel_multiplier=config["MODEL"]["CHANNEL_MULTIPLIER"],
                                                       spatial_bin=config["MODEL"]["SPATIAL_BIN"],
                                                       batch_norm=config["MODEL"]["BATCH_NORM"],
                                                       low_resolution_size=config["MODEL"]["LOW_RESOLUTION_SIZE"])
    cr_model = cr_model.to(device)

    # Compile the model
    if config["MODEL"]["COMPILED"]:
        cr_model = torch.compile(cr_model)

    return cr_model


def define_loss(config: Any, device: torch.device) -> [nn.MSELoss, model.COLORLoss, model.TVLoss]:
    if config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "L1Loss":
        pixel_criterion = nn.L1Loss()
    elif config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "MSELoss":
        pixel_criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['PIXEL_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["COLOR_LOSS"]["NAME"] == "vanilla":
        color_criterion = model.COLORLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['COLOR_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["TV_LOSS"]["NAME"] == "vanilla":
        tv_criterion = model.TVLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['TV_LOSS']['NAME']} is not implemented.")

    pixel_criterion = pixel_criterion.to(device)
    color_criterion = color_criterion.to(device)
    tv_criterion = tv_criterion.to(device)

    return pixel_criterion, color_criterion, tv_criterion


def define_optimizer(g_model: nn.Module, config: Any) -> optim.Adam:
    optimizer = optim.Adam(g_model.parameters(),
                           config["TRAIN"]["HYP"]["LR"],
                           config["TRAIN"]["HYP"]["BETAS"],
                           config["TRAIN"]["HYP"]["EPS"],
                           config["TRAIN"]["HYP"]["WEIGHT_DECAY"])

    return optimizer


def train(
        cr_model: nn.Module,
        degenerated_train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.MSELoss,
        color_criterion: model.COLORLoss,
        tv_criterion: model.TVLoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    """training main function

    Args:
        cr_model (nn.Module): color restoration model
        degenerated_train_prefetcher (CUDARefetcher): training dataset iterator
        pixel_criterion (nn.MSELoss): pixel loss function
        color_criterion (model.COLORLoss): color loss function
        tv_criterion (model.TVLoss): smoothness loss function
        optimizer (optim.Adam): optimizer function
        epoch (int): number of training epochs
        scaler (amp.GradScaler): mixed precision function
        writer (SummaryWriter): training log function
        device (torch.device): evaluation model running device model
        config (Any): configuration file
    """

    # Calculate how many batches of data there are under a dataset iterator
    batches = len(degenerated_train_prefetcher)
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    pixel_losses = AverageMeter("Pixel Loss", ":6.6f", Summary.NONE)
    color_losses = AverageMeter("Color Loss", ":6.6f", Summary.NONE)
    tv_losses = AverageMeter("TV Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, pixel_losses, color_losses, tv_losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Set the model to training mode
    cr_model.train()

    # Define loss function weights
    pixel_loss_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(device)
    color_loss_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["COLOR_LOSS"]["WEIGHT"]).to(device)
    tv_loss_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["TV_LOSS"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    degenerated_train_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = degenerated_train_prefetcher.next()

    while batch_data is not None:
        # Load batches of data
        batch_size = batch_data["input"].shape[0]
        gt_tensor = batch_data["gt"].to(device, non_blocking=True)
        input_tensor = batch_data["input"].to(device, non_blocking=True)

        # image data augmentation
        gt_tensor, input_tensor = random_rotate_torch(gt_tensor, input_tensor, 1, [0, 90, 180, 270])
        gt_tensor, input_tensor = random_vertically_flip_torch(gt_tensor, input_tensor)
        gt_tensor, input_tensor = random_horizontally_flip_torch(gt_tensor, input_tensor)

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # Initialize the generator model gradient
        cr_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            color_map_tensor = cr_model(input_tensor)

            # According to the illumination estimation map, find the reflection map
            cr_tensor = torch.div(input_tensor, color_map_tensor + 1e-4)
            cr_tensor = torch.clamp(cr_tensor, 0.0, 1.0)

            pixel_loss = pixel_criterion(cr_tensor, gt_tensor)
            color_loss = color_criterion(cr_tensor, gt_tensor)
            tv_loss = tv_criterion(cr_tensor, gt_tensor, color_map_tensor, input_tensor)
            pixel_loss = torch.sum(torch.mul(pixel_loss_weight, pixel_loss))
            color_loss = torch.sum(torch.mul(color_loss_weight, color_loss))
            tv_loss = torch.sum(torch.mul(tv_loss_weight, tv_loss))

        # Backpropagation
        scaler.scale(pixel_loss).backward()
        # update model weights
        scaler.step(optimizer)
        scaler.update()

        # record the loss value
        pixel_losses.update(pixel_loss.item(), batch_size)
        color_losses.update(color_loss.item(), batch_size)
        tv_losses.update(tv_loss.item(), batch_size)

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Color_Loss", color_loss.item(), iters)
            writer.add_scalar("Train/TV_Loss", tv_loss.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = degenerated_train_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1


if __name__ == "__main__":
    main()
