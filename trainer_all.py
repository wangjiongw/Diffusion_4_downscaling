import argparse
import logging
import os
import pickle
import warnings
from collections import OrderedDict, defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from tensorboardX import SummaryWriter
from torch.nn.functional import l1_loss, mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from configs import Config

# from x2_data.mydataset_patch import BigDataset_train
# from data.mydataset_patch import SR3_Dataset_patch
from data.cndataset_patch import SR3_CNDataset_patch_preload as SR3_CNDataset_patch

matplotlib.use("Agg")
import glob
import random

from utils import (
    accumulate_statistics,
    construct_and_save_wbd_plots,
    construct_mask,
    dict2str,
    get_optimizer,
    psnr,
    set_seeds,
    setup_logger,
)

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    set_seeds()  # For reproducability.
    accelerator = Accelerator()
    device = accelerator.device
    print("Device:", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        choices=["train", "val"],
        help="Run either training or validation(inference).",
        default="train",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-var", "--variable_name", type=str, default=None)
    parser.add_argument("--area", type=str, choices=("china", "gansu"), default="china", help="area to select data")
    parser.add_argument(
        "--cma_root", type=str, default=None, help="mount point of cma data"
    )
    parser.add_argument(
        "--era5_root", type=str, default=None, help="mount point of era5 data"
    )
    args = parser.parse_args()

    if args.gpu_ids is None:
        args.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]

    variable_name = args.variable_name
    configs = Config(args)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    setup_logger(None, configs.log, "train", screen=True)
    setup_logger("val", configs.log, "val")

    logger = logging.getLogger("base")
    val_logger = logging.getLogger("val")
    logger.info(dict2str(configs.get_hyperparameters_as_dict()))
    tb_logger = SummaryWriter(log_dir=configs.tb_logger)

    land_01_path = "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/assets/earth_data/ETOPO_2022_v1_1km_N60_0E70_140_surface.npy"
    mask_path = "/mnt/petrelfs/wangjiong/ai4earth/ClimateHR/assets/earth_data/land_mask_1km_binary.npy"

    train_data = SR3_CNDataset_patch(
        # np.array(target_paths)[train_index],
        args.era5_root,
        args.cma_root,
        land_01_path,
        mask_path,
        scale=configs.data_scale,
        # lr_paths=np.array(lr_paths)[train_index],
        var=variable_name,
        area=args.area,
        patch_size=configs.height,
        year_start=configs.start_date,
        year_end=configs.end_date,
        year_freq=configs.sample_interval,
    )
    val_data = SR3_CNDataset_patch(
        args.era5_root,
        args.cma_root,
        land_01_path,
        mask_path,
        scale=configs.data_scale,
        var=variable_name,
        area=args.area,
        patch_size=configs.height,
        year_start="2021-07-01",
        year_end="2021-08-26-23",
        year_freq=configs.sample_interval,
    )

    logger.info(f"Train Samples: {len(train_data)}, Val Samples: {len(val_data)}.")
    train_loader = DataLoader(
        train_data,
        batch_size=configs.batch_size,
        shuffle=configs.use_shuffle,
        num_workers=configs.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(configs.batch_size / 16),
        shuffle=False,
        num_workers=configs.num_workers,
        drop_last=True,
    )
    train_loader = accelerator.prepare_data_loader(train_loader)
    val_loader = accelerator.prepare_data_loader(val_loader)
    logger.info(
        f"Training [{len(train_loader)} batches] and Validation [{len(val_loader)} batches] dataloaders are ready. "
    )

    # Defining the model.
    optimizer = get_optimizer(configs.optimizer_type)
    diffusion = model.create_model(
        in_channel=configs.in_channel,
        out_channel=configs.out_channel,
        norm_groups=configs.norm_groups,
        inner_channel=configs.inner_channel,
        channel_multiplier=configs.channel_multiplier,
        attn_res=configs.attn_res,
        res_blocks=configs.res_blocks,
        dropout=configs.dropout,
        diffusion_loss=configs.diffusion_loss,
        conditional=configs.conditional,
        gpu_ids=configs.gpu_ids,
        # distributed=configs.distributed,
        distributed=False,
        init_method=configs.init_method,
        train_schedule=configs.train_schedule,
        train_n_timestep=configs.train_n_timestep,
        train_linear_start=configs.train_linear_start,
        train_linear_end=configs.train_linear_end,
        val_schedule=configs.val_schedule,
        val_n_timestep=configs.val_n_timestep,
        val_linear_start=configs.val_linear_start,
        val_linear_end=configs.val_linear_end,
        finetune_norm=configs.finetune_norm,
        optimizer=optimizer,
        amsgrad=configs.amsgrad,
        learning_rate=configs.lr,
        checkpoint=configs.checkpoint,
        resume_state=configs.resume_state,
        phase=configs.phase,
        height=configs.height,
        accelerator=accelerator,
        img_height=configs.img_height if args.area == "china" else configs.height,        # whole image equals to patch size if not whole CN area
        img_width=configs.img_width if args.area == "china" else configs.height,
    )
    logger.info("Model initialization is finished.")

    current_step, current_epoch = diffusion.begin_step, diffusion.begin_epoch
    if configs.resume_state:
        logger.info(
            f"Resuming training from epoch: {current_epoch}, iter: {current_step}."
        )

    logger.info("Starting the registration.")
    diffusion.register_schedule(
        beta_schedule=configs.train_schedule,
        timesteps=configs.train_n_timestep,
        linear_start=configs.train_linear_start,
        linear_end=configs.train_linear_end,
    )

    accumulated_statistics = OrderedDict()

    val_metrics_dict = {"MSE": 0.0, "MAE": 0.0, "MAE_inter": 0.0}
    val_metrics_dict["PSNR_" + variable_name] = 0.0
    val_metrics_dict["PSNR_inter_" + variable_name] = 0.0
    val_metrics_dict["RMSE_" + variable_name] = 0.0
    val_metrics_dict["RMSE_inter_" + variable_name] = 0.0
    val_metrics = OrderedDict(val_metrics_dict)

    # Training.
    logger.info("Starting the training.")
    while current_step < configs.n_iter:
        current_epoch += 1

        for train_data in train_loader:
            current_step += 1

            if current_step > configs.n_iter:
                break

            # Training.
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            diffusion.lr_scheduler_step()  # For lr scheduler updates per iteration.
            accumulate_statistics(diffusion.get_current_log(), accumulated_statistics)

            if current_step % configs.print_freq == 0:
                message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"

                for metric, values in accumulated_statistics.items():
                    mean_value = np.mean(values)
                    message = f"{message}  |  {metric:s}: {mean_value:.5f}"
                    tb_logger.add_scalar(f"{metric}/train", mean_value, current_step)

                logger.info(message)
                tb_logger.add_scalar(f"learning_rate", diffusion.get_lr(), current_step)

                # Visualizing distributions of parameters.
                # for name, param in diffusion.get_named_parameters():
                #     tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), current_step)

                accumulated_statistics = OrderedDict()

            # Validation.
            if current_step % configs.val_freq == 0:
                logger.info(f"Starting validation at Step {current_step}.")
                idx = 0
                result_path = f"{configs.results}/{current_epoch}"
                os.makedirs(result_path, exist_ok=True)
                diffusion.register_schedule(
                    beta_schedule=configs.val_schedule,
                    timesteps=configs.val_n_timestep,
                    linear_start=configs.val_linear_start,
                    linear_end=configs.val_linear_end,
                )

                # make sure visualize at least once
                val_vis_freq = min(configs.val_vis_freq, int(len(val_loader) - 1))
                # A dictionary for storing a list of mean temperatures for each month.
                # month2mean_temperature = defaultdict(list)

                for val_data in val_loader:
                    idx += 1
                    diffusion.feed_data(val_data)
                    # 实验一采用了250，实验二用50
                    diffusion.test(
                        continuous=False,
                        use_ddim=True,
                        ddim_steps=250,
                        use_dpm_solver=False,
                    )  # Continues=False to return only the last timesteps's outcome.

                    # Computing metrics on validation data.
                    visuals = diffusion.get_current_visuals()
                    # Computing MSE and RMSE on original data.
                    # mask = val_data["mask"]
                    mask = ~visuals["HR"].isnan()
                    mse_value = mse_loss(visuals["HR"] * mask, visuals["SR"] * mask)
                    val_metrics["MSE"] += mse_value
                    val_metrics["MAE"] += l1_loss(
                        visuals["HR"] * mask, visuals["SR"] * mask
                    )
                    val_metrics["MAE_inter"] += l1_loss(
                        visuals["HR"] * mask, visuals["INTERPOLATED"] * mask
                    )

                    val_metrics["RMSE_" + variable_name] += torch.sqrt(
                        mse_loss(visuals["HR"] * mask, visuals["SR"] * mask)
                    )
                    val_metrics["RMSE_inter_" + variable_name] += torch.sqrt(
                        mse_loss(visuals["HR"] * mask, visuals["INTERPOLATED"] * mask)
                    )
                    val_metrics["PSNR_" + variable_name] += psnr(
                        visuals["HR"] * mask, visuals["SR"] * mask
                    )
                    val_metrics["PSNR_inter_" + variable_name] += psnr(
                        visuals["HR"] * mask, visuals["INTERPOLATED"] * mask
                    )

                    if idx % val_vis_freq == 0:

                        logger.info(
                            f"[{idx//configs.val_vis_freq}] Visualizing and storing some examples."
                        )

                        sr_candidates = diffusion.generate_multiple_candidates(
                            n=configs.sample_size, ddim_steps=100, use_dpm_solver=True
                        )

                        mean_candidate = sr_candidates.mean(dim=0)  # [B, C, H, W]
                        std_candidate = sr_candidates.std(dim=0)  # [B, C, H, W]
                        bias = mean_candidate - visuals["HR"]

                        # # Choosing the first n_val_vis number of samples to visualize.
                        # variable_id=0
                        random_idx = np.random.randint(
                            0, int(configs.batch_size // 16), configs.n_val_vis
                        )

                        path = f"{result_path}/{current_epoch}_{current_step}_{idx}"
                        figure, axs = plt.subplots(len(random_idx), 9, figsize=(25, 12))
                        if variable_name == "tp":
                            vmin = 0
                            cmap = "BrBG"
                            vmax = 2
                        elif variable_name in ["u10", "v10", "t2m", "sp"]:
                            vmin = 0
                            cmap = "RdBu_r"
                            vmax = 1
                        else:
                            vmin = -2
                            cmap = "RdBu_r"
                            vmax = 2
                            
                        for idx_i, num in enumerate(random_idx):
                            num = min(num, visuals["HR"].shape[0] - 1)
                            
                            axs[idx_i, 0].imshow(
                                visuals["HR"][num, 0], vmin=vmin, vmax=vmax, cmap=cmap
                            )
                            axs[idx_i, 1].imshow(
                                visuals["SR"][num, 0], vmin=vmin, vmax=vmax, cmap=cmap
                            )
                            axs[idx_i, 2].imshow(
                                visuals["INTERPOLATED"][num, 0],
                                vmin=vmin,
                                vmax=vmax,
                                cmap=cmap,
                            )
                            axs[idx_i, 3].imshow(
                                mean_candidate[num, 0], vmin=vmin, vmax=vmax, cmap=cmap
                            )
                            axs[idx_i, 4].imshow(
                                std_candidate[num, 0], vmin=0, vmax=2, cmap="Reds"
                            )
                            axs[idx_i, 5].imshow(
                                np.abs(visuals["HR"][num, 0] - visuals["SR"][num, 0]),
                                vmin=0,
                                vmax=2,
                                cmap="Reds",
                            )
                            axs[idx_i, 6].imshow(
                                np.abs(
                                    visuals["HR"][num, 0]
                                    - visuals["INTERPOLATED"][num, 0]
                                ),
                                vmin=0,
                                vmax=2,
                                cmap="Reds",
                            )
                            axs[idx_i, 7].imshow(
                                np.abs(bias)[num, 0], vmin=0, vmax=2, cmap="Reds"
                            )
                            axs[idx_i, 8].imshow(
                                val_data["mask"][num, 0].cpu().numpy(),
                                vmin=0,
                                vmax=2,
                                cmap="RdBu_r",
                            )
                            axs[idx_i, 8].set_title(
                                "mean_mae:%.3f,inter_mae:%.3f,sr_mae:%.3f"
                                % (
                                    np.abs(bias)[num, 0].mean(),
                                    np.abs(
                                        visuals["HR"][num, 0]
                                        - visuals["INTERPOLATED"][num, 0]
                                    ).mean(),
                                    np.abs(
                                        visuals["HR"][num, 0] - visuals["SR"][num, 0]
                                    ).mean(),
                                )
                            )
                            axs[idx_i, 0].set_ylabel(f"Sample_{num}")
                        for title, col in zip(
                            [
                                "HR",
                                "Diffusion",
                                "INTERPOLATED",
                                "mean",
                                "std",
                                "mae_sr",
                                "mae_inter",
                                "mae_mean",
                            ],
                            range(8),
                        ):
                            axs[0, col].set_title(title)
                        plt.savefig(f"{path}_.png", bbox_inches="tight")
                        plt.close("all")

                val_metrics["MSE"] /= idx
                val_metrics["MAE"] /= idx
                val_metrics["MAE_inter"] /= idx

                val_metrics["RMSE_" + variable_name] /= idx
                val_metrics["RMSE_inter_" + variable_name] /= idx
                val_metrics["PSNR_" + variable_name] /= idx
                val_metrics["PSNR_inter_" + variable_name] /= idx
                diffusion.register_schedule(
                    beta_schedule=configs.train_schedule,
                    timesteps=configs.train_n_timestep,
                    linear_start=configs.train_linear_start,
                    linear_end=configs.train_linear_end,
                )
                message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"
                for metric, value in val_metrics.items():
                    message = f"{message}  |  {metric:s}: {value:.5f}"
                    tb_logger.add_scalar(f"{metric}/val", value, current_step)

                val_logger.info(message)

                val_metrics = val_metrics.fromkeys(
                    val_metrics, 0.0
                )  # Sets all metrics to zero.

            if current_step % configs.save_checkpoint_freq == 0:
                logger.info("Saving models and training states.")
                diffusion.save_network(current_epoch, current_step)

    tb_logger.close()

    logger.info("End of training.")
