import os
import sys
import ast
import shutil
from datetime import datetime
from omegaconf import OmegaConf

import torch
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule, DatasetSamplerCallback
from sgmse.sdes import SDERegistry
from sgmse.model_crp import ScoreModel


def add_config_args(initial_args):
    if initial_args.config:
        with open(initial_args.config, 'r') as f:
            config = OmegaConf.load(f)
        config_args = []
        for key, value in config.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    config_args.append(f'--{key}')
            else:
                config_args.append(f'--{key}')
                config_args.append(str(value))
        sys.argv.extend(config_args)


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config', default='config_crp.yaml', type=str)

    initial_args, _ = parser.parse_known_args()
    add_config_args(initial_args)

    parser.add_argument("--version", type=str, default=None, help="Version of the model.")
    parser.add_argument("--nolog", action='store_true', help="Turn off logging.")
    parser.add_argument("--logger", type=str, default='tensorboard', help="Logger.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--save_ckpt_interval", type=int, default=50000, help="Save checkpoint interval.")

    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_eval_files", type=int, default=100)
    parser.add_argument("--num_data_per_epoch", type=int, default=None, help="Sample data per epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
    parser.add_argument("--scheduler_config", type=ast.literal_eval, default=None, help="The scheduler configuration.")

    temp_args, _ = parser.parse_known_args()

    trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
    trainer_parser.add_argument("-D", "--devices", default=[0], nargs="+", type=int, help="The index of the available devices, e.g. 0 1 2 3")
    trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
    trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")

    # Parse args and separate into groups
    args, _ = parser.parse_known_args()
    arg_groups = get_argparse_groups(parser)

    # Initialize logger, trainer, model, datamodule
    # Set up logger configuration
    if args.nolog:
        logger = None
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        version = args.version + f"_{datetime.now().strftime('%Y%m%d')}"

        if args.logger == "wandb":
            logger = WandbLogger(project="Diffusion", log_model=False, save_dir=args.log_dir, name=args.wandb_name, version=version)
            logger.experiment.log_code(".")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(save_dir=args.log_dir, name=version, version=f"version_{version[-4:]}")
            code_path = join(args.log_dir, version, str(logger.version), "code")
            if not os.path.exists(code_path):
                shutil.copytree('.', code_path)
                if os.path.exists(os.path.join(code_path, '__pycache__')):
                    shutil.rmtree(os.path.join(code_path, '__pycache__'))
                for dir in os.listdir(code_path):
                    dir_path = os.path.join(code_path, dir)
                    if os.path.isdir(dir_path):
                        pycache_path = os.path.join(dir_path, '__pycache__')
                        if os.path.exists(pycache_path):
                            shutil.rmtree(pycache_path)

    # Initialize model
    model = ScoreModel.load_from_checkpoint(
        args.ckpt,
        log_dir=join(args.log_dir, version, str(logger.version))
    )

    model.sde.N = args.N
    model.data_module.batch_size = args.batch_size
    model.data_module.num_eval_files = args.num_eval_files
    model.data_module.num_data_per_epoch = args.num_data_per_epoch
    model.lr = args.lr
    model.scheduler_config = args.scheduler_config

    # Set up callbacks for logger
    if logger is not None:
        if args.logger == "wandb":
            ckpt_dir = join(args.log_dir, str(logger.version), "checkpoints")
        elif args.logger == "tensorboard":
            ckpt_dir = join(logger.log_dir, "checkpoints")
        callbacks = []
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_last=True, filename='{epoch}-last'))
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval))
        if args.num_eval_files:
            callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="valid_loss", mode="min", filename='{epoch}-loss'))
            callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}'))
            callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}'))
        if args.num_data_per_epoch is not None:
            callbacks.append(DatasetSamplerCallback())
    else:
        callbacks = None

    # Initialize the Trainer and the DataModule
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        logger=logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=callbacks
    )

    # Train model
    trainer.fit(model)
