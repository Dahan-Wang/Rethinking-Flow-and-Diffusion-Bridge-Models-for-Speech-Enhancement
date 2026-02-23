import os
import sys
import shutil
from datetime import datetime
from omegaconf import OmegaConf

import torch
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from fdbm.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from fdbm.backbones.shared import BackboneRegistry
from fdbm.data_module import SpecsDataModule
from fdbm.bridge import Bridge, BridgeRegistry
from fdbm.model import BridgeModel, PredictiveModel


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
    parser.add_argument('-C', '--config', default='config.yaml', type=str)

    initial_args, _ = parser.parse_known_args()
    add_config_args(initial_args)

    parser.add_argument("--mode", type=str, choices=["generative", "predictive"], default="generative", help="Mode of the model.")
    parser.add_argument("--version", type=str, default=None, help="Version of the model.")
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
    parser.add_argument("--bridge", type=str, choices=BridgeRegistry.get_all_names(), default="sb")
    parser.add_argument("--nolog", action='store_true', help="Turn off logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--save_ckpt_interval", type=int, default=50000, help="Save checkpoint interval.")

    temp_args, _ = parser.parse_known_args()

    if temp_args.mode == "generative":
        Model = BridgeModel
    elif temp_args.mode == "predictive":
        Model = PredictiveModel


    # Add specific args for BridgeModel, pl.Trainer, the Bridge class and backbone DNN class
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    path_class = BridgeRegistry.get_by_name(temp_args.bridge)
    trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
    trainer_parser.add_argument("-D", "--devices", default=[0], nargs="+", type=int, help="The index of the available devices, e.g. 0 1 2 3")
    trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
    trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")

    Model.add_argparse_args(
         parser.add_argument_group("BridgeModel", description=Model.__name__))
    
    bridge_parser = parser.add_argument_group("Bridge", description="Bridge class")
    Bridge.add_argparse_args(bridge_parser)
    path_class.add_argparse_args(bridge_parser)

    backbone_cls.add_argparse_args(
         parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    # Add data module args
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
         parser.add_argument_group("DataModule", description=data_module_cls.__name__))
    # Parse args and separate into groups
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    # Initialize logger, trainer, model, datamodule
    # Set up logger configuration
    if args.nolog:
        logger = None
    else:
        os.makedirs(args.log_dir, exist_ok=True)

        if args.ckpt is None:
            version = args.version + f"_{datetime.now().strftime('%Y%m%d')}"
        else:
            version = args.ckpt.split('/')[-4]

        logger = TensorBoardLogger(save_dir=args.log_dir, name="", version=version, default_hp_metric=False)
        code_path = join(args.log_dir, version, "code")
        if not os.path.exists(code_path):
            os.makedirs(code_path)
            for file_name in os.listdir('.'):
                if os.path.isfile(file_name) and file_name.endswith(('.py', '.yaml')):
                    shutil.copy2(file_name, code_path)
            if os.path.exists('fdbm'):
                shutil.copytree('fdbm', os.path.join(code_path, 'fdbm'), 
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    # Initialize model
    model = Model(
         backbone=args.backbone, bridge=args.bridge, data_module_cls=data_module_cls,
         log_dir=join(args.log_dir, version),
         **{
              **vars(arg_groups['BridgeModel']),
              **vars(arg_groups['Bridge']),
              **vars(arg_groups['Backbone']),
              **vars(arg_groups['DataModule'])
         }
    )

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        print(f'================={version}=================')

    # Set up callbacks for logger
    if logger is not None:
        ckpt_dir = join(logger.log_dir, "checkpoints")
        callbacks = []
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_last=True, filename='{epoch}-last'))
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval))
        if args.num_eval_files:
            callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="valid_loss", mode="min", filename='{epoch}-loss'))
            callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}'))
            callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}'))
    else:
        callbacks = None

    # Initialize the Trainer and the DataModule
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        logger=logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        gradient_clip_val=3.0,
        callbacks=callbacks
    )

    # Train model
    trainer.fit(model, ckpt_path=args.ckpt)
