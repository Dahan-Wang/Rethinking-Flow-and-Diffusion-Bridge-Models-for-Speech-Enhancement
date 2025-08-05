import sys
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join
from argparse import ArgumentParser
from librosa import resample
from omegaconf import OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import os


def parse_device_arg():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-D', '--device', default='0', help='Index of the available devices, e.g. 0,1,2,3')
    args, _ = parser.parse_known_args()
    world_size = args.device.count(",") + 1
    return args.device, world_size


def setup_distributed_env(world_size=1):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = '0'


device, world_size = parse_device_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = device

from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec

setup_distributed_env(world_size)


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


class TestDataset(nn.Module):
    def __init__(self, noisy_files, test_dir, normalize="noisy", target_sr=16000):
        super(TestDataset, self).__init__()
        self.noisy_files = sorted(noisy_files)
        self.test_dir = test_dir
        self.target_sr = target_sr
        self.normalize = normalize

    def __getitem__(self, idx):
        y, sr = load(self.noisy_files[idx])
        if sr != self.target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=self.target_sr))

        T_orig = y.size(1)
        if self.normalize == "noisy":
            norm_factor = y.abs().max()
        elif self.normalize == "std":
            norm_factor = y.std()
        y = y / norm_factor

        filename = self.noisy_files[idx].replace(self.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        return y[0], T_orig, filename, norm_factor

    def __len__(self):
        return len(self.noisy_files)


def enhance(args):
    # Load score model
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location="cuda:0").to("cuda:0")
    model.t_eps = args.t_eps
    model.eval()

    # Get list of noisy files
    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.flac')))

    # Check if the model is trained on 48 kHz data
    if model.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    noisy_dataset = TestDataset(noisy_files, args.test_dir, target_sr=target_sr)
    noisy_dataloader = torch.utils.data.DataLoader(
        noisy_dataset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    makedirs(args.enhanced_dir, exist_ok=True)

    # Enhance files
    for y, T_orig, filename, norm_factor in tqdm(noisy_dataloader):

        # Prepare DNN input
        Y = model._forward_transform(model._stft(y.to("cuda:0"))).unsqueeze(1)
        Y = pad_spec(Y, mode=pad_mode)

        # Reverse sampling
        if model.sde.__class__.__name__ == 'OUVESDE' or model.sde.__class__.__name__ == 'BBED':
            if args.sampler_type == 'pc':
                sampler = model.get_pc_sampler('reverse_diffusion', args.corrector, Y.to("cuda:0"), N=args.N,
                    corrector_steps=args.corrector_steps, snr=args.snr)
            elif args.sampler_type == 'ode':
                sampler = model.get_ode_sampler(Y.to("cuda:0"), N=args.N)
            else:
                raise ValueError(f"Sampler type {args.sampler_type} not supported")
        elif model.sde.__class__.__name__ == 'SBVESDE' or model.sde.__class__.__name__ == 'SBSDE':
            model.sde.N = args.N
            sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
            sampler = model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type)
        else:
            raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

        sample, _ = sampler()

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig[0])

        for i in range(len(filename)):
            x_hat_ = x_hat[i] * norm_factor[i]
            write(join(args.enhanced_dir, filename[0]), x_hat_.cpu().numpy(), target_sr)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def enhance_worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    setup(rank, world_size)

    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=f"cuda:{rank}")

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.t_eps = args.t_eps
    model.eval()

    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.flac')))

    if model.module.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.module.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    noisy_dataset = TestDataset(noisy_files, args.test_dir, normalize=model.module.data_module.normalize, target_sr=target_sr)
    sampler = torch.utils.data.distributed.DistributedSampler(
        noisy_dataset,
        num_replicas=world_size,
        rank=rank
    )
    noisy_dataloader = torch.utils.data.DataLoader(
        noisy_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        sampler=sampler
    )

    if rank == 0:
        makedirs(args.enhanced_dir, exist_ok=True)

    for y, T_orig, filename, norm_factor in tqdm(noisy_dataloader, disable=(rank != 0)):
        Y = model.module._forward_transform(model.module._stft(y.to(rank))).unsqueeze(1)
        if model.module.backbone.startswith("ncsnpp"):
            Y = pad_spec(Y, mode=pad_mode)

        if model.module.sde.__class__.__name__ == 'OUVESDE' or model.module.sde.__class__.__name__ == 'BBED':
            if args.sampler_type == 'pc':
                sampler = model.module.get_pc_sampler('reverse_diffusion', args.corrector, Y.to(rank), N=args.N,
                    corrector_steps=args.corrector_steps, snr=args.snr)
            elif args.sampler_type == 'ode':
                sampler = model.module.get_ode_sampler(Y.to(rank), N=args.N)
            elif args.sampler_type == 'sgm':
                model.module.sde.N = args.N
                sampler = model.module.get_sgm_sampler(sde=model.module.sde, y=Y.to(rank))
            else:
                raise ValueError(f"Sampler type {args.sampler_type} not supported")
        elif model.module.sde.__class__.__name__ == 'SBVESDE' or model.module.sde.__class__.__name__ == 'SBSDE':
            sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
            model.module.sde.N = args.N
            sampler = model.module.get_sb_sampler(sde=model.module.sde, y=Y.to(rank), sampler_type=sampler_type)
        elif model.module.sde.__class__.__name__ == 'FM':
            model.module.sde.N = args.N
            sampler = model.module.get_fm_sampler(sde=model.module.sde, y=Y.to(rank))
        else:
            raise ValueError(f"SDE {model.module.sde.__class__.__name__} not supported")

        sample, _ = sampler()

        x_hat = model.module.to_audio(sample.squeeze(1), T_orig[0])

        for i in range(len(filename)):
            x_hat_ = x_hat[i] * norm_factor[i].to(x_hat[i].device)
            write(join(args.enhanced_dir, filename[i]), x_hat_.cpu().numpy(), target_sr)

    cleanup()


def Enhance(args):
    n_gpus = args.device.count(",") + 1
    world_size = n_gpus

    if world_size == 1:
        enhance(args)
    else:
        mp.spawn(
            enhance_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config', default='config_infer.yaml', type=str)
    initial_args, _ = parser.parse_known_args()
    add_config_args(initial_args)

    parser.add_argument('-D', '--device', default='0', help='Index of the available devices, e.g. 0,1,2,3')
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics")
    parser.add_argument("--N", type=int, default=50, help="Number of reverse steps")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    parser.add_argument('-B', '--bs', type=int, default=8, help="Batch size for inference")

    args, _ = parser.parse_known_args()
    Enhance(args)
