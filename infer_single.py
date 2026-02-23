'''
Inference: for single file.
'''
import os
import sys
import ast
import torch
from soundfile import write
from torchaudio import load
from argparse import ArgumentParser
from librosa import resample
from omegaconf import OmegaConf


def parse_device_arg():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-D', '--device', default=['0'], nargs='+', help='Index of the available devices, e.g. 0 1 2')
    args, _ = parser.parse_known_args()
    world_size = len(args.device)
    device_str = ','.join(args.device)
    return device_str, world_size


device, _ = parse_device_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = device

# Set CUDA architecture list
from fdbm.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from fdbm.model import BridgeModel
from fdbm.util.other import pad_spec


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


def enhance(args):
    model = BridgeModel.load_from_checkpoint(args.ckpt, map_location="cuda:0").to("cuda:0")
    model.bridge.N = args.N
    model.bridge.sampler_type = args.sampler_type
    model.eval()
    print(f"Model loaded from checkpoint: {args.ckpt}")
    
    # Prepare sampler kwargs
    sampler_kwargs = args.sampler_kwargs if args.sampler_kwargs is not None else {}

    # Check if the model  is trained on 48 kHz data
    if model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    # Load wav file
    print(f"Loading audio file: {args.noisy_file}")
    y, sr = load(args.noisy_file)

    # Resample if necessary
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz")
        y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))

    T_orig = y.size(1)

    # Normalize
    if model.data_module.normalize == "noisy":
        norm_factor = y.abs().max()
    elif model.data_module.normalize == "std":
        norm_factor = y.std()
    y = y / norm_factor

    # Enhancement
    Y = torch.unsqueeze(model._forward_transform(model._stft(y.to("cuda:0"))), 0)
    if model.backbone.startswith("ncsnpp"):
        Y = pad_spec(Y, mode=pad_mode)
    sample = model.bridge.sampler(model, Y, **sampler_kwargs)
    x_hat = model.to_audio(sample.squeeze(), T_orig)

    # Renormalize
    x_hat = x_hat * norm_factor
    if x_hat.abs().max() > 1.0:
        x_hat = x_hat / x_hat.abs().max() * 0.5

    # Write enhanced wav file
    output_path = args.output_file
    print(f"Writing enhanced audio to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write(output_path, x_hat.cpu().numpy(), target_sr)
    
    print("Enhancement completed successfully!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-C', '--config', default='config_infer_single.yaml', type=str, help='Configuration file')
    initial_args, _ = parser.parse_known_args()
    add_config_args(initial_args)

    parser.add_argument("--noisy_file", type=str, required=True, help='Path to the noisy audio file to enhance')
    parser.add_argument("--output_file", type=str, default=None, help='Path to save the enhanced audio (optional, default: same directory as input with _enhanced suffix)')
    parser.add_argument("--ckpt", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="ode_ei", help="Sampler type.")
    parser.add_argument("--sampler_kwargs", type=ast.literal_eval, default=None, help="Dictionary of keyword arguments for the sampler (e.g., corrector_name='ald', snr=0.5)")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")

    args, _ = parser.parse_known_args()

    enhance(args)
