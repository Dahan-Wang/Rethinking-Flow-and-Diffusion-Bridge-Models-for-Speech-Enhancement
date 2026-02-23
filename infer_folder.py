'''
Inference: for folder with multi-GPU support (non-distributed).
'''
import os
import sys
import ast
import glob
import time
import torch
import torch.multiprocessing as mp
from soundfile import write
from torchaudio import load
from argparse import ArgumentParser
from librosa import resample
from omegaconf import OmegaConf
from tqdm import tqdm
from os.path import join, basename, dirname, splitext


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


def get_audio_files(test_dir):
    """Get all audio files from directory."""
    audio_files = []
    audio_files += sorted(glob.glob(join(test_dir, '*.wav')))
    audio_files += sorted(glob.glob(join(test_dir, '**', '*.wav'), recursive=True))
    audio_files += sorted(glob.glob(join(test_dir, '*.flac')))
    audio_files += sorted(glob.glob(join(test_dir, '**', '*.flac'), recursive=True))
    return audio_files


def enhance_files_worker(gpu_id, file_list, args, progress_counter):
    """Worker function for processing files on a specific GPU."""
    # Set device for this worker
    device = f"cuda:{gpu_id}"
    
    # Load model
    model = BridgeModel.load_from_checkpoint(args.ckpt, map_location=device).to(device)
    model.bridge.N = args.N
    model.bridge.sampler_type = args.sampler_type
    model.eval()
    
    # Prepare sampler kwargs
    sampler_kwargs = args.sampler_kwargs if args.sampler_kwargs is not None else {}
    
    # Determine target sample rate and padding mode
    if model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"
    
    # Process each file
    for noisy_file in file_list:
        try:
            # Load audio
            y, sr = load(noisy_file)
            
            # Resample if necessary
            if sr != target_sr:
                y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))
            
            T_orig = y.size(1)
            
            # Normalize
            if model.data_module.normalize == "noisy":
                norm_factor = y.abs().max()
            elif model.data_module.normalize == "std":
                norm_factor = y.std()
            y = y / norm_factor
            
            # Enhancement
            Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(device))), 0)
            if model.backbone.startswith("ncsnpp"):
                Y = pad_spec(Y, mode=pad_mode)

            sample = model.bridge.sampler(model, Y, **sampler_kwargs)
            
            x_hat = model.to_audio(sample.squeeze(), T_orig)
            
            # Renormalize
            x_hat = x_hat * norm_factor
            if x_hat.abs().max() > 1.0:
                x_hat = x_hat / x_hat.abs().max() * 0.95
            
            # Determine output path
            if args.keep_structure:
                # Keep directory structure relative to test_dir
                rel_path = os.path.relpath(noisy_file, args.test_dir)
                output_file = join(args.enhanced_dir, rel_path)
            else:
                # Flatten to output directory
                output_file = join(args.enhanced_dir, basename(noisy_file))
            
            # Create output directory
            os.makedirs(dirname(output_file), exist_ok=True)
            
            # Write enhanced audio
            write(output_file, x_hat.cpu().numpy(), target_sr)
            
            # Update progress counter
            with progress_counter.get_lock():
                progress_counter.value += 1
            
        except Exception as e:
            print(f"\nError processing {noisy_file} on GPU {gpu_id}: {str(e)}")
            # Still update counter for failed files
            with progress_counter.get_lock():
                progress_counter.value += 1
            continue


def split_list(lst, n):
    """Split list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def enhance_folder(args):
    """Main function for folder enhancement."""
    # Get all audio files
    audio_files = get_audio_files(args.test_dir)
    
    if len(audio_files) == 0:
        print(f"No audio files found in {args.test_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files") 
    
    # Create output directory
    os.makedirs(args.enhanced_dir, exist_ok=True)
    
    # Determine number of GPUs to use
    device_list = device.split(',')
    num_gpus = len(device_list)
    print(f"Using {num_gpus} GPU(s): {device}")
    
    # Create shared progress counter
    progress_counter = mp.Value('i', 0)
    
    if num_gpus == 1:
        # Single GPU - process with progress bar
        # Start worker in a separate process to enable progress monitoring
        p = mp.Process(target=enhance_files_worker, args=(0, audio_files, args, progress_counter))
        p.start()
        
        # Monitor progress
        with tqdm(total=len(audio_files), desc="Processing") as pbar:
            last_value = 0
            while p.is_alive():
                current_value = progress_counter.value
                if current_value > last_value:
                    pbar.update(current_value - last_value)
                    last_value = current_value
                time.sleep(0.1)
            
            # Final update
            current_value = progress_counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)
        
        p.join()
    else:
        # Multi-GPU - split files and use multiprocessing
        file_splits = split_list(audio_files, num_gpus)
        
        # Start processes
        processes = []
        for gpu_idx, file_list in enumerate(file_splits):
            if len(file_list) == 0:
                continue
            p = mp.Process(target=enhance_files_worker, args=(gpu_idx, file_list, args, progress_counter))
            p.start()
            processes.append(p)
        
        # Monitor progress with a single progress bar
        with tqdm(total=len(audio_files), desc="Processing") as pbar:
            last_value = 0
            while any(p.is_alive() for p in processes):
                current_value = progress_counter.value
                if current_value > last_value:
                    pbar.update(current_value - last_value)
                    last_value = current_value
                time.sleep(0.1)
            
            # Final update in case we missed any
            current_value = progress_counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
    
    print(f"Enhancement completed! Results saved to {args.enhanced_dir}")


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    parser = ArgumentParser()
    parser.add_argument('-C', '--config', default='config_infer_folder.yaml', type=str, help='Configuration file')
    initial_args, _ = parser.parse_known_args()
    add_config_args(initial_args)

    parser.add_argument('-D', '--device', default='0', help='Index of the available devices, e.g. 0,1,2,3')
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the noisy audio files')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory to save enhanced audio files')
    parser.add_argument("--ckpt", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="ode_ei", help="Sampler type.")
    parser.add_argument("--sampler_kwargs", type=ast.literal_eval, default=None, help="Dictionary of keyword arguments for the sampler")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--keep_structure", action='store_true', help="Keep directory structure from test_dir in enhanced_dir")

    args, _ = parser.parse_known_args()

    enhance_folder(args)
