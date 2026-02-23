from math import ceil
import warnings
import os
import ast

import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
import soundfile as sf

from fdbm.bridge import Bridge
from fdbm.backbones import BackboneRegistry
from fdbm.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
from fdbm.util.schedulers import LinearWarmupCosineAnnealingLR as WarmupLR
from fdbm.loss import MelSpectrogramLoss, PhaseLoss
import numpy as np


class BridgeModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="data_prediction", help="The type of loss function to use.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--pesq_weight", type=float, default=0.0, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        parser.add_argument("--scheduler_config", type=ast.literal_eval, default=None, help="The scheduler configuration.")
        return parser

    def __init__(
        self, backbone, bridge, lr=1e-4, ema_decay=0.999, scheduler_config=None,
        loss_type='data_prediction', l1_weight=0.001, pesq_weight=0.0, t_eps=0.03,
        sr=16000, data_module_cls=None, log_dir=None, num_eval_files=20, **kwargs
    ):
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)

        # Initialize bridge
        self.bridge = Bridge(bridge, **kwargs)

        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.num_eval_files = num_eval_files
        self.sr = sr
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters()
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

        self.valid_sample_dir = os.path.join(log_dir, "valid_samples")
        os.makedirs(self.valid_sample_dir, exist_ok=True)

        self.scheduler_config = scheduler_config

        if self.loss_type == "data_prediction_mel":
            self.loss_fn = MelSpectrogramLoss(
                n_mels=[5, 10, 20, 40, 80, 160, 210],
                win_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                hop_lengths=[8, 16, 32, 64, 128, 256, 512],
                n_ffts=[32, 64, 128, 256, 512, 1024, 2048],
                mag_weight=0.0,
                log_weight=1.0
            )
        elif self.loss_type == "data_prediction_melphase":
            self.loss_fn_mel = MelSpectrogramLoss(
                n_mels=[5, 10, 20, 40, 80, 160, 210],
                win_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                hop_lengths=[8, 16, 32, 64, 128, 256, 512],
                n_ffts=[32, 64, 128, 256, 512, 1024, 2048],
                mag_weight=0.0,
                log_weight=1.0
            )
            self.loss_fn_phase = PhaseLoss(
                nfreqs=self.data_module.n_fft // 2 + 1,
                frames=self.data_module.num_frames
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler_config['scheduler'] == 'fixed':
            return optimizer
        elif self.scheduler_config['scheduler'] == 'warmup':
            scheduler = WarmupLR(
                optimizer, **self.scheduler_config['config']
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        elif self.scheduler_config['scheduler'] == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **self.scheduler_config['config']
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, forward_out, x_t, z, t, mean, x):
        if self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))

            # losses using PESQ
            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(losses_pesq)
                # combine the losses
                loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq
            else:
                loss = losses_tf + self.l1_weight * losses_l1

        elif self.loss_type == "data_prediction_hybrid":
            x_hat = forward_out
            B, C, F, T = x.shape

            x_nc = self._backward_transform(x)
            x_hat_nc = self._backward_transform(x_hat)
            x_mag = torch.abs(x_nc + 1e-12)
            x_hat_mag = torch.abs(x_hat_nc + 1e-12)

            losses_mag = torch.mean(torch.square(x_mag.pow(0.3) - x_hat_mag.pow(0.3)))
            losses_ri = torch.square(torch.norm(x_nc / x_mag.pow(0.7) - x_hat_nc / x_hat_mag.pow(0.7), p=2)) / (B * C * F * T)

            # target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze())
            x_td = self.to_audio(x.squeeze())

            x_td_norm = torch.div(
                torch.sum(x_td * x_hat_td, dim=-1, keepdim=True) * x_td,
                torch.sum(x_td.pow(2), dim=-1, keepdim=True) + 1e-12
            )
            sisnr = torch.log10(torch.div(
                torch.sum(x_td_norm.pow(2), dim=-1, keepdim=True),
                torch.sum((x_hat_td - x_td_norm).pow(2), dim=-1, keepdim=True) + 1e-12
            ).clamp(min=1e-12)).mean()
            lossed_sisnr = -sisnr

            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(losses_pesq)
                loss = 70 * losses_mag + 30 * losses_ri + lossed_sisnr + self.pesq_weight * losses_pesq
            else:
                loss = 70 * losses_mag + 30 * losses_ri + lossed_sisnr

        elif self.loss_type == "data_prediction_mel":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = torch.mean(torch.square(torch.abs(x_hat - x))) * 0.5

            # losses in the mel domain (mel)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_mel = self.loss_fn(x_hat_td, x_td)

            loss = losses_tf + 0.1 * losses_mel

        elif self.loss_type == "data_prediction_melphase":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = torch.mean(torch.square(torch.abs(x_hat - x))) * 0.5

            # losses in the mel domain (mel)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_mel = self.loss_fn_mel(x_hat_td, x_td)

            # losses in the phase domain (phase)
            losses_phase = self.loss_fn_phase(x_hat, x)

            loss = losses_tf + 0.1 * losses_mel + 0.01 * losses_phase

        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss

    def _step(self, batch, batch_idx):
        x, y = batch

        t, mean, z, x_t = self.sample_prior(x, y)
        forward_out = self(x_t, y, t)

        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def sample_prior(self, x, y):
        z = torch.randn_like(x)
        t = torch.rand(x.shape[0], device=x.device) * (self.bridge.T - self.t_eps) + self.t_eps

        mean, std = self.bridge.probability_path(x, y, t)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z

        return t, mean, z, x_t

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        lr = self.lr if self.scheduler_config['scheduler'] == 'fixed' else self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('learning_rate', lr, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self):
        self.data_module.train_set.sample_data_per_epoch()

    def on_validation_epoch_start(self):
        # Evaluate speech enhancement performance on complete audio files at the start of each validation epoch.
        if self.num_eval_files == 0:
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Split the evaluation files among the GPUs
        eval_files_per_gpu = self.num_eval_files // world_size

        clean_files = self.data_module.valid_set.clean_files[:self.num_eval_files]
        noisy_files = self.data_module.valid_set.noisy_files[:self.num_eval_files]

        # Select the files for this GPU
        if rank == world_size - 1:
            clean_files = clean_files[rank*eval_files_per_gpu:]
            noisy_files = noisy_files[rank*eval_files_per_gpu:]
        else:
            clean_files = clean_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]
            noisy_files = noisy_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]

        # Evaluate the performance of the model
        pesq_sum = 0
        si_sdr_sum = 0
        for idx, (clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)):
            # Load the clean and noisy speech
            x, sr_x = load(clean_file)
            x = x.squeeze().numpy()
            y, sr_y = load(noisy_file)
            assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

            # Resample if necessary
            if sr_x != 16000:
                x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
            else:
                x_16k = x

            # Enhance the noisy speech
            x_hat = self.enhance(y)
            if self.sr != 16000:
                x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
            else:
                x_hat_16k = x_hat

            # Save sample outputs (first 3 files on rank 0)
            if idx < 3 and rank == 0:
                name = os.path.join(self.valid_sample_dir, f"{clean_file.split('/')[-1].split('.')[0]}")
                sf.write(f"{name}_epoch{str(self.current_epoch).zfill(3)}_enh.wav", x_hat_16k, 16000)
                if self.current_epoch == 0:
                    sf.write(f"{name}_noisy.wav", y.squeeze().numpy(), 16000)
                    sf.write(f"{name}_clean.wav", x_16k, 16000)

            # Compute metrics
            if not np.isnan(x_hat).any():
                pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb')
                si_sdr_sum += si_sdr(x, x_hat)

        pesq_avg = pesq_sum / len(clean_files) if len(clean_files) > 0 else 0
        si_sdr_avg = si_sdr_sum / len(clean_files) if len(clean_files) > 0 else 0

        self.log('pesq', pesq_avg, on_step=False, on_epoch=True, sync_dist=True)
        self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)

    def forward(self, x_t, y, t):
        return self.dnn(x_t, y, t)

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
    
    def enhance(self, y, **sampler_kwargs):
        T_orig = y.size(1)
        if self.data_module.normalize == "noisy":
            norm_factor = y.abs().max().item()
        elif self.data_module.normalize == "std":
            norm_factor = torch.std(y, dim=-1).item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        if self.backbone.startswith("ncsnpp"): Y = pad_spec(Y)

        sample = self.bridge.sampler(self, Y, **sampler_kwargs)
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()

        return x_hat

    def load_state_dict_for_dnn(self, checkpoint_path):
        print(f"Load partially pre-trained parameters from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.dnn.load_state_dict(checkpoint['model'], strict=False)


class PredictiveModel(BridgeModel):
    def _step(self, batch, batch_idx):
        x, y = batch
        forward_out = self(y)
        loss = self._loss(forward_out, None, None, None, None, x)
        return loss

    def forward(self, y):
        return self.dnn(y)

    def enhance(self, y, **kwargs):
        T_orig = y.size(1)
        if self.data_module.normalize == "noisy":
            norm_factor = y.abs().max().item()
        elif self.data_module.normalize == "std":
            norm_factor = torch.std(y, dim=-1).item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        if self.backbone.startswith("ncsnpp"): Y = pad_spec(Y)

        sample = self(Y)
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()

        return x_hat


class FinetuningModel(BridgeModel):
    def _step(self, batch, batch_idx):
        x, y = batch
        forward_out = self._sampling_step(y)

        loss = self._loss(forward_out, None, None, None, None, x)
        return loss

    def _sampling_step(self, y, **kwargs):
        xt = self.bridge.prior_sampling(y)
        time_steps = torch.linspace(self.bridge.start_time, self.bridge.end_time, self.bridge.N + 1, device=y.device)
        time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)

        for t in time_steps[1:]:
            time = t * torch.ones(xt.shape[0], device=xt.device)

            if t == time_steps[-1]:
                current_estimate = self(xt, y, time_prev)
            else:
                with torch.no_grad():
                    current_estimate = self(xt, y, time_prev).detach()


            weight_xt, weight_s, weight_y = self.bridge.path.sampling_param_ode(time, time_prev, xt.shape[0], xt.device)
            weight_xt = weight_xt[:, None, None, None]
            weight_s = weight_s[:, None, None, None]
            weight_y = weight_y[:, None, None, None]

            xt = weight_xt * xt + weight_s * current_estimate + weight_y * y

            time_prev = time

        return xt

    def lr_scheduler_step(self, scheduler, metric, optimizer_idx, *args, **kwargs):
        scheduler.step()
