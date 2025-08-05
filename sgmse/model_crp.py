from math import ceil
import warnings
import os
import ast
import random

import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
import soundfile as sf

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry, FrontendRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
from sgmse.util.schedulers import LinearWarmupCosineAnnealingLR as WarmupLR
from sgmse.loss_utils import MelSpectrogramLoss
import numpy as np
from sgmse.model import ScoreModel as BaseModel


class ScoreModel(BaseModel):
    def _step(self, batch, batch_idx):
        x, y = batch

        forward_out = self._sb_sampling_step(y)

        loss = self._loss(forward_out, None, None, None, None, x)
        return loss

    def _sb_sampling_step(self, y):
        return self._sb_ode_sampling_step_randN_1(y)

    def _sb_ode_sampling_step_randN_1(self, y, eps=1e-4):
        N = random.randint(1, self.sde.N)
        xt = y
        time_steps = torch.linspace(self.sde.T, eps, N + 1, device=y.device)

        # Initial values
        time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
        sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = self.sde._sigmas_alphas(time_prev)

        for t in time_steps[1:]:
            # Prepare time steps for the whole batch
            time = t * torch.ones(xt.shape[0], device=xt.device)

            # Get noise schedule for current time
            sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self.sde._sigmas_alphas(time)

            # Run DNN
            if t == time_steps[-1]:
                current_estimate = self(xt, y, time_prev)
            else:
                with torch.no_grad():
                    current_estimate = self(xt, y, time_prev).detach()

            # Calculate scaling for the first-order discretization from the paper
            weight_prev = alpha_t * sigma_t * sigma_bart / (alpha_prev * sigma_prev * sigma_bar_prev + self.sde.eps)
            weight_estimate = (
                alpha_t
                / (sigma_T**2 + self.sde.eps)
                * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + self.sde.eps))
            )
            weight_prior_mean = (
                alpha_t
                / (alpha_T * sigma_T**2 + self.sde.eps)
                * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + self.sde.eps))
            )

            # View as [B, C, D, T]
            weight_prev = weight_prev[:, None, None, None]
            weight_estimate = weight_estimate[:, None, None, None]
            weight_prior_mean = weight_prior_mean[:, None, None, None]

            # Update state: weighted sum of previous state, current estimate and prior
            xt = weight_prev * xt + weight_estimate * current_estimate + weight_prior_mean * y

            # Save previous values
            time_prev = time
            alpha_prev = alpha_t
            sigma_prev = sigma_t
            sigma_bar_prev = sigma_bart

        return xt

    def _sb_sde_sampling_step_randN_1(self, y, eps=1e-4):
        """The SB-SDE sampler function."""
        N = random.randint(1, self.sde.N)
        xt = y  # special case for storm_2ch
        time_steps = torch.linspace(self.sde.T, eps, N + 1, device=y.device)

        # Initial values
        time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
        sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = self.sde._sigmas_alphas(time_prev)
        for t in time_steps[1:]:
            # Prepare time steps for the whole batch
            time = t * torch.ones(xt.shape[0], device=xt.device)

            # Get noise schedule for current time
            sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self.sde._sigmas_alphas(time)

            # Run DNN
            if t == time_steps[-1]:
                current_estimate = self(xt, y, time_prev)
            else:
                with torch.no_grad():
                    current_estimate = self(xt, y, time_prev).detach()

            # Calculate scaling for the first-order discretization from the paper
            weight_prev = alpha_t * sigma_t**2 / (alpha_prev * sigma_prev**2 + self.sde.eps)
            tmp = 1 - sigma_t**2 / (sigma_prev**2 + self.sde.eps)
            weight_estimate = alpha_t * tmp
            weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

            # View as [B, C, D, T]
            weight_prev = weight_prev[:, None, None, None]
            weight_estimate = weight_estimate[:, None, None, None]
            weight_z = weight_z[:, None, None, None]

            # Random sample
            z_norm = torch.randn_like(xt)

            if t == time_steps[-1]:
                weight_z = 0.0

            # Update state: weighted sum of previous state, current estimate and noise
            xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm

            # Save previous values
            time_prev = time
            alpha_prev = alpha_t
            sigma_prev = sigma_t
            sigma_bar_prev = sigma_bart

        return xt

    def update_optimizer_and_scheduler(self, new_lr=None, new_scheduler_config=None):
        optimizer = self.optimizers()
        if new_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        if new_scheduler_config is not None:
            if self.scheduler_config['scheduler'] == 'exp':
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


class PredictiveModel(BaseModel):
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
        if self.backbone.startswith("ncsnpp"):
            Y = pad_spec(Y)

        sample = self(Y)
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()

        return x_hat
