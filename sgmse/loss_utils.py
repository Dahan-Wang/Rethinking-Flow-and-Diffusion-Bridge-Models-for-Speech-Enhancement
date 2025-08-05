import typing
from typing import List

import torch
from torch import nn
import functools


class PhaseLoss(nn.Module):
    def __init__(self, nfreqs=256, frames=256):
        super().__init__()
        self.GD_matrix = torch.triu(torch.ones(nfreqs, nfreqs), diagonal=1) - torch.triu(torch.ones(nfreqs, nfreqs), diagonal=2) - torch.eye(nfreqs)
        self.PTD_matrix = torch.triu(torch.ones(frames, frames), diagonal=1) - torch.triu(torch.ones(frames, frames), diagonal=2) - torch.eye(frames)

    @staticmethod
    def unwrap(x):
        return torch.abs(x - 2 * torch.pi * torch.round(x / (2 * torch.pi)))

    def forward(self, spec_est, spec_ref):
        phase_g = torch.angle(spec_est).squeeze(1)
        phase_r = torch.angle(spec_ref).squeeze(1)

        GD_r = torch.matmul(phase_r.permute(0, 2, 1), self.GD_matrix.to(phase_r.device))
        GD_g = torch.matmul(phase_g.permute(0, 2, 1), self.GD_matrix.to(phase_r.device))

        PTD_r = torch.matmul(phase_r, self.PTD_matrix.to(phase_r.device))
        PTD_g = torch.matmul(phase_g, self.PTD_matrix.to(phase_r.device))

        IP_loss = torch.mean(torch.abs(self.unwrap(phase_r - phase_g)))
        GD_loss = torch.mean(torch.abs(self.unwrap(GD_r - GD_g)))
        PTD_loss = torch.mean(torch.abs(self.unwrap(PTD_r - PTD_g)))

        return IP_loss + GD_loss + PTD_loss


class L1Loss(nn.L1Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, y):
        return super().forward(x, y)


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    win_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    hop_lengths : List[int], optional
        Hop length of each window of each STFT, by default [512, 128]
    n_ffts : List[int], optional
        Number of FFT bins of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        win_lengths: List[int] = [2048, 512],
        hop_lengths: List[int] = [512, 128],
        n_ffts: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0
    ):
        super().__init__()
        self.stft_params = [
            {'win_length': w, 'hop_length': h, 'n_fft': n}
            for w, h, n in zip(win_lengths, hop_lengths, n_ffts)
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x, y):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Estimate signal
        y : Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            window = torch.hann_window(s['win_length']).to(x.device)
            X = torch.stft(x.reshape(-1, x.shape[-1]), window=window, return_complex=True, **s)
            Y = torch.stft(y.reshape(-1, y.shape[-1]), window=window, return_complex=True, **s)

            X_mag = X.abs()
            Y_mag = Y.abs()
            loss += self.log_weight * self.loss_fn(
                X_mag.clamp(min=self.clamp_eps).pow(self.pow).log10(),
                Y_mag.clamp(min=self.clamp_eps).pow(self.pow).log10()
            )
            loss += self.mag_weight * self.loss_fn(X_mag, Y_mag)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: List[int] = [150, 80],
        win_lengths: List[int] = [2048, 512],
        hop_lengths: List[int] = [512, 128],
        n_ffts: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None]
    ):
        super().__init__()
        self.stft_params = [
            {'win_length': w, 'hop_length': h, 'n_fft': n}
            for w, h, n in zip(win_lengths, hop_lengths, n_ffts)
        ]
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x, y):
        loss = 0.0
        for n_mels, s in zip(self.n_mels, self.stft_params):
            window = torch.hann_window(s['win_length']).to(x.device)
            X = torch.stft(x.reshape(-1, x.shape[-1]), window=window, return_complex=True, **s)
            Y = torch.stft(y.reshape(-1, y.shape[-1]), window=window, return_complex=True, **s)

            x_mels = self.mel_spectrogram(torch.abs(X), n_mels)
            y_mels = self.mel_spectrogram(torch.abs(Y), n_mels)

            if self.log_weight > 0:
                loss += self.log_weight * self.loss_fn(
                    x_mels.clamp(min=self.clamp_eps).pow(self.pow).log10(),
                    y_mels.clamp(min=self.clamp_eps).pow(self.pow).log10(),
                )
            if self.mag_weight > 0:
                loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None):
        from librosa.filters import mel as librosa_mel_fn
        return librosa_mel_fn(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

    def mel_spectrogram(self, magnitude, n_mels: int = 80, mel_fmin: float = 0.0, mel_fmax: float = None):
        # magnitude: [B, C, F, T]
        nf = magnitude.shape[-2]
        mel_basis = self.get_mel_filters(
            sr=self.sample_rate,
            n_fft=2 * (nf - 1),
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).to(magnitude.device)

        mel_spectrogram = magnitude.transpose(-2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, -2)
        return mel_spectrogram


if __name__ == "__main__":
    x = torch.randn(2, 1, 16000)
    y = torch.randn(2, 1, 16000)

    loss = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 210],
        win_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        hop_lengths=[8, 16, 32, 64, 128, 256, 512],
        n_ffts=[32, 64, 128, 256, 512, 1024, 2048],
        mag_weight=1.0,
        log_weight=0.0
    )
    print(loss(x, y))
    loss = MultiScaleSTFTLoss()
    print(loss(x, y))
    loss = L1Loss()
    print(loss(x, y))
