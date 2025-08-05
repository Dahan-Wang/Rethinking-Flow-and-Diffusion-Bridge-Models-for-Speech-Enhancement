import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as optimize
import torch.nn.functional as Func
# from scipy.signal import kaiser
from scipy.signal.windows import kaiser
import math
from abc import abstractmethod
from torch.optim import lr_scheduler


class PQMF(torch.nn.Module):
    """PQMF module.

    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122

    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=None, beta=9.0):
        
        """Initilize PQMF module.

        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.

        """
        super(PQMF, self).__init__()
        # configurations
        self.subbands = subbands
        self.taps = taps 
        self.beta = beta 
        self.cutoff_ratio = cutoff_ratio
        if cutoff_ratio == None:
            self.optimize_cutoff_ratio()

        # build analysis & synthesis filter coefficients
        h_proto = self.design_prototype_filter(self.taps, self.cutoff_ratio, self.beta)
        h_analysis = np.zeros((self.subbands, len(h_proto)))
        h_synthesis = np.zeros((self.subbands, len(h_proto)))
        for k in range(self.subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * self.subbands)) *
                (np.arange(self.taps + 1) - (self.taps / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * self.subbands)) *
                (np.arange(self.taps + 1) - (self.taps / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def _objective(self, cutoff_ratio):
        h_proto = self.design_prototype_filter(self.taps, cutoff_ratio, self.beta)
        conv_h_proto = np.convolve(h_proto, h_proto[::-1], mode='full')
        length_conv_h = conv_h_proto.shape[0]
        half_length = length_conv_h // 2

        check_steps = np.arange((half_length) // (2 * self.subbands)) * 2 * self.subbands
        _phi_new = conv_h_proto[half_length:][check_steps]
        phi_new = np.abs(_phi_new[1:]).max()
        # Since phi_new is not convex, This value should also be considered. 
        diff_zero_coef = np.abs(_phi_new[0] - 1 / (2 * self.subbands))
        
        return phi_new + diff_zero_coef

    def design_prototype_filter(self, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Design prototype filter for PQMF.

        This method is based on `A Kaiser window approach for the design of prototype
        filters of cosine modulated filterbanks`_.

        Args:
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.

        Returns:
            ndarray: Impluse response of prototype filter (taps + 1,).

        .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
            https://ieeexplore.ieee.org/abstract/document/681427

        """
        # check the arguments are valid
        assert taps % 2 == 0, "The number of taps mush be even number."
        assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

        # make initial filter
        omega_c = np.pi * cutoff_ratio
        with np.errstate(invalid='ignore'):
            h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
                / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
        h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

        # apply kaiser window
        w = kaiser(taps + 1, beta)
        h = h_i * w

        return h
    
    def optimize_cutoff_ratio(self):
        ret = optimize.minimize(self._objective, np.array([0.01]), 
                            bounds=optimize.Bounds(0.01, 0.99))
        opt_cutoff_ratio = ret.x[0]
        self.cutoff_ratio = opt_cutoff_ratio
        # print("optimized cutoff ratio = {:.08f} for {} subbbands with beta value {}".format(opt_cutoff_ratio, self.subbands, self.beta))
    
    def analysis(self, x):
        """Analysis with PQMF.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).

        """
        x = Func.conv1d(self.pad_fn(x), self.analysis_filter)
        return Func.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.

        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).

        Returns:
            Tensor: Output tensor (B, 1, T).

        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = Func.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return Func.conv1d(self.pad_fn(x), self.synthesis_filter)

    
def sfi_multiR_stft(x, sr, 
                    fft_lens=[0.050, 0.025, 0.010],
                    hop_lens=[0.005, 0.0025, 0.001],
                    win_lens=[0.025, 0.0125, 0.005]):
    """
    x: (B,t), signal in time domain
    sr: sampling rate
    """
    output = []
    for fl, hl, wl in zip(fft_lens, hop_lens, win_lens):
        x_spec = torch.stft(x, int(fl*sr), int(hl*sr), int(wl*sr), torch.hann_window(int(sr*wl)).to(x.device), return_complex=True)  # (B,F,T)
        output += [x_spec]
    return output


def sfi_multiB_stft(x, sr, n_bands=3):
    """
    x: (B,t), signal in time domain
    sr: sampling rate
    """
    pqmf = PQMF(n_bands).to(x.device)
    xs = pqmf.analysis(x[:,None])
    output = []
    for i in range(n_bands):
        n_fft = int(0.032 * sr//n_bands)
        x_spec = torch.stft(xs[:,i], n_fft, n_fft//2, n_fft, torch.hann_window(n_fft).to(x.device), return_complex=True)  # (B,F,T)
        output += [x_spec]
    return output


def sfi_fullsub_multiR_stft(x, sr, n_bands=3,
                            fft_lens=[0.050, 0.025, 0.010],
                            hop_lens=[0.005, 0.0025, 0.001],
                            win_lens=[0.025, 0.0125, 0.005]):
    """
    x: (B,t), signals in time domain
    sr: sampling rate
    """
    pqmf = PQMF(n_bands).to(x.device)
    xs = pqmf.analysis(x[:,None])  # (B,n,t)
    output_s = []
    for i in range(n_bands):
        output = sfi_multiR_stft(xs[:,i], sr//n_bands, fft_lens, hop_lens, win_lens)
        output_s += output
    output_f = sfi_multiR_stft(x, sr, fft_lens, hop_lens, win_lens)
    output_fs = output_f + output_s
    return output_fs
    

class BaseLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self) -> list[float]:
        """Returns the current learning rate for each parameter group."""
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self, **kwargs) -> None:
        """Reinitializes the learning rate scheduler."""
        raise NotImplementedError


class LinearWarmupCosineAnnealingLR(BaseLRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_until_step, max_lr, min_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_until_step = decay_until_step
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def compute_lr(step, warmup_steps, decay_until_step, max_lr, min_lr):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        if step > decay_until_step:
            return min_lr
        if warmup_steps <= step < decay_until_step:
            decay_ratio = (step - warmup_steps) / (decay_until_step - warmup_steps)
            assert 0.0 <= decay_ratio <= 1.0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)
        else:
            return min_lr

    def get_lr(self) -> list[float]:
        """Returns the current learning rate for each parameter group."""
        step = self.last_epoch
        return (
            self.compute_lr(step, self.warmup_steps, self.decay_until_step, self.max_lr, self.min_lr)
            for _ in self.optimizer.param_groups
        )


if __name__ == "__main__":
    import soundfile as sf
    
    device=torch.device("cpu")
    pqmf = PQMF(subbands=3).to(device)
    x, fs = sf.read('./test/train_clean.wav', dtype='float32')
    x = torch.from_numpy(x[None, None]).to(device)
    xs = pqmf.analysis(x)
    
    # sf.write('./test/train_clean_sub1.wav', xs[0,0].cpu().detach().numpy(), fs//3)
    # sf.write('./test/train_clean_sub2.wav', xs[0,1].cpu().detach().numpy(), fs//3)
    # sf.write('./test/train_clean_sub3.wav', xs[0,2].cpu().detach().numpy(), fs//3)
    print(x.shape)
    print(xs.shape)
    
    
    print("===========Multi-resolution STFT================")
    mrx_lst = sfi_multiR_stft(x[:,0], sr=fs)
    for item in mrx_lst:
        print(item.shape)
    
    print("===========Multi-band STFT================")
    mbx_lst = sfi_multiB_stft(x[:,0], sr=fs)
    for item in mbx_lst:
        print(item.shape)
            