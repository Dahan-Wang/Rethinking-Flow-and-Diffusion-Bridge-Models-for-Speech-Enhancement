import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

# from espnet2.enh.separator.abs_separator import AbsSeparator
# from espnet2.torch_utils.get_layer_from_string import get_layer

from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple
import difflib

from .ncsnpp_utils import layers, layerspp

from .shared import BackboneRegistry

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler


class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:

        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError


class TFGridNet(AbsSeparator):
    """Offline TFGridNetV3. Compared with TFGridNet, TFGridNetV3 speeds up the code
        by vectorizing multiple heads in self-attention, and better dealing with
        Deconv1D in each intra- and inter-block when emb_ks == emb_hs.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in ICASSP, 2023.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.
    Specifically, use:
        std_ = std(mix)
        mix = mix / std_
        tgt = tgt / std_

    Args:
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNetV3 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_attn_qk_output_channel: output channels of point-wise conv2d for getting key and query
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNetV3 model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        n_srcs=1,
        n_imics=2,
        n_layers=6,
        lstm_hidden_units=200,
        attn_n_head=4,
        attn_qk_output_channel=2,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        time_embedding_type='fourier',
        fourier_scale=16,
        **kwargs
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.emb_dim = emb_dim

        in_ch = 2 * n_imics
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

        # timestep/noise_level embedding
        self.time_embedding_type = time_embedding_type
        if time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            self.get_time_emb = layerspp.GaussianFourierProjection(
                embedding_size=emb_dim, scale=fourier_scale
            )
            time_emb_dim = 2 * emb_dim
        elif time_embedding_type == 'positional':
            time_emb_dim = emb_dim
        else:
            raise ValueError(f'embedding type {time_embedding_type} unknown.')

        self.time_emb_fc = nn.Sequential(
            nn.Linear(time_emb_dim, emb_dim * 4), nn.SiLU(), nn.Linear(emb_dim * 4, emb_dim * 4), nn.SiLU()
        )
        self.time_emb_blocks = nn.ModuleList(
            [nn.Linear(emb_dim * 4, emb_dim) for _ in range(n_layers)]
        )

    def forward(self, x, y, t):
        """
        x, y: [B, 1, F, T], complex; t: [B,]
        out: [B, 1, F, T], complex
        """
        input = torch.cat((x.real, x.imag, y.real, y.imag), dim=1)  # [B, 4, F, T]

        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.get_time_emb(torch.log(used_sigmas))

        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = layers.get_timestep_embedding(timesteps, self.emb_dim)

        else:
            raise ValueError(f'embedding type {self.time_embedding_type} unknown.')

        temb = self.time_emb_fc(temb)  # [B, emb_dim*4]

        batch = input.permute(0, 1, 3, 2)  # [B, 4, T, F]
        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.time_emb_blocks[ii](temb)[:, :, None, None] + batch
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch).reshape(
            batch.shape[0], self.n_srcs, 2, batch.shape[2], batch.shape[3]
        )  # [B, n_srcs, 2, T, F]
        batch = torch.view_as_complex(batch.permute(0, 1, 4, 3, 2).contiguous())  # [B, n_srcs, T, F]

        return batch

    @property
    def num_spk(self):
        return self.n_srcs


class GridNetV3Block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"

        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        # use constant E not to be dependent on the number of frequency bins
        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC(
                (n_head, emb_dim // n_head), eps=eps
            ),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetV2Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        batch = inter_rnn

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](batch))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](batch))  # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, )
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x


@BackboneRegistry.register("tfgridnet_5l32c100")
class TFGridNet_5l32c100(TFGridNet):
    def __init__(self, **kwargs):
        super().__init__(n_layers=5, emb_dim=32, lstm_hidden_units=100, **kwargs)

    def forward(self, x, y, t):
        return super().forward(x, y, t)

    @staticmethod
    def add_argparse_args(parser):
        return parser


@BackboneRegistry.register("tfgridnet_4l32c80")
class TFGridNet_4l32c80(TFGridNet):
    def __init__(self, **kwargs):
        super().__init__(n_layers=4, emb_dim=32, lstm_hidden_units=80, **kwargs)

    def forward(self, x, y, t):
        return super().forward(x, y, t)

    @staticmethod
    def add_argparse_args(parser):
        return parser


if __name__ == "__main__":
    model = TFGridNet_5l32c100().eval()
    x = torch.view_as_complex(torch.randn(1, 1, 256, 256, 2))
    y = torch.view_as_complex(torch.randn(1, 1, 256, 256, 2))
    t = torch.randn(1)
    out = model(x, y, t)
    print(out.shape)

    from ptflops import get_model_complexity_info

    class Mymodel(TFGridNet_5l32c100):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.view_as_complex(torch.randn(1, 1, 257, 63, 2))
            y = torch.view_as_complex(torch.randn(1, 1, 257, 63, 2))
            t = torch.randn(1)
            return super().forward(x, y, t)

    mymodel = Mymodel()
    flops, params = get_model_complexity_info(
        mymodel, (1, 257, 63), as_strings=True,
        print_per_layer_stat=True, verbose=False,
        flops_units='GMac', param_units='M')
    print(flops, params)
