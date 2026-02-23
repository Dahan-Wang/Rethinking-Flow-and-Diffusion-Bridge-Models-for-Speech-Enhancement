import abc

import torch
import numpy as np

from fdbm.util.registry import Registry


PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, bridge, model):
        super().__init__()
        self.bridge = bridge
        self.model = model

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


@PredictorRegistry.register('euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, bridge, model):
        super().__init__(bridge, model)

    def update_fn(self, x, y, t, stepsize):
        dt = -stepsize
        z = torch.randn_like(x)
        s = self.model(x, y, t)
        drift, diffusion = self.bridge.path.sde(t, x, s, y) 
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * torch.sqrt(-dt) * z
        return x, x_mean


@PredictorRegistry.register('none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, y, t, *args):
        return x, x
