import abc
import torch

from scipy import integrate
from fdbm.util.other import to_flattened_numpy, from_flattened_numpy

from fdbm.util.registry import Registry
from fdbm.util.predictors import PredictorRegistry
from fdbm.util.correctors import CorrectorRegistry

BridgeRegistry = Registry("Bridge")


class Bridge():
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=5, help="The number of steps during sampling. 5 by default.")
        parser.add_argument("--T", type=float, default=1.0, help="The total time duration of the path. 1.0 by default.")
        parser.add_argument("--sampler_type", type=str, default="ode_ei", choices=["ode_ei", "sde_ei", "ode_int", "pc"], help="The sampler type to use. 'ode_ei' by default.")
        parser.add_argument("--sampling_eps", type=float, default=1e-4, help="The minimum process time for sampling. 1e-4 by default.")
        return parser

    def __init__(self, path, N=5, T=1.0, sampler_type="ode_ei", sampling_eps=1e-4, **kwargs):
        path_cls = BridgeRegistry.get_by_name(path)
        self.path = path_cls(T=T, **kwargs)
        self.N = N
        self.T = T
        self.sampler_type = sampler_type

        if self.path.sampling_direction == "forward": 
            self.start_time = sampling_eps
            self.end_time = self.path.T # 1.0
        elif self.path.sampling_direction == "reverse":
            self.start_time = self.path.T # 1.0
            self.end_time = sampling_eps

    def _std(self, t):
       return self.path.sigma_t(t)

    def probability_path(self, s, y, t):
        a_t, b_t, sigma_t = self.path.path_param(t)
        mean = a_t[:, None, None, None] * s + b_t[:, None, None, None] * y
        return mean, sigma_t

    def prior_sampling(self, y):
        _, b_start, sigma_start = self.path.path_param(self.start_time * torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        x_start = y * b_start[:, None, None, None] + z * sigma_start[:, None, None, None]
        return x_start

    def score_fn(self, t, x, s, y):
        mean, sigma = self.probability_path(s, y, t)
        score = - (x - mean) / (sigma[:, None, None, None]**2 + 1e-8)
        return score
    
    def sampler(self, model, y, **kwargs):
        if self.sampler_type == "ode_ei":
            return self.ode_sampler_ei(model, y, **kwargs)
        elif self.sampler_type == "sde_ei":
            return self.sde_sampler_ei(model, y, **kwargs)
        elif self.sampler_type == "ode_int":
            return self.ode_sampler_int(model, y, **kwargs)
        elif self.sampler_type == "pc":
            return self.pc_sampler(model, y, **kwargs)

    def ode_sampler_ei(self, model, y, **kwargs):
        # Exponential integrator based ODE sampler.
        with torch.no_grad():
            xt = self.prior_sampling(y)
            time_steps = torch.linspace(self.start_time, self.end_time, self.N + 1, device=y.device)
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)

            for t in time_steps[1:]:
                time = t * torch.ones(xt.shape[0], device=xt.device)

                current_estimate = model(xt, y, time_prev)

                weight_xt, weight_s, weight_y = self.path.sampling_param_ode_ei(time, time_prev, xt.shape[0], xt.device)
                weight_xt = weight_xt[:, None, None, None]
                weight_s = weight_s[:, None, None, None]
                weight_y = weight_y[:, None, None, None]

                xt = weight_xt * xt + weight_s * current_estimate + weight_y * y

                time_prev = time

        return xt

    def sde_sampler_ei(self, model, y, **kwargs):
        # Exponential integrator based SDE sampler.
        with torch.no_grad():
            xt = self.prior_sampling(y)
            time_steps = torch.linspace(self.start_time, self.end_time, self.N + 1, device=y.device)
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)

            for t in time_steps[1:]:
                time = t * torch.ones(xt.shape[0], device=xt.device)

                current_estimate = model(xt, y, time_prev)

                weight_xt, weight_s, weight_z = self.path.sampling_param_sde_ei(time, time_prev, xt.shape[0], xt.device)
                weight_xt = weight_xt[:, None, None, None]
                weight_s = weight_s[:, None, None, None]
                weight_z = weight_z[:, None, None, None]
                if t == time_steps[-1]:
                    weight_z = torch.zeros_like(weight_z)

                z = torch.randn_like(xt)
                xt = weight_xt * xt + weight_s * current_estimate + weight_z * z

                time_prev = time

        return xt

    def ode_sampler_int(self, model, y, rtol=1e-5, atol=1e-5, method='RK45', **kwargs):
        with torch.no_grad():
            x = self.prior_sampling(y)
            
            def ode_func(t, x):
                x_tensor = from_flattened_numpy(x, y.shape).to(y.device).type(torch.complex64)
                t_tensor = torch.ones(y.shape[0], device=y.device) * t

                s = model(x_tensor, y, t_tensor)
                flow = self.path.ode(t_tensor, x_tensor, s, y)
                
                return to_flattened_numpy(flow)

            solution = integrate.solve_ivp(
                ode_func, 
                (self.start_time, self.end_time), 
                to_flattened_numpy(x),
                rtol=rtol, 
                atol=atol, 
                method=method,
                **kwargs
            )

            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(y.device).type(torch.complex64)
            
        return x

    def pc_sampler(self, model, y, predictor_name='reverse_diffusion', corrector_name='ald', 
                   denoise=True, snr=0.5, corrector_steps=1, **kwargs):
        # The predictor-corrector sampler.
        predictor_cls = PredictorRegistry.get_by_name(predictor_name)
        corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
        predictor = predictor_cls(self, model)
        corrector = corrector_cls(self, model, snr=snr, n_steps=corrector_steps)
        
        with torch.no_grad():
            xt = self.prior_sampling(y)
            timesteps = torch.linspace(self.start_time, self.end_time, self.N, device=y.device)
            
            for i in range(self.N):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]
                    
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = corrector.update_fn(xt, y, vec_t)
                xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)

            x_result = xt_mean if denoise else xt
            return x_result


class ProbabilityPath(abc.ABC):
    def __init__(self, T=1.0):
        self.T = T

    @abc.abstractmethod
    def path_param(self, t):
        pass

    @abc.abstractmethod
    def sigma_t(self, t):
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        pass


@BridgeRegistry.register("sb")
class ProbabilityPathSB(ProbabilityPath):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--noise_schedule", type=str, default="bb", choices=["gmax", "vp", "ve", "bb"], help="Noise schedule to use. 'BB' by default.")
        parser.add_argument("--k", type=float, default=2.6, help="Parameter of the diffusion coefficient. 2.6 by default for VE.")
        parser.add_argument("--c", type=float, default=0.4, help="Parameter of the diffusion coefficient. 0.4 by default for VE, 0.3 for VP.")
        parser.add_argument("--beta_0", type=float, default=0.01, help="Parameter of the diffusion coefficient. 0.01 by default for both gmax and VP.")
        parser.add_argument("--beta_1", type=float, default=20.0, help="Parameter of the diffusion coefficient. 20.0 by default for both gmax and VP.")
        parser.add_argument("--rho", type=float, default=1.0, help="Parameter of the diffusion coefficient. 1.0 by default for both BB (CFM).")
        parser.add_argument("--diffusion_coeff_mode", type=str, default="g", choices=["g", "ode"], help="'g' by default. ")
        return parser

    def __init__(self, noise_schedule="bb", k=2.6, c=0.4, beta_0=0.01, beta_1=20.0, rho=1.0, N=5, eps=1e-8, **ignored_kwargs):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.k = k
        self.c = c
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.rho = rho
        self.N = N
        self.eps = eps
        self.sampling_direction = "reverse"
        self.diffusion_coeff_mode = "g" 

    def _rhos_alphas(self, t):
        if self.noise_schedule == "gmax":
            alpha_t = torch.ones_like(t)
            alpha_T = torch.ones_like(t)
            rho_t = torch.sqrt(self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * (t ** 2))
            rho_T = torch.sqrt(torch.tensor(self.beta_0 * self.T + 0.5 * (self.beta_1 - self.beta_0) * (self.T ** 2)))
        elif self.noise_schedule == "vp":
            alpha_t = torch.exp(-0.5 * (self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * (t ** 2)))
            alpha_T = torch.exp(-0.5 * torch.tensor(self.beta_0 * self.T + 0.5 * (self.beta_1 - self.beta_0) * (self.T ** 2)))
            rho_t = torch.sqrt(self.c * (torch.exp(self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * (t ** 2)) - 1))
            rho_T = torch.sqrt((self.c * (torch.exp(torch.tensor(self.beta_0 * self.T + 0.5 * (self.beta_1 - self.beta_0) * (self.T ** 2))) - 1)))
        elif self.noise_schedule == "ve":
            alpha_t = torch.ones_like(t)
            alpha_T = torch.ones_like(t)
            rho_t = torch.sqrt((self.c*(self.k**(2*t)-1.0)) / (2*torch.log(torch.tensor(self.k))))
            rho_T = torch.sqrt((self.c*(self.k**(2*self.T)-1.0)) / (2*torch.log(torch.tensor(self.k))))
        elif self.noise_schedule == "bb":  # sb-cfm
            alpha_t = torch.ones_like(t)
            alpha_T = torch.ones_like(t)
            rho_t = torch.sqrt(torch.tensor(1) * t) * self.rho
            rho_T = torch.ones_like(t) * self.rho

        alpha_bar_t = alpha_t / (alpha_T + self.eps)
        rho_bar_t = torch.sqrt(rho_T**2 - rho_t**2 + self.eps)

        return rho_t, rho_T, rho_bar_t, alpha_t, alpha_T, alpha_bar_t

    def auxiliary_param(self, t):
        if self.noise_schedule == "ve":
            f = 0.0
            g = torch.sqrt(torch.tensor(self.c)) * self.k**(t)
        elif self.noise_schedule == "vp":
            f = -0.5 * (self.beta_0 + (self.beta_1 - self.beta_0) * t)
            g = torch.sqrt(torch.tensor(self.c) * (self.beta_0 + (self.beta_1 - self.beta_0) * t))
        elif self.noise_schedule == "gmax":
            f = 0.0
            g = torch.sqrt(torch.tensor(self.beta_0 + (self.beta_1 - self.beta_0) * t))
        elif self.noise_schedule == "bb":
            f = 0.0
            g = self.rho * torch.ones_like(t)
        return f, g

    def diffusion_coeff(self, g, t):
        if self.diffusion_coeff_mode == "g":
            return g
        elif self.diffusion_coeff_mode == "ode":
            return 0.0 * torch.ones_like(g)

    def sigma_t(self, t):
        rho_t, rho_T, rho_bar_t, alpha_t, alpha_T, alpha_bar_t = self._rhos_alphas(t)
        sigma_t = (alpha_t * rho_bar_t * rho_t) / (rho_T + self.eps)

        mask = (t == 1.0)
        sigma_t = torch.where(mask, torch.zeros_like(sigma_t), sigma_t)

        return sigma_t

    def path_param(self, t):
        rho_t, rho_T, rho_bar_t, alpha_t, alpha_T, alpha_bar_t = self._rhos_alphas(t)
        a_t = alpha_t * rho_bar_t**2 / (rho_T**2 + self.eps)
        b_t = alpha_bar_t * rho_t**2 / (rho_T**2 + self.eps)
        sigma_t = (alpha_t * rho_bar_t * rho_t) / (rho_T + self.eps)

        mask = (t == 1.0)
        a_t = torch.where(mask, torch.zeros_like(a_t), a_t)
        b_t = torch.where(mask, torch.ones_like(b_t), b_t)
        sigma_t = torch.where(mask, torch.zeros_like(sigma_t), sigma_t)
        
        return a_t, b_t, sigma_t

    def ode(self, t, x, s, y):
        rho, rho_T, rho_bar, alpha, alpha_T, alpha_bar = self._rhos_alphas(t)
        f, g = self.auxiliary_param(t)

        weight_xt = f + g**2 * (rho_bar**2 - rho**2) / (2 * alpha**2 * rho**2 * rho_bar**2 + self.eps)
        weight_s = - g**2 / (2 * alpha * rho**2 + self.eps)
        weight_y = alpha_bar * g**2 / (2 * alpha**2 * rho_bar**2 + self.eps)

        flow = weight_xt * x + weight_s * s + weight_y * y
        return flow

    def sde(self, t, x, s, y):
        # Reverse SDE
        rho, rho_T, rho_bar, alpha, alpha_T, alpha_bar = self._rhos_alphas(t)
        f, g = self.auxiliary_param(t)
        gd = self.diffusion_coeff(g, t)

        weight_xt = f + ((g**2 + gd**2) * rho_bar**2 - (g**2 - gd**2) * rho**2) / (2 * alpha**2 * rho**2 * rho_bar**2 + self.eps)
        weight_s = - (g**2 + gd**2) / (2 * alpha * rho**2 + self.eps)
        weight_y = alpha_bar * (g**2 - gd**2) / (2 * alpha**2 * rho_bar**2 + self.eps)

        drift = weight_xt * x + weight_s * s + weight_y * y
        diffusion = gd
        return drift, diffusion

    def sampling_param_ode_ei(self, t_curr, t_prev, batch_size, device):
        time_prev = t_prev * torch.ones(batch_size, device=device)
        time_curr = t_curr * torch.ones(batch_size, device=device)
        rho_prev, rho_T, rho_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = self._rhos_alphas(time_prev)
        rho_curr, rho_T, rho_bar_curr, alpha_curr, alpha_T, alpha_bar_curr = self._rhos_alphas(time_curr)

        weight_xt = alpha_curr * rho_curr * rho_bar_curr / (alpha_prev * rho_prev * rho_bar_prev + self.eps)
        weight_s = (
            alpha_curr / (rho_T ** 2 + self.eps)
            * (rho_bar_curr ** 2 - rho_bar_prev * rho_curr * rho_bar_curr / (rho_prev + self.eps))
        )
        weight_y = (
            alpha_curr / (alpha_T * rho_T ** 2 + self.eps)
            * (rho_curr ** 2 - rho_prev * rho_curr * rho_bar_curr / (rho_bar_prev + self.eps))
        )

        return weight_xt, weight_s, weight_y

    def sampling_param_sde_ei(self, t_curr, t_prev, batch_size, device):
        time_prev = t_prev * torch.ones(batch_size, device=device)
        time_curr = t_curr * torch.ones(batch_size, device=device)
        rho_prev, rho_T, rho_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = self._rhos_alphas(time_prev)
        rho_curr, rho_T, rho_bar_curr, alpha_curr, alpha_T, alpha_bar_curr = self._rhos_alphas(time_curr)

        weight_xt = alpha_curr * rho_curr **2 / (alpha_prev * rho_prev**2 + self.eps)
        tmp = 1 - rho_curr**2 / (rho_prev**2 + self.eps)
        weight_s = alpha_curr * tmp
        weight_z = alpha_curr * rho_curr * torch.sqrt(tmp)

        return weight_xt, weight_s, weight_z


@BridgeRegistry.register("fm")
class ProbabilityPathFM(ProbabilityPath):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sigma_max", type=float, default=1.0, help="Sigma_max. 1.0 by default.")
        parser.add_argument("--sigma_min", type=float, default=0.01, help="Sigma_min. 0.01 by default.")
        parser.add_argument("--noise_schedule", type=str, default="ot", help="sb is the same as sbbb (sb-cfm), omitted.")
        return parser

    def __init__(self, sigma_max=1.0, sigma_min=0.01, noise_schedule='ot', eps=1e-8, **ignored_kwargs):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.noise_schedule = noise_schedule
        self.eps = eps
        self.sampling_direction = "forward"

    def sigma_t(self, t):
        sigma_t = t * self.sigma_min + (1 - t) * self.sigma_max
        return sigma_t

    def path_param(self, t):
        a_t = t
        b_t = 1 - t
        sigma_t = self.sigma_t(t)
        
        return a_t, b_t, sigma_t

    def ode(self, t, x, s, y):
        sigma_t = self.sigma_t(t)[:, None, None, None]
        flow = ( (self.sigma_min - self.sigma_max) * x + self.sigma_max * s - self.sigma_min * y ) / (sigma_t + self.eps)
        return flow
    
    def sampling_param_ode_ei(self, t_curr, t_prev, batch_size, device):
        # The same as Euler method for OT-CFM.
        time_prev = t_prev * torch.ones(batch_size, device=device)
        time_curr = t_curr * torch.ones(batch_size, device=device)
        t_diff = time_curr - time_prev
        sigma_curr = self.sigma_t(time_curr)
        sigma_prev = self.sigma_t(time_prev)

        weight_xt = sigma_curr / (sigma_prev + self.eps)
        weight_s = self.sigma_max * t_diff / (sigma_prev + self.eps)
        weight_y = - self.sigma_min * t_diff / (sigma_prev + self.eps)

        return weight_xt, weight_s, weight_y


class ProbabilityPathSGM(ProbabilityPath):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--k", type=float, default=2.6, help="base factor for diffusion term")
        parser.add_argument("--theta", type=float, default=0.52, help="root scale factor for diffusion term.")
        parser.add_argument("--noise_schedule", type=str, default="ouve", choices=["ouve", "bbve", "ouvp", "bbvp"], help="sb is the same as sbbb (sb-cfm), omitted.")
        return parser

    def __init__(self, noise_schedule='ot', eps=1e-8, **ignored_kwargs):
        super().__init__()
        # TODO
        pass

    