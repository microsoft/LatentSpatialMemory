import math
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn


class SchedulerInterface(ABC):
    """
    Base class for diffusion noise schedule.
    """

    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    @abstractmethod
    def add_noise(
        self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """
        pass

    def convert_x0_to_noise(
        self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's x0 prediction to noise predidction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(
            lambda x: x.double().to(x0.device), [x0, xt, self.alphas_cumprod]
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 predidction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(
            lambda x: x.double().to(noise.device), [noise, xt, self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 predidction.
        velocity: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device),
            [velocity, xt, self.alphas_cumprod],
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t**0.5) * xt - (beta_prod_t**0.5) * velocity
        return x0_pred.to(original_dtype)


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self, num_inference_steps=100, denoising_strength=1.0, training=False
    ):
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        elif timestep.ndim > 1:
            timestep = timestep.flatten(0, -1)
        device = model_output.device
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        timestep = timestep.to(device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        device = noise.device
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        timestep = timestep.to(device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        device = timestep.device
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(device)
        self.timesteps = self.timesteps.to(device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


class DDPMSchedulerSimple(nn.Module):
    """
    A compact, self-contained DDPM noise scheduler with an API similar to your FlowMatchScheduler:
    - set_timesteps(): build an inference time grid
    - step(): one reverse step x_t -> x_{t-1}
    - add_noise(): forward corruption q(x_t | x_0)
    - training_target(): build supervision target for the denoiser
    - training_weight(): optional loss re-weighting (e.g., Min-SNR)

    Key options:
      num_train_timesteps: total training steps (default 1000; valid indices [0..999])
      beta_schedule: {"linear","scaled_linear","squaredcos_cap_v2"}
      prediction_type: {"epsilon","v_prediction","sample"}  # ε, v, or x0
      variance_type: {"fixed_small","fixed_large","fixed_small_log"}
      timestep_spacing: {"leading","trailing","linspace"}  # how to pick inference steps
      rescale_betas_zero_snr: if True, last step SNR≈0 (999 is ~pure noise)
      extra_one_step: if True, allow t == num_train_timesteps (pure noise endpoint)
      snr_gamma: if set, use Min-SNR weighting in training_weight()
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: Literal[
            "linear", "scaled_linear", "squaredcos_cap_v2"
        ] = "squaredcos_cap_v2",
        prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon",
        variance_type: Literal[
            "fixed_small", "fixed_large", "fixed_small_log"
        ] = "fixed_small",
        timestep_spacing: Literal["leading", "trailing", "linspace"] = "leading",
        rescale_betas_zero_snr: bool = False,
        extra_one_step: bool = False,
        snr_gamma: Optional[float] = None,
    ):
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.timestep_spacing = timestep_spacing
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.extra_one_step = extra_one_step
        self.snr_gamma = snr_gamma

        # ---- build training buffers (betas, alphas, etc.) ----
        betas = self._make_betas(self.num_train_timesteps, beta_schedule)
        if rescale_betas_zero_snr:
            # Rescale betas so that final SNR≈0 (diffusers trick)
            betas = self._rescale_zero_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        # Register as buffers for device/dtype management
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev, persistent=False
        )

        # Precompute common terms
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas), persistent=False
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-12),
            persistent=False,
        )
        # log variants (for "fixed_small_log")
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(self.posterior_variance, min=1e-20)),
            persistent=False,
        )

        # coefficients for posterior mean: q(x_{t-1}|x_t,x_0) = N(μ_t, Σ_t)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-12),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod + 1e-12),
            persistent=False,
        )

        # Build a default inference grid
        self.set_timesteps(num_inference_steps=50)

    # ---------- public API ----------

    def set_timesteps(self, num_inference_steps: int = 50):
        """
        Build a sequence of inference timesteps (ints in [0..T-1], descending).
        """
        T = self.num_train_timesteps
        if self.timestep_spacing == "leading":
            # like diffusers' "leading": choose evenly in [0..T-1], include 0
            timesteps = (
                torch.linspace(T - 1, 0, steps=num_inference_steps).round().long()
            )
        elif self.timestep_spacing == "trailing":
            # like "trailing": avoid early times, bias towards later timesteps
            edges = torch.linspace(0, T, steps=num_inference_steps + 1)
            right = torch.floor(edges[1:]).long().clamp(max=T - 1)
            timesteps = torch.flip(right, dims=[0])
        else:  # "linspace": plain linspace endpoints included
            timesteps = (
                torch.linspace(T - 1, 0, steps=num_inference_steps).round().long()
            )
        self.register_buffer("timesteps", timesteps, persistent=False)

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        """
        One ancestral step: given x_t and model_output, produce x_{t-1}.
        model_output meaning depends on `prediction_type`:
          - "epsilon":  predicted noise ε
          - "v_prediction": v = α_t ε - √(1-ᾱ_t) x_0  (same as Stable Diffusion v-pred)
          - "sample": predicted x_0
        """
        t = self._sanitize_timestep(timestep, allow_T=False)  # DDPM uses [0..T-1]
        # gather scalars per-batch
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        # predict x0 and eps consistently
        if self.prediction_type == "epsilon":
            eps = model_output
            x0 = (sample - sqrt_one_minus_alphas_cumprod_t * eps) / (
                alphas_cumprod_t.sqrt() + 1e-12
            )
        elif self.prediction_type == "sample":
            x0 = model_output
            eps = (sample - alphas_cumprod_t.sqrt() * x0) / (
                sqrt_one_minus_alphas_cumprod_t + 1e-12
            )
        elif self.prediction_type == "v_prediction":
            # v = ᾱ_t^{1/2} ε - (1-ᾱ_t)^{1/2} x0  ->  recover x0, eps
            v = model_output
            x0 = alphas_cumprod_t.sqrt() * sample - sqrt_one_minus_alphas_cumprod_t * v
            x0 = x0 / (alphas_cumprod_t + 1e-12).sqrt()
            eps = (sample - alphas_cumprod_t.sqrt() * x0) / (
                sqrt_one_minus_alphas_cumprod_t + 1e-12
            )
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x0
            + self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * sample
        )

        # variance
        if self.variance_type == "fixed_small":
            var = self.posterior_variance[t].view(-1, 1, 1, 1)
            log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        elif self.variance_type == "fixed_small_log":
            var = None
            log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        elif self.variance_type == "fixed_large":
            # use beta_t as variance (Ho et al. "learned variance" ablation's large variant)
            var = self.betas[t].view(-1, 1, 1, 1)
            log_var = torch.log(var.clamp_min(1e-20))
        else:
            raise ValueError(f"Unknown variance_type: {self.variance_type}")

        # sample x_{t-1}; for t == 0, return mean (no noise)
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
        noise = torch.randn_like(sample, generator=generator)
        if var is None:
            std = torch.exp(0.5 * log_var)
        else:
            std = var.clamp_min(1e-20).sqrt()
        prev_sample = posterior_mean + nonzero_mask * std * noise
        return prev_sample

    @torch.no_grad()
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N( sqrt(ᾱ_t) x_0, (1-ᾱ_t) I )
        If extra_one_step=True and t == T, we return pure noise.
        """
        t = self._sanitize_timestep(timestep, allow_T=self.extra_one_step)
        # pure-noise endpoint (only if allowed)
        is_T = t == self.num_train_timesteps
        t_clamped = t.clamp_max(self.num_train_timesteps - 1)

        mean = self.sqrt_alphas_cumprod[t_clamped].view(-1, 1, 1, 1) * original_samples
        std = self.sqrt_one_minus_alphas_cumprod[t_clamped].view(-1, 1, 1, 1)
        xt = mean + std * noise
        if is_T.any():
            xt[is_T] = noise[is_T]  # x_T := ε
        return xt.type_as(noise)

    def training_target(
        self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervision target for denoiser under chosen prediction_type.
        - "epsilon": target = ε
        - "sample":  target = x0
        - "v_prediction": target = v = ᾱ_t^{1/2} ε - (1-ᾱ_t)^{1/2} x0
        """
        t = self._sanitize_timestep(timestep, allow_T=False)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "sample":
            # x0 = (x_t - √(1-ᾱ_t) ε) / √(ᾱ_t)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x0 = (sample - sqrt_one_minus * noise) / (alphas_cumprod_t.sqrt() + 1e-12)
            return x0
        elif self.prediction_type == "v_prediction":
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            v = alphas_cumprod_t.sqrt() * noise - sqrt_one_minus * (
                (sample - sqrt_one_minus * noise) / (alphas_cumprod_t.sqrt() + 1e-12)
            )
            return v
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Optional loss weight. If snr_gamma is provided, use Min-SNR weighting:
            w = min( snr, snr_gamma ) / snr
        where snr = ᾱ_t / (1-ᾱ_t)
        Otherwise return ones.
        """
        t = self._sanitize_timestep(timestep, allow_T=False)
        if self.snr_gamma is None:
            return torch.ones_like(t, dtype=torch.float32)

        abar = self.alphas_cumprod[t]
        snr = abar / (1.0 - abar + 1e-12)
        w = torch.minimum(
            snr, torch.tensor(self.snr_gamma, device=snr.device, dtype=snr.dtype)
        ) / (snr + 1e-12)
        return w.to(torch.float32)

    # ---------- helpers ----------

    def _sanitize_timestep(self, t: torch.Tensor, allow_T: bool) -> torch.Tensor:
        """
        Ensure t is long, on the right device, and clamped to valid range.
        If allow_T=True, permit t==T as a pure-noise endpoint; otherwise clamp to [0..T-1].
        """
        T = self.num_train_timesteps
        t = t.to(device=self.alphas.device, dtype=torch.long)
        if allow_T:
            return t.clamp_(0, T)
        else:
            return t.clamp_(0, T - 1)

    @staticmethod
    def _make_betas(T: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            # classic Ho et al. linear from 1e-4 to 0.02
            return torch.linspace(1e-4, 2e-2, T, dtype=torch.float32)
        elif schedule == "scaled_linear":
            # scaled linear as used by some implementations
            return torch.linspace(1e-4, 2e-2, T, dtype=torch.float32) ** 0.5
        elif schedule == "squaredcos_cap_v2":
            # Nichol & Dhariwal cosine schedule (approx)
            def _alpha_bar(t):
                s = 0.008
                return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2

            ts = torch.linspace(0, 1, T + 1, dtype=torch.float32)
            abar = _alpha_bar(ts)
            betas = 1 - (abar[1:] / abar[:-1]).clamp(min=1e-8)
            return betas.clamp(1e-8, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {schedule}")

    @staticmethod
    def _rescale_zero_snr(betas: torch.Tensor) -> torch.Tensor:
        """
        Rescale betas so that final ᾱ_T is ~0 (SNR≈0). Mirrors diffusers' trick.
        """
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        # Map abar linearly to reach ~0 at the end
        abar_new = (abar - abar[-1]) / (abar[0] - abar[-1] + 1e-12)
        alphas_new = abar_new.clone()
        alphas_new[1:] = abar_new[1:] / abar_new[:-1].clamp_min(1e-12)
        betas_new = 1.0 - alphas_new
        return betas_new.clamp(1e-8, 0.999)
