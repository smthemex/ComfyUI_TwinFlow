import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from typing import List, Callable, Union, Dict
from collections import OrderedDict, defaultdict


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """
    Step the EMA model parameters towards the current model parameters.
    
    Args:
        ema_model (nn.Module): The exponential moving average model (Teacher).
        model (nn.Module): The current training model (Student).
        decay (float): The decay rate for the moving average.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            # param_ema = decay * param_ema + (1 - decay) * param_curr
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)
            

class RCGM(torch.nn.Module):
    """
    Recursive Consistent Generation Model (RCGM).

    This class implements the backbone for 'Any-step Generation via N-th Order 
    Recursive Consistent Velocity Field Estimation'. It serves as the foundation 
    for consistency training by estimating higher-order trajectories.

    References:
        - RCGM: https://github.com/LINs-lab/RCGM/blob/main/assets/paper.pdf
        - UCGM (Sampler): https://arxiv.org/abs/2505.07447 (Unified Continuous Generative Models)
    """
    def __init__(
        self,
        ema_decay_rate: float = 0.99, # Recomended: >=0.99 for estimate_order >=2
        estimate_order: int = 2,
        enhanced_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.emd = ema_decay_rate
        self.eso = estimate_order # N-th order estimation (RCGM paper)
        
        assert self.eso >= 1, "Only support estimate_order >= 1"

        self.cmd = 0
        self.mod = None # EMA Model container
        self.enr = enhanced_ratio # CFG Guidance ratio

    # ------------------------------------------------------------------
    # Flow Matching Schedule / OT-Flow Coefficients
    # Interpolation: x_t = alpha(t) * z + gamma(t) * x
    # ------------------------------------------------------------------
    def alpha_in(self, t): return t          # Coefficient for noise z
    def gamma_in(self, t): return 1 - t      # Coefficient for data x
    def alpha_to(self, t): return 1          # d(alpha)/dt
    def gamma_to(self, t): return -1         # d(gamma)/dt

    def l2_loss(self, pred, target):
        """Standard L2 (MSE) Loss flattened over spatial dimensions."""
        loss = (pred.float() - target.float()) ** 2
        return loss.flatten(1).mean(dim=1).to(pred.dtype)

    def loss_func(self, pred, target):
        return self.l2_loss(pred, target)

    @torch.no_grad()
    def get_refer_predc(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor],
    ):
        """
        Get reference predictions with and without conditions (Classifier-Free Guidance).
        Restores RNG state to ensure noise consistency between forward passes.
        """
        torch.cuda.set_rng_state(rng_state)
        # Unconditional forward (using empty condition 'e')
        refer_x, refer_z, refer_v, _ = self.forward(model, x_t, t, tt, **dict(c=e))
        
        torch.cuda.set_rng_state(rng_state)
        # Conditional forward (using condition 'c')
        predc_x, predc_z, predc_v, _ = self.forward(model, x_t, t, tt, **dict(c=c))
        
        return refer_x, refer_z, refer_v, predc_x, predc_z, predc_v

    @torch.no_grad()
    def enhance_target(
        self,
        target: torch.Tensor,
        ratio: float,
        pred_w_c: torch.Tensor,
        pred_wo_c: torch.Tensor,
    ):
        """
        Enhance the training target using Classifier-Free Guidance (CFG).
        Target' = Target + w * (Prediction_cond - Prediction_uncond)
        """
        target = target + ratio * (pred_w_c - pred_wo_c)
        return target

    @torch.no_grad()
    def prepare_inputs(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: List[torch.Tensor],
    ):
        """
        Prepare inputs for Flow Matching training.
        Constructs x_t (noisy data) and target vector field.
        """
        size = [x.size(0)] + [1] * (len(x.shape) - 1)
        t = torch.rand(size=size).to(x)
        c = [torch.zeros_like(t)] if c is None else c
        
        # Aux time variable tt < t for consistency estimation
        tt = t - torch.rand_like(t) * t

        # Construct Flow Matching Targets
        z = torch.randn_like(x)
        # x_t = t * z + (1-t) * x
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        # v_t = z - x (Target velocity)
        target = z * self.alpha_to(t) + x * self.gamma_to(t)

        return x_t, z, x, t, tt, c, target

    @torch.no_grad()
    def multi_fwd(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        N: int,
    ):
        """
        Used to calculate the recursive consistency target.
        """
        pred = 0
        ts = [t * (1 - i / (N)) + tt * (i / (N)) for i in range(N + 1)]
        
        # Euler integration loop
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            torch.cuda.set_rng_state(rng_state)
            hx, hz, F_c, _ = self.forward(model, x_t, t_c, t_n, **dict(c=c))
            x_t = self.alpha_in(t_n) * hz + self.gamma_in(t_n) * hx
            pred = pred + F_c * (t_c - t_n)
            
        return hx, hz, pred

    @torch.no_grad()
    def get_rcgm_target(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        F_th_t: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        N: int,
    ):
        """
        Calculates the RCGM consistency target using N-th order estimation.
        
        Ref: 'Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation'
        Uses a small temporal perturbation (Delta t = 0.01) to enforce local consistency.
        """
        # Delta t = 0.01 as mentioned in RCGM paper
        t_m = (t - 0.01).clamp_min(tt)
        x_t = x_t - target * 0.01 # First order step
        
        # N-step integration from t_m to tt
        _, _, Ft_tar = self.multi_fwd(rng_state, model, x_t, t_m, tt, c, N)
        
        # Weighting for boundary conditions near t=tt
        mask = t < (tt + 0.01)
        cof_l = torch.where(mask, torch.ones_like(t), 100 * (t - tt))
        cof_r = torch.where(mask, 1 / (t - tt), torch.ones_like(t) * 100)
        
        # Reconstruct velocity field target from integral
        Ft_tar = (F_th_t * cof_l - Ft_tar * cof_r) - target
        Ft_tar = F_th_t.data - (Ft_tar).clamp(min=-1.0, max=1.0)
        return Ft_tar

    @torch.no_grad()
    def update_ema(
        self,
        model: nn.Module,
    ):
        """Updates the EMA (Teacher) model."""
        if self.emd > 0.0 and self.emd < 1.0:
            self.mod = self.mod or deepcopy(model).requires_grad_(False).train()
            update_ema(self.mod, model, decay=self.cmd)
            # Warmup logic for EMA decay
            self.cmd += (1 - self.cmd) * (self.emd - self.cmd) * 0.5
        elif self.emd == 0.0:
            self.mod = model
        elif self.emd == 1.0:
            self.mod = self.mod or deepcopy(model).requires_grad_(False).train()

    def training_step(
        self,
        model: Union[nn.Module, Callable],
        x: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor] = None,
    ):
        """
        Base RCGM training step (Any-step generation).
        """
        x_t, z, t, tt, c, target = self.prepare_inputs(model, x, c)
        loss, rng_state = 0, torch.cuda.get_rng_state()
        x_wc_t, z_wc_t, F_th_t, den_t = self.forward(model, x_t, t, tt, **dict(c=c))

        self.update_ema(model)

        # Enhance target (Guidance)
        if (self.enr > 0.0):
            _, _, refer_v, _, _, predc_v = self.get_refer_predc(
                rng_state, self.mod, x_t, t, t, c, e
            )
            target = self.enhance_target(target, self.enr, predc_v, refer_v)
        
        # Calculate RCGM Consistency Target
        target = self.get_rcgm_target(
            rng_state, self.mod, F_th_t, target, x_t, t, tt, c, self.eso,
        )

        loss = self.loss_func(F_th_t, target).mean()
        return loss

    def forward(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
        **model_kwargs,
    ):
        """
        Forward pass.
        Returns:
            x_hat: Reconstructed data (x0)
            z_hat: Reconstructed noise (x1)
            F_t: Predicted velocity field v_t
            dent: Denominator (normalization term)
        """
        dent = -1 # dent = alpha(t)*gamma'(t) - gamma(t)*alpha'(t) for linear flow
        F_t = model(
            x_t, 
            t=torch.ones(x_t.size(0), device=x_t.device) * (t).flatten(), 
            tt=torch.ones(x_t.size(0), device=x_t.device) * tt.flatten(), 
            **model_kwargs)
        t = torch.abs(t)
        
        # Invert flow to recover x and z
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def kumaraswamy_transform(self, t, a, b, c):
        """
        Kumaraswamy distribution transform for time step discretization.
        Used to concentrate sampling steps in regions of high curvature.
        """
        return (1 - (1 - t**a) ** b) ** c

    '''
    Sampler: UCGM (Unified Continuous Generative Models)
    
    This sampler is highly compatible with RCGM and TwinFlow frameworks.
    Since RCGM and TwinFlow are designed to train "Any-step" models (capable of 
    functioning effectively as both one-step/few-step generators and multi-step 
    generators), this unified sampler enables seamless switching between these 
    regimes without structural changes.
    
    Reference: https://arxiv.org/abs/2505.07447
    Adapted from: https://github.com/LINs-lab/UCGM/blob/main/methodes/unigen.py
    '''
    @torch.no_grad()
    def sampling_loop(
        self,
        inital_noise_z: torch.FloatTensor,
        sampling_model: Union[nn.Module, Callable],
        sampling_steps: int = 1,
        stochast_ratio: float = 0.0,
        extrapol_ratio: float = 0.0,
        sampling_order: int = 1,
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
        rfba_gap_steps: list = [0.0, 0.0],
        **model_kwargs,
    ):
        """
        Executes the UCGM sampling loop.
        
        Args:
            inital_noise_z: Initial Gaussian noise.
            sampling_model: The trained Any-step model (RCGM/TwinFlow).
            sampling_steps: 1 for One-step generation, >1 for Multi-step refinement.
            ...
        """
        input_dtype = inital_noise_z.dtype
        assert sampling_order in [1, 2]
        num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps

        # Time step discretization (with Kumaraswamy transform)
        num_steps = num_steps + 1 if (rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
        t_steps = torch.linspace(
            rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64
        ).to(inital_noise_z)
        t_steps = t_steps[:-1] if (rfba_gap_steps[1] - 0.0) == 0.0 else t_steps
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)
        t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])

        # Buffer for extrapolation
        x_hats, z_hats, buffer_freq = [], [], 1

        # Main sampling loop
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z.cpu()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # 1. First order prediction (Euler)
            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(input_dtype),
                t_cur.to(input_dtype),
                torch.zeros_like(t_cur), # tt=0 for few-step (one-step)
                # t_next.to(input_dtype), # any-step mode
                # t_cur.to(input_dtype),  # multi-step mode
                **model_kwargs,
            )
            samples.append(x_hat.cpu())
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

            # Extrapolation logic (optional)
            if buffer_freq > 0 and extrapol_ratio > 0:
                z_hats.append(z_hat)
                x_hats.append(x_hat)
                if i > buffer_freq:
                    z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                    x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                    z_hats.pop(0), x_hats.pop(0)

            # Stochastic injection (SDE-like behavior)
            if stochast_ratio == "SDE":
                stochast_ratio = (
                    torch.sqrt((t_next - t_cur).abs())
                    * torch.sqrt(2 * self.alpha_in(t_cur))
                    / self.alpha_in(t_next)
                )
                stochast_ratio = torch.clamp(stochast_ratio ** (1 / 0.50), min=0, max=1)
                noi = torch.randn(x_cur.size()).to(x_cur)
            else:
                noi = torch.randn(x_cur.size()).to(x_cur) if stochast_ratio > 0 else 0.0
            
            x_next = self.gamma_in(t_next) * x_hat + self.alpha_in(t_next) * (
                z_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
            )

            # 2. Second order correction (Heun)
            if sampling_order == 2 and i < num_steps - 1:
                x_pri, z_pri, _, _ = self.forward(
                    sampling_model,
                    x_next.to(input_dtype),
                    t_next.to(input_dtype),
                    **model_kwargs,
                )
                x_pri, z_pri = x_pri.to(torch.float64), z_pri.to(torch.float64)

                x_next = x_cur * self.gamma_in(t_next) / self.gamma_in(t_cur) + (
                    self.alpha_in(t_next)
                    - self.gamma_in(t_next)
                    * self.alpha_in(t_cur)
                    / self.gamma_in(t_cur)
                ) * (0.5 * z_hat + 0.5 * z_pri)

            x_cur = x_next

        return torch.stack(samples, dim=0).to(input_dtype)



class TwinFlow(RCGM):
    """
    TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows.
    
    This class implements the TwinFlow framework, which augments RCGM with 
    Self-Adversarial Flows (L_adv) and Flow Distillation (L_rectify) to achieve 
    high-quality one-step generation.

    Reference: https://arxiv.org/abs/2512.05150
    """
    def __init__(
        self,
        using_twinflow: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.utf = using_twinflow

    @torch.no_grad()
    def dist_match(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: List[torch.Tensor],
    ):
        """
        Distribution Matching (L_rectify helper).
        
        Matches the distribution of the generated 'fake' flow (reverse time) 
        against the 'real' flow (forward time) to align velocity fields.
        """
        z = torch.randn_like(x)
        size = [x.size(0)] + [1] * (len(x.shape) - 1)
        t = torch.rand(size=size).to(x)
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        
        # Forward passes for fake (negative time) and real (positive time) trajectories
        fake_s, _, fake_v, _ = self.forward(model, x_t, -t, -t, **dict(c=c))
        real_s, _, real_v, _ = self.forward(model, x_t,  t,  t, **dict(c=c))
        
        F_grad = (fake_v - real_v)
        x_grad = (fake_s - real_s)
        return x_grad, F_grad

    def training_step(
        self,
        model: Union[nn.Module, Callable],
        x: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor] = None,
    ):
        """
        TwinFlow Training Step.
        Combines RCGM loss with TwinFlow-specific losses (L_adv, L_rectify).
        """
        x_t, z, x, t, tt, c, target = self.prepare_inputs(model, x, c)
        loss, rng_state = 0, torch.cuda.get_rng_state()
        _, _, F_th_t, _ = self.forward(model, x_t, t, tt, **dict(c=c))

        self.update_ema(model)

        # Enhance Target (CFG Guidance)
        if (self.enr > 0.0):
            _, _, refer_v, _, _, predc_v = self.get_refer_predc(
                rng_state, self.mod, x_t, t, t, c, e
            )
            target = self.enhance_target(target, self.enr, predc_v, refer_v)
        
        # -----------------------------------------------------------
        # 1. RCGM Base Loss (L_base)
        # -----------------------------------------------------------
        rcgm_target = self.get_rcgm_target(
            rng_state, self.mod, F_th_t, target.clone(), x_t, t, tt, c, self.eso,
        )

        loss = self.loss_func(F_th_t, rcgm_target).mean()

        # -----------------------------------------------------------
        # TwinFlow Specific Losses
        # -----------------------------------------------------------
        if self.utf is True:
            # [Optional] Real Velocity Loss
            # Ensures real velocity is learned well (redundant if RCGM loss is perfect)
            _, _, F_pred, _ = self.forward(model, x_t, t, t, **dict(c=c))
            loss += self.loss_func(F_pred, target).mean()

            # One-step generation forward pass (z -> x_fake)
            # z = torch.randn_like(z); t = rand... (Re-sampling noise/time if needed)
            x_fake, _, F_fake, _ = self.forward(
                model, z, torch.ones_like(t), torch.zeros_like(t), **dict(c=c)
            )

            # 2. Fake Velocity Loss / Self-Adversarial (L_adv)
            # Training fake velocity at t in [-1, 0] to match target flow
            x_t_fake = z * self.alpha_in(t) + x_fake.detach() * self.gamma_in(t)
            target_fake = z * self.alpha_to(t) + x_fake.detach() * self.gamma_to(t)
            _, _, F_th_t_fake, _ = self.forward(model, x_t_fake, -t, -t, **dict(c=c))
            loss += self.loss_func(F_th_t_fake, target_fake).mean()

            # 3. Distribution Matching / Rectification Loss (L_rectify)
            # Aligns the generated flow with the 'correct' gradient direction
            _, F_grad = self.dist_match(model, x_fake, c)
            loss += self.loss_func(F_fake, (F_fake - F_grad).detach()).mean()

            # [Optional] Consistency mapping (t=1 to tt=0)
            rcgm_target = self.get_rcgm_target(
                rng_state, self.mod, F_fake, target.clone(), z, torch.ones_like(t), torch.zeros_like(t), c, self.eso,
            )
            loss += self.loss_func(F_fake, rcgm_target).mean()

        '''
        NOTE ON EFFICIENCY:
        The code above demonstrates the complete TwinFlow logic with all loss terms 
        (RCGM L_base, L_adv, L_rectify) calculated in a single step.
        
        In practice, calculating multiple forward passes with gradients is computationally expensive.
        For large-scale training, it is recommended to:
        1. Split the batch into sub-batches.
        2. Apply different loss terms to different sub-batches (e.g. 50% Base, 25% Adv, 25% Rectify).
        3. Optimize redundant calculations.
        '''
        return loss