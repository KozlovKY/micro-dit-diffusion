"""
Consistency Distillation Loss for micro-diffusion.

This module implements consistency distillation training for DiT models,
where a student model learns to predict teacher model outputs at different
timesteps for faster inference.

Key features:
- Teacher model is deep-copied and frozen (avoid aliasing with student)
- Correct dtype handling for VAE encoding (uses DATA_TYPES mapping)
- Uses torch.no_grad() when encoding with frozen VAE
- Robust `.to(device)` handling for ConsistencyDistillationLoss and contained teacher
- Ensures teacher remains in eval() mode and frozen during training
- Supports both single-step and multi-boundary consistency distillation
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler


class ConsistencyDistillationLoss(nn.Module):
    """
    Consistency Distillation Loss for DiT models.
    
    This loss function implements consistency distillation training where
    a student model learns to predict teacher model outputs at different
    timesteps for faster inference.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        target_model: Optional[nn.Module] = None,
        noise_scheduler: DDPMScheduler = None,
        loss_type: str = "l2",
        huber_c: float = 0.001,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.9,
        num_scales: int = 18,
        teacher_solver: str = "heun",
        k_max: int = 3,
        include_terminal: bool = True,
        terminal_weight: float = 2.0,
    ):
        super().__init__()
        
        self.teacher_model = copy.deepcopy(teacher_model)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()
        if target_model is not None:
            self.target_model = target_model
            for p in self.target_model.parameters():
                p.requires_grad = False
            self.target_model.eval()
        else:
            self.target_model = None

        self.noise_scheduler = noise_scheduler
        self.loss_type = loss_type
        self.huber_c = huber_c
        # Karras config
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sigma_data = float(sigma_data)
        self.num_scales = int(num_scales)
        self.teacher_solver = teacher_solver
        self.k_max = int(k_max)
        self.include_terminal = bool(include_terminal)
        self.terminal_weight = float(terminal_weight)
        alphas_cumprod = getattr(self.noise_scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            alphas_cumprod = torch.tensor(self.noise_scheduler.config.alphas_cumprod, dtype=torch.float32)
        else:
            alphas_cumprod = torch.as_tensor(alphas_cumprod, dtype=torch.float32)
        
        self.register_buffer("alpha_schedule", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sigma_schedule", torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        sigmas = self.get_sigmas_karras(self.num_scales, self.sigma_min, self.sigma_max, self.rho)
        self.register_buffer("karras_sigmas", sigmas, persistent=False)
        
        self._device = torch.device("cpu")
    
    def to(self, device):
        """Move loss function and teacher to device."""
        device = torch.device(device)
        self.teacher_model = self.teacher_model.to(device)
        self._device = device
        super().to(device)
        return self

    @staticmethod
    def get_sigmas_karras(n: int, sigma_min: float, sigma_max: float, rho: float, device: Optional[torch.device] = None) -> torch.Tensor:
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # Don't add zero at the end - use small value instead
        return torch.cat([sigmas, torch.full_like(sigmas[:1], 1e-8)])

    def karras_scalings(self, sigma: torch.Tensor, x_like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extra_dims = (1,) * (x_like.ndim - 1)
        sigma = sigma.to(x_like.dtype).reshape(-1, *extra_dims)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        sigma_sq = sigma ** 2
        sigma_data_sq = self.sigma_data ** 2
        
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq + eps)
        denom = (sigma_sq + sigma_data_sq + eps).sqrt()
        c_out = sigma * self.sigma_data / denom
        c_in = 1.0 / denom
        return c_skip, c_out, c_in

    def karras_scalings_boundary(self, sigma: torch.Tensor, x_like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extra_dims = (1,) * (x_like.ndim - 1)
        sigma = sigma.to(x_like.dtype).to(x_like.device).reshape(-1, *extra_dims)
        
        eps = 1e-8
        sigma_diff = sigma - self.sigma_min
        sigma_sq = sigma ** 2
        sigma_data_sq = self.sigma_data ** 2
        
        c_skip = sigma_data_sq / (sigma_diff ** 2 + sigma_data_sq + eps)
        denom_sqrt = (sigma_sq + sigma_data_sq + eps).sqrt()
        c_out = sigma_diff * self.sigma_data / denom_sqrt
        c_in = 1.0 / denom_sqrt
        return c_skip, c_out, c_in

    def call_model_raw(self, model: nn.Module, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, use_boundary: bool = True) -> torch.Tensor:
        if use_boundary:
            c_skip, c_out, c_in = self.karras_scalings_boundary(sigma, x)
        else:
            c_skip, c_out, c_in = self.karras_scalings(sigma, x)


        # Clamp sigma to prevent log(0) = -inf
        sigma_clamped = torch.clamp(sigma, min=1e-8)
        c_noise = sigma_clamped.log() / 4
        
        model_input = (c_in * x).to(x.dtype)
        
        if torch.isinf(model_input).any() or (model_input.abs() > 1e6).any():
            model_input = torch.clamp(model_input, min=-1e6, max=1e6)
        
        out = model(model_input, c_noise.flatten(), y, mask_ratio=0.0)
        if isinstance(out, dict):
            out = out.get("sample", out)
        return out

    def denoised_from_raw(self, x: torch.Tensor, sigma: torch.Tensor, raw: torch.Tensor, boundary: bool = False) -> torch.Tensor:
        if boundary:
            c_skip, c_out, _ = self.karras_scalings_boundary(sigma, x)
        else:
            c_skip, c_out, _ = self.karras_scalings(sigma, x)
        result = c_skip.to(raw.device) * x.to(raw.device) + c_out.to(raw.device) * raw
        return result

    def teacher_denoise(self, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw = self.call_model_raw(self.teacher_model, x, sigma, y, use_boundary=False)
        denoised = self.denoised_from_raw(x, sigma, raw, boundary=False)
        return denoised



    def forward(
            self, 
            student_model: nn.Module, 
            latents: torch.Tensor, 
            text_embeddings: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass for Multi-Boundary Consistency Distillation loss.
            """
            bsz = latents.shape[0]
            device = latents.device
            dims = latents.ndim
            
            max_idx = self.num_scales - 1 - (self.k_max - 1)
            max_idx = max(max_idx, 1)
            indices = torch.randint(0, max_idx, (bsz,), device=device)
            
            sigma_t = self.karras_sigmas[indices]
            
            # nans cause of log(0) = -inf
            if (sigma_t > 100.0).any() or (sigma_t < 1e-6).any():
                sigma_t = torch.clamp(sigma_t, min=1e-6, max=100.0)

            noise = torch.randn_like(latents)
            x_t = latents + noise * sigma_t.reshape(-1, *([1] * (dims - 1)))
            y = text_embeddings
            
            dropout_state = torch.get_rng_state()
            if torch.cuda.is_available():
                cuda_dropout_state = torch.cuda.get_rng_state()

            student_raw = self.call_model_raw(student_model, x_t, sigma_t, y, use_boundary=True)
            distiller_t = self.denoised_from_raw(x_t, sigma_t, student_raw, boundary=True)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Multi-boundary loop (k=1 to k_max)
            
            for k in range(1, self.k_max + 1):
                x_tk = x_t.clone()
                
                for j in range(k):
                    sigma_prev = self.karras_sigmas[indices + j]
                    sigma_next_j = self.karras_sigmas[indices + j + 1]
                    
                    if self.teacher_solver == "heun":
                        denoised_prev = self.teacher_denoise(x_tk, sigma_prev, y)
                        sigma_prev_clamped = torch.clamp(sigma_prev, min=1e-8)
                        d = (x_tk - denoised_prev) / sigma_prev_clamped.reshape(-1, *([1] * (dims - 1)))
                        x_euler = x_tk + d * (sigma_next_j - sigma_prev).reshape(-1, *([1] * (dims - 1)))
                        denoised_next = self.teacher_denoise(x_euler, sigma_next_j, y)
                        sigma_next_j_clamped = torch.clamp(sigma_next_j, min=1e-8)
                        d_next = (x_euler - denoised_next) / sigma_next_j_clamped.reshape(-1, *([1] * (dims - 1)))
                        d_prime = 0.5 * (d + d_next)
                        x_tk = x_tk + d_prime * (sigma_next_j - sigma_prev).reshape(-1, *([1] * (dims - 1)))
                    else: # Euler
                        denoised_prev = self.teacher_denoise(x_tk, sigma_prev, y)
                        sigma_prev_clamped = torch.clamp(sigma_prev, min=1e-8)
                        d = (x_tk - denoised_prev) / sigma_prev_clamped.reshape(-1, *([1] * (dims - 1)))
                        x_tk = x_tk + d * (sigma_next_j - sigma_prev).reshape(-1, *([1] * (dims - 1)))

                sigma_tk = self.karras_sigmas[indices + k]

                torch.set_rng_state(dropout_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(cuda_dropout_state)

                if self.target_model is not None:
                    target_raw_k = self.call_model_raw(self.target_model, x_tk, sigma_tk, y, use_boundary=True)
                    distiller_target_k = self.denoised_from_raw(x_tk, sigma_tk, target_raw_k, boundary=True)
                else:
                    target_raw_k = self.call_model_raw(self.teacher_model, x_tk, sigma_tk, y, use_boundary=False)
                    distiller_target_k = self.denoised_from_raw(x_tk, sigma_tk, target_raw_k, boundary=False)
                
                if self.loss_type == "l2":
                    snr = sigma_t ** -2
                    snr = torch.clamp(snr, min=1e-8, max=1e8)
                    weight = snr + (1.0 / (self.sigma_data ** 2))
                    weight = weight.reshape(-1, *([1] * (distiller_t.ndim - 1)))
                    weight = torch.clamp(weight, min=1e-8, max=1e8)
                    loss_k = (weight * (distiller_t.float() - distiller_target_k.float()) ** 2).mean()
                elif self.loss_type == "huber":
                    diff = distiller_t.float() - distiller_target_k.float()
                    loss_k = torch.mean(
                        torch.sqrt(diff ** 2 + self.huber_c ** 2) - self.huber_c
                    )
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")
                
            
                total_loss = total_loss + loss_k / k

            # (x_t -> x_0)
            if self.include_terminal:
                x_t_final = x_t.clone()
                remaining = (self.num_scales - 1) - indices
                
                max_remaining = remaining.max().item()
                
                for j in range(max_remaining):
                    step_mask = (j < remaining).float().reshape(-1, *([1] * (dims - 1)))
                    
                    sigma_prev = self.karras_sigmas[(indices + j).clamp_max(self.num_scales - 1)]
                    sigma_next_j = self.karras_sigmas[(indices + j + 1).clamp_max(self.num_scales)]
                    
                    if self.teacher_solver == "heun":
                        denoised_prev = self.teacher_denoise(x_t_final, sigma_prev, y)
                        sigma_prev_clamped = torch.clamp(sigma_prev, min=1e-8)
                        d = (x_t_final - denoised_prev) / sigma_prev_clamped.reshape(-1, *([1] * (dims - 1)))
                        x_euler = x_t_final + step_mask * (d * (sigma_next_j - sigma_prev).reshape(-1, *([1] * (dims - 1))))
                        denoised_next = self.teacher_denoise(x_euler, sigma_next_j, y)
                        sigma_next_j_clamped = torch.clamp(sigma_next_j, min=1e-8)
                        d_next = (x_euler - denoised_next) / sigma_next_j_clamped.reshape(-1, *([1] * (dims - 1)))
                        d_prime = 0.5 * (d + d_next)
                        x_t_final = x_t_final + step_mask * (d_prime * (sigma_next_j - sigma_prev).reshape(-1, *([1] * (dims - 1))))
                    else:
                        denoised_prev = self.teacher_denoise(x_t_final, sigma_prev, y)
                        sigma_prev_clamped = torch.clamp(sigma_prev, min=1e-8)
                        d = (x_t_final - denoised_prev) / sigma_prev_clamped.reshape(-1, *([1] * (dims - 1)))
                        x_t_final = x_t_final + step_mask * (d * (sigma_next_j - sigma_prev).reshape(-1, *([1] * (dims - 1))))

                sigma_final = torch.clamp(self.karras_sigmas[-1], min=1e-6) # sigma ~ 0, but clampe

                torch.set_rng_state(dropout_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(cuda_dropout_state)

                if self.target_model is not None:
                    target_raw_f = self.call_model_raw(self.target_model, x_t_final, sigma_final, y, use_boundary=True)
                    distiller_target_f = self.denoised_from_raw(x_t_final, sigma_final, target_raw_f, boundary=True)
                else:
                    target_raw_f = self.call_model_raw(self.teacher_model, x_t_final, sigma_final, y, use_boundary=False)
                    distiller_target_f = self.denoised_from_raw(x_t_final, sigma_final, target_raw_f, boundary=False)
            
                if self.loss_type == "l2":
                    snr = sigma_t ** -2
                    snr = torch.clamp(snr, min=1e-8, max=1e8)
                    weight = snr + (1.0 / (self.sigma_data ** 2))
                    weight = weight.reshape(-1, *([1] * (distiller_t.ndim - 1)))
                    weight = torch.clamp(weight, min=1e-8, max=1e8)
                    loss_f = (weight * (distiller_t.float() - distiller_target_f.float()) ** 2).mean()
                elif self.loss_type == "huber":
                    diff = distiller_t.float() - distiller_target_f.float()
                    loss_f = torch.mean(
                        torch.sqrt(diff ** 2 + self.huber_c ** 2) - self.huber_c
                    )
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")

                total_loss = total_loss + self.terminal_weight * loss_f
            return total_loss


    # def forward(
    #     self, 
    #     student_model: nn.Module, 
    #     latents: torch.Tensor, 
    #     text_embeddings: torch.Tensor,
    # ) -> torch.Tensor:
    #     bsz = latents.shape[0]
    #     device = latents.device
    #     dims = latents.ndim
        
    #     indices = torch.randint(0, self.num_scales - 1, (bsz,), device=device)
    #     sigma_t = self.karras_sigmas[indices]
    #     sigma_next = self.karras_sigmas[indices + 1]

    #     noise = torch.randn_like(latents)
    #     x_t = latents + noise * sigma_t.reshape(-1, *([1] * (dims - 1)))

    #     y = text_embeddings

    #     dropout_state = torch.get_rng_state()
    #     if torch.cuda.is_available():
    #         cuda_dropout_state = torch.cuda.get_rng_state()

    #     student_raw = self.call_model_raw(student_model, x_t, sigma_t, y, use_boundary=True)
    #     distiller = self.denoised_from_raw(x_t, sigma_t, student_raw, boundary=True)

    #     with torch.no_grad():
    #         if self.teacher_solver == "heun":
    #             denoised_t = self.teacher_denoise(x_t, sigma_t, y)
    #             d = (x_t - denoised_t) / sigma_t.reshape(-1, *([1] * (dims - 1)))
    #             x_euler = x_t + d * (sigma_next - sigma_t).reshape(-1, *([1] * (dims - 1)))
                
    #             denoised_next = self.teacher_denoise(x_euler, sigma_next, y)
    #             d_next = (x_euler - denoised_next) / sigma_next.reshape(-1, *([1] * (dims - 1)))
                
    #             d_prime = 0.5 * (d + d_next)
    #             x_t2 = x_t + d_prime * (sigma_next - sigma_t).reshape(-1, *([1] * (dims - 1)))
    #         else:
    #             denoised_t = self.teacher_denoise(x_t, sigma_t, y)
    #             d = (x_t - denoised_t) / sigma_t.reshape(-1, *([1] * (dims - 1)))
    #             x_t2 = x_t + d * (sigma_next - sigma_t).reshape(-1, *([1] * (dims - 1)))

    #     torch.set_rng_state(dropout_state)
    #     if torch.cuda.is_available():
    #         torch.cuda.set_rng_state(cuda_dropout_state)

    #     with torch.no_grad():
    #         if self.target_model is not None:
    #             target_raw = self.call_model_raw(self.target_model, x_t2, sigma_next, y, use_boundary=True)
    #             distiller_target = self.denoised_from_raw(x_t2, sigma_next, target_raw, boundary=True)
    #         else:
    #             target_raw = self.call_model_raw(self.teacher_model, x_t2, sigma_next, y, use_boundary=False)
    #             distiller_target = self.denoised_from_raw(x_t2, sigma_next, target_raw, boundary=False)
        
    #     distiller_target = distiller_target.detach()

    #     if self.loss_type == "l2":
    #         snr = sigma_t ** -2
    #         weight = snr + (1.0 / (self.sigma_data ** 2))
    #         weight = weight.reshape(-1, *([1] * (distiller.ndim - 1)))
    #         loss = (weight * (distiller.float() - distiller_target.float()) ** 2).mean()
    #     elif self.loss_type == "huber":
    #         loss = torch.mean(
    #             torch.sqrt((distiller.float() - distiller_target.float()) ** 2 + self.huber_c ** 2) - self.huber_c
    #         )
    #     else:
    #         raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
    #     return loss
