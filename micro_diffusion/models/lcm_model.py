"""
Consistency-distillation-enabled LatentDiffusion model for micro-diffusion.

This module extends the base LatentDiffusion to support consistency
distillation training (student learns to match teacher predictions
at adjacent timesteps for fast inference).

Key guarantees:
- teacher model is deep-copied and frozen (avoid aliasing with student)
- correct dtype handling for VAE encoding (uses DATA_TYPES mapping)
- uses torch.no_grad() when encoding with frozen VAE
- robust `.to(device)` handling for distillation loss and contained teacher
- ensures teacher remains in eval() mode and frozen during training
"""

import copy
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from .model import LatentDiffusion, DATA_TYPES
from .consistency_distillation import ConsistencyDistillationLoss

class LCMLatentDiffusion(LatentDiffusion):
    """
    LatentDiffusion model with Consistency Distillation support.

    This model supports consistency function sampling using EMA target model.
    Only consistency function sampling methods are available.
    """

    def __init__(
        self,
        dit: nn.Module,
        vae,
        text_encoder,
        tokenizer,
        # Consistency distillation parameters
        use_consistency_distillation: bool = False,
        consistency_teacher_model: Optional[nn.Module] = None,  # Для ODE solver
        consistency_target_model: Optional[nn.Module] = None,    # EMA копия student для loss target
        consistency_noise_scheduler: Optional[DDPMScheduler] = None,
        consistency_config: Optional[Dict[str, Any]] = None,
        # Base model parameters
        **base_kwargs
    ):
        super().__init__(
            dit=dit,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            **base_kwargs
        )

        self.use_consistency_distillation = use_consistency_distillation
        self.consistency_config = consistency_config or {}
        self._uncond_cache: dict[tuple[int, str, str], torch.Tensor] = {}

        if self.use_consistency_distillation:
            if consistency_teacher_model is None:
                consistency_teacher_model = copy.deepcopy(self.dit)

            if consistency_target_model is None:
                consistency_target_model = copy.deepcopy(self.dit)

            if consistency_noise_scheduler is None:
                raise ValueError(
                    "consistency_noise_scheduler must be provided when use_consistency_distillation=True"
                )

            self.consistency_loss_fn = ConsistencyDistillationLoss(
                teacher_model=consistency_teacher_model,
                target_model=consistency_target_model,  
                noise_scheduler=consistency_noise_scheduler,
                loss_type=self.consistency_config.get('loss_type', 'l2'),
                huber_c=self.consistency_config.get('huber_c', 0.001),
                sigma_min=self.consistency_config.get('sigma_min', self.edm_config.sigma_min),
                sigma_max=self.consistency_config.get('sigma_max', self.edm_config.sigma_max),
                rho=self.consistency_config.get('rho', self.edm_config.rho),
                sigma_data=self.consistency_config.get('sigma_data', self.edm_config.sigma_data),
                num_scales=self.consistency_config.get('num_scales', self.edm_config.num_steps),
                teacher_solver=self.consistency_config.get('teacher_solver', 'heun'),
            )
            
            # self.consistency_loss_fn = ConsistencyDistillationLossEDM(
            #     teacher_model=consistency_teacher_model,
            #     target_model=consistency_target_model,
            #     edm_config=self.edm_config,
            #     num_ddim_timesteps=self.consistency_config.get('num_ddim_timesteps', 50),
            #     loss_type=self.consistency_config.get('loss_type', 'l2'),
            #     huber_c=self.consistency_config.get('huber_c', 0.001),
            #     guidance_scale=self.consistency_config.get('guidance_scale', 7.5),
            #     train_mask_ratio=self.consistency_config.get('train_mask_ratio', 0.0),
            #     eval_mask_ratio=self.consistency_config.get('eval_mask_ratio', 0.0),
            # )
            
        else:
            self.consistency_loss_fn = None
        

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional LCM or consistency distillation.

        Args:
            batch: Input batch dictionary

        Returns:
            Tuple of (loss, latents, conditioning)
        """
        if self.consistency_loss_fn is not None:
            return self._forward_with_distillation(batch, self.consistency_loss_fn)
        else:
            return super().forward(batch)

    def _forward_with_distillation(
        self, 
        batch: dict, 
        loss_fn: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with distillation loss (LCM or Consistency).

        Args:
            batch: Input batch dictionary
            loss_fn: Distillation loss function (LCM or Consistency)

        Returns:
            Tuple of (loss, latents, conditioning)
        """
        if self.precomputed_latents and self.image_latents_key in batch:
            latents = batch[self.image_latents_key]
        else:
            with torch.no_grad():
                images = batch[self.image_key]
                images = images.to(DATA_TYPES[self.dtype])
                latents = self.vae.encode(images)['latent_dist'].sample().data
                latents *= self.latent_scale

        if self.precomputed_latents and self.text_latents_key in batch:
            conditioning = batch[self.text_latents_key]
        else:
            captions = batch[self.text_key]
            captions = captions.view(-1, captions.shape[-1])
            if 'attention_mask' in batch:
                conditioning = self.text_encoder.encode(
                    captions,
                    attention_mask=batch['attention_mask'].view(-1, captions.shape[-1])
                )[0]
            else:
                conditioning = self.text_encoder.encode(captions)[0]

        if 'drop_caption_mask' in batch.keys():
            conditioning *= batch['drop_caption_mask'].view(
                [-1] + [1] * (len(conditioning.shape) - 1)
            )
        # def _get_uncond_embeddings(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        #     cache_key = (batch_size, str(device), str(dtype))
        #     if cache_key in self._uncond_cache:
        #         uncond = self._uncond_cache[cache_key]
        #         # страховка на случай смены девайса/типа
        #         return uncond.to(device=device, dtype=dtype)

        #     # Токенизируем пустые строки той же длины батча
        #     tokens = self.tokenizer.tokenize([""] * batch_size)
        #     input_ids = tokens['input_ids'].to(device)
        #     attention_mask = tokens['attention_mask'].to(device) if 'attention_mask' in tokens else None

        #     uncond = self.text_encoder.encode(input_ids, attention_mask)[0]
        #     uncond = uncond.to(device=device, dtype=dtype)
        #     self._uncond_cache[cache_key] = uncond.detach()
        #     return uncond

        # uncond_embeddings = _get_uncond_embeddings(
        #     batch_size=conditioning.shape[0],
        #     device=conditioning.device,
        #     dtype=conditioning.dtype,
        # )

        loss = loss_fn(
            student_model=self.dit,
            latents=latents.float(),
            text_embeddings=conditioning.float(),
            #uncond_text_embeddings=uncond_embeddings.float(),
        )

        return (loss, latents, conditioning)

    def to(self, device):
        """Move model and distillation-related components to device."""
        super().to(device)
        
        # Move distillation loss functions (Consistency)
        for loss_fn in [self.consistency_loss_fn]:
            if loss_fn is not None:
                loss_fn.to(device)
                # Move target model separately if it exists
                if hasattr(loss_fn, 'target_model') and loss_fn.target_model is not None:
                    loss_fn.target_model = loss_fn.target_model.to(device)
        return self

    def train(self, mode: bool = True):
        """Set training mode; keep all teachers in eval and frozen."""
        super().train(mode)
        
        for loss_fn in [self.consistency_loss_fn]:
            if loss_fn is not None:
                if hasattr(loss_fn, 'teacher_model') and loss_fn.teacher_model is not None:
                    loss_fn.teacher_model.eval()
                    for p in loss_fn.teacher_model.parameters():
                        p.requires_grad = False
                if hasattr(loss_fn, 'target_model') and loss_fn.target_model is not None:
                    loss_fn.target_model.eval()
                    for p in loss_fn.target_model.parameters():
                        p.requires_grad = False
        return self

    def eval(self):
        """Set evaluation mode and ensure all teachers remain in eval."""
        super().eval()
        
        # Ensure all teachers and targets remain in eval
        for loss_fn in [self.consistency_loss_fn]:
            if loss_fn is not None:
                if hasattr(loss_fn, 'teacher_model') and loss_fn.teacher_model is not None:
                    loss_fn.teacher_model.eval()
                if hasattr(loss_fn, 'target_model') and loss_fn.target_model is not None:
                    loss_fn.target_model.eval()
        return self

    # ===== Consistency-aware inference (boundary scalings, no CFG) =====
    def _karras_sigmas(self, n: int, device: torch.device) -> torch.Tensor:
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.edm_config.sigma_min ** (1.0 / self.edm_config.rho)
        max_inv_rho = self.edm_config.sigma_max ** (1.0 / self.edm_config.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.edm_config.rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    def _karras_scalings_boundary(self, sigma: torch.Tensor, x_like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extra_dims = (1,) * (x_like.ndim - 1)
        sigma = sigma.to(x_like.dtype).to(x_like.device).reshape(-1, *extra_dims)
        c_skip = (self.edm_config.sigma_data ** 2) / ((sigma - self.edm_config.sigma_min) ** 2 + self.edm_config.sigma_data ** 2)
        denom_sqrt = (sigma ** 2 + self.edm_config.sigma_data ** 2).sqrt()
        c_out = (sigma - self.edm_config.sigma_min) * self.edm_config.sigma_data / denom_sqrt
        c_in = 1.0 / denom_sqrt
        return c_skip, c_out, c_in

    def _denoise_boundary(self, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Student uses boundary scalings as in training
        c_skip, c_out, c_in = self._karras_scalings_boundary(sigma, x)
        c_noise = sigma.log() / 4
        out = self.dit((c_in * x).to(x.dtype), c_noise.flatten(), y, mask_ratio=0.0)
        F_x = out["sample"] if isinstance(out, dict) else out
        c_skip = c_skip.to(F_x.device)
        x = x.to(F_x.device)
        c_out = c_out.to(F_x.device)
        return c_skip * x + c_out * F_x

    @torch.no_grad()
    def _consistency_function_sampler_loop(self, x: torch.Tensor, y: torch.Tensor, steps: int) -> torch.Tensor:
        """Consistency Function sampling using transition:

        x_next = C_out + sigma_next * (x_cur - C_out) / sigma_cur

        where C_out is boundary-scaled denoised estimate produced by the EMA distiller.
        """
        # Create Karras sigmas (monotone to 0)
        sigmas = self._karras_sigmas(steps if steps is not None else self.edm_config.num_steps, x.device)
        x_t = x.to(torch.float32) * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1]

            s_in = x_t.new_ones([x_t.shape[0]], dtype=x_t.dtype, device=x_t.device)
            C_out = self._denoise_boundary(x_t, sigma_t * s_in, y)

            sigma_t_b = (sigma_t * s_in).reshape(-1, 1, 1, 1)
            sigma_next_b = (sigma_next * s_in).reshape(-1, 1, 1, 1)
            eps_pred = (x_t - C_out) / sigma_t_b
            x_t = C_out + sigma_next_b * eps_pred

        return x_t

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        guidance_scale: Optional[float] = 1.0,
        num_inference_steps: Optional[int] = 4,
        sampler: Optional[str] = 'consistency_fn',
        seed: Optional[int] = None,
        return_only_latents: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        device = self.vae.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)

        if tokenized_prompts is None:
            out = self.tokenizer.tokenize(prompt)
            tokenized_prompts = out['input_ids']
            attention_mask = out['attention_mask'] if 'attention_mask' in out else None
        text_embeddings = self.text_encoder.encode(
            tokenized_prompts.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None
        )[0]

        latents = torch.randn(
            (len(text_embeddings), self.dit.in_channels, self.latent_res, self.latent_res),
            device=device,
            generator=rng_generator,
        )

        if self.consistency_loss_fn is not None:
            target_model = getattr(self.consistency_loss_fn, 'target_model', None)
            if isinstance(target_model, nn.Module):
                original_dit = self.dit
                self.dit = target_model

        if sampler == 'consistency_fn':
            latents = self._consistency_function_sampler_loop(latents, text_embeddings, num_inference_steps)
        else:
            if self.consistency_loss_fn is None:
                return super().generate(
                    prompt=prompt,
                    tokenized_prompts=tokenized_prompts,
                    attention_mask=attention_mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    sampler=sampler,
                    seed=seed,
                    return_only_latents=return_only_latents,
                    **kwargs
                )

        if self.consistency_loss_fn is not None and original_dit is not None:
            self.dit = original_dit

        if return_only_latents:
            return latents

        latents = 1 / self.latent_scale * latents
        torch_dtype = DATA_TYPES[self.dtype]
        image = self.vae.decode(latents.to(torch_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float().detach()
        return image




def create_lcm_latent_diffusion(
    # Base model parameters
    vae_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    text_encoder_name: str = 'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 
    dit_arch: str = 'MicroDiT_XL_2',
    latent_res: int = 32,
    in_channels: int = 4,
    pos_interp_scale: float = 1.0,
    dtype: str = 'bfloat16',
    precomputed_latents: bool = True,
    p_mean: float = -0.6,
    p_std: float = 1.2,
    train_mask_ratio: float = 0.,
    # Optional pretrained DiT checkpoint (state_dict) to load
    pretrained_dit_ckpt: Optional[str] = None,
    # Consistency distillation parameters
    use_consistency_distillation: bool = False,
    consistency_teacher_model: Optional[nn.Module] = None,  # Для ODE solver
    consistency_target_model: Optional[nn.Module] = None,    # EMA копия student для loss target
    consistency_noise_scheduler: Optional[DDPMScheduler] = None,
    consistency_config: Optional[Dict[str, Any]] = None,
) -> LCMLatentDiffusion:
    """
    Create LatentDiffusion model with optional consistency distillation.

    This function wraps create_latent_diffusion and then constructs
    an LCMLatentDiffusion object, optionally wiring the consistency loss/scheduler.
    """
    # Import here to avoid circular imports
    from .model import create_latent_diffusion

    base_model = create_latent_diffusion(
        vae_name=vae_name,
        text_encoder_name=text_encoder_name,
        dit_arch=dit_arch,
        latent_res=latent_res,
        in_channels=in_channels,
        pos_interp_scale=pos_interp_scale,
        dtype=dtype,
        precomputed_latents=precomputed_latents,
        p_mean=p_mean,
        p_std=p_std,
        train_mask_ratio=train_mask_ratio,
    )

    if pretrained_dit_ckpt is not None:
        print(f"Loading pretrained DiT checkpoint from {pretrained_dit_ckpt}")
        state_dict = torch.load(pretrained_dit_ckpt, map_location='cpu')
        base_model.dit.load_state_dict(state_dict)

    lcm_model = LCMLatentDiffusion(
        dit=base_model.dit,
        vae=base_model.vae,
        text_encoder=base_model.text_encoder,
        tokenizer=base_model.tokenizer,
        image_key=base_model.image_key,
        text_key=base_model.text_key,
        image_latents_key=base_model.image_latents_key,
        text_latents_key=base_model.text_latents_key,
        precomputed_latents=base_model.precomputed_latents,
        dtype=base_model.dtype,
        latent_res=base_model.latent_res,
        p_mean=base_model.edm_config.P_mean,
        p_std=base_model.edm_config.P_std,
        train_mask_ratio=base_model.train_mask_ratio,
        # Consistency distillation parameters
        use_consistency_distillation=use_consistency_distillation,
        consistency_teacher_model=consistency_teacher_model,
        consistency_target_model=consistency_target_model,
        consistency_noise_scheduler=consistency_noise_scheduler,
        consistency_config=consistency_config,
    )

    return lcm_model
