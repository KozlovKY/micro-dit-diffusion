# Callbacks adopted from https://github.com/mosaicml/diffusion/tree/main/diffusion/callbacks
from typing import Dict, List, Optional, Sequence
import torch
from torch.nn.parallel import DistributedDataParallel
from composer import Callback, Logger, State
from composer.core import TimeUnit, get_precision_context
import torch.distributed as dist
import os

class LogDiffusionImages(Callback):
    """Logs images generated from the evaluation prompts to a logger.

    Args:
        prompts (List[str]): List of prompts to use for evaluation.
        sampling_steps (int): Number of inference steps to use during sampling.
        guidance_scale (float): Guidance scale in classifier free guidance (scale=1 implies no classifier free guidance).
        seed (int): Random seed to use for generation. Set a seed for reproducible generation.
    """
    def __init__(self, prompts: List[str], sampling_steps: int = 30, guidance_scale: float = 5.0, seed: Optional[int] = 1138,
                 sampler: str = 'edm'):
        self.prompts = prompts
        self.sampling_steps = sampling_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.sampler = sampler

    def eval_batch_end(self, state: State, logger: Logger):
        # Only log once per eval epoch
        if state.eval_timestamp.get(TimeUnit.BATCH).value == 1:
            # Get the model object if it has been wrapped by DDP
            model = state.model.module if isinstance(state.model, DistributedDataParallel) else state.model

            # Generate images
            with get_precision_context(state.precision):
                images = model.generate(
                    self.prompts,
                    num_inference_steps=self.sampling_steps,
                    guidance_scale=self.guidance_scale,
                    sampler=self.sampler,
                    seed=self.seed)
            
            # Log images to tensorboard/wandb
            for prompt, image in zip(self.prompts, images):
                logger.log_images(
                    images=image,
                    name=prompt[:100],
                    step=state.timestamp.batch.value,
                    use_table=False)


class NaNCatcher(Callback):
    """Catches NaNs in the loss and raises an error if one is found."""

    def after_loss(self, state: State, logger: Logger):
        """Check if loss is NaN and raise an error if so."""
        if isinstance(state.loss, torch.Tensor):
            if torch.isnan(state.loss).any():
                raise RuntimeError('Train loss contains a NaN.')
        elif isinstance(state.loss, Sequence):
            for loss in state.loss:
                if torch.isnan(loss).any():
                    raise RuntimeError('Train loss contains a NaN.')
        elif isinstance(state.loss, Dict):
            for k, v in state.loss.items():
                if torch.isnan(v).any():
                    raise RuntimeError(f'Train loss {k} contains a NaN.')
        else:
            raise TypeError(f'Loss is of type {type(state.loss)}, but should be a tensor or a list of tensors')


class EMATargetUpdate(Callback):
    def __init__(self, ema: float = 0.999, update_every: int = 1, warmup_steps: int = 0):
        self.ema = float(ema)
        self.update_every = int(update_every)
        self.warmup_steps = int(warmup_steps)

    def _get_models(self, state: State):
        """Helper to safely retrieve student and target models."""
        model = state.model.module if isinstance(state.model, DistributedDataParallel) else state.model
        loss_fn = getattr(model, 'consistency_loss_fn', None)
        target_model = getattr(loss_fn, 'target_model', None)
        student = getattr(model, 'dit', None)
        return student, target_model

    def batch_end(self, state: State, logger: Logger) -> None:
        step = state.timestamp.batch.value
        student, target_model = self._get_models(state)
        
        if student is None or target_model is None:
            return

        should_update = (step >= self.warmup_steps) and ((step % self.update_every) == 0)

        if should_update:
            with torch.no_grad():
                for t_param, s_param in zip(target_model.parameters(), student.parameters()):
                    t_param.data.mul_(self.ema).add_(s_param.data.to(t_param.device), alpha=(1.0 - self.ema))

                student_bufs = dict(student.named_buffers())
                for b_name, t_buf in target_model.named_buffers():
                    s_buf = student_bufs.get(b_name, None)
                    if s_buf is not None and t_buf.dtype.is_floating_point and s_buf.shape == t_buf.shape:
                        t_buf.data.mul_(self.ema).add_(s_buf.data.to(t_buf.device), alpha=(1.0 - self.ema))
