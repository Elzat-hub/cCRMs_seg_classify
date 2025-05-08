# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import bisect

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from typing import List


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'plateau':
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            patience_t=config.TRAIN.LR_SCHEDULER.PATIENCE,
            lr_min=config.TRAIN.MIN_LR,
            mode='min',
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cooldown_t=0,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0,
                 t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect.bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

class PlateauLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_rate: float = 0.1,
                 patience_t: int = 10,
                 lr_min: float = 0,
                 mode: str = 'min',
                 threshold: float = 1e-4,
                 cooldown_t: int = 0,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0,
                 t_in_epochs: bool = True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize: bool = True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_rate = decay_rate
        self.patience_t = patience_t
        self.lr_min = lr_min
        self.mode = mode
        self.threshold = threshold
        self.cooldown_t = cooldown_t
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.t = 0

        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == 'min' and current < best - self.threshold:
            return True
        if self.mode == 'max' and current > best + self.threshold:
            return True
        return False

    def get_epoch_values(self, epoch: int) -> List[float]:
        if self.t_in_epochs:
            self.t = epoch
        return self._get_lr(self.t)

    def get_update_values(self, num_updates: int) -> List[float]:
        if not self.t_in_epochs:
            self.t = num_updates
        return self._get_lr(self.t)

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [max(v * (self.decay_rate ** (self.num_bad_epochs // self.patience_t)), self.lr_min)
                   for v in self.base_values]
        return lrs
    
    def step(self, metric: float, epoch: int = None) -> None:
        if epoch is not None:
            self.t = epoch
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metric, self.best):
                self.best = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > 0 and self.num_bad_epochs % self.patience_t == 0:
                self.cooldown_counter = self.cooldown_t

        # 更新学习率
        new_lrs = self._get_lr(self.t)
        self.update_groups(new_lrs)

    def state_dict(self) -> dict:
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
            't': self.t,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.t = state_dict['t']