"""策略状态读写与优化辅助函数。

这里放置梯度博弈和 LLM 博弈共享的策略应用逻辑，避免两个入口重复处理关税、配额和历史切片。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .config import Country, OptimizationConfig, TriggerConfig
from .runtime.model import TORCH_DTYPE
from .runtime.sim import CountryState, TwoCountryDynamicSimulator


def make_optimizer(params: List[torch.Tensor], cfg: OptimizationConfig) -> torch.optim.Optimizer:
    """按配置构造 PyTorch 优化器。"""
    if cfg.optimizer == "Adam":
        return torch.optim.Adam(params, lr=float(cfg.learning_rate))
    if cfg.optimizer == "SGD":
        return torch.optim.SGD(params, lr=float(cfg.learning_rate))
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def randn_like(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """兼容部分 torch 版本中 randn_like 不支持 generator 的情况。"""
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def current_tariff_rate(sim: TwoCountryDynamicSimulator, country: Country) -> torch.Tensor:
    mult = sim.home_import_multiplier if country == "H" else sim.foreign_import_multiplier
    base = sim.baseline_import_multiplier[country]
    return (mult / (base + 1e-6) - 1.0).detach().clone().view(-1)


def current_quota_multiplier(sim: TwoCountryDynamicSimulator, country: Country) -> torch.Tensor:
    return sim.export_multiplier[country].detach().clone().view(-1)


def policy_dict_from_vector(x: torch.Tensor, active_sectors: List[int]) -> Dict[int, float]:
    x = x.detach().view(-1).cpu()
    return {int(s): float(x[int(s)].item()) for s in active_sectors}


def apply_policy_differentiable(
    sim_fork: TwoCountryDynamicSimulator,
    *,
    country: Country,
    tau_rate: torch.Tensor,
    quota_mult: torch.Tensor,
) -> None:
    """在可微仿真副本上施加关税和出口配额策略。"""
    if country == "H":
        sim_fork.home_import_multiplier = sim_fork.baseline_import_multiplier["H"] * (1.0 + tau_rate)
        sim_fork.export_multiplier["H"] = quota_mult
        sim_fork._update_export_base("H")
    else:
        sim_fork.foreign_import_multiplier = sim_fork.baseline_import_multiplier["F"] * (1.0 + tau_rate)
        sim_fork.export_multiplier["F"] = quota_mult
        sim_fork._update_export_base("F")


def history_slice(sim: TwoCountryDynamicSimulator, start_idx: int) -> Dict[str, List[CountryState]]:
    return {"H": list(sim.history["H"][start_idx:]), "F": list(sim.history["F"][start_idx:])}


def apply_trigger(sim: TwoCountryDynamicSimulator, trigger: TriggerConfig) -> None:
    if trigger.tariff:
        sim.apply_import_tariff(trigger.country, dict(trigger.tariff), note="Trigger: tariff")
    if trigger.quota:
        sim.apply_export_control(trigger.country, dict(trigger.quota), note="Trigger: quota")


# Backward-compatible aliases for callers that imported helper names from grad_game.
_make_optimizer = make_optimizer
_randn_like = randn_like
_current_tariff_rate = current_tariff_rate
_current_quota_multiplier = current_quota_multiplier
_policy_dict_from_vector = policy_dict_from_vector
_apply_policy_differentiable = apply_policy_differentiable
_history_slice = history_slice
_apply_trigger = apply_trigger


__all__ = [
    "TORCH_DTYPE",
    "_apply_policy_differentiable",
    "_apply_trigger",
    "_current_quota_multiplier",
    "_current_tariff_rate",
    "_history_slice",
    "_make_optimizer",
    "_policy_dict_from_vector",
    "_randn_like",
    "apply_policy_differentiable",
    "apply_trigger",
    "current_quota_multiplier",
    "current_tariff_rate",
    "history_slice",
    "make_optimizer",
    "policy_dict_from_vector",
    "randn_like",
]
