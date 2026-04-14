"""可微目标函数。

目标函数同时服务梯度优化和 LLM 对照实验，按收入增长、贸易余额和价格稳定性聚合单国效用。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .config import ObjectiveConfig
from .runtime.sim import CountryState


class DifferentiableObjective:
    """从仿真历史中计算单国效用，并保留 PyTorch 梯度链路。"""

    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.w_income, self.w_tb, self.w_stab = config.weights

    def compute(self, history: Dict[str, List[CountryState]], actor: str, opponent: str) -> torch.Tensor:
        """Return one utility scalar, or a vector when the simulator is batched."""
        states_self = history[actor]
        states_opp = history[opponent]

        def get_seq(states, attr):
            return torch.stack([getattr(s, attr) for s in states[1:]])

        income_self = get_seq(states_self, "income")
        income_self_0 = states_self[0].income
        growth_self = (income_self / (income_self_0 + 1e-6)) - 1.0
        score_income_self = growth_self.mean(dim=0)

        def compute_tb(s):
            exp_val = (s.export_actual * s.price).sum(dim=-1)
            imp_val = (s.imp_price * (s.X_imp.sum(dim=-2) + s.C_imp)).sum(dim=-1)
            return exp_val - imp_val

        tb_self_seq = torch.stack([compute_tb(s) for s in states_self[1:]])
        scale = income_self_0 + 1.0
        score_tb_self = (tb_self_seq / scale).mean(dim=0)

        def get_price_idx(s, p0):
            return (s.price / (p0 + 1e-6)).mean(dim=-1)

        p0 = states_self[0].price
        p_idx_seq = torch.stack([get_price_idx(s, p0) for s in states_self[1:]])
        score_stab_self = -torch.std(p_idx_seq, dim=0)

        j_std = self.w_income * score_income_self + self.w_tb * score_tb_self + self.w_stab * score_stab_self

        if self.config.type == "relative":
            income_opp = get_seq(states_opp, "income")
            income_opp_0 = states_opp[0].income
            growth_opp = (income_opp / (income_opp_0 + 1e-6)) - 1.0
            score_income_opp = growth_opp.mean(dim=0)
            tb_opp_seq = torch.stack([compute_tb(s) for s in states_opp[1:]])
            score_tb_opp = (tb_opp_seq / (income_opp_0 + 1.0)).mean(dim=0)
            return (
                self.w_income * (score_income_self - score_income_opp)
                + self.w_tb * (score_tb_self - score_tb_opp)
                + self.w_stab * score_stab_self
            )

        return j_std


__all__ = ["DifferentiableObjective"]
