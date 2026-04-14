"""Gradient best-response game experiment for the current trade_game package.

Run:
    python -m trade_game.grad_game
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .config import (
    ConstraintsConfig,
    Country,
    GameConfig,
    ObjectiveConfig,
    OptimizationConfig,
    ParamsSource,
    TriggerConfig,
)
from .experiment_plots import plot_game_analysis, plot_supply_demand_gap
from .objective import DifferentiableObjective
from .params import load_params
from .policy_helpers import (
    _apply_policy_differentiable,
    _apply_trigger,
    _current_quota_multiplier,
    _current_tariff_rate,
    _history_slice,
    _make_optimizer,
    _policy_dict_from_vector,
    _randn_like,
)
from .runtime.model import DEFAULT_DEVICE, TORCH_DTYPE
from .runtime.sim import TwoCountryDynamicSimulator, bootstrap_simulator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_load_params = load_params


def default_grad_config() -> GameConfig:
    """Default real-IO gradient game config used by this entrypoint."""
    experiment_name = "grad_real_io_CN2017_US2017_H50_theta0002_warmup10000_r10"
    return GameConfig(
        name=experiment_name,
        params_source=ParamsSource(type="io_final", home_id="CN2017", foreign_id="US2017"),
        normalize_gap_by_supply=True,
        theta_price=0.002,
        rounds=10,
        decision_interval=10,
        lookahead_periods=12,
        warmup_periods=10000,
        trigger_settle_periods=0,
        trigger=TriggerConfig(country="H", tariff={2: 0.5}),
        opt_config=OptimizationConfig(
            learning_rate=0.01,
            iterations=50,
            optimizer="Adam",
            multi_start=1,
            start_strategy="current",
            select="sum",
        ),
        constraints=ConstraintsConfig(active_sectors=[2, 3], reciprocal_coeff=0, max_tariff=1.0, min_quota=0.0),
        objective=ObjectiveConfig(type="standard", weights=(1.0, 1.0, 1.0)),
        plot=True,
        output_dir=experiment_name.replace("grad_", "results_grad_", 1),
    )


def _optimize_single_side_gradient(
    sim: TwoCountryDynamicSimulator,
    *,
    config: GameConfig,
    target_country: Country,
    prev_opp_tau: torch.Tensor,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Single-country best response with the opponent policy held fixed."""
    objective = DifferentiableObjective(config.objective)
    lookahead = int(config.lookahead_periods)
    n_sectors = int(sim.params.home.alpha.shape[0])
    K = max(int(config.opt_config.multi_start), 1)

    opponent: Country = "F" if target_country == "H" else "H"

    active_mask = torch.zeros(n_sectors, dtype=torch.bool, device=DEFAULT_DEVICE)
    for s in config.constraints.active_sectors:
        active_mask[int(s)] = True

    init_tau_self = _current_tariff_rate(sim, target_country).to(DEFAULT_DEVICE)
    init_quota_self = _current_quota_multiplier(sim, target_country).to(DEFAULT_DEVICE)
    fixed_tau_opp = _current_tariff_rate(sim, opponent).to(DEFAULT_DEVICE)
    fixed_quota_opp = _current_quota_multiplier(sim, opponent).to(DEFAULT_DEVICE)

    if K == 1:
        tau_param = init_tau_self.clone().detach().requires_grad_(True)
        quota_param = init_quota_self.clone().detach().requires_grad_(True)
        opp_tau_batch = fixed_tau_opp.view(-1)
        opp_quota_batch = fixed_quota_opp.view(-1)
    else:
        gen = torch.Generator(device=DEFAULT_DEVICE)
        gen.manual_seed(int(config.opt_config.seed))

        def _repeat(x: torch.Tensor) -> torch.Tensor:
            return x.view(1, -1).repeat(K, 1)

        tau_param = _repeat(init_tau_self).detach()
        quota_param = _repeat(init_quota_self).detach()
        opp_tau_batch = _repeat(fixed_tau_opp)
        opp_quota_batch = _repeat(fixed_quota_opp)

        max_t = float(config.constraints.max_tariff)
        min_q = float(config.constraints.min_quota)

        def _lb(prev_opp: torch.Tensor) -> torch.Tensor:
            if float(config.constraints.reciprocal_coeff) <= 0:
                return torch.zeros_like(prev_opp, device=DEFAULT_DEVICE).view(1, -1).repeat(K, 1)
            lb = float(config.constraints.reciprocal_coeff) * prev_opp.to(DEFAULT_DEVICE).view(1, -1).repeat(K, 1)
            return torch.clamp(lb, 0.0, max_t)

        lb_self = _lb(prev_opp_tau)
        ub_self = torch.full_like(lb_self, max_t)

        strat = config.opt_config.start_strategy
        if strat == "random":
            tau_param = lb_self + (max_t - lb_self) * torch.rand(
                (K, n_sectors), generator=gen, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE
            )
            quota_param = min_q + (1.0 - min_q) * torch.rand(
                (K, n_sectors), generator=gen, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE
            )
        elif strat == "noisy_current":
            sigma = float(max(config.opt_config.start_noise, 0.0))
            tau_width = torch.clamp(max_t - lb_self, min=1e-6)
            tau_param = torch.clamp(
                tau_param + sigma * tau_width * _randn_like(tau_param, generator=gen),
                min=lb_self,
                max=ub_self,
            )
            q_width = max(1.0 - min_q, 1e-6)
            quota_param = torch.clamp(
                quota_param + sigma * q_width * _randn_like(quota_param, generator=gen),
                min=min_q,
                max=1.0,
            )
        elif strat == "current":
            pass
        else:
            raise ValueError(f"Unknown start_strategy: {strat}")

        tau_param.requires_grad_(True)
        quota_param.requires_grad_(True)

    opt = _make_optimizer([tau_param, quota_param], config.opt_config)
    max_t = torch.tensor(float(config.constraints.max_tariff), device=DEFAULT_DEVICE)
    min_q = float(config.constraints.min_quota)

    def lower_bound_tau(prev_opp: torch.Tensor) -> torch.Tensor:
        if float(config.constraints.reciprocal_coeff) <= 0:
            return torch.zeros(n_sectors, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
        lb = float(config.constraints.reciprocal_coeff) * prev_opp.to(DEFAULT_DEVICE)
        return torch.clamp(lb, 0.0, float(config.constraints.max_tariff))

    lb_self_vec = lower_bound_tau(prev_opp_tau)

    for _ in range(int(config.opt_config.iterations)):
        opt.zero_grad(set_to_none=True)

        sim_fork = sim.fork_differentiable()
        if K > 1:
            sim_fork.home_state = sim_fork.home_state.ensure_batch(K)
            sim_fork.foreign_state = sim_fork.foreign_state.ensure_batch(K)
            sim_fork.batch_size = K
            sim_fork.history = {"H": [sim_fork.home_state], "F": [sim_fork.foreign_state]}

        clamped_tau = torch.clamp(tau_param, min=lb_self_vec, max=max_t)
        final_tau = torch.where(active_mask, clamped_tau, init_tau_self)
        clamped_quota = torch.clamp(quota_param, min=min_q, max=1.0)
        final_quota = torch.where(active_mask, clamped_quota, init_quota_self)

        opp_tau_expanded = torch.where(
            active_mask,
            opp_tau_batch,
            _current_tariff_rate(sim, opponent).to(DEFAULT_DEVICE),
        )
        opp_quota_expanded = torch.where(
            active_mask,
            opp_quota_batch,
            _current_quota_multiplier(sim, opponent).to(DEFAULT_DEVICE),
        )

        _apply_policy_differentiable(sim_fork, country=target_country, tau_rate=final_tau, quota_mult=final_quota)
        _apply_policy_differentiable(sim_fork, country=opponent, tau_rate=opp_tau_expanded, quota_mult=opp_quota_expanded)

        sim_fork.run(lookahead)
        J_self = objective.compute(sim_fork.history, target_country, opponent)
        J_total = J_self.sum() if J_self.dim() > 0 else J_self

        grad = torch.autograd.grad(-J_total, [tau_param, quota_param])
        tau_param.grad = grad[0]
        quota_param.grad = grad[1]
        torch.nn.utils.clip_grad_norm_([tau_param, quota_param], max_norm=1.0)
        opt.step()

    tau_final = torch.clamp(tau_param.detach(), min=lb_self_vec, max=max_t)
    quota_final = torch.clamp(quota_param.detach(), min=min_q, max=1.0)
    tau_vec = torch.where(active_mask, tau_final, init_tau_self)
    quota_vec = torch.where(active_mask, quota_final, init_quota_self)

    sim_pred = sim.fork_differentiable()
    if K > 1:
        sim_pred.home_state = sim_pred.home_state.ensure_batch(K)
        sim_pred.foreign_state = sim_pred.foreign_state.ensure_batch(K)
        sim_pred.batch_size = K
        sim_pred.history = {"H": [sim_pred.home_state], "F": [sim_pred.foreign_state]}

    opp_tau_expanded = torch.where(active_mask, opp_tau_batch, _current_tariff_rate(sim, opponent).to(DEFAULT_DEVICE))
    opp_quota_expanded = torch.where(active_mask, opp_quota_batch, _current_quota_multiplier(sim, opponent).to(DEFAULT_DEVICE))
    _apply_policy_differentiable(sim_pred, country=target_country, tau_rate=tau_vec, quota_mult=quota_vec)
    _apply_policy_differentiable(sim_pred, country=opponent, tau_rate=opp_tau_expanded, quota_mult=opp_quota_expanded)
    sim_pred.run(lookahead)

    J_pred_all = objective.compute(sim_pred.history, target_country, opponent).detach()

    if K == 1:
        J_best = float(J_pred_all.cpu().item())
    else:
        best_idx = int(torch.argmax(J_pred_all).cpu().item())
        J_best = float(J_pred_all[best_idx].cpu().item())
        tau_vec = tau_vec[best_idx]
        quota_vec = quota_vec[best_idx]

    return (
        _policy_dict_from_vector(tau_vec, config.constraints.active_sectors),
        _policy_dict_from_vector(quota_vec, config.constraints.active_sectors),
        J_best,
    )


def _optimize_static_best_response(
    sim: TwoCountryDynamicSimulator,
    *,
    config: GameConfig,
    prev_tau_H: torch.Tensor,
    prev_tau_F: torch.Tensor,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """Compute a one-step Nash-like static best response for both countries."""
    tau_H, quota_H, J_H = _optimize_single_side_gradient(
        sim, config=config, target_country="H", prev_opp_tau=prev_tau_F
    )
    tau_F, quota_F, J_F = _optimize_single_side_gradient(
        sim, config=config, target_country="F", prev_opp_tau=prev_tau_H
    )

    return (
        {
            "H": {"tariff": tau_H, "quota": quota_H},
            "F": {"tariff": tau_F, "quota": quota_F},
        },
        {"J_H_pred": J_H, "J_F_pred": J_F},
    )


def run_grad_experiment(config: GameConfig) -> TwoCountryDynamicSimulator:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Simulator...")
    logger.info("  params_source: %s", config.params_source.type)
    params_raw = load_params(config.params_source)
    sim = bootstrap_simulator(
        params_raw,
        theta_price=float(config.theta_price),
        normalize_gap_by_supply=bool(config.normalize_gap_by_supply),
    )

    sim.run(int(config.warmup_periods))
    if config.trigger is not None:
        _apply_trigger(sim, config.trigger)
    if int(config.trigger_settle_periods) > 0:
        sim.run(int(config.trigger_settle_periods))

    policy_history: list[Dict[str, Any]] = []
    timing_stats: list[float] = []
    total_start = time.perf_counter()
    objective = DifferentiableObjective(config.objective)

    for r in range(int(config.rounds)):
        logger.info("=== ROUND %s ===", r + 1)

        prev_tau_H = _current_tariff_rate(sim, "H")
        prev_tau_F = _current_tariff_rate(sim, "F")
        prev_quota_H = _current_quota_multiplier(sim, "H")
        prev_quota_F = _current_quota_multiplier(sim, "F")

        round_rec: Dict[str, Any] = {
            "round": r + 1,
            "prev": {
                "H": {
                    "tariff": _policy_dict_from_vector(prev_tau_H, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(prev_quota_H, config.constraints.active_sectors),
                },
                "F": {
                    "tariff": _policy_dict_from_vector(prev_tau_F, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(prev_quota_F, config.constraints.active_sectors),
                },
            },
        }

        t0 = time.perf_counter()
        policies, pred = _optimize_static_best_response(
            sim,
            config=config,
            prev_tau_H=prev_tau_H,
            prev_tau_F=prev_tau_F,
        )
        opt_elapsed = time.perf_counter() - t0
        timing_stats.append(opt_elapsed)

        tau_H, quota_H = policies["H"]["tariff"], policies["H"]["quota"]
        tau_F, quota_F = policies["F"]["tariff"], policies["F"]["quota"]
        round_rec["decision"] = {"H": {"tariff": tau_H, "quota": quota_H}, "F": {"tariff": tau_F, "quota": quota_F}}
        round_rec["predicted"] = pred
        round_rec["opt_time_s"] = float(opt_elapsed)

        logger.info("[Round %s] H tariff=%s quota=%s | pred_J=%s", r + 1, tau_H, quota_H, pred.get("J_H_pred"))
        logger.info("[Round %s] F tariff=%s quota=%s | pred_J=%s", r + 1, tau_F, quota_F, pred.get("J_F_pred"))

        if tau_H:
            sim.apply_import_tariff("H", tau_H, note=f"R{r+1} Decision")
        if quota_H:
            sim.apply_export_control("H", quota_H, note=f"R{r+1} Decision")
        if tau_F:
            sim.apply_import_tariff("F", tau_F, note=f"R{r+1} Decision")
        if quota_F:
            sim.apply_export_control("F", quota_F, note=f"R{r+1} Decision")

        start_idx = len(sim.history["H"]) - 1
        sim.run(int(config.decision_interval))

        realized_hist = _history_slice(sim, start_idx)
        payoff_H = float(objective.compute(realized_hist, "H", "F").detach().cpu().item())
        payoff_F = float(objective.compute(realized_hist, "F", "H").detach().cpu().item())
        round_rec["payoff"] = {"H": payoff_H, "F": payoff_F}

        summary = sim.summarize_history(base_period_idx=int(config.warmup_periods))
        round_rec["metrics"] = {
            "income_H": float(summary["H"]["income"][-1]),
            "trade_balance_H": float(summary["H"]["trade_balance_val"][-1]),
            "price_mean_H": float(summary["H"]["price_mean"][-1]),
            "income_F": float(summary["F"]["income"][-1]),
            "trade_balance_F": float(summary["F"]["trade_balance_val"][-1]),
            "price_mean_F": float(summary["F"]["price_mean"][-1]),
        }

        policy_history.append(round_rec)

    total_elapsed = time.perf_counter() - total_start
    out_path = Path(config.output_dir) / f"{config.name}_policies.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "name": config.name,
                    "params_source": {
                        "type": config.params_source.type,
                        "home_id": config.params_source.home_id,
                        "foreign_id": config.params_source.foreign_id,
                        "tradable_sectors": (
                            None if config.params_source.tradable_sectors is None else list(config.params_source.tradable_sectors)
                        ),
                    },
                    "rounds": int(config.rounds),
                    "theta_price": float(config.theta_price),
                    "normalize_gap_by_supply": bool(config.normalize_gap_by_supply),
                    "decision_interval": int(config.decision_interval),
                    "lookahead_periods": int(config.lookahead_periods),
                    "warmup_periods": int(config.warmup_periods),
                    "trigger_settle_periods": int(config.trigger_settle_periods),
                    "trigger": (
                        None
                        if config.trigger is None
                        else {"country": config.trigger.country, "tariff": config.trigger.tariff, "quota": config.trigger.quota}
                    ),
                    "opt": {
                        "optimizer": config.opt_config.optimizer,
                        "learning_rate": config.opt_config.learning_rate,
                        "iterations": config.opt_config.iterations,
                    },
                    "constraints": {
                        "active_sectors": list(config.constraints.active_sectors),
                        "reciprocal_coeff": float(config.constraints.reciprocal_coeff),
                        "max_tariff": float(config.constraints.max_tariff),
                        "min_quota": float(config.constraints.min_quota),
                    },
                    "objective": {"type": config.objective.type, "weights": tuple(float(x) for x in config.objective.weights)},
                },
                "policies": policy_history,
                "timing": {
                    "total_s": float(total_elapsed),
                    "per_round_s": [float(x) for x in timing_stats],
                    "avg_per_round_s": float(sum(timing_stats) / len(timing_stats)) if timing_stats else 0.0,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved policy history to: %s", out_path)

    print("\n" + "=" * 80)
    print("策略决策摘要")
    print("=" * 80)
    for rec in policy_history:
        print(f"\n轮次 {rec['round']}:")
        print(f"  决策(H): {rec['decision']['H']}")
        print(f"  决策(F): {rec['decision']['F']}")
        print(f"  预测效用: {rec['predicted']}")
        print(f"  事后效用: {rec['payoff']}")
        print(f"  优化耗时: {rec['opt_time_s']:.2f}s")

    print("\n" + "-" * 40)
    print("时间统计")
    print("-" * 40)
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  优化总耗时: {sum(timing_stats):.2f}s")
    if timing_stats:
        print(f"  平均每轮优化: {sum(timing_stats) / len(timing_stats):.2f}s")
    print("=" * 80 + "\n")

    if config.plot:
        plot_path = Path(config.output_dir) / f"{config.name}_analysis.png"
        plot_game_analysis(
            sim.summarize_history(base_period_idx=int(config.warmup_periods)),
            sim.policy_events,
            save_path=str(plot_path),
            warmup_periods=int(config.warmup_periods),
        )
        logger.info("Plot saved to: %s", plot_path)

        gap_plot_path = Path(config.output_dir) / f"{config.name}_supply_demand_gap.png"
        plot_supply_demand_gap(sim, save_path=str(gap_plot_path), warmup_periods=int(config.warmup_periods))
        logger.info("Supply-demand gap plot saved to: %s", gap_plot_path)

    return sim


if __name__ == "__main__":
    run_grad_experiment(default_grad_config())
