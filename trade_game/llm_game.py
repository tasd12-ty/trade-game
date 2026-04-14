"""LLM-driven game experiment for the current trade_game package.

Run:
    python -m trade_game.llm_game
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
    LLMConfig,
    LLMGameConfig,
    ObjectiveConfig,
    OptimizationConfig,
    ParamsSource,
    TriggerConfig,
)
from .experiment_plots import plot_game_analysis, plot_supply_demand_gap
from .llm import LLMPolicyAgent, OpenAIClient
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
)
from .runtime.model import DEFAULT_DEVICE, TORCH_DTYPE
from .runtime.sim import TwoCountryDynamicSimulator, bootstrap_simulator
from .scenarios import default_llm_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_load_params = load_params


def _optimize_single_country_gradient(
    sim: TwoCountryDynamicSimulator,
    country: Country,
    opponent_policy: Dict[str, Dict[int, float]],
    *,
    config: LLMGameConfig,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """Optimize one country's policy while holding the opponent policy fixed."""
    objective = DifferentiableObjective(config.objective)
    lookahead = int(config.lookahead_periods)
    n_sectors = int(sim.params.home.alpha.shape[0])
    opponent: Country = "F" if country == "H" else "H"

    active_mask = torch.zeros(n_sectors, dtype=torch.bool, device=DEFAULT_DEVICE)
    for s in config.constraints.active_sectors:
        active_mask[int(s)] = True

    init_tau = _current_tariff_rate(sim, country).to(DEFAULT_DEVICE)
    init_quota = _current_quota_multiplier(sim, country).to(DEFAULT_DEVICE)

    opp_tau = torch.zeros(n_sectors, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
    opp_quota = torch.ones(n_sectors, device=DEFAULT_DEVICE, dtype=TORCH_DTYPE)
    for s, v in opponent_policy["tariff"].items():
        opp_tau[int(s)] = float(v)
    for s, v in opponent_policy["quota"].items():
        opp_quota[int(s)] = float(v)

    tau_param = init_tau.clone().detach().requires_grad_(True)
    quota_param = init_quota.clone().detach().requires_grad_(True)
    opt = _make_optimizer([tau_param, quota_param], config.opt_config)

    max_t = torch.tensor(float(config.constraints.max_tariff), device=DEFAULT_DEVICE)
    min_q = float(config.constraints.min_quota)

    for _ in range(int(config.opt_config.iterations)):
        opt.zero_grad(set_to_none=True)
        sim_fork = sim.fork_differentiable()

        clamped_tau = torch.clamp(tau_param, min=0.0, max=max_t)
        clamped_quota = torch.clamp(quota_param, min=min_q, max=1.0)
        final_tau = torch.where(active_mask, clamped_tau, init_tau)
        final_quota = torch.where(active_mask, clamped_quota, init_quota)

        _apply_policy_differentiable(sim_fork, country=country, tau_rate=final_tau, quota_mult=final_quota)
        _apply_policy_differentiable(sim_fork, country=opponent, tau_rate=opp_tau, quota_mult=opp_quota)

        sim_fork.run(lookahead)
        J = objective.compute(sim_fork.history, country, opponent)
        grad = torch.autograd.grad(-J, [tau_param, quota_param])
        tau_param.grad = grad[0]
        quota_param.grad = grad[1]
        opt.step()

    tau_final = torch.clamp(tau_param.detach(), min=0.0, max=max_t)
    quota_final = torch.clamp(quota_param.detach(), min=min_q, max=1.0)
    tau_vec = torch.where(active_mask, tau_final, init_tau)
    quota_vec = torch.where(active_mask, quota_final, init_quota)

    sim_pred = sim.fork_differentiable()
    _apply_policy_differentiable(sim_pred, country=country, tau_rate=tau_vec, quota_mult=quota_vec)
    _apply_policy_differentiable(sim_pred, country=opponent, tau_rate=opp_tau, quota_mult=opp_quota)
    sim_pred.run(lookahead)
    J_pred = float(objective.compute(sim_pred.history, country, opponent).detach().cpu().item())

    return (
        {
            "tariff": _policy_dict_from_vector(tau_vec, config.constraints.active_sectors),
            "quota": _policy_dict_from_vector(quota_vec, config.constraints.active_sectors),
        },
        {"J_pred": J_pred},
    )


def _create_llm_client(config: LLMGameConfig):
    """Create the configured OpenAI-compatible LLM client."""
    logger.info("Using OpenAI-compatible client: preset=%s, model=%s", config.llm.preset, config.llm.model)
    return OpenAIClient(
        model=config.llm.model,
        preset=config.llm.preset,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )


def run_llm_experiment(config: LLMGameConfig) -> TwoCountryDynamicSimulator:
    """Run an LLM-vs-LLM or LLM-vs-gradient trade policy game."""
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

    llm_client = _create_llm_client(config)
    agent = LLMPolicyAgent(
        llm_client=llm_client,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

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
        decisions: Dict[str, Dict[str, Any]] = {"H": {}, "F": {}}
        reasoning_log = {"H": "", "F": ""}

        for country in ["H", "F"]:
            opponent = "F" if country == "H" else "H"
            current_tau = prev_tau_H if country == "H" else prev_tau_F
            current_quota = prev_quota_H if country == "H" else prev_quota_F
            opp_tau = prev_tau_F if country == "H" else prev_tau_H
            opp_quota = prev_quota_F if country == "H" else prev_quota_H

            if config.llm_plays == "both" or config.llm_plays == country:
                decision = agent.decide(
                    sim=sim,
                    country=country,
                    round_num=r + 1,
                    opponent_prev_tariff=_policy_dict_from_vector(opp_tau, config.constraints.active_sectors),
                    opponent_prev_quota=_policy_dict_from_vector(opp_quota, config.constraints.active_sectors),
                    current_tariff=_policy_dict_from_vector(current_tau, config.constraints.active_sectors),
                    current_quota=_policy_dict_from_vector(current_quota, config.constraints.active_sectors),
                    active_sectors=list(config.constraints.active_sectors),
                    max_tariff=config.constraints.max_tariff,
                    min_quota=config.constraints.min_quota,
                    reciprocal_coeff=config.constraints.reciprocal_coeff,
                )
                decisions[country] = {
                    "tariff": decision.tariff,
                    "quota": decision.quota,
                    "system_prompt": decision.system_prompt,
                    "user_prompt": decision.user_prompt,
                    "raw_response": decision.raw_response,
                    "reasoning_content": decision.llm_reasoning_content,
                    "token_usage": decision.token_usage,
                }
                reasoning_log[country] = f"[LLM] {decision.reasoning}"
                logger.info("[%s] LLM reasoning: %s", country, decision.reasoning)
            elif config.non_llm_strategy == "gradient":
                opponent_policy = {
                    "tariff": _policy_dict_from_vector(opp_tau, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(opp_quota, config.constraints.active_sectors),
                }
                grad_policy, grad_info = _optimize_single_country_gradient(
                    sim=sim,
                    country=country,
                    opponent_policy=opponent_policy,
                    config=config,
                )
                decisions[country] = {
                    "tariff": grad_policy["tariff"],
                    "quota": grad_policy["quota"],
                }
                reasoning_log[country] = f"[Gradient] J_pred={grad_info['J_pred']:.4f}"
                logger.info("[%s] Gradient optimization: J_pred=%.4f", country, grad_info["J_pred"])
            else:
                decisions[country] = {
                    "tariff": _policy_dict_from_vector(current_tau, config.constraints.active_sectors),
                    "quota": _policy_dict_from_vector(current_quota, config.constraints.active_sectors),
                }
                reasoning_log[country] = "[Fixed] Keeping current policy"

        opt_elapsed = time.perf_counter() - t0
        timing_stats.append(opt_elapsed)

        tau_H, quota_H = decisions["H"]["tariff"], decisions["H"]["quota"]
        tau_F, quota_F = decisions["F"]["tariff"], decisions["F"]["quota"]

        round_rec["decision"] = {
            "H": {"tariff": tau_H, "quota": quota_H},
            "F": {"tariff": tau_F, "quota": quota_F},
        }
        round_rec["reasoning"] = reasoning_log
        round_rec["llm_time_s"] = float(opt_elapsed)
        round_rec["llm_io"] = {}
        for country in ["H", "F"]:
            d = decisions[country]
            if "system_prompt" in d:
                round_rec["llm_io"][country] = {
                    "system_prompt": d.get("system_prompt", ""),
                    "user_prompt": d.get("user_prompt", ""),
                    "raw_response": d.get("raw_response", ""),
                    "reasoning_content": d.get("reasoning_content"),
                    "token_usage": d.get("token_usage"),
                }

        logger.info("[Round %s] H tariff=%s quota=%s", r + 1, tau_H, quota_H)
        logger.info("[Round %s] F tariff=%s quota=%s", r + 1, tau_F, quota_F)

        if tau_H:
            sim.apply_import_tariff("H", tau_H, note=f"R{r+1} LLM Decision")
        if quota_H:
            sim.apply_export_control("H", quota_H, note=f"R{r+1} LLM Decision")
        if tau_F:
            sim.apply_import_tariff("F", tau_F, note=f"R{r+1} LLM Decision")
        if quota_F:
            sim.apply_export_control("F", quota_F, note=f"R{r+1} LLM Decision")

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
                    "llm": {
                        "model": config.llm.model,
                        "preset": config.llm.preset,
                        "temperature": (None if config.llm.temperature is None else float(config.llm.temperature)),
                        "max_tokens": (None if config.llm.max_tokens is None else int(config.llm.max_tokens)),
                        "llm_plays": config.llm_plays,
                    },
                    "non_llm_strategy": str(config.non_llm_strategy),
                    "opt": {
                        "optimizer": config.opt_config.optimizer,
                        "learning_rate": float(config.opt_config.learning_rate),
                        "iterations": int(config.opt_config.iterations),
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
    print("LLM 博弈策略决策摘要")
    print("=" * 80)
    for rec in policy_history:
        print(f"\n轮次 {rec['round']}:")
        print(f"  决策(H): {rec['decision']['H']}")
        print(f"  推理(H): {rec['reasoning']['H']}")
        print(f"  决策(F): {rec['decision']['F']}")
        print(f"  推理(F): {rec['reasoning']['F']}")
        print(f"  事后效用: H={rec['payoff']['H']:.4f}, F={rec['payoff']['F']:.4f}")
        print(f"  LLM 耗时: {rec['llm_time_s']:.2f}s")

    print("\n" + "-" * 40)
    print("时间统计")
    print("-" * 40)
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  LLM 总耗时: {sum(timing_stats):.2f}s")
    if timing_stats:
        print(f"  平均每轮 LLM: {sum(timing_stats) / len(timing_stats):.2f}s")
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
    run_llm_experiment(default_llm_config())
