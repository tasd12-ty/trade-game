"""warmup 稳定性参数搜索。

脚本会先为一个国家组合求解一次初始均衡，再复用该均衡测试多组
theta_price/warmup 候选值。报告指标聚焦政策冲击前的基线 warmup：

- 使用计划出口口径计算各部门供需相对缺口
- 计算最后窗口内各部门价格变异系数
- 计算最后窗口内均价指数变异系数
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from .params import load_io_params
from .runtime.model import normalize_model_params, solve_initial_equilibrium
from .runtime.sim import CountryState, CountrySimulator, TwoCountryDynamicSimulator


ROOT = Path(__file__).resolve().parent.parent


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _pick_last_state(sim: TwoCountryDynamicSimulator, country: str) -> CountryState:
    return sim.history[country][-1]


def _country_gap_metrics(country_sim: CountrySimulator, state: CountryState) -> Dict[str, object]:
    outputs, pX_dom, _pX_imp, pC_dom, _pC_imp, _theta_prod, _theta_cons = country_sim._plan_demands(
        state,
        state.imp_price,
    )
    planned_export = state.export_base
    supply = outputs.detach()
    demand = (pX_dom.sum(dim=-2) + pC_dom + planned_export).detach()
    rel_gap = ((demand - supply) / torch.clamp(supply, min=1e-9)).abs()
    return {
        "supply": supply.detach().cpu().numpy().tolist(),
        "demand": demand.detach().cpu().numpy().tolist(),
        "rel_gap": rel_gap.detach().cpu().numpy().tolist(),
        "max_rel_gap": float(rel_gap.max().item()),
        "mean_rel_gap": float(rel_gap.mean().item()),
    }


def compute_gap_metrics(sim: TwoCountryDynamicSimulator) -> Dict[str, Dict[str, object]]:
    return {
        "H": _country_gap_metrics(sim.home_sim, _pick_last_state(sim, "H")),
        "F": _country_gap_metrics(sim.foreign_sim, _pick_last_state(sim, "F")),
    }


def _price_stack(history: Sequence[CountryState], last_n: int) -> torch.Tensor:
    n = min(int(last_n), len(history) - 1)
    if n < 2:
        raise ValueError("Need at least two periods for price convergence metrics")
    return torch.stack([history[-i].price.detach() for i in range(1, n + 1)])


def check_price_convergence(sim: TwoCountryDynamicSimulator, last_n: int = 100) -> Dict[str, object]:
    try:
        prices_H = _price_stack(sim.history["H"], last_n)
        prices_F = _price_stack(sim.history["F"], last_n)
    except ValueError:
        return {
            "H_price_cv_max": float("inf"),
            "F_price_cv_max": float("inf"),
            "H_price_index_cv": float("inf"),
            "F_price_index_cv": float("inf"),
        }

    cv_H = prices_H.std(dim=0) / prices_H.mean(dim=0).clamp(min=1e-9)
    cv_F = prices_F.std(dim=0) / prices_F.mean(dim=0).clamp(min=1e-9)
    index_H = prices_H.mean(dim=1)
    index_F = prices_F.mean(dim=1)
    index_cv_H = index_H.std() / index_H.mean().clamp(min=1e-9)
    index_cv_F = index_F.std() / index_F.mean().clamp(min=1e-9)
    return {
        "H_price_cv_max": float(cv_H.max().item()),
        "F_price_cv_max": float(cv_F.max().item()),
        "H_price_index_cv": float(index_cv_H.item()),
        "F_price_index_cv": float(index_cv_F.item()),
        "H_final_price": [float(x) for x in sim.history["H"][-1].price.detach().cpu().tolist()],
        "F_final_price": [float(x) for x in sim.history["F"][-1].price.detach().cpu().tolist()],
    }


def _has_nan_or_inf(sim: TwoCountryDynamicSimulator) -> bool:
    for state in (sim.history["H"][-1], sim.history["F"][-1]):
        for tensor in (state.price, state.output, state.income):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return True
    return False


def _round_list(values: Iterable[float], digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in values]


def run_search(
    home_id: str = "CN2017",
    foreign_id: str = "US2017",
    theta_values: Sequence[float] | None = None,
    warmup_values: Sequence[int] | None = None,
    *,
    tradable_sectors: Sequence[int] | None = None,
    normalize_gap_by_supply: bool = True,
    last_n: int = 100,
    chunk_size: int = 500,
    max_solver_iter: int = 400,
    solver_tol: float = 1e-8,
) -> Dict[str, object]:
    if theta_values is None:
        theta_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002]
    if warmup_values is None:
        warmup_values = [1000, 2000, 5000, 10000]

    params_raw = load_io_params(home_id, foreign_id, list(tradable_sectors) if tradable_sectors else None)
    params = normalize_model_params(params_raw)

    print(f"Solving initial equilibrium for H={home_id}, F={foreign_id} ...")
    solve_t0 = time.perf_counter()
    equilibrium = solve_initial_equilibrium(params, max_iterations=max_solver_iter, tolerance=solver_tol)
    solve_elapsed = time.perf_counter() - solve_t0
    eq_info = dict(equilibrium.get("convergence_info", {}))
    eq_info["elapsed_s"] = round(solve_elapsed, 2)
    print(f"Equilibrium: {eq_info}")

    results: List[Dict[str, object]] = []
    total = len(theta_values) * len(warmup_values)
    idx = 0
    for theta_price in theta_values:
        for warmup in warmup_values:
            idx += 1
            print(f"[{idx}/{total}] theta={theta_price:g}, warmup={warmup}")
            t0 = time.perf_counter()
            sim = TwoCountryDynamicSimulator(
                params,
                equilibrium,
                theta_price=float(theta_price),
                normalize_gap_by_supply=bool(normalize_gap_by_supply),
            )

            nan_step = -1
            steps_done = 0
            while steps_done < int(warmup):
                run_n = min(int(chunk_size), int(warmup) - steps_done)
                sim.run(run_n)
                steps_done += run_n
                if _has_nan_or_inf(sim):
                    nan_step = steps_done
                    break

            elapsed = time.perf_counter() - t0
            if nan_step > 0:
                result = {
                    "theta_price": float(theta_price),
                    "warmup": int(warmup),
                    "stable": False,
                    "nan_step": int(nan_step),
                    "elapsed_s": round(elapsed, 2),
                }
                print(f"  unstable at step {nan_step} ({elapsed:.1f}s)")
                results.append(result)
                continue

            gap = compute_gap_metrics(sim)
            price = check_price_convergence(sim, last_n=last_n)
            max_gap_H = float(gap["H"]["max_rel_gap"])
            max_gap_F = float(gap["F"]["max_rel_gap"])
            mean_gap = (float(gap["H"]["mean_rel_gap"]) + float(gap["F"]["mean_rel_gap"])) / 2.0
            max_gap = max(max_gap_H, max_gap_F)
            sector_price_cv_max = max(float(price["H_price_cv_max"]), float(price["F_price_cv_max"]))
            price_index_cv_max = max(float(price["H_price_index_cv"]), float(price["F_price_index_cv"]))

            result = {
                "theta_price": float(theta_price),
                "warmup": int(warmup),
                "stable": True,
                "max_gap": round(max_gap, 8),
                "mean_gap": round(mean_gap, 8),
                "max_gap_H": round(max_gap_H, 8),
                "max_gap_F": round(max_gap_F, 8),
                "gap_H_per_sector": _round_list(gap["H"]["rel_gap"], 6),
                "gap_F_per_sector": _round_list(gap["F"]["rel_gap"], 6),
                "sector_price_cv_max": round(sector_price_cv_max, 8),
                "price_index_cv_max": round(price_index_cv_max, 8),
                "H_price_index_cv": round(float(price["H_price_index_cv"]), 8),
                "F_price_index_cv": round(float(price["F_price_index_cv"]), 8),
                "H_final_price": _round_list(price["H_final_price"], 6),
                "F_final_price": _round_list(price["F_final_price"], 6),
                "elapsed_s": round(elapsed, 2),
            }
            print(
                "  max_gap={:.2%}, mean_gap={:.2%}, sector_price_cv={:.6f}, "
                "price_index_cv={:.6f} ({:.1f}s)".format(
                    max_gap,
                    mean_gap,
                    sector_price_cv_max,
                    price_index_cv_max,
                    elapsed,
                )
            )
            results.append(result)

    stable = [r for r in results if r.get("stable")]
    stable.sort(
        key=lambda r: (
            float(r.get("max_gap", float("inf"))),
            float(r.get("sector_price_cv_max", float("inf"))),
            int(r.get("warmup", 0)),
        )
    )
    return {
        "home_id": home_id,
        "foreign_id": foreign_id,
        "tradable_sectors": list(tradable_sectors) if tradable_sectors else list(range(params.home.alpha.shape[0])),
        "normalize_gap_by_supply": bool(normalize_gap_by_supply),
        "last_n": int(last_n),
        "equilibrium_info": eq_info,
        "results": results,
        "best": stable[0] if stable else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home-id", default="CN2017")
    parser.add_argument("--foreign-id", default="US2017")
    parser.add_argument("--theta-values", default="0.0001,0.0002,0.0005,0.001,0.002")
    parser.add_argument("--warmup-values", default="1000,2000,5000,10000")
    parser.add_argument("--tradable-sectors", default="")
    parser.add_argument("--last-n", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-solver-iter", type=int, default=400)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--no-normalize-gap-by-supply", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results_trade_game_param_search" / "param_search_results.json"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    tradable = _parse_int_list(args.tradable_sectors) if args.tradable_sectors.strip() else None
    payload = run_search(
        home_id=args.home_id,
        foreign_id=args.foreign_id,
        theta_values=_parse_float_list(args.theta_values),
        warmup_values=_parse_int_list(args.warmup_values),
        tradable_sectors=tradable,
        normalize_gap_by_supply=not bool(args.no_normalize_gap_by_supply),
        last_n=args.last_n,
        chunk_size=args.chunk_size,
        max_solver_iter=args.max_solver_iter,
        solver_tol=args.solver_tol,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output}")
    if payload.get("best"):
        best = payload["best"]
        print(
            "Best: theta={theta_price:g}, warmup={warmup}, max_gap={max_gap:.2%}, "
            "sector_price_cv={sector_price_cv_max:.6f}, price_index_cv={price_index_cv_max:.6f}".format(**best)
        )


if __name__ == "__main__":
    main()
