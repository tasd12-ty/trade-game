"""默认实验场景。

这些配置用于命令行入口的开箱运行；需要复现实验时优先在这里或入口文件中显式修改参数。
"""

from __future__ import annotations

from .config import (
    ConstraintsConfig,
    GameConfig,
    LLMConfig,
    LLMGameConfig,
    ObjectiveConfig,
    OptimizationConfig,
    ParamsSource,
    TriggerConfig,
)


def default_grad_config() -> GameConfig:
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


def default_llm_config() -> LLMGameConfig:
    return LLMGameConfig(
        name="llm_vs_gradient",
        params_source=ParamsSource(type="symmetric"),
        normalize_gap_by_supply=True,
        theta_price=0.1,
        rounds=10,
        decision_interval=10,
        lookahead_periods=12,
        warmup_periods=1000,
        trigger_settle_periods=0,
        trigger=TriggerConfig(country="F", tariff={4: 0.5}),
        constraints=ConstraintsConfig(active_sectors=[2, 3], reciprocal_coeff=0, max_tariff=1.0, min_quota=0.0),
        objective=ObjectiveConfig(type="standard", weights=(1.0, 1.0, 1.0)),
        llm=LLMConfig(
            preset="deepseek",
            model="deepseek-chat",
        ),
        llm_plays="H",
        non_llm_strategy="gradient",
        opt_config=OptimizationConfig(
            learning_rate=0.01,
            iterations=200,
            optimizer="Adam",
            multi_start=8,
            start_strategy="noisy_current",
            select="sum",
        ),
        plot=True,
        output_dir="results_0117_llmH_strategyF_F",
    )


__all__ = ["default_grad_config", "default_llm_config"]
