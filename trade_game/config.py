"""贸易博弈实验配置对象。

这里集中定义梯度博弈、LLM 博弈、约束、目标函数和数据源的最小配置结构。
这些 dataclass 是运行入口之间共享的稳定接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


Country = Literal["H", "F"]


@dataclass(frozen=True)
class TriggerConfig:
    """Initial policy shock used to start a game."""

    country: Country = "H"
    tariff: Dict[int, float] = field(default_factory=dict)
    quota: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 0.01
    iterations: int = 50
    optimizer: Literal["Adam", "SGD"] = "Adam"
    multi_start: int = 1
    start_strategy: Literal["current", "noisy_current", "random"] = "current"
    start_noise: float = 0.05
    seed: int = 42
    select: Literal["sum", "min", "H", "F"] = "sum"


@dataclass(frozen=True)
class ConstraintsConfig:
    active_sectors: List[int] = field(default_factory=lambda: [0, 1])
    reciprocal_coeff: float = 0.0
    max_tariff: float = 1.0
    min_quota: float = 0.0


@dataclass(frozen=True)
class ObjectiveConfig:
    type: Literal["standard", "relative"] = "standard"
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ParamsSource:
    """Experiment parameter source."""

    type: Literal["symmetric", "io_final"] = "symmetric"
    home_id: str = "CN2017"
    foreign_id: str = "US2017"
    tradable_sectors: Optional[List[int]] = None


@dataclass(frozen=True)
class GameConfig:
    """Gradient game experiment configuration."""

    name: str = "grad_simultaneous"
    params_source: ParamsSource = ParamsSource()
    theta_price: float = 12500.0
    normalize_gap_by_supply: bool = False
    rounds: int = 10
    decision_interval: int = 10
    lookahead_periods: int = 10
    warmup_periods: int = 10
    trigger_settle_periods: int = 10
    trigger: Optional[TriggerConfig] = None
    opt_config: OptimizationConfig = OptimizationConfig()
    constraints: ConstraintsConfig = ConstraintsConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    plot: bool = True
    output_dir: str = "results_grad"


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible LLM client configuration."""

    model: str = "qwen-plus"
    preset: Literal["qwen", "openai", "deepseek"] = "qwen"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 2048


@dataclass(frozen=True)
class LLMGameConfig:
    """LLM game experiment configuration."""

    name: str = "llm_game"
    params_source: ParamsSource = ParamsSource()
    theta_price: float = 0.1
    normalize_gap_by_supply: bool = True
    rounds: int = 10
    decision_interval: int = 10
    lookahead_periods: int = 10
    warmup_periods: int = 1000
    trigger_settle_periods: int = 0
    trigger: Optional[TriggerConfig] = None
    constraints: ConstraintsConfig = ConstraintsConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    llm: LLMConfig = LLMConfig()
    llm_plays: Literal["H", "F", "both"] = "both"
    non_llm_strategy: Literal["fixed", "gradient"] = "fixed"
    opt_config: OptimizationConfig = OptimizationConfig()
    plot: bool = True
    output_dir: str = "results_llm"


__all__ = [
    "Country",
    "ConstraintsConfig",
    "GameConfig",
    "LLMConfig",
    "LLMGameConfig",
    "ObjectiveConfig",
    "OptimizationConfig",
    "ParamsSource",
    "TriggerConfig",
]
