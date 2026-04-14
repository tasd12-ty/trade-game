"""策略生成 Prompt 模板。

构建包含经济状态、历史策略、约束条件的结构化 prompt。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class SectorContext:
    """单个部门的状态上下文。"""
    sector_id: int
    price: float
    output: float
    demand: float
    supply_demand_gap: float
    import_volume: float
    export_volume: float


@dataclass
class EconomicContext:
    """经济状态上下文。"""
    country: Literal["H", "F"]
    opponent: Literal["H", "F"]
    round_num: int
    
    # 当前状态
    income: float
    trade_balance: float
    price_mean: float
    
    # 对手信息
    opponent_income: float
    opponent_trade_balance: float
    opponent_prev_tariff: Dict[int, float]
    opponent_prev_quota: Dict[int, float]
    
    # 本方当前政策
    current_tariff: Dict[int, float]
    current_quota: Dict[int, float]
    
    # 约束
    active_sectors: List[int]
    max_tariff: float
    min_quota: float
    reciprocal_coeff: float
    
    # 新增：逐部门数据
    sectors: List[SectorContext] = field(default_factory=list)
    
    # 新增：历史摘要
    history_summary: str = ""


class PromptBuilder:
    """Prompt 构建器。"""

    SYSTEM_PROMPT = """你是一个经济贸易政策专家 AI，负责为国家 {country} 制定贸易政策。

## 你的目标
最大化本国的经济效用，包括：
1. **收入增长** - 提高国民收入
2. **贸易差额** - 改善贸易收支
3. **价格稳定** - 维持物价水平

## 政策工具
- **关税率 (tariff)**: 对进口商品征收的税率，范围 [0, {max_tariff}]
  - 较高关税可保护本国产业，但可能引发对方报复
- **出口配额乘数 (quota)**: 限制出口的比例，范围 [{min_quota}, 1.0]
  - 较低配额可作为谈判筹码，但会减少出口收入

## 博弈环境
这是一个两国同步决策博弈，双方在每回合同时做出决策。
对手也在追求其自身利益，可能采取报复性措施。

## 输出格式
请严格按以下 JSON 格式输出，不要包含任何其他内容：
```json
{{
  "reasoning": "简要说明决策理由（1-2句话）",
  "tariff": {{"行业编号": 关税率, ...}},
  "quota": {{"行业编号": 配额乘数, ...}}
}}
```
"""

    USER_PROMPT_TEMPLATE = """## 当前状态（第 {round_num} 回合）

### 本国 ({country}) 经济指标
- 国民收入: {income}
- 贸易差额: {trade_balance}
- 平均物价指数: {price_mean}
- 当前关税: {current_tariff}
- 当前出口配额: {current_quota}

### 对手国 ({opponent}) 信息
- 国民收入: {opponent_income}
- 贸易差额: {opponent_trade_balance}
- 上回合关税: {opponent_prev_tariff}
- 上回合出口配额: {opponent_prev_quota}

{sector_data}

{history_section}

### 约束条件
- 可调整的行业编号: {active_sectors}
- 关税上限: {max_tariff}
- 配额下限: {min_quota}
{reciprocal_note}

请为本国制定本回合的贸易政策。"""

    RECIPROCAL_NOTE = "- 对等约束系数: {coeff} (本方关税不得低于对方上回合关税的 {coeff} 倍)"
    
    SECTOR_DATA_TEMPLATE = """### 本国各部门状态
| 部门 | 价格 | 产出 | 需求 | 供需缺口 | 进口 | 出口 |
|------|------|------|------|----------|------|------|
{sector_rows}"""

    @classmethod
    def _format_sector_data(cls, sectors: List[SectorContext], active_sectors: List[int]) -> str:
        """格式化部门数据表格。"""
        if not sectors:
            return ""
        
        rows = []
        for s in sectors:
            # 标记活跃部门
            marker = "⭐" if s.sector_id in active_sectors else ""
            gap_sign = "+" if s.supply_demand_gap > 0 else ""
            rows.append(
                f"| {s.sector_id}{marker} | {s.price} | {s.output} | "
                f"{s.demand} | {gap_sign}{s.supply_demand_gap} | "
                f"{s.import_volume} | {s.export_volume} |"
            )
        
        return cls.SECTOR_DATA_TEMPLATE.format(sector_rows="\n".join(rows))

    @classmethod
    def build_system_prompt(
        cls,
        country: str,
        max_tariff: float = 1.0,
        min_quota: float = 0.0,
    ) -> str:
        """构建系统 prompt。"""
        return cls.SYSTEM_PROMPT.format(
            country=country,
            max_tariff=max_tariff,
            min_quota=min_quota,
        )

    @classmethod
    def build_user_prompt(cls, ctx: EconomicContext) -> str:
        """构建用户 prompt。"""
        reciprocal_note = ""
        if ctx.reciprocal_coeff > 0:
            reciprocal_note = cls.RECIPROCAL_NOTE.format(coeff=ctx.reciprocal_coeff)
        
        sector_data = cls._format_sector_data(ctx.sectors, ctx.active_sectors)
        
        history_section = ""
        if ctx.history_summary:
            history_section = f"### 历史数据\n{ctx.history_summary}"

        return cls.USER_PROMPT_TEMPLATE.format(
            round_num=ctx.round_num,
            country=ctx.country,
            income=ctx.income,
            trade_balance=ctx.trade_balance,
            price_mean=ctx.price_mean,
            current_tariff=ctx.current_tariff,
            current_quota=ctx.current_quota,
            opponent=ctx.opponent,
            opponent_income=ctx.opponent_income,
            opponent_trade_balance=ctx.opponent_trade_balance,
            opponent_prev_tariff=ctx.opponent_prev_tariff,
            opponent_prev_quota=ctx.opponent_prev_quota,
            sector_data=sector_data,
            history_section=history_section,
            active_sectors=ctx.active_sectors,
            max_tariff=ctx.max_tariff,
            min_quota=ctx.min_quota,
            reciprocal_note=reciprocal_note,
        )

    @classmethod
    def build_prompts(cls, ctx: EconomicContext) -> tuple[str, str]:
        """构建完整的 system + user prompt。
        
        Returns:
            (system_prompt, user_prompt)
        """
        system = cls.build_system_prompt(
            country=ctx.country,
            max_tariff=ctx.max_tariff,
            min_quota=ctx.min_quota,
        )
        user = cls.build_user_prompt(ctx)
        return system, user
