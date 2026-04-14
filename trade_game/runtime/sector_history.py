"""逐部门历史数据记录。

提供详细的经济状态快照，用于 LLM 决策和分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import numpy as np


@dataclass
class SectorRecord:
    """单期单部门的完整记录。"""
    sector_id: int
    price: float              # 部门价格
    output: float             # 产出 (供给)
    demand_domestic: float    # 国内需求 (中间品 + 消费)
    demand_export: float      # 出口需求
    total_demand: float       # 总需求
    supply_demand_gap: float  # 供需缺口 (demand - supply)
    import_volume: float      # 进口量
    import_price: float       # 进口价格
    export_actual: float      # 实际出口量
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_id": self.sector_id,
            "price": round(self.price, 6),
            "output": round(self.output, 6),
            "demand_domestic": round(self.demand_domestic, 6),
            "demand_export": round(self.demand_export, 6),
            "total_demand": round(self.total_demand, 6),
            "supply_demand_gap": round(self.supply_demand_gap, 6),
            "import_volume": round(self.import_volume, 6),
            "import_price": round(self.import_price, 6),
            "export_actual": round(self.export_actual, 6),
        }


@dataclass
class PeriodRecord:
    """单期的完整经济状态。"""
    period: int
    country: str
    
    # 逐部门数据
    sectors: Dict[int, SectorRecord] = field(default_factory=dict)
    
    # 聚合指标
    total_income: float = 0.0
    total_output: float = 0.0
    trade_balance: float = 0.0
    price_mean: float = 0.0
    
    # 当期政策状态
    tariff_rates: Dict[int, float] = field(default_factory=dict)
    quota_rates: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "country": self.country,
            "sectors": {k: v.to_dict() for k, v in self.sectors.items()},
            "total_income": round(self.total_income, 6),
            "total_output": round(self.total_output, 6),
            "trade_balance": round(self.trade_balance, 6),
            "price_mean": round(self.price_mean, 6),
            "tariff_rates": {k: round(v, 4) for k, v in self.tariff_rates.items()},
            "quota_rates": {k: round(v, 4) for k, v in self.quota_rates.items()},
        }
    
    def summary_str(self, max_sectors: Optional[int] = None) -> str:
        """生成简洁的摘要字符串，用于 LLM prompt。"""
        lines = [
            f"期数 {self.period} | 收入: {self.total_income} | 贸易差额: {self.trade_balance} | 均价: {self.price_mean}"
        ]
        
        sector_ids = sorted(self.sectors.keys())
        if max_sectors is not None:
            sector_ids = sector_ids[: int(max_sectors)]
        sector_strs = []
        for sid in sector_ids:
            s = self.sectors[sid]
            gap_sign = "+" if s.supply_demand_gap > 0 else ""
            sector_strs.append(
                f"  部门{sid}: P={s.price}, Y={s.output}, D={s.total_demand}, Gap={gap_sign}{s.supply_demand_gap}"
            )
        if sector_strs:
            lines.extend(sector_strs)
        
        return "\n".join(lines)


def extract_sector_records(
    state,  # CountryState
    n_sectors: int,
    batch_index: int = 0,
) -> Dict[int, SectorRecord]:
    """从 CountryState 提取逐部门记录。"""
    
    def _pick(t: torch.Tensor) -> torch.Tensor:
        """处理 batch 维度：3D -> 2D, 2D -> 1D, or keep 1D/0D as is."""
        if t.dim() == 3:
            return t[batch_index]  # (B, n, n) -> (n, n)
        elif t.dim() == 2 and t.shape[0] > 1 and t.shape[0] == t.shape[1]:
            # 非 batch 的 2D 矩阵，保持原样
            return t
        elif t.dim() == 2:
            return t[batch_index]  # (B, n) -> (n,)
        return t  # 1D or 0D
    
    records = {}
    
    price = _pick(state.price).detach().cpu()
    output = _pick(state.output).detach().cpu()
    X_dom = _pick(state.X_dom).detach().cpu()  # (n, n)
    C_dom = _pick(state.C_dom).detach().cpu()  # (n,)
    X_imp = _pick(state.X_imp).detach().cpu()  # (n, n)
    C_imp = _pick(state.C_imp).detach().cpu()  # (n,)
    export_base = _pick(state.export_base).detach().cpu()
    export_actual = _pick(state.export_actual).detach().cpu()
    imp_price = _pick(state.imp_price).detach().cpu()
    
    for j in range(n_sectors):
        # 国内需求 = 被其他部门使用的中间品 + 消费
        if X_dom.dim() == 2:
            demand_domestic = float(X_dom[:, j].sum() + C_dom[j])
            import_vol = float(X_imp[:, j].sum() + C_imp[j])
        else:
            # 1D tensor（不常见，但处理边缘情况）
            demand_domestic = float(C_dom[j]) if j < len(C_dom) else 0.0
            import_vol = float(C_imp[j]) if j < len(C_imp) else 0.0
        
        demand_export = float(export_base[j])
        total_demand = demand_domestic + demand_export
        supply = float(output[j])
        gap = total_demand - supply
        
        records[j] = SectorRecord(
            sector_id=j,
            price=float(price[j]),
            output=supply,
            demand_domestic=demand_domestic,
            demand_export=demand_export,
            total_demand=total_demand,
            supply_demand_gap=gap,
            import_volume=import_vol,
            import_price=float(imp_price[j]),
            export_actual=float(export_actual[j]),
        )
    
    return records


def build_period_record(
    state,  # CountryState
    country: str,
    period: int,
    n_sectors: int,
    tariff_rates: Optional[Dict[int, float]] = None,
    quota_rates: Optional[Dict[int, float]] = None,
    batch_index: int = 0,
) -> PeriodRecord:
    """构建完整的期数记录。"""
    
    def _pick(t: torch.Tensor) -> torch.Tensor:
        if t.dim() <= 1:
            return t
        return t[batch_index]
    
    sectors = extract_sector_records(state, n_sectors, batch_index)
    
    price = _pick(state.price).detach().cpu()
    output = _pick(state.output).detach().cpu()
    income = _pick(state.income).detach().cpu()
    export_actual = _pick(state.export_actual).detach().cpu()
    imp_price = _pick(state.imp_price).detach().cpu()
    X_imp = _pick(state.X_imp).detach().cpu()
    C_imp = _pick(state.C_imp).detach().cpu()
    
    export_value = float((export_actual * price).sum())
    import_value = float((imp_price * (X_imp.sum(dim=0) + C_imp)).sum())
    trade_balance = export_value - import_value
    
    return PeriodRecord(
        period=period,
        country=country,
        sectors=sectors,
        total_income=float(income),
        total_output=float(output.sum()),
        trade_balance=trade_balance,
        price_mean=float(price.mean()),
        tariff_rates=tariff_rates or {},
        quota_rates=quota_rates or {},
    )
