"""LLM 策略代理。

封装状态提取、prompt 构建、LLM 调用、策略解析的完整流程。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from .llm_client import LLMClient, LLMResponse
from .prompts import PromptBuilder, EconomicContext

logger = logging.getLogger(__name__)

Country = Literal["H", "F"]


@dataclass
class PolicyDecision:
    """策略决策结果。"""
    tariff: Dict[int, float]
    quota: Dict[int, float]
    reasoning: str = ""
    raw_response: str = ""
    llm_reasoning_content: Optional[str] = None  # 思考模型的推理过程
    # 新增：完整的输入输出记录
    system_prompt: str = ""
    user_prompt: str = ""
    token_usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": x, "completion_tokens": y}


class LLMPolicyAgent:
    """LLM 策略生成代理。"""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 2048,
        max_retries: int = 3,
    ):
        """初始化 LLM 策略代理。
        
        Args:
            llm_client: LLM 客户端实例
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            max_retries: 解析失败时的最大重试次数
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 响应中提取 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 markdown 代码块中的 JSON
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # 尝试提取花括号包围的内容
        brace_pattern = r"\{[\s\S]*\}"
        match = re.search(brace_pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return None

    def _validate_and_clamp_policy(
        self,
        policy: Dict[str, Any],
        active_sectors: List[int],
        max_tariff: float,
        min_quota: float,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """校验并裁剪策略到约束范围内。"""
        tariff = {}
        quota = {}
        
        raw_tariff = policy.get("tariff", {})
        raw_quota = policy.get("quota", {})
        
        for sector in active_sectors:
            # 解析 tariff
            t_val = raw_tariff.get(str(sector)) or raw_tariff.get(sector, 0.0)
            try:
                t_val = float(t_val)
            except (ValueError, TypeError):
                t_val = 0.0
            tariff[sector] = max(0.0, min(t_val, max_tariff))
            
            # 解析 quota
            q_val = raw_quota.get(str(sector)) or raw_quota.get(sector, 1.0)
            try:
                q_val = float(q_val)
            except (ValueError, TypeError):
                q_val = 1.0
            quota[sector] = max(min_quota, min(q_val, 1.0))
        
        return tariff, quota

    def _build_context(
        self,
        sim,  # TwoCountryDynamicSimulator
        country: Country,
        round_num: int,
        opponent_prev_tariff: Dict[int, float],
        opponent_prev_quota: Dict[int, float],
        current_tariff: Dict[int, float],
        current_quota: Dict[int, float],
        active_sectors: List[int],
        max_tariff: float,
        min_quota: float,
        reciprocal_coeff: float,
    ) -> EconomicContext:
        """从仿真器构建经济上下文。"""
        from .prompts import SectorContext
        
        opponent: Country = "F" if country == "H" else "H"
        
        summary = sim.summarize_history()
        
        # 获取最新状态
        self_summary = summary[country]
        opp_summary = summary[opponent]
        
        # 提取逐部门数据
        sectors = []
        try:
            # 获取当前状态的详细数据
            detailed = sim.get_detailed_history(country, start_period=-1)
            if detailed:
                latest = detailed[-1]
                for sid, sr in latest.sectors.items():
                    sectors.append(SectorContext(
                        sector_id=sid,
                        price=sr.price,
                        output=sr.output,
                        demand=sr.total_demand,
                        supply_demand_gap=sr.supply_demand_gap,
                        import_volume=sr.import_volume,
                        export_volume=sr.export_actual,
                    ))
        except Exception:
            pass  # 如果获取失败，使用空列表
        
        # 获取历史摘要
        history_summary = ""
        try:
            history_summary = sim.get_recent_history_summary(country, num_periods=5)
        except Exception:
            pass
        
        return EconomicContext(
            country=country,
            opponent=opponent,
            round_num=round_num,
            income=float(self_summary["income"][-1]),
            trade_balance=float(self_summary["trade_balance_val"][-1]),
            price_mean=float(self_summary["price_mean"][-1]),
            opponent_income=float(opp_summary["income"][-1]),
            opponent_trade_balance=float(opp_summary["trade_balance_val"][-1]),
            opponent_prev_tariff=opponent_prev_tariff,
            opponent_prev_quota=opponent_prev_quota,
            current_tariff=current_tariff,
            current_quota=current_quota,
            active_sectors=active_sectors,
            max_tariff=max_tariff,
            min_quota=min_quota,
            reciprocal_coeff=reciprocal_coeff,
            sectors=sectors,
            history_summary=history_summary,
        )

    def decide(
        self,
        sim,  # TwoCountryDynamicSimulator
        country: Country,
        round_num: int,
        opponent_prev_tariff: Dict[int, float],
        opponent_prev_quota: Dict[int, float],
        current_tariff: Dict[int, float],
        current_quota: Dict[int, float],
        active_sectors: List[int],
        max_tariff: float = 1.0,
        min_quota: float = 0.0,
        reciprocal_coeff: float = 0.0,
    ) -> PolicyDecision:
        """为指定国家生成策略决策。
        
        Args:
            sim: 两国动态仿真器
            country: 决策国家 ("H" 或 "F")
            round_num: 当前回合数
            opponent_prev_tariff: 对手上回合关税
            opponent_prev_quota: 对手上回合配额
            current_tariff: 本方当前关税
            current_quota: 本方当前配额
            active_sectors: 可调整的行业列表
            max_tariff: 关税上限
            min_quota: 配额下限
            reciprocal_coeff: 对等约束系数
            
        Returns:
            PolicyDecision 包含 tariff 和 quota 字典
        """
        ctx = self._build_context(
            sim=sim,
            country=country,
            round_num=round_num,
            opponent_prev_tariff=opponent_prev_tariff,
            opponent_prev_quota=opponent_prev_quota,
            current_tariff=current_tariff,
            current_quota=current_quota,
            active_sectors=active_sectors,
            max_tariff=max_tariff,
            min_quota=min_quota,
            reciprocal_coeff=reciprocal_coeff,
        )
        
        system_prompt, user_prompt = PromptBuilder.build_prompts(ctx)
        
        for attempt in range(self.max_retries):
            try:
                gen_kwargs: Dict[str, Any] = {
                    "prompt": user_prompt,
                    "system_prompt": system_prompt,
                }
                if self.temperature is not None:
                    gen_kwargs["temperature"] = float(self.temperature)
                if self.max_tokens is not None:
                    gen_kwargs["max_tokens"] = int(self.max_tokens)
                response = self.llm_client.generate(**gen_kwargs)
                
                parsed = self._extract_json(response.content)
                if parsed is None and response.reasoning_content:
                    parsed = self._extract_json(response.reasoning_content)
                if parsed is None:
                    logger.warning(f"Attempt {attempt + 1}: Failed to parse JSON from LLM response")
                    continue
                
                tariff, quota = self._validate_and_clamp_policy(
                    parsed,
                    active_sectors=active_sectors,
                    max_tariff=max_tariff,
                    min_quota=min_quota,
                )
                
                # 提取 token 使用信息
                token_usage = None
                if response.usage:
                    token_usage = {
                        "prompt_tokens": response.usage.get("prompt_tokens", 0),
                        "completion_tokens": response.usage.get("completion_tokens", 0),
                        "total_tokens": response.usage.get("total_tokens", 0),
                    }
                
                return PolicyDecision(
                    tariff=tariff,
                    quota=quota,
                    reasoning=parsed.get("reasoning", ""),
                    raw_response=response.content,
                    llm_reasoning_content=response.reasoning_content,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    token_usage=token_usage,
                )
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: LLM call failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        # 如果所有重试都失败，返回保守策略
        logger.warning("All retries exhausted, returning conservative policy")
        return PolicyDecision(
            tariff={s: 0.1 for s in active_sectors},
            quota={s: 0.9 for s in active_sectors},
            reasoning="Fallback: LLM 响应解析失败，采用保守策略",
            raw_response="",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
