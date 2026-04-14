"""LLM 客户端抽象层。

支持 OpenAI API 兼容的服务（包括 Qwen、DeepSeek 等国产模型）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应结构。"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    reasoning_content: Optional[str] = None  # 思考模型的推理过程


class LLMClient(ABC):
    """LLM 客户端抽象基类。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 2048,
        **kwargs,
    ) -> LLMResponse:
        """生成 LLM 响应。"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API 兼容客户端。
    
    支持：
    - OpenAI (gpt-4, gpt-3.5-turbo)
    - Qwen (qwen-plus, qwen-turbo, qwq-32b-preview)
    - DeepSeek (deepseek-chat, deepseek-reasoner)
    - 其他 OpenAI 兼容 API
    """

    # 预设的 API 配置
    PRESETS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "env_key": "DASHSCOPE_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "env_key": "DEEPSEEK_API_KEY",
        },
    }

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        preset: str = "qwen",
    ):
        """初始化 OpenAI 兼容客户端。

        Args:
            model: 模型名称
            api_key: API 密钥（可从环境变量读取）
            base_url: API 基础 URL（可从预设读取）
            preset: 预设配置 ("openai", "qwen", "deepseek")
        """
        self.model = model
        
        # 确定 base_url
        if base_url:
            self.base_url = base_url
        elif preset in self.PRESETS:
            self.base_url = self.PRESETS[preset]["base_url"]
        else:
            self.base_url = "https://api.openai.com/v1"
        
        # 确定 API key
        if api_key:
            self.api_key = api_key
        elif preset in self.PRESETS:
            self.api_key = os.environ.get(self.PRESETS[preset]["env_key"], "")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not self.api_key:
            logger.warning(f"No API key found for preset '{preset}'. Set the environment variable or pass api_key.")

        # 延迟导入 openai
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 2048,
        **kwargs,
    ) -> LLMResponse:
        """调用 LLM 生成响应。"""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            req: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                **kwargs,
            }
            if temperature is not None:
                req["temperature"] = float(temperature)
            if max_tokens is not None:
                req["max_tokens"] = int(max_tokens)

            response = self.client.chat.completions.create(**req)
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # 提取思考模型的推理内容（如 qwq, deepseek-reasoner）
            reasoning_content = None
            if hasattr(choice.message, "reasoning_content"):
                reasoning_content = choice.message.reasoning_content
            
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                reasoning_content=reasoning_content,
            )
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise


class MockLLMClient(LLMClient):
    """模拟 LLM 客户端，用于测试。"""

    def __init__(self, default_tariff: float = 0.2, default_quota: float = 0.8):
        self.default_tariff = default_tariff
        self.default_quota = default_quota

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """返回模拟的策略决策 JSON。"""
        # 从 prompt 中提取 active_sectors
        sectors = [0, 1]  # 默认
        active_match = re.search(r"可调整的行业编号.*?(\[[\d,\s]+\])", prompt)
        if active_match:
            try:
                sectors = json.loads(active_match.group(1))
            except:
                pass

        # 生成策略
        tariff = {str(s): round(self.default_tariff + 0.05 * s, 2) for s in sectors}
        quota = {str(s): round(self.default_quota - 0.05 * s, 2) for s in sectors}
        
        response_json = {
            "reasoning": "作为模拟响应，采用温和的贸易政策：适度关税保护本国产业，同时维持较高出口配额以保持贸易关系。",
            "tariff": tariff,
            "quota": quota,
        }
        
        return LLMResponse(
            content=json.dumps(response_json, ensure_ascii=False, indent=2),
            model="mock-llm",
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 100, "total_tokens": len(prompt) // 4 + 100},
        )
