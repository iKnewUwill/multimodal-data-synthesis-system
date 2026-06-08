"""LLM 配置 - 支持双模型（qwen3.7-max 主力 + qwen3-8b 策略采样）"""

import os
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class LLMConfig(BaseModel):
    """LLM 配置类 - 用于 qwen3.7-max（数据生成主力）"""

    # API 配置
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="API Key"
    )

    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        description="API Base URL"
    )

    # 模型配置
    model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", os.getenv("PRO_MODEL_NAME", "gpt-4-vision-preview")),
        description="使用的模型名称"
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="温度参数"
    )

    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="最大生成 token 数"
    )

    timeout: int = Field(
        default=60,
        ge=1,
        description="请求超时时间（秒）"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="最大重试次数"
    )

    # 按角色细分的温度设置（qwen3.7-max 各角色）
    proposer_temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="提议者温度（需要多样性）"
    )

    proposer_max_tokens: int = Field(
        default=2048,
        ge=1,
        description="提议者最大 token 数"
    )

    positive_solver_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="正样本求解者温度（需要稳定性）"
    )

    positive_solver_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="正样本求解者最大 token 数"
    )

    negative_solver_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="负样本求解者温度（需要多样化错误）"
    )

    negative_solver_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="负样本求解者最大 token 数"
    )

    validator_temperature: float = Field(
        default=0.05,
        ge=0.0,
        le=2.0,
        description="验证者温度（评判需要高度一致）"
    )

    validator_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="验证者最大 token 数"
    )

    def save_to_file(self, filepath: Path = None):
        """Save config to JSON"""
        filepath = filepath or Path("config/saved/llm_config.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.model_dump_json(indent=2, exclude_none=True), encoding='utf-8')

    @classmethod
    def load_from_file(cls, filepath: Path = None) -> "LLMConfig":
        """Load config from JSON"""
        filepath = filepath or Path("config/saved/llm_config.json")
        if filepath.exists():
            return cls.parse_file(filepath)
        return cls()


class StrategyLLMConfig(BaseModel):
    """策略模型配置 - 用于 qwen3-8b（自然推理采样，无错误注入引导）"""

    # API 配置（可独立配置不同端点）
    api_key: str = Field(
        default_factory=lambda: os.getenv("STRATEGY_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        description="策略模型 API Key"
    )

    base_url: str = Field(
        default_factory=lambda: os.getenv("STRATEGY_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        description="策略模型 API Base URL"
    )

    model_name: str = Field(
        default_factory=lambda: os.getenv("STRATEGY_LLM_MODEL_NAME", os.getenv("NORMAL_MODEL_NAME", "qwen3-8b")),
        description="策略模型名称"
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="策略模型采样温度（模拟真实推理多样性）"
    )

    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="策略模型最大生成 token 数"
    )

    timeout: int = Field(
        default=60,
        ge=1,
        description="请求超时时间（秒）"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="最大重试次数"
    )

    enable_thinking: bool = Field(
        default=False,
        description="是否启用思考模式（qwen3-8b 非流式需要设为 False）"
    )

    def get_extra_body(self) -> dict:
        """Get extra body params for API call"""
        return {"enable_thinking": self.enable_thinking} if not self.enable_thinking else {}

    def save_to_file(self, filepath: Path = None):
        """Save config to JSON"""
        filepath = filepath or Path("config/saved/strategy_llm_config.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.model_dump_json(indent=2, exclude_none=True), encoding='utf-8')

    @classmethod
    def load_from_file(cls, filepath: Path = None) -> "StrategyLLMConfig":
        """Load config from JSON"""
        filepath = filepath or Path("config/saved/strategy_llm_config.json")
        if filepath.exists():
            return cls.parse_file(filepath)
        return cls()


def get_llm_config() -> LLMConfig:
    """Get LLM config instance"""
    return LLMConfig.load_from_file()


def get_strategy_llm_config() -> StrategyLLMConfig:
    """Get strategy model config instance"""
    return StrategyLLMConfig.load_from_file()


llm_config = get_llm_config()  # Backward compatibility
strategy_llm_config = get_strategy_llm_config()
