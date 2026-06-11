"""系统设置配置"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class SystemSettings(BaseModel):
    """系统设置"""
    
    # 项目根目录
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="项目根目录"
    )
    
    # 数据目录
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        description="数据目录"
    )
    
    # 上传目录
    UPLOAD_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "uploads",
        description="图片上传目录"
    )
    
    # 输出目录
    OUTPUT_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "outputs",
        description="合成数据输出目录"
    )
    
    # 数据库路径
    DATABASE_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "synthesis.db",
        description="SQLite数据库路径"
    )
    
    # 最大加载任务数
    MAX_LOAD_TASKS: int = Field(default=100, description="页面刷新时最大加载任务数")
    
    # 日志目录
    LOG_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs",
        description="日志目录"
    )
    
    # 日志级别
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    
    # 最大迭代次数（每个任务的提问数量）
    MAX_ITERATIONS: int = Field(default=5, description="每个任务生成的问题数量（每个问题包含1标准答案+1正样本+1负样本+N策略样本）")

    # 难度范围（随机难度）
    MIN_DIFFICULTY: float = Field(default=0.1, description="随机难度下限")
    MAX_DIFFICULTY: float = Field(default=1.0, description="随机难度上限")
    
    # 验证通过阈值
    VALIDATION_THRESHOLD: float = Field(default=0.8, description="验证通过的语义相似度阈值")

    # 模型调用重试次数
    MAX_RETRIES: int = Field(default=5, ge=1, le=10, description="模型调用 + JSON 解析重试次数")

    # 负样本生成比例
    NEGATIVE_SAMPLE_RATIO: float = Field(default=0.5, ge=0.0, le=1.0, description="负样本生成比例（0-1之间，默认0.5表示50%的样本为负样本）")

    # 策略模型采样配置
    STRATEGY_SAMPLE_COUNT: int = Field(default=3, ge=1, le=10, description="策略模型(qwen3-8b)每个问题的采样次数")

    # Execution & Performance Settings
    PARALLEL_TASK_COUNT: int = Field(
        default=3,
        ge=1,
        le=10,
        description="并行任务数量（同时运行的合成任务数）"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        # 创建必要的目录
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    class Config:
        arbitrary_types_allowed = True


# 全局设置实例
settings = SystemSettings()
