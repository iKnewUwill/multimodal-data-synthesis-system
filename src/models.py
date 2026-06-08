"""数据模型定义"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class TaskType(str, Enum):
    """任务类型枚举"""
    FINANCIAL_QA = "金融财务问答"


class FileType(str, Enum):
    """文件类型枚举"""
    IMAGE = "image"
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    EXCEL = "excel"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "待处理"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "处理失败"


class FileInfo(BaseModel):
    """文件信息（通用）"""
    path: str = Field(..., description="文件路径")
    filename: str = Field(..., description="文件名")
    file_type: FileType = Field(..., description="文件类型")
    file_size: int = Field(default=0, description="文件大小（字节）")
    content: Optional[str] = Field(None, description="文件内容（文本类文件）")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")


class ImageInfo(BaseModel):
    """图片信息（保持向后兼容）"""
    path: str = Field(..., description="图片路径")
    filename: str = Field(..., description="图片文件名")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")


class FinancialTaskInput(BaseModel):
    """金融财务任务输入"""
    task_id: str = Field(default_factory=lambda: f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    证券代码: str = Field(..., description="证券代码")
    公司名称: str = Field(..., description="公司名称")
    统计截止日期: str = Field(..., description="统计截止日期")
    评估维度: str = Field(..., description="评估维度")
    financial_data: Dict[str, Any] = Field(..., description="financial_data数据")
    is_positive_sample: bool = Field(default=True, description="是否为正样本（True=正样本, False=负样本/错误注入）")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class QAPair(BaseModel):
    """问答对（保持向后兼容）"""
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="难度等级")
    iteration: int = Field(..., ge=1, description="所属迭代轮次")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FinancialQAResult(BaseModel):
    """金融财务问答结果（保持向后兼容）"""
    question: str = Field(..., description="问题")
    analysis_process: Dict[str, Any] = Field(..., description="分析过程")
    conclusion: str = Field(..., description="分析结论")
    difficulty: float = Field(..., ge=0.0, le=1.0)
    iteration: int
    is_positive_sample: bool = Field(default=True, description="是否为正样本")
    sample_source: str = Field(default="positive_solver", description="样本来源：positive_solver/negative_solver/strategy_model")
    created_at: datetime = Field(default_factory=datetime.now)


class ValidationResult(BaseModel):
    """验证结果"""
    is_valid: bool = Field(..., description="是否通过验证")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="语义相似度分数")
    reason: str = Field(..., description="验证理由")


class ProposerOutput(BaseModel):
    """提议者输出 - 问题 + 标准答案"""
    question: str = Field(..., description="生成的问题")
    answer: str = Field(..., description="参考答案/结论")
    analysis_process: Dict[str, str] = Field(default_factory=dict, description="分析过程（分步骤）")
    conclusion: str = Field(default="", description="分析结论")


class SolverOutput(BaseModel):
    """求解者输出 - 金融财务分析"""
    answer: str = Field(..., description="预测答案/结论")
    analysis_process: Dict[str, str] = Field(default_factory=dict, description="分析过程（分步骤）")
    conclusion: str = Field(default="", description="分析结论")


class StrategySampleResult(BaseModel):
    """策略模型采样结果 - qwen3-8b 单次自然推理输出"""
    sample_index: int = Field(..., description="采样序号（1-based）")
    solver_output: SolverOutput = Field(..., description="策略模型推理输出")
    validation: Optional[ValidationResult] = Field(None, description="采样验证结果")
    is_valid: bool = Field(default=False, description="是否被标注为正样本")


# === 新输出结构 ===

class SampleWithValidation(BaseModel):
    """单个带验证的样本"""
    sample_source: str = Field(..., description="样本来源：positive_solver / negative_solver / strategy_model")
    is_positive_sample: bool = Field(..., description="是否为正样本")
    conclusion: str = Field(default="", description="分析结论")
    analysis_process: Dict[str, str] = Field(default_factory=dict, description="分析过程")
    validation: Optional[ValidationResult] = Field(None, description="验证结果")
    sample_index: int = Field(default=1, description="采样序号（策略样本用）")


class QuestionSampleSet(BaseModel):
    """一个问题对应的完整样本集

    结构：1个标准答案 + 1个正样本 + 1个负样本 + N个策略模型样本
    """
    difficulty: float = Field(..., description="随机难度等级")
    question: str = Field(..., description="分析问题")
    # 标准答案（由 Proposer 与问题一同生成）
    standard_answer: str = Field(..., description="标准答案结论")
    standard_analysis_process: Dict[str, str] = Field(default_factory=dict, description="标准答案分析过程")
    # 正样本（qwen3.7-max 求解者）
    positive_sample: Optional[SampleWithValidation] = Field(None, description="正样本")
    # 负样本（qwen3.7-max 负样本求解者·错误注入）
    negative_sample: Optional[SampleWithValidation] = Field(None, description="负样本")
    # 策略模型样本（qwen3-8b 自然推理）
    strategy_samples: List[SampleWithValidation] = Field(default_factory=list, description="策略模型采样结果")


class FinancialTaskResult(BaseModel):
    """金融财务任务结果"""
    task_id: str
    证券代码: str
    公司名称: str
    评估维度: str
    financial_data: Dict[str, Any]
    status: TaskStatus
    # 新结构：每个问题一个 QuestionSampleSet
    sample_sets: List[QuestionSampleSet] = Field(default_factory=list, description="每个问题的样本集")
    valid_qa_count: int = 0
    completed_at: Optional[datetime] = None


class IterationState(BaseModel):
    """迭代状态"""
    iteration: int = Field(..., description="当前迭代轮次")
    difficulty: float = Field(..., description="随机难度等级")
    proposed_qa: Optional[ProposerOutput] = Field(None, description="提议的问答对（问题+标准答案）")
    # 正样本求解者输出（qwen3.7-max）
    positive_solved_output: Optional[SolverOutput] = Field(None, description="正样本求解者的输出")
    positive_validation: Optional[ValidationResult] = Field(None, description="正样本验证结果")
    # 负样本求解者输出（qwen3.7-max）
    negative_solved_output: Optional[SolverOutput] = Field(None, description="负样本求解者的输出")
    negative_validation: Optional[ValidationResult] = Field(None, description="负样本验证结果")
    # 策略模型采样结果（qwen3-8b，多次采样）
    strategy_samples: list = Field(default_factory=list, description="策略模型采样结果列表")
    # 向后兼容
    solved_output: Optional[SolverOutput] = Field(None, description="求解的输出（向后兼容）")
    validation: Optional[ValidationResult] = Field(None, description="验证结果（向后兼容）")
    is_positive_sample: bool = Field(default=True, description="是否为正样本")
    status: str = Field(default="pending", description="状态")
    error: Optional[str] = Field(None, description="错误信息")


class SynthesisTask(BaseModel):
    """数据合成任务"""
    task_id: str = Field(..., description="任务ID")
    task_type: str = Field(..., description="任务类型")
    task_description: Optional[str] = Field(None, description="任务描述")
    max_iterations: int = Field(default=5, description="生成的问题数量")
    is_positive_sample: bool = Field(default=True, description="是否为正样本")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    # 金融财务相关字段
    证券代码: Optional[str] = Field(None, description="证券代码")
    公司名称: Optional[str] = Field(None, description="公司名称")
    评估维度: Optional[str] = Field(None, description="评估维度")
    financial_data: Optional[Dict[str, Any]] = Field(None, description="财务数据")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SynthesisResult(BaseModel):
    """合成结果（向后兼容）"""
    task_id: str = Field(..., description="任务ID")
    task_type: str = Field(..., description="任务类型")
    qa_pairs: List[QAPair] = Field(default_factory=list, description="生成的问答对")
    iterations: List[IterationState] = Field(default_factory=list, description="迭代状态")
    total_iterations: int = Field(default=0, description="总迭代次数")
    valid_qa_count: int = Field(default=0, description="有效问答对数量")
    completed_at: Optional[datetime] = Field(None, description="完成时间")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentState(BaseModel):
    """Agent 状态（用于 LangGraph）"""
    # 任务信息
    task: SynthesisTask

    # 收集结果（新结构）
    sample_sets: List[QuestionSampleSet] = Field(default_factory=list)

    # 历史问答对（向后兼容）
    history_qa_pairs: List[Any] = Field(default_factory=list)

    # 当前迭代
    current_iteration: int = Field(default=0)

    # 当前难度（随机）
    current_difficulty: float = Field(default=0.3)

    # 当前迭代状态
    current_state: Optional[IterationState] = None

    # 所有迭代状态
    all_iterations: List[IterationState] = Field(default_factory=list)

    # 是否完成
    is_finished: bool = Field(default=False)

    # 错误信息
    error: Optional[str] = Field(None)

    class Config:
        arbitrary_types_allowed = True
