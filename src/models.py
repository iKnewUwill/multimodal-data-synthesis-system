"""数据模型定义"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class TaskType(str, Enum):
    """任务类型枚举"""
    FINANCIAL_QA = "金融财务问答"
    # 以下为图片类任务（暂时注释，保留向后兼容）
    # IMAGE_DESCRIPTION = "图片描述类"
    # IMAGE_QA = "图片问答类"
    # MULTI_IMAGE_COMPARISON = "多图比较类"
    # VISUAL_REASONING = "视觉推理类"
    # DETAIL_RECOGNITION = "细节识别类"
    # SCENE_UNDERSTANDING = "场景理解类"
    # TEXT_RECOGNITION = "文字识别类"
    # COUNTING = "计数统计类"
    # DOCUMENT_QA = "文档问答类"
    # DATA_ANALYSIS = "数据分析类"
    # TEXT_SUMMARY = "文本摘要类"
    # CUSTOM = "自定义"


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
    关键指标: Dict[str, Any] = Field(..., description="关键指标数据")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class QAPair(BaseModel):
    """问答对"""
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
    """金融财务问答结果"""
    question: str = Field(..., description="问题")
    analysis_process: Dict[str, Any] = Field(..., description="分析过程")
    conclusion: str = Field(..., description="分析结论")
    difficulty: float = Field(..., ge=0.0, le=1.0)
    iteration: int
    created_at: datetime = Field(default_factory=datetime.now)

class FinancialTaskResult(BaseModel):
    """金融财务任务结果"""
    task_id: str
    证券代码: str
    公司名称: str
    评估维度: str
    status: TaskStatus
    qa_pairs: List[FinancialQAResult] = Field(default_factory=list)
    total_iterations: int = 0
    valid_qa_count: int = 0
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None


class ValidationResult(BaseModel):
    """验证结果"""
    is_valid: bool = Field(..., description="是否通过验证")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="语义相似度分数")
    reason: str = Field(..., description="验证理由")


class ProposerOutput(BaseModel):
    """提议者输出 - 金融财务分析"""
    question: str = Field(..., description="生成的问题")
    answer: str = Field(..., description="参考答案/结论")
    analysis_process: Dict[str, str] = Field(default_factory=dict, description="分析过程（分步骤）")
    conclusion: str = Field(default="", description="分析结论")


class SolverOutput(BaseModel):
    """求解者输出 - 金融财务分析"""
    answer: str = Field(..., description="预测答案/结论")
    analysis_process: Dict[str, str] = Field(default_factory=dict, description="分析过程（分步骤）")
    conclusion: str = Field(default="", description="分析结论")

class IterationState(BaseModel):
    """迭代状态"""
    iteration: int = Field(..., description="当前迭代轮次")
    difficulty: float = Field(..., description="当前难度等级")
    proposed_qa: Optional[ProposerOutput] = Field(None, description="提议的问答对")
    solved_output: Optional[SolverOutput] = Field(None, description="求解的输出")
    validation: Optional[ValidationResult] = Field(None, description="验证结果")
    status: str = Field(default="pending", description="状态：pending/proposing/solving/validating/completed/failed")
    error: Optional[str] = Field(None, description="错误信息")


class SynthesisTask(BaseModel):
    """数据合成任务"""
    task_id: str = Field(..., description="任务ID")
    task_type: str = Field(..., description="任务类型")
    task_description: Optional[str] = Field(None, description="任务描述")
    images: List[ImageInfo] = Field(default_factory=list, description="图片列表")
    files: List[FileInfo] = Field(default_factory=list, description="文件列表（通用）")
    max_iterations: int = Field(default=10, description="最大迭代次数")
    initial_difficulty: float = Field(default=0.3, description="初始难度")
    difficulty_increment: float = Field(default=0.1, description="难度递增")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    # 金融财务相关字段（可选，用于向后兼容）
    证券代码: Optional[str] = Field(None, description="证券代码")
    公司名称: Optional[str] = Field(None, description="公司名称")
    评估维度: Optional[str] = Field(None, description="评估维度")
    financial_data: Optional[Dict[str, Any]] = Field(None, description="财务数据")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SynthesisResult(BaseModel):
    """合成结果"""
    task_id: str = Field(..., description="任务ID")
    task_type: str = Field(..., description="任务类型")
    images: List[ImageInfo] = Field(..., description="图片列表")
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

    # 图片路径列表
    image_paths: List[str] = Field(default_factory=list)

    # 文件内容列表（文本类文件）
    file_contents: List[str] = Field(default_factory=list)

    # 历史问答对（已验证通过的）- 支持两种类型
    history_qa_pairs: List[Any] = Field(default_factory=list)

    # 当前迭代
    current_iteration: int = Field(default=0)

    # 当前难度
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
