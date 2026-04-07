"""Multi-Agent 系统实现 - 金融财务分析"""

import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config.llm_config import llm_config
from config.prompts import prompts_config
from config.settings import settings
from src.models import (
    ProposerOutput, SolverOutput, ValidationResult,
    QAPair, IterationState
)
from src.utils import extract_json_from_text, setup_logger


# 设置日志
logger = setup_logger("agents", settings.LOG_DIR, settings.LOG_LEVEL)


class MultimodalLLMClient:
    """LLM 客户端 - 支持金融数据"""
    
    def __init__(self, config=None):
        self.config = config or llm_config
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
    
    def call_with_financial_data(
        self,
        system_prompt: str,
        user_prompt: str,
        financial_data_str: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """调用 LLM 处理金融数据"""
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # 构建用户消息
            content = [{"type": "text", "text": user_prompt}]

            # 添加金融数据内容
            if financial_data_str:
                content.append({
                    "type": "text",
                    "text": f"\n财务数据：\n{financial_data_str}"
                })

            messages.append({"role": "user", "content": content})

            # 调用 API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            raise


class ProposerAgent:
    """提议者 Agent - 生成金融分析问答对"""
    
    def __init__(self, llm_client: MultimodalLLMClient):
        self.llm_client = llm_client
        self.prompts_config = prompts_config
    
    def propose(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        difficulty: float = 0.3,
        history_qa_pairs: Optional[List[QAPair]] = None
    ) -> ProposerOutput:
        """基于金融数据生成新的问答对"""
        logger.info(f"提议者开始生成问答对 - 难度: {difficulty}")

        try:
            # 格式化 Prompt
            system_prompt, user_prompt = self.prompts_config.format_proposer_prompt(
                task_type=task_type or "金融财务问答",
                difficulty_level=difficulty,
                history_qa_pairs=[qa.model_dump() for qa in history_qa_pairs] if history_qa_pairs else None
            )

            # 将financial_data转换为JSON字符串
            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            # 调用 LLM
            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str
            )

            # 解析响应 - 适配新的中文格式
            result = extract_json_from_text(response)

            # 映射中文字段到英文结构（保持向后兼容）
            output = ProposerOutput(
                question=result.get("问题", result.get("question", "")),
                answer=result.get("分析结论", result.get("answer", ""))
            )
            # 存储完整的分析结果到extra字段
            if "分析过程" in result:
                output.analysis_process = result["分析过程"]
            if "分析结论" in result:
                output.conclusion = result["分析结论"]

            logger.info(f"提议者生成问题: {output.question[:50]}...")
            return output

        except Exception as e:
            logger.error(f"提议者执行失败: {str(e)}")
            raise


class SolverAgent:
    """求解者 Agent - 回答金融分析问题"""
    
    def __init__(self, llm_client: MultimodalLLMClient):
        self.llm_client = llm_client
        self.prompts_config = prompts_config
    
    def solve(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None
    ) -> SolverOutput:
        """基于金融数据回答问题"""
        logger.info(f"求解者开始回答问题: {question[:50] if question else 'N/A'}...")

        try:
            # 格式化 Prompt
            system_prompt, user_prompt = self.prompts_config.format_solver_prompt(
                question=question or ""
            )

            # 将financial_data转换为JSON字符串
            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            # 调用 LLM
            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str
            )

            # 解析响应 - 适配新的中文格式
            result = extract_json_from_text(response)

            # 映射字段
            output = SolverOutput(
                answer=result.get("分析结论", result.get("answer", ""))
            )
            # 存储完整的分析结果
            if "分析过程" in result:
                output.analysis_process = result["分析过程"]
            if "分析结论" in result:
                output.conclusion = result["分析结论"]

            logger.info(f"求解者生成答案: {output.answer[:50]}...")
            return output

        except Exception as e:
            logger.error(f"求解者执行失败: {str(e)}")
            raise


class ValidatorAgent:
    """验证者 Agent - 验证答案的正确性"""
    
    def __init__(self, llm_client: MultimodalLLMClient):
        self.llm_client = llm_client
        self.prompts_config = prompts_config
        self.validation_threshold = settings.VALIDATION_THRESHOLD
    
    def validate(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        reference_answer: Optional[str] = None,
        predicted_answer: Optional[str] = None
    ) -> ValidationResult:
        """验证答案的正确性"""
        logger.info("验证者开始验证答案")

        try:
            # 格式化 Prompt
            system_prompt, user_prompt = self.prompts_config.format_validator_prompt(
                question=question or "",
                reference_answer=reference_answer or "",
                predicted_answer=predicted_answer or ""
            )

            # 将financial_data转换为JSON字符串
            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            # 调用 LLM
            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=0.1  # 验证时使用较低温度
            )

            # 解析响应
            result = extract_json_from_text(response)

            validation = ValidationResult(
                is_valid=result.get("is_valid", False),
                similarity_score=result.get("similarity_score", 0.0),
                reason=result.get("reason", "")
            )

            logger.info(
                f"验证结果: {'通过' if validation.is_valid else '未通过'} "
                f"(相似度: {validation.similarity_score:.2f})"
            )

            return validation

        except Exception as e:
            logger.error(f"验证者执行失败: {str(e)}")
            raise
