"""Multi-Agent 系统实现 - 金融财务分析（双模型架构）"""

import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config.llm_config import llm_config, strategy_llm_config
from config.prompts import prompts_config
from config.settings import settings
from src.models import (
    ProposerOutput, SolverOutput, ValidationResult,
    QAPair, IterationState, StrategySampleResult
)
from src.utils import extract_json_from_text, setup_logger


# 设置日志
logger = setup_logger("agents", settings.LOG_DIR, settings.LOG_LEVEL)


class MultimodalLLMClient:
    """LLM 客户端 - 支持金融数据（用于 qwen3.7-max 主力模型）"""

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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """调用 LLM 处理金融数据"""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            content = [{"type": "text", "text": user_prompt}]

            content.append({
                "type": "text",
                "text": f"\n财务数据：\n{financial_data_str}"
            })

            messages.append({"role": "user", "content": content})

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            raise


class StrategyModelClient:
    """策略模型客户端 - 用于 qwen3-8b（自然推理采样，无错误注入引导）"""

    def __init__(self, config=None):
        self.config = config or strategy_llm_config
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """调用策略模型进行自然推理"""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            content = [{"type": "text", "text": user_prompt}]

            content.append({
                "type": "text",
                "text": f"\n财务数据：\n{financial_data_str}"
            })

            messages.append({"role": "user", "content": content})

            extra_body = self.config.get_extra_body() if hasattr(self.config, 'get_extra_body') else {}
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                extra_body=extra_body if extra_body else None
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"策略模型调用失败: {str(e)}")
            raise


class ProposerAgent:
    """提议者 Agent - 生成金融分析问答对（qwen3.7-max）"""

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
            system_prompt, user_prompt = self.prompts_config.format_proposer_prompt(
                task_type=task_type or "金融财务问答",
                difficulty_level=difficulty,
                history_qa_pairs=[qa.model_dump() for qa in history_qa_pairs] if history_qa_pairs else None
            )

            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=llm_config.proposer_temperature,
                max_tokens=llm_config.proposer_max_tokens
            )

            result = extract_json_from_text(response)

            output = ProposerOutput(
                question=result.get("问题", result.get("question", "")),
                answer=result.get("分析结论", result.get("answer", "")),
                analysis_process=result["分析过程"],
                conclusion=result["分析结论"]
            )

            logger.info(f"提议者生成问题: {output.question[:50]}...")
            return output

        except Exception as e:
            logger.error(f"提议者执行失败: {str(e)}")
            raise


class SolverAgent:
    """求解者 Agent - 回答金融分析问题（qwen3.7-max）"""

    def __init__(self, llm_client: MultimodalLLMClient):
        self.llm_client = llm_client
        self.prompts_config = prompts_config

    def solve(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None
    ) -> SolverOutput:
        """正样本求解 - 基于金融数据生成正确答案"""
        logger.info(f"正样本求解者开始回答问题: {question[:50] if question else 'N/A'}...")

        try:
            system_prompt, user_prompt = self.prompts_config.format_solver_prompt(
                question=question or ""
            )

            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=llm_config.positive_solver_temperature,
                max_tokens=llm_config.positive_solver_max_tokens
            )

            result = extract_json_from_text(response)

            output = SolverOutput(
                answer=result.get("分析结论", result.get("answer", "")),
                analysis_process=result["分析过程"],
                conclusion=result["分析结论"]
            )

            logger.info(f"正样本求解者生成答案: {output.answer[:50]}...")
            return output

        except Exception as e:
            logger.error(f"正样本求解者执行失败: {str(e)}")
            raise

    def solve_negative(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None
    ) -> SolverOutput:
        """负样本求解 - 基于金融数据生成包含错误的答案"""
        logger.info(f"负样本求解者开始回答问题: {question[:50] if question else 'N/A'}...")

        try:
            system_prompt, user_prompt = self.prompts_config.format_negative_solver_prompt(
                question=question or ""
            )

            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=llm_config.negative_solver_temperature,
                max_tokens=llm_config.negative_solver_max_tokens
            )

            result = extract_json_from_text(response)

            output = SolverOutput(
                answer=result.get("分析结论", result.get("answer", "")),
                analysis_process=result["分析过程"],
                conclusion=result["分析结论"]
            )

            logger.info(f"负样本求解者生成答案: {output.answer[:50]}...")
            return output

        except Exception as e:
            logger.error(f"负样本求解者执行失败: {str(e)}")
            raise


class StrategyModelSampler:
    """策略模型采样器 - 使用 qwen3-8b 进行自然推理（无错误注入引导）"""

    def __init__(self, strategy_client: StrategyModelClient):
        self.strategy_client = strategy_client
        self.prompts_config = prompts_config

    def sample(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        sample_index: int = 1
    ) -> SolverOutput:
        """策略模型自然推理采样"""
        logger.info(f"策略模型采样 #{sample_index}: {question[:50] if question else 'N/A'}...")

        try:
            system_prompt, user_prompt = self.prompts_config.format_sampling_prompt(
                question=question or ""
            )

            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            response = self.strategy_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=strategy_llm_config.temperature,
                max_tokens=strategy_llm_config.max_tokens
            )

            result = extract_json_from_text(response)

            output = SolverOutput(
                answer=result.get("分析结论", result.get("answer", "")),
                analysis_process=result["分析过程"],
                conclusion=result["分析结论"]
            )

            logger.info(f"策略模型采样 #{sample_index} 完成: {output.answer[:50]}...")
            return output

        except Exception as e:
            logger.error(f"策略模型采样 #{sample_index} 失败: {str(e)}")
            raise


class ValidatorAgent:
    """验证者 Agent - 验证答案的正确性（qwen3.7-max）"""

    def __init__(self, llm_client: MultimodalLLMClient):
        self.llm_client = llm_client
        self.prompts_config = prompts_config
        self.validation_threshold = settings.VALIDATION_THRESHOLD

    def validate(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        reference_answer: Optional[str] = None,
        predicted_answer: Optional[str] = None,
        is_positive_sample: bool = True
    ) -> ValidationResult:
        """验证答案的正确性"""
        logger.info("验证者开始验证答案")

        try:
            if is_positive_sample:
                system_prompt, user_prompt = self.prompts_config.format_validator_prompt(
                    question=question or "",
                    reference_answer=reference_answer or "",
                    predicted_answer=predicted_answer or ""
                )
            else:
                system_prompt, user_prompt = self.prompts_config.format_negative_validator_prompt(
                    question=question or "",
                    reference_answer=reference_answer or "",
                    predicted_answer=predicted_answer or ""
                )

            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=llm_config.validator_temperature,
                max_tokens=llm_config.validator_max_tokens
            )

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

    def validate_strategy_sample(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        reference_answer: Optional[str] = None,
        predicted_answer: Optional[str] = None
    ) -> ValidationResult:
        """验证策略模型采样结果（8B样本标注）"""
        logger.info("验证者开始标注策略模型样本")

        try:
            system_prompt, user_prompt = self.prompts_config.format_sampling_validator_prompt(
                question=question or "",
                reference_answer=reference_answer or "",
                predicted_answer=predicted_answer or ""
            )

            financial_data_str = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else None

            response = self.llm_client.call_with_financial_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                financial_data_str=financial_data_str,
                temperature=llm_config.validator_temperature,
                max_tokens=llm_config.validator_max_tokens
            )

            result = extract_json_from_text(response)

            validation = ValidationResult(
                is_valid=result.get("is_valid", False),
                similarity_score=result.get("similarity_score", 0.0),
                reason=result.get("reason", "")
            )

            logger.info(
                f"策略样本标注: {'正样本' if validation.is_valid else '负样本'} "
                f"(相似度: {validation.similarity_score:.2f})"
            )

            return validation

        except Exception as e:
            logger.error(f"策略样本验证失败: {str(e)}")
            raise
