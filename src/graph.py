"""基于 LangGraph 的工作流图 - 双模型架构（qwen3.7-max + qwen3-8b）
每个问题：1个标准答案 + 1个正样本 + 1个负样本 + N个策略模型样本
随机难度，问题间相互独立
"""

import logging
import random
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime

from config.settings import settings
from src.models import (
    AgentState, IterationState,
    ProposerOutput, ValidationResult, StrategySampleResult,
    FinancialTaskInput, FinancialTaskResult, TaskStatus,
    SynthesisTask, TaskType,
    SampleWithValidation, QuestionSampleSet
)
from src.agents import (
    MultimodalLLMClient, StrategyModelClient,
    ProposerAgent, SolverAgent, ValidatorAgent, StrategyModelSampler
)
from src.utils import setup_logger

logger = setup_logger("graph", settings.LOG_DIR, settings.LOG_LEVEL)


class MultimodalSynthesisGraph:
    """多模态数据合成工作流图（双模型架构）

    每个问题（迭代）独立生成，随机难度，输出结构：
    - 1个标准答案（qwen3.7-max Proposer，与问题一起生成）
    - 1个正样本（qwen3.7-max 求解者）
    - 1个负样本（qwen3.7-max 负样本求解者·错误注入）
    - N个策略模型样本（qwen3-8b 自然推理，默认3个）
    """

    def __init__(self, llm_config=None, strategy_config=None):
        self.llm_client = MultimodalLLMClient(llm_config)
        self.strategy_client = StrategyModelClient(strategy_config)

        self.proposer = ProposerAgent(self.llm_client)
        self.solver = SolverAgent(self.llm_client)
        self.validator = ValidatorAgent(self.llm_client)
        self.strategy_sampler = StrategyModelSampler(self.strategy_client)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)

        workflow.add_node("check_continue", self._check_continue)
        workflow.add_node("propose", self._propose_node)
        workflow.add_node("solve_positive", self._solve_positive_node)
        workflow.add_node("solve_negative", self._solve_negative_node)
        workflow.add_node("sample_strategy", self._sample_strategy_node)
        workflow.add_node("validate_positive", self._validate_positive_node)
        workflow.add_node("validate_negative", self._validate_negative_node)
        workflow.add_node("validate_strategy", self._validate_strategy_node)
        workflow.add_node("update_state", self._update_state_node)

        workflow.set_entry_point("check_continue")

        workflow.add_conditional_edges(
            "check_continue",
            self._should_continue,
            {"continue": "propose", "end": END}
        )
        workflow.add_edge("propose", "solve_positive")
        workflow.add_edge("solve_positive", "solve_negative")
        workflow.add_edge("solve_negative", "sample_strategy")
        workflow.add_edge("sample_strategy", "validate_positive")
        workflow.add_edge("validate_positive", "validate_negative")
        workflow.add_edge("validate_negative", "validate_strategy")
        workflow.add_edge("validate_strategy", "update_state")
        workflow.add_edge("update_state", "check_continue")

        return workflow.compile()

    def _random_difficulty(self) -> float:
        """生成随机难度"""
        return round(random.uniform(settings.MIN_DIFFICULTY, settings.MAX_DIFFICULTY), 2)

    def _check_continue(self, state: AgentState) -> AgentState:
        """检查是否继续迭代 —— 随机难度，不回看历史"""
        logger.info(f"检查是否继续 - 当前问题: {state.current_iteration}/{state.task.max_iterations}")

        if state.current_iteration < state.task.max_iterations:
            state.current_iteration += 1
            state.current_difficulty = self._random_difficulty()

            state.current_state = IterationState(
                iteration=state.current_iteration,
                difficulty=state.current_difficulty,
                status="pending"
            )

            logger.info(f"开始问题 {state.current_iteration}，随机难度: {state.current_difficulty:.2f}")

        return state

    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续"""
        if state.is_finished or state.current_iteration > state.task.max_iterations:
            logger.info(f"全部 {state.task.max_iterations} 个问题生成完成")
            return "end"
        return "continue"

    def _propose_failed(self, state: AgentState) -> bool:
        """Check if proposal failed"""
        return state.current_state.proposed_qa is None

    def _propose_node(self, state: AgentState) -> AgentState:
        """提议者节点（qwen3.7-max）- 生成问题 + 标准答案，不传递历史"""
        try:
            logger.info(f"[问题 {state.current_iteration}] 提议者生成问题+标准答案（难度: {state.current_difficulty:.2f}）")
            state.current_state.status = "proposing"

            # 不传递历史问答对 —— 每个问题独立
            output = self.proposer.propose(
                financial_data=state.task.financial_data,
                task_type=state.task.task_type,
                difficulty=state.current_difficulty,
                history_qa_pairs=None
            )

            state.current_state.proposed_qa = output
            logger.info(f"[问题 {state.current_iteration}] 提议者完成: {output.question[:50]}...")

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 提议者失败: {str(e)}")
            state.current_state.status = "failed"
            state.current_state.error = str(e)

        return state

    def _solve_positive_node(self, state: AgentState) -> AgentState:
        """正样本求解者节点（qwen3.7-max）"""
        if self._propose_failed(state):
            return state

        try:
            logger.info(f"[问题 {state.current_iteration}] 正样本求解者")
            state.current_state.status = "solving_positive"

            output = self.solver.solve(
                financial_data=state.task.financial_data,
                question=state.current_state.proposed_qa.question
            )
            state.current_state.positive_solved_output = output

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 正样本求解失败: {str(e)}")
            state.current_state.status = "failed"
            state.current_state.error = str(e)

        return state

    def _solve_negative_node(self, state: AgentState) -> AgentState:
        """负样本求解者节点（qwen3.7-max）"""
        if self._propose_failed(state):
            return state

        try:
            logger.info(f"[问题 {state.current_iteration}] 负样本求解者")
            state.current_state.status = "solving_negative"

            output = self.solver.solve_negative(
                financial_data=state.task.financial_data,
                question=state.current_state.proposed_qa.question
            )
            state.current_state.negative_solved_output = output

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 负样本求解失败: {str(e)}")
            state.current_state.status = "failed"
            state.current_state.error = str(e)

        return state

    def _sample_strategy_node(self, state: AgentState) -> AgentState:
        """策略模型采样节点（qwen3-8b）- 多次自然推理"""
        if self._propose_failed(state):
            return state

        try:
            sample_count = settings.STRATEGY_SAMPLE_COUNT
            logger.info(f"[问题 {state.current_iteration}] 策略模型采样 x{sample_count}")
            state.current_state.status = "sampling_strategy"

            strategy_samples = []
            for i in range(1, sample_count + 1):
                try:
                    sample_output = self.strategy_sampler.sample(
                        financial_data=state.task.financial_data,
                        question=state.current_state.proposed_qa.question,
                        sample_index=i
                    )
                    strategy_samples.append(StrategySampleResult(
                        sample_index=i,
                        solver_output=sample_output
                    ))
                except Exception as sample_err:
                    logger.error(f"[问题 {state.current_iteration}] 策略样本 #{i} 失败: {str(sample_err)[:100]}")
            state.current_state.strategy_samples = strategy_samples
            logger.info(f"[问题 {state.current_iteration}] 策略模型采样完成 ({len(strategy_samples)}/{sample_count}个成功)")

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 策略模型采样失败: {str(e)}")
            state.current_state.status = "failed"
            state.current_state.error = str(e)

        return state

    def _validate_positive_node(self, state: AgentState) -> AgentState:
        """正样本验证节点"""
        if state.current_state.positive_solved_output is None:
            return state

        try:
            logger.info(f"[问题 {state.current_iteration}] 正样本验证")
            state.current_state.status = "validating_positive"

            validation = self.validator.validate(
                financial_data=state.task.financial_data,
                question=state.current_state.proposed_qa.question,
                reference_answer=state.current_state.proposed_qa.answer,
                predicted_answer=state.current_state.positive_solved_output.answer,
                is_positive_sample=True
            )
            state.current_state.positive_validation = validation

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 正样本验证失败: {str(e)}")

        return state

    def _validate_negative_node(self, state: AgentState) -> AgentState:
        """负样本验证节点"""
        if state.current_state.negative_solved_output is None:
            return state

        try:
            logger.info(f"[问题 {state.current_iteration}] 负样本验证")
            state.current_state.status = "validating_negative"

            validation = self.validator.validate(
                financial_data=state.task.financial_data,
                question=state.current_state.proposed_qa.question,
                reference_answer=state.current_state.proposed_qa.answer,
                predicted_answer=state.current_state.negative_solved_output.answer,
                is_positive_sample=False
            )
            state.current_state.negative_validation = validation

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 负样本验证失败: {str(e)}")

        return state

    def _validate_strategy_node(self, state: AgentState) -> AgentState:
        """策略模型采样验证节点"""
        if not state.current_state.strategy_samples:
            return state

        try:
            logger.info(f"[问题 {state.current_iteration}] 策略样本验证")
            state.current_state.status = "validating_strategy"

            reference_answer = state.current_state.proposed_qa.answer

            for sample in state.current_state.strategy_samples:
                validation = self.validator.validate_strategy_sample(
                    financial_data=state.task.financial_data,
                    question=state.current_state.proposed_qa.question,
                    reference_answer=reference_answer,
                    predicted_answer=sample.solver_output.answer
                )
                sample.validation = validation
                sample.is_valid = validation.is_valid

            valid_count = sum(1 for s in state.current_state.strategy_samples if s.is_valid)
            logger.info(f"[问题 {state.current_iteration}] 策略验证: {valid_count}/{len(state.current_state.strategy_samples)} 正样本")

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 策略样本验证失败: {str(e)}")

        return state

    def _build_sample_with_validation(
        self, source: str, is_positive: bool, conclusion: str,
        analysis_process: dict, validation: ValidationResult = None,
        sample_index: int = 1
    ) -> SampleWithValidation:
        """构建带验证的样本对象"""
        return SampleWithValidation(
            sample_source=source,
            is_positive_sample=is_positive,
            conclusion=conclusion,
            analysis_process=analysis_process,
            validation=validation,
            sample_index=sample_index
        )

    def _update_state_node(self, state: AgentState) -> AgentState:
        """更新状态节点 - 构建 QuestionSampleSet"""
        try:
            if self._propose_failed(state):
                logger.warning(f"[问题 {state.current_iteration}] 提议失败，跳过")
                state.current_state.status = "failed"
                state.all_iterations.append(state.current_state)
                state.is_finished = state.current_iteration >= state.task.max_iterations
                return state

            q = state.current_state.proposed_qa
            question = q.question
            difficulty = state.current_difficulty

            # 构建 QuestionSampleSet
            sample_set = QuestionSampleSet(
                difficulty=difficulty,
                question=question,
                standard_answer=q.answer,
                standard_analysis_process=q.analysis_process
            )

            # 1. 正样本（qwen3.7-max 求解者）
            if state.current_state.positive_solved_output is not None:
                ps = state.current_state.positive_solved_output
                sample_set.positive_sample = self._build_sample_with_validation(
                    source="positive_solver",
                    is_positive=True,
                    conclusion=ps.conclusion,
                    analysis_process=ps.analysis_process,
                    validation=state.current_state.positive_validation
                )

            # 2. 负样本（qwen3.7-max 负样本求解者·错误注入）
            if state.current_state.negative_solved_output is not None:
                ns = state.current_state.negative_solved_output
                sample_set.negative_sample = self._build_sample_with_validation(
                    source="negative_solver",
                    is_positive=False,
                    conclusion=ns.conclusion,
                    analysis_process=ns.analysis_process,
                    validation=state.current_state.negative_validation
                )

            # 3. 策略模型样本（qwen3-8b）
            for s in state.current_state.strategy_samples:
                sv = s.solver_output
                sample_set.strategy_samples.append(self._build_sample_with_validation(
                    source="strategy_model",
                    is_positive=s.is_valid,
                    conclusion=sv.conclusion,
                    analysis_process=sv.analysis_process,
                    validation=s.validation,
                    sample_index=s.sample_index
                ))

            state.sample_sets.append(sample_set)
            state.current_state.status = "completed"
            state.all_iterations.append(state.current_state)

            total_samples = 1  # standard answer
            if sample_set.positive_sample:
                total_samples += 1
            if sample_set.negative_sample:
                total_samples += 1
            total_samples += len(sample_set.strategy_samples)

            logger.info(
                f"[问题 {state.current_iteration}] 完成: 1标准答案 + "
                f"{'1' if sample_set.positive_sample else '0'}正样本 + "
                f"{'1' if sample_set.negative_sample else '0'}负样本 + "
                f"{len(sample_set.strategy_samples)}策略样本 = {total_samples}条"
            )

            state.is_finished = state.current_iteration >= state.task.max_iterations

        except Exception as e:
            logger.error(f"[问题 {state.current_iteration}] 更新状态失败: {str(e)}")
            state.error = str(e)
            state.is_finished = True

        return state

    def run(self, task_input: FinancialTaskInput, max_iterations: int = None) -> FinancialTaskResult:
        """运行工作流

        每个迭代生成一个问题及其全部样本。
        max_iterations = 每个任务生成的问题数量。
        """
        actual_max_iterations = max_iterations or settings.MAX_ITERATIONS

        logger.info("=" * 50)
        logger.info("开始金融财务数据合成工作流（双模型架构·随机难度·独立问题）")
        logger.info(f"任务ID: {task_input.task_id}")
        logger.info(f"公司: {task_input.公司名称} ({task_input.证券代码})")
        logger.info(f"评估维度: {task_input.评估维度}")
        logger.info(f"问题数量: {actual_max_iterations}")
        logger.info(f"每个问题: 1标准答案 + 1正样本 + 1负样本 + {settings.STRATEGY_SAMPLE_COUNT}策略样本")
        logger.info("=" * 50)

        try:
            task_input.status = TaskStatus.PROCESSING
            task_input.started_at = datetime.now()

            task_financial_data = dict(task_input.financial_data) if task_input.financial_data else {}

            initial_state = AgentState(
                task=SynthesisTask(
                    task_id=task_input.task_id,
                    task_type=TaskType.FINANCIAL_QA.value,
                    证券代码=task_input.证券代码,
                    公司名称=task_input.公司名称,
                    评估维度=task_input.评估维度,
                    financial_data=task_financial_data,
                    is_positive_sample=task_input.is_positive_sample,
                    max_iterations=actual_max_iterations
                )
            )

            final_state = self.graph.invoke(initial_state)

            if isinstance(final_state, dict):
                sample_sets = final_state.get('sample_sets', [])
                all_iterations = final_state.get('all_iterations', [])
            else:
                sample_sets = final_state.sample_sets
                all_iterations = final_state.all_iterations

            # 统计
            total_positive = sum(1 for ss in sample_sets if ss.positive_sample is not None)
            total_negative = sum(1 for ss in sample_sets if ss.negative_sample is not None)
            total_strategy = sum(len(ss.strategy_samples) for ss in sample_sets)
            total_samples = len(sample_sets) + total_positive + total_negative + total_strategy

            result = FinancialTaskResult(
                task_id=task_input.task_id,
                证券代码=task_input.证券代码,
                公司名称=task_input.公司名称,
                评估维度=task_input.评估维度,
                financial_data=task_input.financial_data,
                status=TaskStatus.COMPLETED,
                sample_sets=sample_sets,
                valid_qa_count=total_samples,
                completed_at=datetime.now()
            )

            logger.info("=" * 50)
            logger.info("工作流完成")
            logger.info(f"问题数量: {len(sample_sets)}")
            logger.info(f"总样本数: {result.valid_qa_count}")
            logger.info(f"  - 标准答案: {len(sample_sets)}")
            logger.info(f"  - 正样本（235B求解者）: {total_positive}")
            logger.info(f"  - 负样本（235B负样本求解者）: {total_negative}")
            logger.info(f"  - 策略模型样本（8B自然推理）: {total_strategy}")
            logger.info("=" * 50)

            return result

        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return FinancialTaskResult(
                task_id=task_input.task_id,
                证券代码=task_input.证券代码,
                公司名称=task_input.公司名称,
                评估维度=task_input.评估维度,
                financial_data=task_input.financial_data,
                status=TaskStatus.FAILED,
                valid_qa_count=0,
                completed_at=datetime.now()
            )
