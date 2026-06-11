"""UI Handlers - Event handlers for Gradio interface"""

import json
import random
import concurrent.futures
from queue import Queue
import threading
from typing import List, Generator
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd
from src.models import FinancialTaskInput, FinancialTaskResult, TaskStatus
from src.task_manager import TaskManager
from src.graph import MultimodalSynthesisGraph
from src.utils import save_json
from config.settings import settings
from config.llm_config import llm_config, strategy_llm_config
from config.prompts import prompts_config
from services.html_generator import HTMLGenerator
from web_ui.data_converters import TaskDataConverter


class UIHandlers:
    """Event handlers for UI interactions"""

    def __init__(self, task_manager: TaskManager, log_queue: Queue, log_lock: threading.Lock):
        self.task_manager = task_manager
        self.log_queue = log_queue
        self.log_lock = log_lock
        self.is_running = False

    def load_json_file(self, file_path: str):
        """Handler for file upload

        Loads JSON file and appends tasks to the task manager.
        """
        if not file_path:
            empty_dataframe = TaskDataConverter.tasks_to_dataframe([])
            return (
                "❌ 请先上传文件", [], "0", "0", "0", "0", "0",
                empty_dataframe
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tasks = []
            for item in data:
                is_positive = random.random() > settings.NEGATIVE_SAMPLE_RATIO
                task = FinancialTaskInput(
                    证券代码=item.get("证券代码", ""),
                    公司名称=item.get("公司名称", ""),
                    统计截止日期=item.get("统计截止日期", ""),
                    评估维度=item.get("评估维度", ""),
                    financial_data=item.get("financial_data", {}),
                    is_positive_sample=is_positive
                )
                self.task_manager.add_task(task)
                tasks.append(task)

            all_tasks = self.task_manager.get_all_tasks()
            task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)

            new_count = len(tasks)
            total_count = len(all_tasks)

            return (
                f"✅ 成功追加 {new_count} 个任务（共计 {total_count} 个）",
                all_tasks,
                str(total_count),
                str(total_count),
                "0",
                "0",
                "0",
                task_dataframe
            )
        except Exception as e:
            empty_dataframe = TaskDataConverter.tasks_to_dataframe([])
            return (
                f"❌ 加载失败: {str(e)}", [], "0", "0", "0", "0", "0",
                empty_dataframe
            )

    def refresh_task_list(self):
        """Handler for refresh button"""
        tasks = self.task_manager.get_all_tasks()

        total = len(tasks)
        pending = len([t for t in tasks if t.status == TaskStatus.PENDING])
        processing = len([t for t in tasks if t.status == TaskStatus.PROCESSING])
        completed = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in tasks if t.status == TaskStatus.FAILED])

        task_dataframe = TaskDataConverter.tasks_to_dataframe(tasks)

        return (
            str(total),
            str(pending),
            str(processing),
            str(completed),
            str(failed),
            task_dataframe,
            "✅ 任务列表已刷新"
        )

    def view_task_detail(self, task_id: str):
        """Handler for view detail button."""
        return HTMLGenerator.task_detail_html(task_id, self.task_manager)

    def handle_dataframe_selection(self, dataframe_value: pd.DataFrame, selection_data: gr.SelectData):
        """Handler for DataFrame row selection"""
        if dataframe_value is None or len(dataframe_value) == 0:
            return "<div class='scrollable-box'>请选择一个任务查看详情</div>"

        if "暂无任务" in str(dataframe_value.iloc[0]["状态"]):
            return "<div class='scrollable-box'>暂无任务，请先上传文件</div>"

        try:
            row_index = selection_data.index[0] if hasattr(selection_data, 'index') and isinstance(selection_data.index, list) else selection_data.index
            task_id = dataframe_value.iloc[row_index]["任务ID"]

            if not task_id or str(task_id).strip() == "":
                return "<div class='scrollable-box'>❌ 无效的任务ID</div>"

            return HTMLGenerator.task_detail_html(str(task_id), self.task_manager)

        except (IndexError, KeyError) as e:
            return f"<div class='scrollable-box'>❌ 获取任务详情失败: {str(e)}</div>"
        except Exception as e:
            return f"<div class='scrollable-box'>❌ 处理选择时发生错误: {str(e)}</div>"

    def _process_single_task(self, task: FinancialTaskInput, max_iter: int) -> FinancialTaskResult:
        """Process a single task (thread-safe)."""
        try:
            graph = MultimodalSynthesisGraph(llm_config, strategy_llm_config)

            log_msg = f"[INFO] 开始处理任务: {task.公司名称} ({task.证券代码})\n"
            with self.log_lock:
                self.log_queue.put(log_msg)

            result = graph.run(task, max_iterations=max_iter)

            output_file = settings.OUTPUT_DIR / f"{task.task_id}.json"
            save_json(result.model_dump(mode='json'), output_file)

            log_msg = f"[SUCCESS] 完成任务: {task.公司名称} - 生成 {result.valid_qa_count} 个样本\n"
            with self.log_lock:
                self.log_queue.put(log_msg)

            return result

        except Exception as e:
            log_msg = f"[ERROR] 任务失败: {task.公司名称} - {str(e)}\n"
            with self.log_lock:
                self.log_queue.put(log_msg)

            return FinancialTaskResult(
                task_id=task.task_id,
                证券代码=task.证券代码,
                公司名称=task.公司名称,
                评估维度=task.评估维度,
                status=TaskStatus.FAILED,
                completed_at=datetime.now()
            )

    def start_batch_processing(self, max_iter, parallel_num) -> Generator:
        """Handler for batch processing - generator for Gradio streaming."""
        if not self.task_manager.get_task_list_for_display():
            empty_dataframe = TaskDataConverter.tasks_to_dataframe([])
            yield (
                "0", "0", "0", "0", "0",
                empty_dataframe,
                "<div class='log-box'>❌ 请先上传任务文件</div>",
                0,
                "<div class='status-badge status-failed'>❌ 请先上传任务文件</div>"
            )
            return

        self.is_running = True

        pending_tasks = self.task_manager.filter_tasks(status=TaskStatus.PENDING)
        total_count = len(pending_tasks)

        if total_count == 0:
            all_tasks = self.task_manager.get_all_tasks()
            completed_count = len([t for t in all_tasks if t.status == TaskStatus.COMPLETED])
            failed_count = len([t for t in all_tasks if t.status == TaskStatus.FAILED])

            task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)
            yield (
                str(len(all_tasks)),
                "0",
                "0",
                str(completed_count),
                str(failed_count),
                task_dataframe,
                "<div class='log-box'>⚠️ 没有待处理的任务</div>",
                100,
                "<div class='status-badge status-completed'>✅ 所有任务已完成</div>"
            )
            return

        log_html = "<div class='log-box'>"
        log_html += f"[INFO] 开始批量处理 {total_count} 个任务\n"
        log_html += f"[INFO] 并行数: {int(parallel_num)}\n"
        log_html += f"[INFO] 最大迭代次数: {int(max_iter)}\n"
        log_html += f"[INFO] 策略模型采样次数/迭代: {settings.STRATEGY_SAMPLE_COUNT}\n"
        log_html += "</div>"

        initial_dataframe = TaskDataConverter.tasks_to_dataframe(self.task_manager.get_all_tasks())
        yield (
            str(len(self.task_manager.get_all_tasks())),
            str(total_count),
            "0",
            "0",
            "0",
            initial_dataframe,
            log_html,
            0,
            "<div class='status-badge status-running'>🚀 开始批量处理</div>"
        )

        completed_count = 0
        failed_count = 0
        task_dataframe = None
        log_html = ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_num)) as executor:
            future_to_task = {}
            for task in pending_tasks:
                if not self.is_running:
                    break

                self.task_manager.update_task_status(task.task_id, TaskStatus.PROCESSING)
                future = executor.submit(self._process_single_task, task, int(max_iter))
                future_to_task[future] = task

            for future in concurrent.futures.as_completed(future_to_task):
                if not self.is_running:
                    break

                task = future_to_task[future]
                try:
                    result = future.result()

                    if result.status == TaskStatus.COMPLETED:
                        self.task_manager.update_task_status(task.task_id, TaskStatus.COMPLETED)
                        completed_count += 1
                    else:
                        self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                        failed_count += 1

                    all_tasks = self.task_manager.get_all_tasks()
                    task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)

                    log_html = "<div class='log-box'>"
                    with self.log_lock:
                        while not self.log_queue.empty():
                            log_line = self.log_queue.get()
                            if "[ERROR]" in log_line:
                                log_html += f"<span class='log-error'>{log_line}</span>"
                            elif "[SUCCESS]" in log_line:
                                log_html += f"<span class='log-success'>{log_line}</span>"
                            else:
                                log_html += f"<span class='log-info'>{log_line}</span>"
                    log_html += "</div>"

                    progress = int((completed_count + failed_count) / total_count * 100)

                    yield (
                        str(len(all_tasks)),
                        str(total_count - completed_count - failed_count),
                        str(len([f for f in future_to_task if f.running()])),
                        str(completed_count),
                        str(failed_count),
                        task_dataframe,
                        log_html,
                        progress,
                        f"<div class='status-badge status-running'>🔄 处理中: {completed_count + failed_count}/{total_count}</div>"
                    )

                except Exception as e:
                    failed_count += 1
                    log_msg = f"[ERROR] 任务异常: {task.公司名称} - {str(e)}\n"
                    with self.log_lock:
                        self.log_queue.put(log_msg)

        self.is_running = False

        all_tasks = self.task_manager.get_all_tasks()
        task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)

        log_html = "<div class='log-box'>"
        with self.log_lock:
            temp_logs = []
            while not self.log_queue.empty():
                temp_logs.append(self.log_queue.get())

            for log_line in temp_logs:
                self.log_queue.put(log_line)

            for log_line in temp_logs:
                if "[ERROR]" in log_line:
                    log_html += f"<span class='log-error'>{log_line}</span>"
                elif "[SUCCESS]" in log_line:
                    log_html += f"<span class='log-success'>{log_line}</span>"
                else:
                    log_html += f"<span class='log-info'>{log_line}</span>"
        log_html += "</div>"

        yield (
            str(len(all_tasks)),
            "0",
            "0",
            str(completed_count),
            str(failed_count),
            task_dataframe,
            log_html,
            100,
            f"<div class='status-badge status-completed'>✅ 批量处理完成！成功: {completed_count}, 失败: {failed_count}</div>"
        )

    def stop_processing(self):
        """Handler for stop button."""
        self.is_running = False
        return "⏹️ 已停止处理"

    def _retry_single_task(self, task: FinancialTaskInput):
        """Retry missing samples for a single failed task (thread-safe)."""
        import json as json_mod

        result_file = settings.OUTPUT_DIR / f"{task.task_id}.json"
        if not result_file.exists():
            log_msg = f"[ERROR] 任务 {task.公司名称}: 结果文件不存在，跳过\n"
            with self.log_lock:
                self.log_queue.put(log_msg)
            return FinancialTaskResult(
                task_id=task.task_id,
                证券代码=task.证券代码,
                公司名称=task.公司名称,
                评估维度=task.评估维度,
                status=TaskStatus.FAILED,
                completed_at=datetime.now()
            )

        try:
            with open(result_file, "r", encoding="utf-8") as f:
                existing_data = json_mod.load(f)
        except Exception as e:
            log_msg = f"[ERROR] 任务 {task.公司名称}: JSON 读取失败 - {str(e)[:80]}\n"
            with self.log_lock:
                self.log_queue.put(log_msg)
            return FinancialTaskResult(
                task_id=task.task_id,
                证券代码=task.证券代码,
                公司名称=task.公司名称,
                评估维度=task.评估维度,
                status=TaskStatus.FAILED,
                completed_at=datetime.now()
            )

        log_msg = f"[INFO] 开始部分重跑: {task.公司名称} ({task.证券代码})\n"
        with self.log_lock:
            self.log_queue.put(log_msg)

        try:
            graph = MultimodalSynthesisGraph(llm_config, strategy_llm_config)
            result = graph.retry_samples(task, existing_data)

            output_file = settings.OUTPUT_DIR / f"{task.task_id}.json"
            save_json(result.model_dump(mode='json'), output_file)

            if result.status == TaskStatus.COMPLETED:
                log_msg = f"[SUCCESS] 部分重跑完成: {task.公司名称} - {result.valid_qa_count} 个样本\n"
            else:
                log_msg = f"[ERROR] 部分重跑失败: {task.公司名称}\n"

            with self.log_lock:
                self.log_queue.put(log_msg)
            return result

        except Exception as e:
            log_msg = f"[ERROR] 部分重跑异常: {task.公司名称} - {str(e)[:100]}\n"
            with self.log_lock:
                self.log_queue.put(log_msg)
            return FinancialTaskResult(
                task_id=task.task_id,
                证券代码=task.证券代码,
                公司名称=task.公司名称,
                评估维度=task.评估维度,
                status=TaskStatus.FAILED,
                completed_at=datetime.now()
            )

    def retry_failed_tasks(self, max_iter, parallel_num) -> Generator:
        """部分重跑：只重跑失败任务中缺失/损坏的样本（正样本/负样本/策略样本）"""
        failed_tasks = self.task_manager.filter_tasks(status=TaskStatus.FAILED)
        legacy_failed = self.task_manager.filter_tasks(status="失败")
        seen_ids = {t.task_id for t in failed_tasks}
        for t in legacy_failed:
            if t.task_id not in seen_ids:
                failed_tasks.append(t)

        if not failed_tasks:
            try:
                all_tasks = self.task_manager.get_all_tasks()
            except Exception:
                all_tasks = []
            task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)
            yield (
                str(len(all_tasks)),
                "0", "0",
                str(len([t for t in all_tasks if t.status == TaskStatus.COMPLETED])) if all_tasks else "0",
                "0",
                task_dataframe,
                "<div class='log-box'>⚠️ 没有失败的任务需要重试</div>",
                100,
                "<div class='status-badge status-completed'>✅ 没有失败的任务</div>"
            )
            return

        self.is_running = True
        total_count = len(failed_tasks)

        log_html = "<div class='log-box'>"
        log_html += f"[INFO] 开始部分重跑 {total_count} 个失败任务（仅重跑缺失样本）\n"
        log_html += f"[INFO] 并行数: {int(parallel_num)}\n"
        log_html += f"[INFO] 粒度：正样本/负样本/策略样本 按需重跑\n"
        log_html += "</div>"

        initial_dataframe = TaskDataConverter.tasks_to_dataframe(self.task_manager.get_all_tasks())
        yield (
            str(len(self.task_manager.get_all_tasks())),
            str(total_count),
            "0",
            "0",
            "0",
            initial_dataframe,
            log_html,
            0,
            "<div class='status-badge status-running'>🔄 开始部分重跑</div>"
        )

        completed_count = 0
        still_failed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_num)) as executor:
            future_to_task = {}
            for task in failed_tasks:
                if not self.is_running:
                    break
                self.task_manager.update_task_status(task.task_id, TaskStatus.PROCESSING)
                future = executor.submit(self._retry_single_task, task)
                future_to_task[future] = task

            for future in concurrent.futures.as_completed(future_to_task):
                if not self.is_running:
                    break

                task = future_to_task[future]
                try:
                    result = future.result()

                    if result.status == TaskStatus.COMPLETED:
                        self.task_manager.update_task_status(task.task_id, TaskStatus.COMPLETED)
                        completed_count += 1
                    else:
                        self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                        still_failed_count += 1

                    all_tasks = self.task_manager.get_all_tasks()
                    task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)

                    log_html = "<div class='log-box'>"
                    with self.log_lock:
                        while not self.log_queue.empty():
                            log_line = self.log_queue.get()
                            if "[ERROR]" in log_line:
                                log_html += f"<span class='log-error'>{log_line}</span>"
                            elif "[SUCCESS]" in log_line:
                                log_html += f"<span class='log-success'>{log_line}</span>"
                            else:
                                log_html += f"<span class='log-info'>{log_line}</span>"
                    log_html += "</div>"

                    progress = int((completed_count + still_failed_count) / total_count * 100)

                    yield (
                        str(len(all_tasks)),
                        str(total_count - completed_count - still_failed_count),
                        str(len([f for f in future_to_task if f.running()])),
                        str(completed_count),
                        str(still_failed_count),
                        task_dataframe,
                        log_html,
                        progress,
                        f"<div class='status-badge status-running'>🔄 部分重跑中: {completed_count + still_failed_count}/{total_count}</div>"
                    )

                except Exception as e:
                    still_failed_count += 1
                    self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                    log_msg = f"[ERROR] 重跑任务异常: {task.公司名称} - {str(e)[:100]}\n"
                    with self.log_lock:
                        self.log_queue.put(log_msg)

        self.is_running = False

        all_tasks = self.task_manager.get_all_tasks()
        task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)

        log_html = "<div class='log-box'>"
        with self.log_lock:
            temp_logs = []
            while not self.log_queue.empty():
                temp_logs.append(self.log_queue.get())
            for log_line in temp_logs:
                self.log_queue.put(log_line)
            for log_line in temp_logs:
                if "[ERROR]" in log_line:
                    log_html += f"<span class='log-error'>{log_line}</span>"
                elif "[SUCCESS]" in log_line:
                    log_html += f"<span class='log-success'>{log_line}</span>"
                else:
                    log_html += f"<span class='log-info'>{log_line}</span>"
        log_html += "</div>"

        yield (
            str(len(all_tasks)),
            "0",
            "0",
            str(completed_count),
            str(still_failed_count),
            task_dataframe,
            log_html,
            100,
            f"<div class='status-badge status-completed'>✅ 部分重跑完成！成功: {completed_count}, 仍失败: {still_failed_count}</div>"
        )

    def save_llm_config(
        self,
        api_key, base_url, model_name,
        proposer_temp, proposer_max_tok,
        pos_solver_temp, pos_solver_max_tok,
        neg_solver_temp, neg_solver_max_tok,
        validator_temp, validator_max_tok,
        strategy_api_key, strategy_base_url, strategy_model_name,
        strategy_temp, strategy_max_tok, strategy_enable_thinking
    ):
        """Handler for saving full LLM config (main + strategy models)"""
        try:
            # Save main model config
            llm_config.api_key = api_key
            llm_config.base_url = base_url
            llm_config.model_name = model_name
            llm_config.proposer_temperature = proposer_temp
            llm_config.proposer_max_tokens = int(proposer_max_tok)
            llm_config.positive_solver_temperature = pos_solver_temp
            llm_config.positive_solver_max_tokens = int(pos_solver_max_tok)
            llm_config.negative_solver_temperature = neg_solver_temp
            llm_config.negative_solver_max_tokens = int(neg_solver_max_tok)
            llm_config.validator_temperature = validator_temp
            llm_config.validator_max_tokens = int(validator_max_tok)
            llm_config.save_to_file()

            # Save strategy model config
            strategy_llm_config.api_key = strategy_api_key or api_key
            strategy_llm_config.base_url = strategy_base_url or base_url
            strategy_llm_config.model_name = strategy_model_name
            strategy_llm_config.temperature = strategy_temp
            strategy_llm_config.max_tokens = int(strategy_max_tok)
            strategy_llm_config.enable_thinking = bool(strategy_enable_thinking)
            strategy_llm_config.save_to_file()

            return "✅ LLM 配置已保存（主力模型 + 策略模型）！"
        except Exception as e:
            return f"❌ 保存失败：{str(e)}"

    def save_prompts_config_full(
        self,
        p_sys, p_user,
        s_sys, s_user,
        ns_sys, ns_user,
        v_sys, v_user,
        nv_sys, nv_user,
        sampling_sys, sampling_user,
        sampling_val_sys, sampling_val_user
    ):
        """Handler for full prompts config save (includes sampling prompts)"""
        try:
            prompts_config.proposer_system_prompt = p_sys
            prompts_config.proposer_user_prompt = p_user
            prompts_config.solver_system_prompt = s_sys
            prompts_config.solver_user_prompt = s_user
            prompts_config.negative_solver_system_prompt = ns_sys
            prompts_config.negative_solver_user_prompt = ns_user
            prompts_config.validator_system_prompt = v_sys
            prompts_config.validator_user_prompt = v_user
            prompts_config.negative_validator_system_prompt = nv_sys
            prompts_config.negative_validator_user_prompt = nv_user
            prompts_config.sampling_system_prompt = sampling_sys
            prompts_config.sampling_user_prompt = sampling_user
            prompts_config.sampling_validator_system_prompt = sampling_val_sys
            prompts_config.sampling_validator_user_prompt = sampling_val_user
            prompts_config.save_to_file()
            return "✅ Prompt 配置已保存（含策略模型采样提示词）！"
        except Exception as e:
            return f"❌ 保存失败：{str(e)}"

    def update_negative_sample_ratio(self, ratio: float):
        """Update negative sample ratio in settings"""
        try:
            settings.NEGATIVE_SAMPLE_RATIO = ratio
            return f"✅ 负样本比例已更新为 {ratio:.2f}"
        except Exception as e:
            return f"❌ 更新失败：{str(e)}"

    def update_strategy_sample_count(self, count: float):
        """Update strategy sample count in settings"""
        try:
            settings.STRATEGY_SAMPLE_COUNT = int(count)
            return f"✅ 策略模型采样次数已更新为 {int(count)}"
        except Exception as e:
            return f"❌ 更新失败：{str(e)}"

    # === 输出完整性检查 ===

    # 目标结构必需字段定义
    _REQUIRED_TOP_LEVEL = [
        "task_id", "证券代码", "公司名称", "评估维度",
        "financial_data", "status", "sample_sets", "valid_qa_count", "completed_at"
    ]
    _REQUIRED_SAMPLE_SET = [
        "difficulty", "question", "standard_answer", "standard_analysis_process",
        "positive_sample", "negative_sample", "strategy_samples"
    ]
    _REQUIRED_SAMPLE = [
        "sample_source", "is_positive_sample", "conclusion",
        "analysis_process", "validation", "sample_index"
    ]
    _REQUIRED_VALIDATION = ["is_valid", "similarity_score", "reason"]

    @classmethod
    def _check_empty(cls, value) -> bool:
        """Return True if value is considered empty/missing."""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, dict) and len(value) == 0:
            return True
        return False

    @classmethod
    def _check_analysis_process(cls, process: dict, label: str, ss_idx: int) -> list:
        """检查分析过程的步骤是否完整（步骤1/2/3及内容非空）"""
        issues = []
        if not isinstance(process, dict) or len(process) == 0:
            issues.append(f"  问题{ss_idx + 1} {label}: analysis_process 为空")
            return issues
        for step_key in ("步骤1", "步骤2", "步骤3"):
            if step_key not in process:
                issues.append(f"  问题{ss_idx + 1} {label}: 缺少 [{step_key}]")
            elif cls._check_empty(process[step_key]):
                issues.append(f"  问题{ss_idx + 1} {label}: [{step_key}] 内容为空")
        return issues

    @classmethod
    def _validate_sample(cls, sample: dict, label: str, ss_idx: int) -> list:
        """Validate a single sample, return list of issue strings."""
        issues = []
        if sample is None:
            issues.append(f"  问题{ss_idx + 1} {label}: 样本为 null")
            return issues
        for field in cls._REQUIRED_SAMPLE:
            if field not in sample:
                issues.append(f"  问题{ss_idx + 1} {label}: 缺少字段 [{field}]")
            elif field == "analysis_process":
                issues.extend(
                    cls._check_analysis_process(sample[field], f"{label}.analysis_process", ss_idx)
                )
            elif field == "is_positive_sample":
                # 布尔字段，只检查是否存在（None = 缺失，False/True = 有效值）
                if sample[field] is None:
                    issues.append(f"  问题{ss_idx + 1} {label}: 字段 [is_positive_sample] 为 null")
            elif cls._check_empty(sample[field]):
                issues.append(f"  问题{ss_idx + 1} {label}: 字段 [{field}] 为空")
        # Validate nested validation
        val = sample.get("validation")
        if val and isinstance(val, dict):
            for vf in cls._REQUIRED_VALIDATION:
                if vf not in val:
                    issues.append(f"  问题{ss_idx + 1} {label}.validation: 缺少字段 [{vf}]")
                elif vf == "is_valid":
                    if val[vf] is None:
                        issues.append(f"  问题{ss_idx + 1} {label}.validation: 字段 [is_valid] 为 null")
                elif cls._check_empty(val[vf]):
                    issues.append(f"  问题{ss_idx + 1} {label}.validation: 字段 [{vf}] 为空")
        elif val is None:
            issues.append(f"  问题{ss_idx + 1} {label}.validation: 为 null")
        return issues

    def validate_completed_tasks(self) -> str:
        """检查所有已完成任务的输出 JSON，排查必要字段为空值的情况。
        对存在问题的任务，自动将其状态置为失败，以便重跑按钮处理。"""
        import json as json_mod

        completed_tasks = self.task_manager.filter_tasks(status=TaskStatus.COMPLETED)
        if not completed_tasks:
            return "✅ 没有已完成的任务需要检查"

        all_issues = []
        checked_count = 0
        ok_count = 0
        failed_count = 0

        for task in completed_tasks:
            result_file = settings.OUTPUT_DIR / f"{task.task_id}.json"
            if not result_file.exists():
                all_issues.append(f"❌ {task.公司名称} ({task.证券代码}): 结果文件不存在 [{task.task_id}.json]")
                self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                failed_count += 1
                continue

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
            except Exception as e:
                all_issues.append(f"❌ {task.公司名称} ({task.证券代码}): JSON 解析失败 - {str(e)}")
                self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                failed_count += 1
                continue

            checked_count += 1
            task_issues = []

            # 顶层字段
            for field in self._REQUIRED_TOP_LEVEL:
                if field not in data:
                    task_issues.append(f"  顶层: 缺少字段 [{field}]")
                elif field not in ("sample_sets",):
                    if self._check_empty(data[field]):
                        task_issues.append(f"  顶层: 字段 [{field}] 为空")

            sample_sets = data.get("sample_sets", [])
            if not isinstance(sample_sets, list) or len(sample_sets) == 0:
                task_issues.append("  顶层: sample_sets 为空列表")
            else:
                for ss_idx, ss in enumerate(sample_sets):
                    if not isinstance(ss, dict):
                        task_issues.append(f"  问题{ss_idx + 1}: 不是有效对象")
                        continue
                    for field in self._REQUIRED_SAMPLE_SET:
                        if field not in ss:
                            task_issues.append(f"  问题{ss_idx + 1}: 缺少字段 [{field}]")
                        elif field == "strategy_samples":
                            strat = ss.get("strategy_samples")
                            if strat is None:
                                task_issues.append(f"  问题{ss_idx + 1}: strategy_samples 为 null")
                            elif not isinstance(strat, list):
                                task_issues.append(f"  问题{ss_idx + 1}: strategy_samples 不是列表")
                            elif len(strat) == 0:
                                task_issues.append(f"  问题{ss_idx + 1}: strategy_samples 为空（0个样本）")
                            elif len(strat) < 3:
                                task_issues.append(f"  问题{ss_idx + 1}: strategy_samples 样本数不足（少于3个）")
                        elif field in ("positive_sample", "negative_sample"):
                            task_issues.extend(
                                self._validate_sample(ss.get(field), field, ss_idx)
                            )
                        elif field == "standard_analysis_process":
                            task_issues.extend(
                                self._check_analysis_process(
                                    ss.get(field, {}), "standard_analysis_process", ss_idx
                                )
                            )
                        else:
                            if self._check_empty(ss[field]):
                                task_issues.append(f"  问题{ss_idx + 1}: 字段 [{field}] 为空")

                    # 检查 strategy_samples 内每个样本
                    strat_samples = ss.get("strategy_samples", [])
                    if isinstance(strat_samples, list):
                        for si, strat in enumerate(strat_samples):
                            task_issues.extend(
                                self._validate_sample(strat, f"strategy_samples[{si}]", ss_idx)
                            )

            if task_issues:
                all_issues.append(
                    f"⚠️ {task.公司名称} ({task.证券代码}) [{task.task_id}]:\n"
                    + "\n".join(task_issues)
                )
                # 将有问题的任务状态置为失败，以便重跑
                self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                failed_count += 1
            else:
                ok_count += 1

        if not all_issues:
            return f"✅ 全部 {checked_count} 个已完成任务的输出 JSON 完整性检查通过"

        report = f"检查完成：{checked_count} 个已完成任务，{ok_count} 个通过，{len(all_issues)} 个存在问题（已自动标记为失败）\n\n"
        report += f"可点击「重新处理失败任务」按钮对这 {failed_count} 个任务进行部分重跑（仅重跑缺失样本）\n\n"
        report += "\n\n".join(all_issues)
        return report
