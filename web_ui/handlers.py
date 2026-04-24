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
from config.llm_config import llm_config
from config.prompts import prompts_config
from services.html_generator import HTMLGenerator
from web_ui.data_converters import TaskDataConverter


class UIHandlers:
    """Event handlers for UI interactions"""
    
    def __init__(self, task_manager: TaskManager, log_queue: Queue, log_lock: threading.Lock):
        """Initialize UI handlers.
        
        Args:
            task_manager: TaskManager instance for task operations
            log_queue: Queue for thread-safe log messages
            log_lock: Lock for thread-safe log access
        """
        self.task_manager = task_manager
        self.log_queue = log_queue
        self.log_lock = log_lock
        self.is_running = False
    
    def load_json_file(self, file_path: str):
        """Handler for file upload (web_ui.py lines 532-587)

        Loads JSON file and appends tasks to the task manager.

        Args:
            file_path: Path to the JSON file to load

        Returns:
            Tuple of status message, tasks list, and task counts
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

            # Create tasks (append mode)
            tasks = []
            for item in data:
                # 按 NEGATIVE_SAMPLE_RATIO 比例随机标记负样本（样本级粒度）
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

            # Get all tasks
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
        """Handler for refresh button (web_ui.py lines 589-625)
        
        Refreshes the task list display with current status counts.
        
        Returns:
            Tuple of task counts, task list HTML, and status message
        """
        tasks = self.task_manager.get_all_tasks()
        
        # Count by status
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
        """Handler for view detail button.
        
        Displays detailed information about a specific task including QA results.
        
        Args:
            task_id: ID of the task to display
            
        Returns:
            HTML string with task details
        """
        return HTMLGenerator.task_detail_html(task_id, self.task_manager)
    
    def handle_dataframe_selection(self, dataframe_value: pd.DataFrame, selection_data: gr.SelectData):
        """Handler for DataFrame row selection

        Args:
            dataframe_value: Current DataFrame value (passed first due to how Gradio handles inputs)
            selection_data: Gradio selection event data

        Returns:
            HTML string with task details
        """
        # Handle no selection or empty dataframe
        if dataframe_value is None or len(dataframe_value) == 0:
            return "<div class='scrollable-box'>请选择一个任务查看详情</div>"

        # Handle custom message row (empty state)
        if "暂无任务" in str(dataframe_value.iloc[0]["状态"]):
            return "<div class='scrollable-box'>暂无任务，请先上传文件</div>"

        try:
            # Get task_id from selected row using selection data
            row_index = selection_data.index[0] if hasattr(selection_data, 'index') and isinstance(selection_data.index, list) else selection_data.index
            task_id = dataframe_value.iloc[row_index]["任务ID"]

            # Validate task_id
            if not task_id or str(task_id).strip() == "":
                return "<div class='scrollable-box'>❌ 无效的任务ID</div>"

            # Delegate to existing detail generation
            return HTMLGenerator.task_detail_html(str(task_id), self.task_manager)

        except (IndexError, KeyError) as e:
            return f"<div class='scrollable-box'>❌ 获取任务详情失败: {str(e)}</div>"
        except Exception as e:
            return f"<div class='scrollable-box'>❌ 处理选择时发生错误: {str(e)}</div>"
    
    def _process_single_task(self, task: FinancialTaskInput, max_iter: int) -> FinancialTaskResult:
        """Process a single task (thread-safe).
        
        Args:
            task: The task to process
            max_iter: Maximum iterations for the synthesis graph
            
        Returns:
            FinancialTaskResult with processing outcome
        """
        try:
            # Create graph instance
            graph = MultimodalSynthesisGraph(llm_config)
            
            # Log message
            log_msg = f"[INFO] 开始处理任务: {task.公司名称} ({task.证券代码})\n"
            with self.log_lock:
                self.log_queue.put(log_msg)
            
            # Call graph.run() with max_iterations parameter
            result = graph.run(task, max_iterations=max_iter)
            
            # Save result
            output_file = settings.OUTPUT_DIR / f"{task.task_id}.json"
            save_json(result.dict(), output_file)
            
            log_msg = f"[SUCCESS] 完成任务: {task.公司名称} - 生成 {result.valid_qa_count} 个问答对\n"
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
        """Handler for batch processing (web_ui.py lines 745-891)
        
        This is a generator function for Gradio streaming updates.
        Processes multiple tasks in parallel using ThreadPoolExecutor.
        
        Args:
            max_iter: Maximum iterations per task
            parallel_num: Number of parallel workers
            
        Yields:
            Tuple of task counts, displays, progress, and status
        """
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
        
        # Get pending tasks
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
        
        # Initial status
        log_html = "<div class='log-box'>"
        log_html += f"[INFO] 开始批量处理 {total_count} 个任务\n"
        log_html += f"[INFO] 并行数: {int(parallel_num)}\n"
        log_html += f"[INFO] 最大迭代次数: {int(max_iter)}\n"
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
        
        # Use ThreadPoolExecutor for parallel processing
        completed_count = 0
        failed_count = 0
        task_dataframe = None
        log_html = ""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_num)) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in pending_tasks:
                if not self.is_running:
                    break
                
                # Update task status
                self.task_manager.update_task_status(task.task_id, TaskStatus.PROCESSING)
                future = executor.submit(self._process_single_task, task, int(max_iter))
                future_to_task[future] = task
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_task):
                if not self.is_running:
                    break
                
                task = future_to_task[future]
                try:
                    result = future.result()
                    
                    # Update task status
                    if result.status == TaskStatus.COMPLETED:
                        self.task_manager.update_task_status(task.task_id, TaskStatus.COMPLETED)
                        completed_count += 1
                    else:
                        self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                        failed_count += 1
                    
                    # Update task list display
                    all_tasks = self.task_manager.get_all_tasks()
                    task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)
                    
                    # Update log display (real-time updates)
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
                    
                    # Calculate progress
                    progress = int((completed_count + failed_count) / total_count * 100)
                    
                    # Yield update
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
        
        # Complete
        self.is_running = False
        final_status = "completed" if failed_count == 0 else "partial"
        
        # Final update
        all_tasks = self.task_manager.get_all_tasks()
        task_dataframe = TaskDataConverter.tasks_to_dataframe(all_tasks)
        
        # Final log display
        log_html = "<div class='log-box'>"
        with self.log_lock:
            # Create a temporary list to preserve all logs
            temp_logs = []
            while not self.log_queue.empty():
                temp_logs.append(self.log_queue.get())
            
            # Put them back
            for log_line in temp_logs:
                self.log_queue.put(log_line)
                
            # Display them
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
        """Handler for stop button.
        
        Stops the batch processing.
        
        Returns:
            Status message indicating processing stopped
        """
        self.is_running = False
        return "⏹️ 已停止处理"
    
    def save_llm_config(self, api_key, base_url, model_name, temp, max_tok):
        """Handler for LLM config save (web_ui.py lines 919-929)
        
        Saves LLM configuration to the config object and persists to file.
        
        Args:
            api_key: API key for the LLM
            base_url: Base URL for the LLM API
            model_name: Name of the model to use
            temp: Temperature setting
            max_tok: Maximum tokens setting
            
        Returns:
            Status message indicating success or failure
        """
        try:
            llm_config.api_key = api_key
            llm_config.base_url = base_url
            llm_config.model_name = model_name
            llm_config.temperature = temp
            llm_config.max_tokens = int(max_tok)
            llm_config.save_to_file()  # Persist to file
            return "✅ LLM 配置已保存！"
        except Exception as e:
            return f"❌ 保存失败：{str(e)}"
    
    def save_prompts_config(self, p_sys, p_user, s_sys, s_user, v_sys, v_user):
        """Handler for prompts config save (web_ui.py lines 943-954)
        
        Saves prompt configuration to the config object and persists to file.
        
        Args:
            p_sys: Proposer system prompt
            p_user: Proposer user prompt
            s_sys: Solver system prompt
            s_user: Solver user prompt
            v_sys: Validator system prompt
            v_user: Validator user prompt
            
        Returns:
            Status message indicating success or failure
        """
        try:
            prompts_config.proposer_system_prompt = p_sys
            prompts_config.proposer_user_prompt = p_user
            prompts_config.solver_system_prompt = s_sys
            prompts_config.solver_user_prompt = s_user
            prompts_config.validator_system_prompt = v_sys
            prompts_config.validator_user_prompt = v_user
            prompts_config.save_to_file()  # Persist to file
            return "✅ Prompt 配置已保存！"
        except Exception as e:
            return f"❌ 保存失败：{str(e)}"

    def save_prompts_config_full(self, p_sys, p_user, s_sys, s_user, ns_sys, ns_user, v_sys, v_user, nv_sys, nv_user):
        """Handler for full prompts config save (including negative sample prompts)
        
        Args:
            p_sys: Proposer system prompt
            p_user: Proposer user prompt
            s_sys: Solver system prompt
            s_user: Solver user prompt
            ns_sys: Negative solver system prompt
            ns_user: Negative solver user prompt
            v_sys: Validator system prompt
            v_user: Validator user prompt
            nv_sys: Negative validator system prompt
            nv_user: Negative validator user prompt
            
        Returns:
            Status message indicating success or failure
        """
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
            prompts_config.save_to_file()
            return "✅ Prompt 配置已保存（含负样本提示词）！"
        except Exception as e:
            return f"❌ 保存失败：{str(e)}"
    
    def update_negative_sample_ratio(self, ratio: float):
        """Update negative sample ratio in settings
        
        Args:
            ratio: New negative sample ratio (0.0-1.0)
            
        Returns:
            Status message
        """
        try:
            settings.NEGATIVE_SAMPLE_RATIO = ratio
            return f"✅ 负样本比例已更新为 {ratio:.2f}"
        except Exception as e:
            return f"❌ 更新失败：{str(e)}"
