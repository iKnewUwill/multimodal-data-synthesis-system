"""Web UI - 基于 Gradio 的可视化界面（金融财务数据合成）"""

import os
import json
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import threading
from queue import Queue
import concurrent.futures

from config.llm_config import llm_config
from config.prompts import prompts_config
from config.settings import settings
from src.models import (
    FinancialTaskInput, FinancialTaskResult, TaskStatus, TaskType
)
from src.graph import MultimodalSynthesisGraph
from src.task_manager import TaskManager
from src.utils import save_json


class MultimodalSynthesisUI:
    """多模态数据合成 Web UI"""

    # 自定义 CSS
    CUSTOM_CSS = """
    .container {
        max-width: 1400px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: #ffffff;
        color: #333333;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px solid #e0e0e0;
    }
    .config-box {
        background: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    .progress-dashboard {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px solid #90a4ae;
    }
    .scrollable-box {
        max-height: 600px;
        overflow-y: auto;
        padding: 15px;
        background: #fafafa;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    .task-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .task-item:hover {
        background: #f0f8ff;
        border-color: #2196F3;
    }
    .task-pending {
        background: #fff9c4;
        border-left: 4px solid #ffc107;
    }
    .task-processing {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .task-completed {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    .task-failed {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .status-running {
        background: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .status-completed {
        background: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .status-failed {
        background: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .log-box {
        font-family: 'Courier New', monospace;
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
    }
    .log-info {
        color: #4fc3f7;
    }
    .log-success {
        color: #66bb6a;
    }
    .log-error {
        color: #ef5350;
    }
    """

    def __init__(self):
        self.graph = None
        self.task_manager = TaskManager()
        self.is_running = False
        self.log_queue = Queue()
        self.log_lock = threading.Lock()
    
    def create_interface(self):
        """创建 Gradio 界面"""

        with gr.Blocks(title="金融财务数据合成系统") as interface:
            
            # 标题
            gr.HTML("""
            <div class="header">
                <h1>🤖 金融财务数据合成系统</h1>
                <p>基于 Multi-Agent 的高质量金融财务分析训练数据合成平台</p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: 批量任务处理
                with gr.Tab("📊 批量任务处理"):
                    with gr.Row():
                        # 左侧：配置区域
                        with gr.Column(scale=1):
                            gr.Markdown("### 📁 数据上传")
                            
                            file_input = gr.File(
                                label="上传 JSON 文件（金融数据）",
                                file_types=[".json"],
                                type="filepath"
                            )

                            load_status = gr.Markdown("")
                            
                            gr.Markdown("### ⚙️ 任务配置")
                            
                            with gr.Group():
                                max_iterations = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label="最大迭代次数",
                                    info="每个任务生成的问答对数量"
                                )
                                
                                parallel_count = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=3,
                                    step=1,
                                    label="并行任务数",
                                    info="同时处理的任务数量"
                                )
                            
                            with gr.Row():
                                start_btn = gr.Button("🚀 开始批量处理", variant="primary", size="lg")
                                stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")
                                refresh_btn = gr.Button("🔄 刷新任务列表", variant="secondary", size="lg")

                            stop_status = gr.Markdown("")
                        
                        # 右侧：任务列表和日志
                        with gr.Column(scale=2):
                            # 任务统计
                            with gr.Group():
                                gr.Markdown("### 📊 任务统计")
                                with gr.Row():
                                    total_tasks = gr.Textbox(
                                        label="总任务数",
                                        value="0",
                                        interactive=False,
                                        scale=1
                                    )
                                    pending_tasks = gr.Textbox(
                                        label="待处理",
                                        value="0",
                                        interactive=False,
                                        scale=1
                                    )
                                    processing_tasks = gr.Textbox(
                                        label="处理中",
                                        value="0",
                                        interactive=False,
                                        scale=1
                                    )
                                    completed_tasks = gr.Textbox(
                                        label="已完成",
                                        value="0",
                                        interactive=False,
                                        scale=1
                                    )
                                    failed_tasks = gr.Textbox(
                                        label="失败",
                                        value="0",
                                        interactive=False,
                                        scale=1
                                    )
                            
                            # 任务列表
                            gr.Markdown("### 📋 任务列表")
                            task_list_display = gr.HTML(
                                "<div class='scrollable-box'>暂无任务</div>"
                            )
                            
                            # 实时日志
                            gr.Markdown("### 📝 实时日志")
                            log_display = gr.HTML(
                                "<div class='log-box'>等待开始...</div>"
                            )
                            
                            # 进度条
                            progress_bar = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                label="整体进度",
                                interactive=False
                            )
                            
                            status_text = gr.Markdown(
                                "<div class='status-badge'>⏸️ 等待开始</div>",
                                elem_classes=["progress-dashboard"]
                            )
                
                # Tab 2: LLM 配置
                with gr.Tab("🔧 LLM 配置"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### API 配置")
                            
                            api_key_input = gr.Textbox(
                                label="API Key",
                                value=llm_config.api_key,
                                type="password"
                            )
                            
                            base_url_input = gr.Textbox(
                                label="Base URL",
                                value=llm_config.base_url
                            )
                            
                            model_name_input = gr.Textbox(
                                label="模型名称",
                                value=llm_config.model_name
                            )
                            
                            temperature_input = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=llm_config.temperature,
                                step=0.1,
                                label="Temperature"
                            )
                            
                            max_tokens_input = gr.Slider(
                                minimum=512,
                                maximum=4096,
                                value=llm_config.max_tokens,
                                step=256,
                                label="Max Tokens"
                            )
                            
                            save_llm_config_btn = gr.Button("💾 保存 LLM 配置", variant="primary")
                            llm_config_status = gr.Markdown("")
                
                # Tab 3: Prompt 配置
                with gr.Tab("📝 Prompt 配置"):
                    with gr.Accordion("💡 提议者 Prompt", open=True):
                        proposer_system = gr.Textbox(
                            label="系统 Prompt",
                            value=prompts_config.proposer_system_prompt,
                            lines=15,
                            max_lines=25
                        )
                        proposer_user = gr.Textbox(
                            label="用户 Prompt 模板",
                            value=prompts_config.proposer_user_prompt,
                            lines=12,
                            max_lines=20
                        )
                    
                    with gr.Accordion("🤔 求解者 Prompt", open=False):
                        solver_system = gr.Textbox(
                            label="系统 Prompt",
                            value=prompts_config.solver_system_prompt,
                            lines=12,
                            max_lines=20
                        )
                        solver_user = gr.Textbox(
                            label="用户 Prompt 模板",
                            value=prompts_config.solver_user_prompt,
                            lines=10,
                            max_lines=15
                        )
                    
                    with gr.Accordion("✅ 验证者 Prompt", open=False):
                        validator_system = gr.Textbox(
                            label="系统 Prompt",
                            value=prompts_config.validator_system_prompt,
                            lines=12,
                            max_lines=20
                        )
                        validator_user = gr.Textbox(
                            label="用户 Prompt 模板",
                            value=prompts_config.validator_user_prompt,
                            lines=12,
                            max_lines=20
                        )
                    
                    save_prompts_btn = gr.Button("💾 保存 Prompt 配置", variant="primary")
                    prompts_status = gr.Markdown("")
            
            # 事件处理
            def load_json_file(file_path):
                """加载 JSON 文件（追加模式）"""
                if not file_path:
                    return "❌ 请先上传文件", [], "0", "0", "0", "0", "0", "<div class='scrollable-box'>暂无任务</div>"
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get existing task count
                    existing_count = len(self.task_manager.get_all_tasks())
                    
                    # 创建任务（APPEND, don't clear）
                    tasks = []
                    for item in data:
                        task = FinancialTaskInput(
                            证券代码=item.get("证券代码", ""),
                            公司名称=item.get("公司名称", ""),
                            统计截止日期=item.get("统计截止日期", ""),
                            评估维度=item.get("评估维度", ""),
                            关键指标=item.get("关键指标", {})
                        )
                        self.task_manager.add_task(task)
                        tasks.append(task)
                    
                    # Get ALL tasks (old + new)
                    all_tasks = self.task_manager.get_all_tasks()
                    
                    # 更新显示
                    task_list_html = "<div class='scrollable-box'>"
                    for t in all_tasks:
                        status_class = f"task-{t.status.value}"
                        task_list_html += f"""
                        <div class='task-item {status_class}'>
                            <p><strong>{t.公司名称}</strong> ({t.证券代码})</p>
                            <p>评估维度: {t.评估维度}</p>
                            <p>状态: {t.status.value}</p>
                        </div>
                        """
                    task_list_html += "</div>"
                    
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
                        task_list_html
                    )
                except Exception as e:
                    return f"❌ 加载失败: {str(e)}", [], "0", "0", "0", "0", "0", "<div class='scrollable-box'>暂无任务</div>"
            
            def refresh_task_list():
                """刷新任务列表"""
                tasks = self.task_manager.get_all_tasks()
                
                # 统计各状态任务数
                total = len(tasks)
                pending = len([t for t in tasks if t.status == TaskStatus.PENDING])
                processing = len([t for t in tasks if t.status == TaskStatus.PROCESSING])
                completed = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
                failed = len([t for t in tasks if t.status == TaskStatus.FAILED])
                
                # 生成任务列表 HTML
                task_list_html = "<div class='scrollable-box'>"
                if tasks:
                    for task in tasks:
                        status_class = f"task-{task.status.value}"
                        task_list_html += f"""
                        <div class='task-item {status_class}'>
                            <p><strong>{task.公司名称}</strong> ({task.证券代码})</p>
                            <p>评估维度: {task.评估维度}</p>
                            <p>状态: {task.status.value}</p>
                        </div>
                        """
                else:
                    task_list_html += "<p>暂无任务</p>"
                task_list_html += "</div>"
                
                return (
                    str(total),
                    str(pending),
                    str(processing),
                    str(completed),
                    str(failed),
                    task_list_html,
                    "✅ 任务列表已刷新"
                )
            
            file_input.change(
                fn=load_json_file,
                inputs=[file_input],
                outputs=[load_status, gr.State(), total_tasks, pending_tasks, processing_tasks, 
                        completed_tasks, failed_tasks, task_list_display]
            )
            
            refresh_btn.click(
                fn=refresh_task_list,
                outputs=[total_tasks, pending_tasks, processing_tasks, completed_tasks, failed_tasks, task_list_display, load_status]
            )
            
            def process_single_task(self_ref, task: FinancialTaskInput, max_iter: int) -> FinancialTaskResult:
                """处理单个任务（线程安全）"""
                try:
                    # 创建图实例
                    graph = MultimodalSynthesisGraph(llm_config)
                    
                    # 记录日志
                    log_msg = f"[INFO] 开始处理任务: {task.公司名称} ({task.证券代码})\n"
                    with self_ref.log_lock:
                        self_ref.log_queue.put(log_msg)
                    
                    # 调用 graph.run() 并传入 max_iterations 参数（Bug 1 修复）
                    result = graph.run(task, max_iterations=max_iter)
                    
                    # 保存结果
                    output_file = settings.OUTPUT_DIR / f"{task.task_id}.json"
                    save_json(result.dict(), output_file)
                    
                    log_msg = f"[SUCCESS] 完成任务: {task.公司名称} - 生成 {result.valid_qa_count} 个问答对\n"
                    with self_ref.log_lock:
                        self_ref.log_queue.put(log_msg)
                    
                    return result
                    
                except Exception as e:
                    log_msg = f"[ERROR] 任务失败: {task.公司名称} - {str(e)}\n"
                    with self_ref.log_lock:
                        self_ref.log_queue.put(log_msg)
                    
                    return FinancialTaskResult(
                        task_id=task.task_id,
                        证券代码=task.证券代码,
                        公司名称=task.公司名称,
                        评估维度=task.评估维度,
                        status=TaskStatus.FAILED,
                        completed_at=datetime.now()
                    )
            
            def start_batch_processing(max_iter, parallel_num):
                """开始批量处理"""
                if not self.task_manager.get_task_list_for_display():
                    yield (
                        "0", "0", "0", "0", "0",
                        "<div class='scrollable-box'>暂无任务</div>",
                        "<div class='log-box'>❌ 请先上传任务文件</div>",
                        0,
                        "<div class='status-badge status-failed'>❌ 请先上传任务文件</div>"
                    )
                    return
                
                self.is_running = True
                
                # 获取待处理任务
                pending_tasks = self.task_manager.filter_tasks(status=TaskStatus.PENDING)
                total_count = len(pending_tasks)
                
                if total_count == 0:
                    yield (
                        str(len(self.task_manager.get_all_tasks())),
                        "0",
                        "0",
                        str(len([t for t in self.task_manager.get_all_tasks().values() if t.status == TaskStatus.COMPLETED])),
                        str(len([t for t in self.task_manager.get_all_tasks().values() if t.status == TaskStatus.FAILED])),
                        "<div class='scrollable-box'>没有待处理的任务</div>",
                        "<div class='log-box'>⚠️ 没有待处理的任务</div>",
                        100,
                        "<div class='status-badge status-completed'>✅ 所有任务已完成</div>"
                    )
                    return
                
                # 初始状态
                log_html = "<div class='log-box'>"
                log_html += f"[INFO] 开始批量处理 {total_count} 个任务\n"
                log_html += f"[INFO] 并行数: {int(parallel_num)}\n"
                log_html += f"[INFO] 最大迭代次数: {int(max_iter)}\n"
                log_html += "</div>"
                
                yield (
                    str(len(self.task_manager.get_all_tasks())),
                    str(total_count),
                    "0",
                    "0",
                    "0",
                    "<div class='scrollable-box'>处理中...</div>",
                    log_html,
                    0,
                    "<div class='status-badge status-running'>🚀 开始批量处理</div>"
                )
                
                # 使用 ThreadPoolExecutor 进行真正的并行处理（Bug 3 修复）
                completed_count = 0
                failed_count = 0
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel_num)) as executor:
                    # 提交所有任务
                    future_to_task = {}
                    for task in pending_tasks:
                        if not self.is_running:
                            break
                        
                        # 更新任务状态
                        self.task_manager.update_task_status(task.task_id, TaskStatus.PROCESSING)
                        future = executor.submit(process_single_task, self, task, int(max_iter))
                        future_to_task[future] = task
                    
                    # 处理完成的任务
                    for future in concurrent.futures.as_completed(future_to_task):
                        if not self.is_running:
                            break
                        
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            
                            # 更新任务状态
                            if result.status == TaskStatus.COMPLETED:
                                self.task_manager.update_task_status(task.task_id, TaskStatus.COMPLETED)
                                completed_count += 1
                            else:
                                self.task_manager.update_task_status(task.task_id, TaskStatus.FAILED)
                                failed_count += 1
                            
                            # 更新任务列表显示
                            task_list_html = "<div class='scrollable-box'>"
                            for t in self.task_manager.get_all_tasks():
                                status_class = f"task-{t.status.value}"
                                task_list_html += f"""
                                <div class='task-item {status_class}'>
                                    <p><strong>{t.公司名称}</strong> ({t.证券代码})</p>
                                    <p>评估维度: {t.评估维度}</p>
                                    <p>状态: {t.status.value}</p>
                                </div>
                                """
                            task_list_html += "</div>"
                            
                            # 更新日志显示（Bug 2 修复 - 实时更新）
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
                            
                            # 计算进度
                            progress = int((completed_count + failed_count) / total_count * 100)
                            
                            # Yield 更新
                            yield (
                                str(len(self.task_manager.get_all_tasks())),
                                str(total_count - completed_count - failed_count),
                                str(len([f for f in future_to_task if f.running()])),
                                str(completed_count),
                                str(failed_count),
                                task_list_html,
                                log_html,
                                progress,
                                f"<div class='status-badge status-running'>🔄 处理中: {completed_count + failed_count}/{total_count}</div>"
                            )
                            
                        except Exception as e:
                            failed_count += 1
                            log_msg = f"[ERROR] 任务异常: {task.公司名称} - {str(e)}\n"
                            with self.log_lock:
                                self.log_queue.put(log_msg)
                
                # 完成
                self.is_running = False
                final_status = "completed" if failed_count == 0 else "partial"
                
                yield (
                    str(len(self.task_manager.get_all_tasks())),
                    "0",
                    "0",
                    str(completed_count),
                    str(failed_count),
                    task_list_html,
                    log_html,
                    100,
                    f"<div class='status-badge status-completed'>✅ 批量处理完成！成功: {completed_count}, 失败: {failed_count}</div>"
                )
            
            start_btn.click(
                fn=start_batch_processing,
                inputs=[max_iterations, parallel_count],
                outputs=[
                    total_tasks,
                    pending_tasks,
                    processing_tasks,
                    completed_tasks,
                    failed_tasks,
                    task_list_display,
                    log_display,
                    progress_bar,
                    status_text
                ]
            )
            
            def stop_processing():
                """停止处理"""
                self.is_running = False
                return "⏹️ 已停止处理"
            
            stop_btn.click(
                fn=stop_processing,
                outputs=[stop_status]
            )
            
            def save_llm_config_func(api_key, base_url, model_name, temp, max_tok):
                """保存 LLM 配置"""
                try:
                    llm_config.api_key = api_key
                    llm_config.base_url = base_url
                    llm_config.model_name = model_name
                    llm_config.temperature = temp
                    llm_config.max_tokens = int(max_tok)
                    return "✅ LLM 配置已保存！"
                except Exception as e:
                    return f"❌ 保存失败：{str(e)}"
            
            save_llm_config_btn.click(
                fn=save_llm_config_func,
                inputs=[
                    api_key_input,
                    base_url_input,
                    model_name_input,
                    temperature_input,
                    max_tokens_input
                ],
                outputs=[llm_config_status]
            )
            
            def save_prompts_func(p_sys, p_user, s_sys, s_user, v_sys, v_user):
                """保存 Prompt 配置"""
                try:
                    prompts_config.proposer_system_prompt = p_sys
                    prompts_config.proposer_user_prompt = p_user
                    prompts_config.solver_system_prompt = s_sys
                    prompts_config.solver_user_prompt = s_user
                    prompts_config.validator_system_prompt = v_sys
                    prompts_config.validator_user_prompt = v_user
                    return "✅ Prompt 配置已保存！"
                except Exception as e:
                    return f"❌ 保存失败：{str(e)}"
            
            save_prompts_btn.click(
                fn=save_prompts_func,
                inputs=[
                    proposer_system,
                    proposer_user,
                    solver_system,
                    solver_user,
                    validator_system,
                    validator_user
                ],
                outputs=[prompts_status]
            )
            
            # Auto-load historical tasks on interface load
            interface.load(
                fn=lambda: refresh_task_list(),
                outputs=[total_tasks, pending_tasks, processing_tasks, completed_tasks, failed_tasks, task_list_display, load_status]
            )
        
        return interface


def launch_ui():
    """启动 Web UI"""
    ui = MultimodalSynthesisUI()
    interface = ui.create_interface()

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=ui.CUSTOM_CSS
    )


if __name__ == "__main__":
    launch_ui()
