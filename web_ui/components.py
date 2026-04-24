"""UI Components - Modular component builders for Gradio interface"""

import gradio as gr
from typing import Tuple, Dict
from config.llm_config import LLMConfig
from config.prompts import PromptsConfig
from config.settings import settings


class UIComponents:
    """UI component builders - static methods for creating interface sections"""
    
    @staticmethod
    def build_batch_tab() -> Tuple[Dict[str, gr.Component], Dict[str, gr.Component]]:
        """Build batch processing tab
        
        Returns:
            Tuple of (inputs_dict, outputs_dict)
        """
        with gr.Column():
            # Left: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### 📁 数据上传")
                file_input = gr.File(label="上传 JSON 文件（金融数据）", file_types=[".json"], type="filepath")
                load_status = gr.Markdown("")
                
                gr.Markdown("### ⚙️ 任务配置")
                with gr.Group():
                    max_iterations = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="最大迭代次数", info="每个任务生成的问答对数量")
                    parallel_count = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="并行任务数", info="同时处理的任务数量")
                    negative_sample_ratio = gr.Slider(minimum=0.0, maximum=1.0, value=settings.NEGATIVE_SAMPLE_RATIO, step=0.05, label="负样本比例", info="被标记为负样本的样本占比（粒度为样本级别）")
                
                with gr.Row():
                    start_btn = gr.Button("🚀 开始批量处理", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")
                    refresh_btn = gr.Button("🔄 刷新任务列表", variant="secondary", size="lg")
                stop_status = gr.Markdown("")
            
            # Right: Task list and logs
            with gr.Column(scale=2):
                # Task statistics
                with gr.Group():
                    gr.Markdown("### 📊 任务统计")
                    with gr.Row():
                        total_tasks = gr.Textbox(label="总任务数", value="0", interactive=False, scale=1)
                        pending_tasks = gr.Textbox(label="待处理", value="0", interactive=False, scale=1)
                        processing_tasks = gr.Textbox(label="处理中", value="0", interactive=False, scale=1)
                        completed_tasks = gr.Textbox(label="已完成", value="0", interactive=False, scale=1)
                        failed_tasks = gr.Textbox(label="失败", value="0", interactive=False, scale=1)
                
                gr.Markdown("### 📋 任务列表（点击行查看详情）")
                task_dataframe = gr.DataFrame(
                    headers=["状态", "公司名称", "证券代码", "评估维度", "任务ID"],
                    datatype=["str", "str", "str", "str", "str"],
                    row_count=10,
                    col_count=(5, "fixed"),
                    interactive=False,
                    label="",
                    wrap=True,
                    show_search="search",
                    max_height=400
                )
                
                gr.Markdown("### 🔍 任务详情")
                task_detail_display = gr.HTML("<div class='scrollable-box'>点击上方任务列表中的任务查看详情</div>")
                
                gr.Markdown("### 📝 实时日志")
                log_display = gr.HTML("<div class='log-box'>等待开始...</div>")
                
                progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="整体进度", interactive=False)
                status_text = gr.Markdown("<div class='status-badge'>⏸️ 等待开始</div>", elem_classes=["progress-dashboard"])
        
        inputs = {
            'file_input': file_input,
            'max_iterations': max_iterations,
            'parallel_count': parallel_count,
            'negative_sample_ratio': negative_sample_ratio,
            'start_btn': start_btn,
            'stop_btn': stop_btn,
            'refresh_btn': refresh_btn
        }
        
        outputs = {
            'load_status': load_status,
            'stop_status': stop_status,
            'total_tasks': total_tasks,
            'pending_tasks': pending_tasks,
            'processing_tasks': processing_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'task_dataframe': task_dataframe,
            'task_detail_display': task_detail_display,
            'log_display': log_display,
            'progress_bar': progress_bar,
            'status_text': status_text
        }
        
        return inputs, outputs
    
    @staticmethod
    def build_llm_config_tab(llm_config: LLMConfig) -> Dict[str, gr.Component]:
        """Build LLM configuration tab"""
        with gr.Column():
            gr.Markdown("### API 配置")
            api_key_input = gr.Textbox(label="API Key", value=llm_config.api_key, type="password")
            base_url_input = gr.Textbox(label="Base URL", value=llm_config.base_url)
            model_name_input = gr.Textbox(label="模型名称", value=llm_config.model_name)
            temperature_input = gr.Slider(minimum=0.0, maximum=2.0, value=llm_config.temperature, step=0.1, label="Temperature")
            max_tokens_input = gr.Slider(minimum=512, maximum=4096, value=llm_config.max_tokens, step=256, label="Max Tokens")
            save_llm_config_btn = gr.Button("💾 保存 LLM 配置", variant="primary")
            llm_config_status = gr.Markdown("")
        
        return {
            'api_key_input': api_key_input,
            'base_url_input': base_url_input,
            'model_name_input': model_name_input,
            'temperature_input': temperature_input,
            'max_tokens_input': max_tokens_input,
            'save_llm_config_btn': save_llm_config_btn,
            'llm_config_status': llm_config_status
        }
    
    @staticmethod
    def build_prompts_config_tab(prompts_config: PromptsConfig) -> Dict[str, gr.Component]:
        """Build Prompts configuration tab"""
        with gr.Accordion("💡 提议者 Prompt", open=True):
            proposer_system = gr.Textbox(label="系统 Prompt", value=prompts_config.proposer_system_prompt, lines=15, max_lines=25)
            proposer_user = gr.Textbox(label="用户 Prompt 模板", value=prompts_config.proposer_user_prompt, lines=12, max_lines=20)
        
        with gr.Accordion("🤔 求解者 Prompt", open=False):
            solver_system = gr.Textbox(label="系统 Prompt", value=prompts_config.solver_system_prompt, lines=12, max_lines=20)
            solver_user = gr.Textbox(label="用户 Prompt 模板", value=prompts_config.solver_user_prompt, lines=10, max_lines=15)
        
        with gr.Accordion("⚠️ 负样本求解者 Prompt", open=False):
            neg_solver_system = gr.Textbox(label="系统 Prompt（负样本错误注入）", value=prompts_config.negative_solver_system_prompt, lines=15, max_lines=25)
            neg_solver_user = gr.Textbox(label="用户 Prompt 模板（负样本）", value=prompts_config.negative_solver_user_prompt, lines=10, max_lines=15)
        
        with gr.Accordion("✅ 验证者 Prompt", open=False):
            validator_system = gr.Textbox(label="系统 Prompt", value=prompts_config.validator_system_prompt, lines=12, max_lines=20)
            validator_user = gr.Textbox(label="用户 Prompt 模板", value=prompts_config.validator_user_prompt, lines=12, max_lines=20)
        
        with gr.Accordion("🔍 负样本验证者 Prompt", open=False):
            neg_validator_system = gr.Textbox(label="系统 Prompt（负样本验证）", value=prompts_config.negative_validator_system_prompt, lines=12, max_lines=20)
            neg_validator_user = gr.Textbox(label="用户 Prompt 模板（负样本验证）", value=prompts_config.negative_validator_user_prompt, lines=10, max_lines=15)
        
        save_prompts_btn = gr.Button("💾 保存 Prompt 配置", variant="primary")
        prompts_status = gr.Markdown("")
        
        return {
            'proposer_system': proposer_system,
            'proposer_user': proposer_user,
            'solver_system': solver_system,
            'solver_user': solver_user,
            'neg_solver_system': neg_solver_system,
            'neg_solver_user': neg_solver_user,
            'validator_system': validator_system,
            'validator_user': validator_user,
            'neg_validator_system': neg_validator_system,
            'neg_validator_user': neg_validator_user,
            'save_prompts_btn': save_prompts_btn,
            'prompts_status': prompts_status
        }
