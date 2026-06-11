"""UI Components - Modular component builders for Gradio interface"""

import gradio as gr
from typing import Tuple, Dict
from config.llm_config import LLMConfig, StrategyLLMConfig
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
            with gr.Column(scale=1):
                gr.Markdown("### 📁 数据上传")
                file_input = gr.File(label="上传 JSON 文件（金融数据）", file_types=[".json"], type="filepath")
                load_status = gr.Markdown("")

                gr.Markdown("### ⚙️ 任务配置")
                with gr.Group():
                    max_iterations = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="问题数量", info="每个任务生成的问题数（每个问题产出 1标准答案+1正样本+1负样本+N策略样本，随机难度）")
                    parallel_count = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="并行任务数", info="同时处理的任务数量")
                    negative_sample_ratio = gr.Slider(minimum=0.0, maximum=1.0, value=settings.NEGATIVE_SAMPLE_RATIO, step=0.05, label="负样本比例", info="被标记为负样本的样本占比（粒度为样本级别）")
                    strategy_sample_count = gr.Slider(minimum=1, maximum=10, value=settings.STRATEGY_SAMPLE_COUNT, step=1, label="策略模型采样次数", info="qwen3-8b 每个问题的自然推理采样次数（默认3次）")

                with gr.Row():
                    start_btn = gr.Button("🚀 开始批量处理", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")
                    retry_failed_btn = gr.Button("🔄 重新处理失败任务", variant="secondary", size="lg")
                    refresh_btn = gr.Button("🔃 刷新任务列表", variant="secondary", size="lg")
                    validate_btn = gr.Button("🔍 检查已完成任务", variant="secondary", size="lg")
                stop_status = gr.Markdown("")

            with gr.Column(scale=2):
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
            'strategy_sample_count': strategy_sample_count,
            'start_btn': start_btn,
            'stop_btn': stop_btn,
            'retry_failed_btn': retry_failed_btn,
            'refresh_btn': refresh_btn,
            'validate_btn': validate_btn
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
    def build_llm_config_tab(llm_config: LLMConfig, strategy_config: StrategyLLMConfig = None) -> Dict[str, gr.Component]:
        """Build LLM configuration tab with dual model support"""
        components = {}

        with gr.Column():
            # 主力模型配置（qwen3.7-max）
            gr.Markdown("### 🔧 主力模型配置（qwen3.7-max — 提议者/求解者/验证者）")
            with gr.Group():
                api_key_input = gr.Textbox(label="API Key", value=llm_config.api_key, type="password")
                base_url_input = gr.Textbox(label="Base URL", value=llm_config.base_url)
                model_name_input = gr.Textbox(label="模型名称", value=llm_config.model_name)

                gr.Markdown("#### 按角色温度设置")
                with gr.Row():
                    proposer_temp = gr.Slider(minimum=0.0, maximum=2.0, value=llm_config.proposer_temperature, step=0.05, label="提议者 Temperature", info="需要多样性 (建议0.7-0.9)")
                    proposer_max_tok = gr.Slider(minimum=512, maximum=8192, value=llm_config.proposer_max_tokens, step=256, label="提议者 Max Tokens")
                with gr.Row():
                    pos_solver_temp = gr.Slider(minimum=0.0, maximum=2.0, value=llm_config.positive_solver_temperature, step=0.05, label="正样本求解者 Temperature", info="需要稳定性 (建议0.1-0.3)")
                    pos_solver_max_tok = gr.Slider(minimum=512, maximum=8192, value=llm_config.positive_solver_max_tokens, step=256, label="正样本求解者 Max Tokens")
                with gr.Row():
                    neg_solver_temp = gr.Slider(minimum=0.0, maximum=2.0, value=llm_config.negative_solver_temperature, step=0.05, label="负样本求解者 Temperature", info="需要多样化错误 (建议0.6-0.8)")
                    neg_solver_max_tok = gr.Slider(minimum=512, maximum=8192, value=llm_config.negative_solver_max_tokens, step=256, label="负样本求解者 Max Tokens")
                with gr.Row():
                    validator_temp = gr.Slider(minimum=0.0, maximum=2.0, value=llm_config.validator_temperature, step=0.05, label="验证者 Temperature", info="评判需高度一致 (建议0.0-0.1)")
                    validator_max_tok = gr.Slider(minimum=256, maximum=4096, value=llm_config.validator_max_tokens, step=128, label="验证者 Max Tokens")

            # 策略模型配置（qwen3-8b）
            gr.Markdown("### 🧪 策略模型配置（qwen3-8b — 自然推理采样）")
            with gr.Group():
                strategy_api_key = gr.Textbox(
                    label="API Key",
                    value=strategy_config.api_key if strategy_config else "",
                    type="password",
                    info="留空则使用主力模型 API Key"
                )
                strategy_base_url = gr.Textbox(
                    label="Base URL",
                    value=strategy_config.base_url if strategy_config else "",
                    info="留空则使用主力模型 Base URL"
                )
                strategy_model_name = gr.Textbox(
                    label="模型名称",
                    value=strategy_config.model_name if strategy_config else "qwen3-8b"
                )
                with gr.Row():
                    strategy_temp = gr.Slider(minimum=0.0, maximum=2.0, value=strategy_config.temperature if strategy_config else 0.7, step=0.05, label="Temperature", info="模拟真实推理多样性 (建议0.6-0.8)")
                    strategy_max_tok = gr.Slider(minimum=512, maximum=8192, value=strategy_config.max_tokens if strategy_config else 4096, step=256, label="Max Tokens")
                strategy_enable_thinking = gr.Checkbox(
                    value=strategy_config.enable_thinking if strategy_config and hasattr(strategy_config, 'enable_thinking') else False,
                    label="启用 Thinking 模式",
                    info="qwen3-8b 非流式调用需关闭此选项"
                )

            save_llm_config_btn = gr.Button("💾 保存全部 LLM 配置", variant="primary")
            llm_config_status = gr.Markdown("")

        components.update({
            'api_key_input': api_key_input,
            'base_url_input': base_url_input,
            'model_name_input': model_name_input,
            'proposer_temp': proposer_temp,
            'proposer_max_tok': proposer_max_tok,
            'pos_solver_temp': pos_solver_temp,
            'pos_solver_max_tok': pos_solver_max_tok,
            'neg_solver_temp': neg_solver_temp,
            'neg_solver_max_tok': neg_solver_max_tok,
            'validator_temp': validator_temp,
            'validator_max_tok': validator_max_tok,
            'strategy_api_key': strategy_api_key,
            'strategy_base_url': strategy_base_url,
            'strategy_model_name': strategy_model_name,
            'strategy_temp': strategy_temp,
            'strategy_max_tok': strategy_max_tok,
            'strategy_enable_thinking': strategy_enable_thinking,
            'save_llm_config_btn': save_llm_config_btn,
            'llm_config_status': llm_config_status
        })

        return components

    @staticmethod
    def build_prompts_config_tab(prompts_config: PromptsConfig) -> Dict[str, gr.Component]:
        """Build Prompts configuration tab (includes sampling prompts)"""
        with gr.Accordion("💡 提议者 Prompt", open=True):
            proposer_system = gr.Textbox(label="系统 Prompt", value=prompts_config.proposer_system_prompt, lines=15, max_lines=25)
            proposer_user = gr.Textbox(label="用户 Prompt 模板", value=prompts_config.proposer_user_prompt, lines=12, max_lines=20)

        with gr.Accordion("🤔 求解者 Prompt（正样本）", open=False):
            solver_system = gr.Textbox(label="系统 Prompt", value=prompts_config.solver_system_prompt, lines=12, max_lines=20)
            solver_user = gr.Textbox(label="用户 Prompt 模板", value=prompts_config.solver_user_prompt, lines=10, max_lines=15)

        with gr.Accordion("⚠️ 负样本求解者 Prompt", open=False):
            neg_solver_system = gr.Textbox(label="系统 Prompt（负样本错误注入）", value=prompts_config.negative_solver_system_prompt, lines=15, max_lines=25)
            neg_solver_user = gr.Textbox(label="用户 Prompt 模板（负样本）", value=prompts_config.negative_solver_user_prompt, lines=10, max_lines=15)

        with gr.Accordion("✅ 验证者 Prompt（正样本）", open=False):
            validator_system = gr.Textbox(label="系统 Prompt", value=prompts_config.validator_system_prompt, lines=12, max_lines=20)
            validator_user = gr.Textbox(label="用户 Prompt 模板", value=prompts_config.validator_user_prompt, lines=12, max_lines=20)

        with gr.Accordion("🔍 负样本验证者 Prompt", open=False):
            neg_validator_system = gr.Textbox(label="系统 Prompt（负样本验证）", value=prompts_config.negative_validator_system_prompt, lines=12, max_lines=20)
            neg_validator_user = gr.Textbox(label="用户 Prompt 模板（负样本验证）", value=prompts_config.negative_validator_user_prompt, lines=10, max_lines=15)

        with gr.Accordion("🧪 策略模型采样 Prompt（qwen3-8b）", open=False):
            sampling_system = gr.Textbox(label="系统 Prompt（8B自然推理，无引导）", value=prompts_config.sampling_system_prompt, lines=10, max_lines=18)
            sampling_user = gr.Textbox(label="用户 Prompt 模板（8B采样）", value=prompts_config.sampling_user_prompt, lines=8, max_lines=12)

        with gr.Accordion("🏷️ 策略模型验证者 Prompt（8B样本标注）", open=False):
            sampling_val_system = gr.Textbox(label="系统 Prompt（8B样本正/负标注）", value=prompts_config.sampling_validator_system_prompt, lines=12, max_lines=20)
            sampling_val_user = gr.Textbox(label="用户 Prompt 模板（8B样本验证）", value=prompts_config.sampling_validator_user_prompt, lines=12, max_lines=20)

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
            'sampling_system': sampling_system,
            'sampling_user': sampling_user,
            'sampling_val_system': sampling_val_system,
            'sampling_val_user': sampling_val_user,
            'save_prompts_btn': save_prompts_btn,
            'prompts_status': prompts_status
        }
