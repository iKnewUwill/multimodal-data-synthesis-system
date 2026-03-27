"""Web UI - 基于 Gradio 的可视化界面"""

import os
import json
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

from config.llm_config import llm_config
from config.prompts import prompts_config
from config.settings import settings
from src.models import (
    SynthesisTask, ImageInfo, AgentState,
    TaskType, SynthesisResult
)
from src.graph import MultimodalSynthesisGraph
from src.utils import generate_task_id, save_json, read_document_content, get_file_type


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
    .iteration-box {
        border-left: 4px solid #333333;
        padding: 15px;
        margin: 10px 0;
        background: #ffffff;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    .proposer-output {
        background: #e3f2fd;
        border-left: 5px solid #2196F3;
        border: 2px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .solver-output {
        background: #f3e5f5;
        border-left: 5px solid #9c27b0;
        border: 2px solid #9c27b0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .validator-output {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
        border: 2px solid #4caf50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .failed-output {
        background: #ffebee;
        border-left: 5px solid #f44336;
        border: 2px solid #f44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
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
    .gallery-container {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 10px;
        background: #fafafa;
    }
    """

    def __init__(self):
        self.graph = None
        self.current_task_id = None
    
    def create_interface(self):
        """创建 Gradio 界面"""

        with gr.Blocks(title="多模态数据合成系统") as interface:
            
            # 标题
            gr.HTML("""
            <div class="header">
                <h1>🤖 多模态数据合成系统</h1>
                <p>基于 Multi-Agent 的高质量多模态训练数据合成平台</p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: 数据合成
                with gr.Tab("📊 数据合成"):
                    with gr.Row():
                        # 左侧：配置区域
                        with gr.Column(scale=1):
                            gr.Markdown("### 📁 文件上传")
                            file_input = gr.File(
                                label="上传文件（支持图片和文档）",
                                file_count="multiple",
                                file_types=["image", ".txt", ".csv", ".json", ".md", ".xlsx", ".pdf"]
                            )

                            uploaded_files_display = gr.Gallery(
                                label="已上传的文件",
                                show_label=True,
                                columns=2,
                                height="auto",
                                elem_classes=["gallery-container"]
                            )
                            
                            gr.Markdown("### ⚙️ 任务配置")
                            
                            with gr.Group():
                                task_type = gr.Dropdown(
                                    choices=[t.value for t in TaskType],
                                    value=TaskType.IMAGE_QA.value,
                                    label="任务类型",
                                    info="选择要合成的数据类型"
                                )
                                
                                custom_task_input = gr.Textbox(
                                    label="自定义任务类型（仅当选择'自定义'时）",
                                    placeholder="例如：图表数据分析类",
                                    visible=False
                                )
                                
                                task_description = gr.Textbox(
                                    label="任务描述（可选）",
                                    placeholder="为任务类型添加额外的描述信息...",
                                    lines=3
                                )
                                
                                max_iterations = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label="最大迭代次数",
                                    info="生成的问答对数量"
                                )
                                
                                initial_difficulty = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.5,
                                    value=0.3,
                                    step=0.1,
                                    label="初始难度",
                                    info="第一个问题的难度（0-1）"
                                )
                                
                                difficulty_increment = gr.Slider(
                                    minimum=0.05,
                                    maximum=0.2,
                                    value=0.1,
                                    step=0.05,
                                    label="难度递增步长",
                                    info="每次迭代增加的难度"
                                )
                            
                            start_btn = gr.Button("🚀 开始合成", variant="primary", size="lg")
                            stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")
                        
                        # 右侧：结果显示区域
                        with gr.Column(scale=2):
                            # 实时进度看板
                            with gr.Group():
                                gr.Markdown("### 📊 实时进度看板")
                                with gr.Row():
                                    current_iteration = gr.Textbox(
                                        label="当前迭代",
                                        value="0/0",
                                        interactive=False,
                                        scale=1
                                    )
                                    current_difficulty = gr.Textbox(
                                        label="当前难度",
                                        value="--",
                                        interactive=False,
                                        scale=1
                                    )
                                    valid_count = gr.Textbox(
                                        label="已生成问答对",
                                        value="0",
                                        interactive=False,
                                        scale=1
                                    )
                                
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
                            
                            # 实时过程（可滚动）
                            gr.Markdown("### 🔄 Agent 执行过程")
                            iteration_display = gr.HTML(
                                "<div class='scrollable-box'>等待开始...</div>"
                            )
                            
                            # 已验证的问答对（可滚动）
                            gr.Markdown("### ✅ 已验证的问答对")
                            validated_qa_display = gr.HTML(
                                "<div class='scrollable-box'>暂无数据</div>"
                            )
                            
                            # 导出结果
                            gr.Markdown("### 💾 导出结果")
                            with gr.Row():
                                export_json_btn = gr.Button("📥 导出 JSON")
                                export_path = gr.Textbox(
                                    label="导出路径",
                                    interactive=False
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
            def update_custom_task_visibility(task_type_value):
                """根据任务类型显示/隐藏自定义输入框"""
                return gr.update(visible=(task_type_value == "自定义"))
            
            task_type.change(
                fn=update_custom_task_visibility,
                inputs=[task_type],
                outputs=[custom_task_input]
            )
            
            def handle_file_upload(files):
                """处理文件上传（支持图片和文档）"""
                if not files:
                    return []

                # 处理文件列表
                file_info = []
                image_files = []

                for file in files:
                    file_path = file.name if hasattr(file, 'name') else file
                    file_type = get_file_type(file_path)

                    if file_type == 'image':
                        image_files.append(file_path)
                        file_info.append(("图片", file_path))

                    # 文档文件处理逻辑移到 start_synthesis 中

                # 返回图片文件用于 Gallery 显示
                return image_files if image_files else []

            file_input.change(
                fn=handle_file_upload,
                inputs=[file_input],
                outputs=[uploaded_files_display]
            )
            
            def start_synthesis(
                files,
                task_type_value,
                custom_task_value,
                task_desc,
                max_iter,
                init_diff,
                diff_inc
            ):
                """开始数据合成（支持图片和文档）"""
                if not files:
                    yield (
                        "0/0", "--", "0", 0,
                        "<div class='status-badge status-failed'>❌ 请先上传文件</div>",
                        "<div class='scrollable-box'>请先上传文件</div>",
                        "<div class='scrollable-box'>暂无数据</div>"
                    )
                    return

                # 确定任务类型
                final_task_type = custom_task_value if task_type_value == "自定义" else task_type_value

                # 处理上传的文件
                image_infos = []
                image_paths = []
                file_contents = []
                file_info_list = []

                for file in files:
                    file_path = file.name if hasattr(file, 'name') else file
                    file_type = get_file_type(file_path)

                    # 复制文件到上传目录
                    filename = Path(file_path).name
                    dest_path = settings.UPLOAD_DIR / filename
                    import shutil
                    shutil.copy(file_path, dest_path)

                    if file_type == 'image':
                        # 处理图片文件
                        image_infos.append(ImageInfo(
                            path=str(dest_path),
                            filename=filename
                        ))
                        image_paths.append(str(dest_path))

                    else:
                        # 处理文档文件
                        content = read_document_content(str(dest_path))
                        file_contents.append(content)
                        file_info_list.append({
                            "filename": filename,
                            "file_type": file_type,
                            "path": str(dest_path),
                            "content_preview": content[:200] + "..." if len(content) > 200 else content
                        })

                # 创建任务
                task_id = generate_task_id()

                # 构建任务描述
                enhanced_desc = task_desc if task_desc else ""
                if file_info_list:
                    file_info_text = f"\n上传文件: {', '.join([f['filename'] for f in file_info_list])}"
                    enhanced_desc += file_info_text

                task = SynthesisTask(
                    task_id=task_id,
                    task_type=final_task_type,
                    task_description=enhanced_desc if enhanced_desc else None,
                    images=image_infos,
                    max_iterations=int(max_iter),
                    initial_difficulty=init_diff,
                    difficulty_increment=diff_inc
                )

                # 初始化状态
                initial_state = AgentState(
                    task=task,
                    image_paths=image_paths,
                    file_contents=file_contents,
                    current_difficulty=init_diff
                )

                # 创建图实例
                graph = MultimodalSynthesisGraph(llm_config)

                # 初始状态
                file_summary = f"<p>🖼️ 图片数量：{len(image_paths)}</p>" if image_paths else ""
                doc_summary = f"<p>📄 文档数量：{len(file_contents)}</p>" if file_contents else ""

                yield (
                    f"0/{int(max_iter)}", f"{init_diff:.2f}", "0", 0,
                    f"<div class='status-badge status-running'>🚀 开始合成 - 任务ID: {task_id}</div>",
                    f"<div class='scrollable-box'><p>📋 任务类型：{final_task_type}</p>{file_summary}{doc_summary}<p>🔢 最大迭代：{max_iter}</p></div>",
                    "<div class='scrollable-box'>暂无数据</div>"
                )

                try:
                    # 运行合成过程
                    state = initial_state
                    all_iterations_html = ""

                    for iteration in range(1, int(max_iter) + 1):
                        # 手动执行每个步骤
                        state.current_iteration = iteration
                        state.current_difficulty = min(
                            init_diff + (iteration - 1) * diff_inc,
                            settings.MAX_DIFFICULTY
                        )

                        # 计算进度
                        progress_percent = int((iteration / int(max_iter)) * 100)

                        # 构建当前迭代的 HTML
                        iteration_html = f"<div style='margin-bottom: 20px; border: 2px solid #ddd; padding: 10px; border-radius: 8px;'>"
                        iteration_html += f"<h3>🔄 迭代 {iteration} - 难度: {state.current_difficulty:.2f}</h3>"

                        # 提议者
                        iteration_html += '<div class="proposer-output">'
                        iteration_html += "<h4>💡 提议者</h4>"

                        try:
                            proposer_output = graph.proposer.propose(
                                image_paths=image_paths if image_paths else None,
                                file_contents=file_contents if file_contents else None,
                                task_type=final_task_type,
                                difficulty=state.current_difficulty,
                                history_qa_pairs=state.history_qa_pairs
                            )

                            iteration_html += f"<p><strong>问题：</strong>{proposer_output.question}</p>"
                            iteration_html += f"<p><strong>参考答案：</strong>{proposer_output.answer}</p>"
                            iteration_html += "</div>"

                            # 更新状态并 yield
                            all_iterations_html = iteration_html + all_iterations_html
                            yield (
                                f"{iteration}/{int(max_iter)}",
                                f"{state.current_difficulty:.2f}",
                                str(len(state.history_qa_pairs)),
                                progress_percent,
                                "<div class='status-badge status-running'>⏳ 提议者工作中...</div>",
                                f"<div class='scrollable-box'>{all_iterations_html}</div>",
                                f"<div class='scrollable-box'>已生成 {len(state.history_qa_pairs)} 对问答</div>"
                            )

                            # 求解者
                            iteration_html += '<div class="solver-output">'
                            iteration_html += "<h4>🤔 求解者</h4>"

                            solver_output = graph.solver.solve(
                                image_paths=image_paths if image_paths else None,
                                file_contents=file_contents if file_contents else None,
                                question=proposer_output.question
                            )

                            iteration_html += f"<p><strong>预测答案：</strong>{solver_output.answer}</p>"
                            iteration_html += "</div>"

                            all_iterations_html = iteration_html.replace("</div>", "</div>", 1) + all_iterations_html.split("</div>", 1)[1] if "</div>" in all_iterations_html else iteration_html
                            yield (
                                f"{iteration}/{int(max_iter)}",
                                f"{state.current_difficulty:.2f}",
                                str(len(state.history_qa_pairs)),
                                progress_percent,
                                "<div class='status-badge status-running'>⏳ 求解者工作中...</div>",
                                f"<div class='scrollable-box'>{all_iterations_html}</div>",
                                f"<div class='scrollable-box'>已生成 {len(state.history_qa_pairs)} 对问答</div>"
                            )

                            # 验证者
                            iteration_html += '<div class="validator-output">'
                            iteration_html += "<h4>✅ 验证者</h4>"

                            validation = graph.validator.validate(
                                image_paths=image_paths if image_paths else None,
                                file_contents=file_contents if file_contents else None,
                                question=proposer_output.question,
                                reference_answer=proposer_output.answer,
                                predicted_answer=solver_output.answer
                            )

                            iteration_html += f"<p><strong>验证结果：</strong>{'✅ 通过' if validation.is_valid else '❌ 未通过'}</p>"
                            iteration_html += f"<p><strong>相似度分数：</strong>{validation.similarity_score:.2f}</p>"
                            iteration_html += f"<p><strong>理由：</strong>{validation.reason}</p>"
                            iteration_html += "</div>"
                            iteration_html += "</div>"

                            # 更新历史
                            validated_html = "<div class='scrollable-box'>"
                            if validation.is_valid:
                                from src.models import QAPair
                                qa_pair = QAPair(
                                    question=proposer_output.question,
                                    answer=proposer_output.answer,
                                    difficulty=state.current_difficulty,
                                    iteration=iteration
                                )
                                state.history_qa_pairs.append(qa_pair)

                                # 更新已验证的问答对显示
                                validated_html += f"<h3>✅ 已生成 {len(state.history_qa_pairs)} 对问答</h3>"
                                for i, qa in enumerate(state.history_qa_pairs, 1):
                                    validated_html += f"<div style='background: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #1976d2;'>"
                                    validated_html += f"<h4>问答对 {i}（难度：{qa.difficulty:.2f}）</h4>"
                                    validated_html += f"<p><strong>Q：</strong>{qa.question}</p>"
                                    validated_html += f"<p><strong>A：</strong>{qa.answer}</p>"
                                    validated_html += "</div>"
                            else:
                                validated_html += f"<p>已生成 {len(state.history_qa_pairs)} 对问答</p>"
                            validated_html += "</div>"

                            all_iterations_html = iteration_html + all_iterations_html
                            yield (
                                f"{iteration}/{int(max_iter)}",
                                f"{state.current_difficulty:.2f}",
                                str(len(state.history_qa_pairs)),
                                progress_percent,
                                f"<div class='status-badge status-running'>🔄 迭代 {iteration}/{int(max_iter)} 完成</div>",
                                f"<div class='scrollable-box'>{all_iterations_html}</div>",
                                validated_html
                            )

                        except Exception as e:
                            iteration_html += f'<div class="failed-output">'
                            iteration_html += f"<h4>❌ 错误</h4><p>{str(e)}</p>"
                            iteration_html += "</div></div>"
                            all_iterations_html = iteration_html + all_iterations_html
                            yield (
                                f"{iteration}/{int(max_iter)}",
                                f"{state.current_difficulty:.2f}",
                                str(len(state.history_qa_pairs)),
                                progress_percent,
                                "<div class='status-badge status-failed'>❌ 执行失败</div>",
                                f"<div class='scrollable-box'>{all_iterations_html}</div>",
                                validated_html if 'validated_html' in locals() else "<div class='scrollable-box'>暂无数据</div>"
                            )

                    # 完成
                    result = SynthesisResult(
                        task_id=task_id,
                        task_type=final_task_type,
                        images=image_infos,
                        qa_pairs=state.history_qa_pairs,
                        total_iterations=int(max_iter),
                        valid_qa_count=len(state.history_qa_pairs),
                        completed_at=datetime.now()
                    )

                    output_file = settings.OUTPUT_DIR / f"{task_id}.json"
                    save_json(result.dict(), output_file)

                    yield (
                        f"{int(max_iter)}/{int(max_iter)}",
                        f"{state.current_difficulty:.2f}",
                        str(len(state.history_qa_pairs)),
                        100,
                        f"<div class='status-badge status-completed'>✅ 合成完成！有效问答对: {len(state.history_qa_pairs)}</div>",
                        f"<div class='scrollable-box'>{all_iterations_html}</div>",
                        validated_html
                    )

                except Exception as e:
                    yield (
                        "错误", "--", "0", 0,
                        f"<div class='status-badge status-failed'>❌ 合成失败: {str(e)}</div>",
                        f"<div class='scrollable-box'><p style='color: red;'>错误: {str(e)}</p></div>",
                        "<div class='scrollable-box'>暂无数据</div>"
                    )
            
            start_btn.click(
                fn=start_synthesis,
                inputs=[
                    file_input,
                    task_type,
                    custom_task_input,
                    task_description,
                    max_iterations,
                    initial_difficulty,
                    difficulty_increment
                ],
                outputs=[
                    current_iteration, 
                    current_difficulty, 
                    valid_count, 
                    progress_bar, 
                    status_text, 
                    iteration_display, 
                    validated_qa_display
                ]
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
