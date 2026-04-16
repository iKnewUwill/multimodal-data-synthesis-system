"""Main UI Application - Assembles components and wires event handlers"""

import gradio as gr
from queue import Queue
import threading

from src.task_manager import TaskManager
from config.llm_config import get_llm_config
from config.prompts import get_prompts_config
from web_ui.styles import CUSTOM_CSS
from web_ui.components import UIComponents
from web_ui.handlers import UIHandlers


class MultimodalSynthesisUI:
    """Main UI class - orchestrates components and handlers"""
    
    def __init__(self):
        self.task_manager = TaskManager()
        self.log_queue = Queue()
        self.log_lock = threading.Lock()
        self.llm_config = get_llm_config()
        self.prompts_config = get_prompts_config()
        self.handlers = UIHandlers(self.task_manager, self.log_queue, self.log_lock)
    
    def create_interface(self):
        """Assemble interface from components + wire handlers"""
        with gr.Blocks(title="金融财务数据合成系统", css=CUSTOM_CSS) as interface:
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🤖 金融财务数据合成系统</h1>
                <p>基于 Multi-Agent 的高质量金融财务分析训练数据合成平台</p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: Batch Processing
                with gr.Tab("📊 批量任务处理"):
                    batch_inputs, batch_outputs = UIComponents.build_batch_tab()
                    self._wire_batch_handlers(batch_inputs, batch_outputs)
                
                # Tab 2: LLM Config
                with gr.Tab("🔧 LLM 配置"):
                    llm_inputs = UIComponents.build_llm_config_tab(self.llm_config)
                    self._wire_llm_config_handlers(llm_inputs)
                
                # Tab 3: Prompts Config
                with gr.Tab("📝 Prompt 配置"):
                    prompts_inputs = UIComponents.build_prompts_config_tab(self.prompts_config)
                    self._wire_prompts_config_handlers(prompts_inputs)
            
            # Auto-load historical tasks on interface load
            interface.load(
                fn=self.handlers.refresh_task_list,
                outputs=[
                    batch_outputs['total_tasks'],
                    batch_outputs['pending_tasks'],
                    batch_outputs['processing_tasks'],
                    batch_outputs['completed_tasks'],
                    batch_outputs['failed_tasks'],
                    batch_outputs['task_dataframe'],
                    batch_outputs['load_status']
                ]
            )
        
        return interface
    
    def _wire_batch_handlers(self, inputs, outputs):
        """Connect batch processing handlers to components"""
        # File upload
        inputs['file_input'].change(
            fn=self.handlers.load_json_file,
            inputs=[inputs['file_input']],
            outputs=[
                outputs['load_status'],
                gr.State(),
                outputs['total_tasks'],
                outputs['pending_tasks'],
                outputs['processing_tasks'],
                outputs['completed_tasks'],
                outputs['failed_tasks'],
                outputs['task_dataframe']
            ]
        )
        
        # Refresh button
        inputs['refresh_btn'].click(
            fn=self.handlers.refresh_task_list,
            outputs=[
                outputs['total_tasks'],
                outputs['pending_tasks'],
                outputs['processing_tasks'],
                outputs['completed_tasks'],
                outputs['failed_tasks'],
                outputs['task_dataframe'],
                outputs['load_status']
            ]
        )
        
        # Start button
        inputs['start_btn'].click(
            fn=self.handlers.start_batch_processing,
            inputs=[inputs['max_iterations'], inputs['parallel_count']],
            outputs=[
                outputs['total_tasks'],
                outputs['pending_tasks'],
                outputs['processing_tasks'],
                outputs['completed_tasks'],
                outputs['failed_tasks'],
                outputs['task_dataframe'],
                outputs['log_display'],
                outputs['progress_bar'],
                outputs['status_text']
            ]
        )
        
        # Stop button
        inputs['stop_btn'].click(
            fn=self.handlers.stop_processing,
            outputs=[outputs['stop_status']]
        )
        
        # DataFrame selection - direct click interaction
        outputs['task_dataframe'].select(
            fn=self.handlers.handle_dataframe_selection,
            inputs=[outputs['task_dataframe']],
            outputs=[outputs['task_detail_display']]
        )
    
    def _wire_llm_config_handlers(self, inputs):
        """Connect LLM config handlers"""
        inputs['save_llm_config_btn'].click(
            fn=self.handlers.save_llm_config,
            inputs=[
                inputs['api_key_input'],
                inputs['base_url_input'],
                inputs['model_name_input'],
                inputs['temperature_input'],
                inputs['max_tokens_input']
            ],
            outputs=[inputs['llm_config_status']]
        )
    
    def _wire_prompts_config_handlers(self, inputs):
        """Connect prompts config handlers"""
        inputs['save_prompts_btn'].click(
            fn=self.handlers.save_prompts_config,
            inputs=[
                inputs['proposer_system'],
                inputs['proposer_user'],
                inputs['solver_system'],
                inputs['solver_user'],
                inputs['validator_system'],
                inputs['validator_user']
            ],
            outputs=[inputs['prompts_status']]
        )


def launch_ui():
    """Launch Web UI - Entry point for backward compatibility"""
    ui = MultimodalSynthesisUI()
    interface = ui.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    launch_ui()
