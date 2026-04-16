"""UI Styles - Custom CSS for Gradio interface"""

CUSTOM_CSS = """
.container { max-width: 1400px; margin: auto; }
.header { text-align: center; padding: 20px; background: #ffffff; color: #333333; border-radius: 10px; margin-bottom: 20px; border: 2px solid #e0e0e0; }
.config-box { background: #ffffff; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e0e0e0; }
.progress-dashboard { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 2px solid #90a4ae; }
.scrollable-box { max-height: 600px; overflow-y: auto; padding: 15px; background: #fafafa; border-radius: 8px; border: 2px solid #e0e0e0; }
.task-item { padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #e0e0e0; cursor: pointer; transition: all 0.3s; }
.task-item:hover { background: #f0f8ff; border-color: #2196F3; transform: translateX(5px); box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
.task-item.selected { background: #2196F3 !important; color: white !important; border-color: #1976D2 !important; box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3); }
.task-item.selected:hover { background: #1976D2 !important; }
.task-item.selected p { color: white !important; }
.task-pending { background: #fff9c4; border-left: 4px solid #ffc107; }
.task-processing { background: #e3f2fd; border-left: 4px solid #2196F3; }
.task-completed { background: #d4edda; border-left: 4px solid #28a745; }
.task-failed { background: #f8d7da; border-left: 4px solid #dc3545; }
.status-badge { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; margin: 5px; }
.status-running { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
.status-completed { background: #d4edda; color: #155724; border: 2px solid #28a745; }
.status-failed { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
.qa-item { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.qa-question { font-weight: bold; color: #1976D2; margin-bottom: 8px; }
.qa-answer { color: #424242; margin-bottom: 8px; line-height: 1.6; }
.qa-metrics { font-size: 0.9em; color: #757575; background: #f5f5f5; padding: 5px 10px; border-radius: 4px; display: inline-block; }
.log-box { font-family: 'Courier New', monospace; background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; max-height: 400px; overflow-y: auto; }
.log-info { color: #4fc3f7; }
.log-success { color: #66bb6a; }
.log-error { color: #ef5350; }
"""
