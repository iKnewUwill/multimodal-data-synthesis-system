"""HTML generation service for task display."""
import json
from pathlib import Path

from src.models import FinancialTaskInput, TaskStatus
from src.task_manager import TaskManager
from config.settings import settings


class HTMLGenerator:
    """Centralized HTML generation for task display."""
    
    @staticmethod
    def task_detail_html(task_id: str, task_manager: TaskManager) -> str:
        """Generate task detail HTML with QA results.
        
        Args:
            task_id: ID of the task to display
            task_manager: TaskManager instance to retrieve task data
            
        Returns:
            HTML string for task detail display
        """
        if not task_id or task_id == "None" or not task_id.strip():
            return "<div class='scrollable-box'>请选择一个任务查看详情</div>"
        
        try:
            task = task_manager.get_task(task_id)
            if not task:
                return "<div class='scrollable-box'>任务不存在</div>"
            
            # Basic task information
            detail_html = f"""
            <div class='scrollable-box'>
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h3 style='margin: 0; color: white;'>📊 任务基本信息</h3>
                </div>
                <div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 15px;'>
                    <p><strong>🏢 公司名称:</strong> {task.公司名称}</p>
                    <p><strong>📈 证券代码:</strong> {task.证券代码}</p>
                    <p><strong>📋 评估维度:</strong> {task.评估维度}</p>
                    <p><strong>⏰ 统计截止日期:</strong> {task.统计截止日期}</p>
                    <p><strong>📊 状态:</strong> <span style='padding: 3px 10px; border-radius: 12px; background: {'#28a745' if task.status == TaskStatus.COMPLETED else '#ffc107'}; color: white;'>{task.status.value}</span></p>
                </div>
            """
            
            # If task is completed, try to load QA results
            if task.status == TaskStatus.COMPLETED:
                try:
                    result_file = settings.OUTPUT_DIR / f"{task_id}.json"
                    if result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                        
                        qa_count = result_data.get('valid_qa_count', 0)
                        total_iterations = result_data.get('total_iterations', 0)
                        
                        detail_html += f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                            <h3 style='margin: 0; color: white;'>🎯 QA 生成结果</h3>
                        </div>
                        <div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 15px;'>
                            <p><strong>✅ 有效QA对数量:</strong> {qa_count}</p>
                            <p><strong>🔄 总迭代次数:</strong> {total_iterations}</p>
                            <p><strong>📈 成功率:</strong> {qa_count/total_iterations*100 if total_iterations > 0 else 0:.1f}%</p>
                        </div>
                        """
                        
                        qa_pairs = result_data.get('qa_pairs', [])
                        if qa_pairs:
                            detail_html += "<h4 style='color: #667eea; margin-top: 20px;'>💬 问答对详情（显示前5个）</h4>"
                            
                            for i, qa in enumerate(qa_pairs[:5], 1):
                                question = qa.get('question', '')
                                analysis_process = qa.get('analysis_process', {})
                                conclusion = qa.get('conclusion', '')
                                difficulty = qa.get('difficulty', 0)
                                iteration = qa.get('iteration', 0)
                                created_at = qa.get('created_at', '')

                                # Get validation reason from analysis_process
                                validation_reason = analysis_process.get('validation_reason', '')
                                predicted_answer = analysis_process.get('predicted_answer', '')
                                predicted_analysis_process_dict = analysis_process.get('predicted_analysis_process', '')
                                predicted_analysis_process = '\n'.join(f'{k} : {predicted_analysis_process_dict[k]}' for k in predicted_analysis_process_dict)
                                
                                reference_answer = analysis_process.get('reference_answer', '')
                                reference_analysis_process_dict = analysis_process.get('reference_analysis_process', '')
                                reference_analysis_process = '\n'.join(f'{k} : {reference_analysis_process_dict[k]}' for k in reference_analysis_process_dict)
                                is_valid = analysis_process.get('is_valid', '')

                                detail_html += f"""
                                <div class='qa-item'>
                                    <div class='qa-question'>❓ Q{i}: {question}</div>
                                    <div class='qa-answer'>
                                        <div style='background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='color: #1976D2; font-weight: bold; margin-bottom: 5px;'>🤖 预测答案:</div>
                                            <div style='margin-left: 10px;'>{predicted_answer}</div>
                                        </div>
                                        <div style='background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='color: #1976D2; font-weight: bold; margin-bottom: 5px;'>🤖 预测过程:</div>
                                            <div style='margin-left: 10px;'>{predicted_analysis_process}</div>
                                        </div>
                                        <div style='background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='color: #388E3C; font-weight: bold; margin-bottom: 5px;'>📝 参考答案:</div>
                                            <div style='margin-left: 10px;'>{reference_answer}</div>
                                        </div>
                                        <div style='background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='color: #388E3C; font-weight: bold; margin-bottom: 5px;'>📝 参考过程:</div>
                                            <div style='margin-left: 10px;'>{reference_analysis_process}</div>
                                        </div>
                                        <div style='background: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='color: #F57C00; font-weight: bold; margin-bottom: 5px;'>✨ 最终结论:</div>
                                            <div style='margin-left: 10px;'>{conclusion}</div>
                                        </div>
                                        <div style='background: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='color: #F57C00; font-weight: bold; margin-bottom: 5px;'>✨ 是否通过验证:</div>
                                            <div style='margin-left: 10px;'>{"是" if is_valid else "否"}</div>
                                        </div>
                                    </div>
                                    <div class='qa-metrics'>
                                        <strong>📊 质量指标:</strong>
                                        难度系数: {difficulty:.2f} |
                                        迭代次数: {iteration} |
                                        创建时间: {created_at}
                                    </div>
                                    <div style='background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 5px 0; font-size: 0.9em;'>
                                        <div style='color: #1976D2; font-weight: bold; margin-bottom: 5px;'>🔍 验证理由:</div>
                                        <div style='margin-left: 10px;'>{validation_reason[:200]}{'...' if len(validation_reason) > 200 else ''}</div>
                                    </div>
                                </div>
                                """
                            
                            if len(qa_pairs) > 5:
                                detail_html += f"<p style='text-align: center; color: #757575;'>... 还有 {len(qa_pairs) - 5} 个问答对（完整结果保存在输出文件中）</p>"
                        else:
                            detail_html += "<p style='color: #757575; text-align: center;'>暂无问答对数据</p>"
                    else:
                        detail_html += "<div style='background: #fff3cd; color: #856404; padding: 15px; border-radius: 8px;'>⚠️ 结果文件未找到，任务可能尚未完成或文件已丢失</div>"
                
                except Exception as e:
                    detail_html += f"<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px;'>❌ 加载结果文件失败: {str(e)}</div>"
            else:
                detail_html += "<div style='background: #e3f2fd; color: #1976D2; padding: 15px; border-radius: 8px;'>⏳ 任务尚未完成，请等待任务处理完成后再查看QA结果</div>"
            
            detail_html += "</div>"
            return detail_html
        
        except Exception as e:
            return f"<div class='scrollable-box' style='background: #f8d7da; color: #721c24;'>❌ 显示任务详情时发生错误: {str(e)}</div>"
