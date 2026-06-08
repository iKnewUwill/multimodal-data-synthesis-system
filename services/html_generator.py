"""HTML generation service for task display."""
import json
from pathlib import Path

from src.models import FinancialTaskInput, TaskStatus
from src.task_manager import TaskManager
from config.settings import settings


class HTMLGenerator:
    """Centralized HTML generation for task display."""

    @staticmethod
    def _source_badge(source: str, is_positive: bool, sample_index: int = 0) -> str:
        """Generate sample source badge"""
        if source == "positive_solver":
            color = "#388E3C"
            bg = "#e8f5e9"
            label = "正样本求解器 (qwen3.7-max)"
        elif source == "negative_solver":
            color = "#C62828"
            bg = "#fce4ec"
            label = "负样本求解器 (qwen3.7-max·错误注入)"
        elif source == "strategy_model":
            if is_positive:
                color = "#1565C0"
                bg = "#e3f2fd"
                label = f"策略模型 #{sample_index} (qwen3-8b) → 正样本"
            else:
                color = "#E65100"
                bg = "#fff3e0"
                label = f"策略模型 #{sample_index} (qwen3-8b) → 负样本（真实错误）"
        else:
            color = "#757575"
            bg = "#f5f5f5"
            label = source
        return f"<span style='background:{bg};color:{color};padding:3px 10px;border-radius:12px;font-weight:bold'>{label}</span>"

    @staticmethod
    def task_detail_html(task_id: str, task_manager: TaskManager) -> str:
        """Generate task detail HTML with sample sets."""
        if not task_id or task_id == "None" or not task_id.strip():
            return "<div class='scrollable-box'>请选择一个任务查看详情</div>"

        try:
            task = task_manager.get_task(task_id)
            if not task:
                return "<div class='scrollable-box'>任务不存在</div>"

            detail_html = f"""
            <div class='scrollable-box'>
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h3 style='margin: 0; color: white;'>任务基本信息</h3>
                </div>
                <div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 15px;'>
                    <p><strong>公司名称:</strong> {task.公司名称}</p>
                    <p><strong>证券代码:</strong> {task.证券代码}</p>
                    <p><strong>评估维度:</strong> {task.评估维度}</p>
                    <p><strong>统计截止日期:</strong> {task.统计截止日期}</p>
                    <p><strong>状态:</strong> <span style='padding: 3px 10px; border-radius: 12px; background: {'#28a745' if task.status == TaskStatus.COMPLETED else '#ffc107'}; color: white;'>{task.status.value}</span></p>
                </div>
            """

            if task.status == TaskStatus.COMPLETED:
                try:
                    result_file = settings.OUTPUT_DIR / f"{task_id}.json"
                    if result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)

                        # Check if new sample_sets structure exists
                        sample_sets = result_data.get('sample_sets', [])
                        if sample_sets:
                            detail_html += f"""
                            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                <h3 style='margin: 0; color: white;'>生成结果: {len(sample_sets)} 个问题</h3>
                            </div>
                            """

                            for set_idx, ss in enumerate(sample_sets):
                                difficulty = ss.get('difficulty', 0)
                                question = ss.get('question', '')
                                standard_answer = ss.get('standard_answer', '')
                                standard_process = ss.get('standard_analysis_process', {})

                                detail_html += f"""
                                <div style='background: white; border: 2px solid #667eea; border-radius: 10px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                                    <div style='background: #667eea; color: white; padding: 8px 15px; border-radius: 6px; margin-bottom: 12px; display: flex; justify-content: space-between;'>
                                        <strong>问题 {set_idx + 1}</strong>
                                        <span>难度: {difficulty:.2f}（随机）</span>
                                    </div>

                                    <div style='background: #f3e5f5; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #9c27b0;'>
                                        <div style='color: #6a1b9a; font-weight: bold; margin-bottom: 5px;'>问题:</div>
                                        <div>{question}</div>
                                    </div>

                                    <div style='background: #e8f5e9; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #4caf50;'>
                                        <div style='color: #2e7d32; font-weight: bold; margin-bottom: 5px;'>标准答案（Proposer 与问题一同生成）:</div>
                                        <div style='margin-bottom: 8px;'><strong>结论:</strong> {standard_answer}</div>
                                        {HTMLGenerator._format_steps(standard_process)}
                                    </div>
                                """

                                # Positive sample
                                ps = ss.get('positive_sample')
                                if ps:
                                    pv = ps.get('validation', {})
                                    detail_html += f"""
                                    <div style='background: #e8f5e9; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid #66bb6a;'>
                                        {HTMLGenerator._source_badge(ps.get('sample_source', 'positive_solver'), True)}
                                        <div style='margin-top: 8px;'><strong>结论:</strong> {ps.get('conclusion', '')}</div>
                                        {HTMLGenerator._format_steps(ps.get('analysis_process', {}))}
                                        <div style='background: #c8e6c9; padding: 6px 10px; border-radius: 4px; margin-top: 8px; font-size: 0.9em;'>
                                            验证: {'通过' if pv.get('is_valid') else '未通过'} | 相似度: {pv.get('similarity_score', 'N/A')}
                                        </div>
                                    </div>
                                    """

                                # Negative sample
                                ns = ss.get('negative_sample')
                                if ns:
                                    nv = ns.get('validation', {})
                                    detail_html += f"""
                                    <div style='background: #fce4ec; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid #ef5350;'>
                                        {HTMLGenerator._source_badge(ns.get('sample_source', 'negative_solver'), False)}
                                        <div style='margin-top: 8px;'><strong>结论:</strong> {ns.get('conclusion', '')}</div>
                                        {HTMLGenerator._format_steps(ns.get('analysis_process', {}))}
                                        <div style='background: #ffcdd2; padding: 6px 10px; border-radius: 4px; margin-top: 8px; font-size: 0.9em;'>
                                            验证: {'有效负样本' if nv.get('is_valid') else '无效'} | 相似度: {nv.get('similarity_score', 'N/A')}
                                        </div>
                                    </div>
                                    """

                                # Strategy samples
                                strategy_samples = ss.get('strategy_samples', [])
                                for si, s in enumerate(strategy_samples):
                                    sv = s.get('validation', {})
                                    is_pos = s.get('is_positive_sample', False)
                                    idx = s.get('sample_index', si + 1)
                                    bg = '#e3f2fd' if is_pos else '#fff3e0'
                                    border = '#42a5f5' if is_pos else '#ff9800'
                                    detail_html += f"""
                                    <div style='background: {bg}; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid {border};'>
                                        {HTMLGenerator._source_badge(s.get('sample_source', 'strategy_model'), is_pos, idx)}
                                        <div style='margin-top: 8px;'><strong>结论:</strong> {s.get('conclusion', '')}</div>
                                        {HTMLGenerator._format_steps(s.get('analysis_process', {}))}
                                        <div style='background: rgba(0,0,0,0.05); padding: 6px 10px; border-radius: 4px; margin-top: 8px; font-size: 0.9em;'>
                                            标注: {'正样本（推理正确）' if is_pos else '负样本（真实错误）'} | 相似度: {sv.get('similarity_score', 'N/A')}
                                        </div>
                                        <div style='color: #666; font-size: 0.85em; margin-top: 4px;'>验证理由: {sv.get('reason', '')[:150]}...</div>
                                    </div>
                                    """

                                detail_html += "</div>"  # Close sample set div

                            if len(sample_sets) > 3:
                                detail_html += f"<p style='text-align: center; color: #757575;'>... 还有 {len(sample_sets) - 3} 个问题（完整结果保存在输出文件中）</p>"
                        else:
                            detail_html += "<p style='color: #757575;'>暂无结果数据</p>"
                    else:
                        detail_html += "<div style='background: #fff3cd; color: #856404; padding: 15px; border-radius: 8px;'>结果文件未找到</div>"
                except Exception as e:
                    detail_html += f"<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px;'>加载结果失败: {str(e)}</div>"
            else:
                detail_html += "<div style='background: #e3f2fd; color: #1976D2; padding: 15px; border-radius: 8px;'>任务尚未完成</div>"

            detail_html += "</div>"
            return detail_html

        except Exception as e:
            return f"<div class='scrollable-box' style='background: #f8d7da; color: #721c24;'>显示任务详情时发生错误: {str(e)}</div>"

    @staticmethod
    def _format_steps(analysis_process: dict) -> str:
        """Format analysis steps as HTML"""
        if not analysis_process:
            return ""
        html = "<div style='margin-left: 10px; font-size: 0.9em; color: #555;'>"
        for step_name in sorted(analysis_process.keys()):
            html += f"<div style='margin: 4px 0;'><strong>{step_name}:</strong> {analysis_process[step_name]}</div>"
        html += "</div>"
        return html
