"""Data conversion utilities for Gradio UI components"""

from typing import List
import pandas as pd
from src.models import FinancialTaskInput, TaskStatus


class TaskDataConverter:
    """Convert task data to DataFrame format for Gradio UI"""
    
    @staticmethod
    def tasks_to_dataframe(tasks: List[FinancialTaskInput]) -> pd.DataFrame:
        """Convert FinancialTaskInput list to pandas DataFrame
        
        Args:
            tasks: List of financial tasks
            
        Returns:
            DataFrame with columns: 状态, 公司名称, 证券代码, 评估维度, 任务ID
            If empty, returns single-row DataFrame with custom message
        """
        # Define column structure
        columns = ["状态", "公司名称", "证券代码", "评估维度", "任务ID"]
        
        # Handle empty state with custom message
        if not tasks:
            return pd.DataFrame([{
                "状态": "暂无任务，请上传文件",
                "公司名称": "",
                "证券代码": "",
                "评估维度": "",
                "任务ID": ""
            }], columns=columns)
        
        # Convert tasks to DataFrame rows
        data = []
        for task in tasks:
            # Map status to emoji
            status_emoji = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.PROCESSING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌"
            }.get(task.status, "📋")
            
            data.append({
                "状态": f"{status_emoji} {task.status.value}",
                "公司名称": task.公司名称,
                "证券代码": task.证券代码,
                "评估维度": task.评估维度,
                "任务ID": task.task_id
            })
        
        return pd.DataFrame(data, columns=columns)
    
    @staticmethod
    def get_task_id_from_selection(df: pd.DataFrame, selection) -> str:
        """Extract task_id from selected DataFrame row
        
        Args:
            df: DataFrame containing task data
            selection: Selection data (can be None, pd.NA, or row index)
            
        Returns:
            Task ID string, or None/empty string if no valid selection
        """
        # Handle no selection
        if selection is None or pd.isna(selection):
            return ""
        
        # Handle empty DataFrame or custom message row
        if len(df) == 0 or "暂无任务" in str(df.iloc[0]["状态"]):
            return ""
        
        try:
            # Get the row index (could be int or list)
            row_idx = selection if isinstance(selection, int) else (selection[0] if isinstance(selection, list) else int(selection))
            
            # Extract task_id from the selected row
            task_id = df.iloc[row_idx]["任务ID"]
            return task_id if task_id else ""
            
        except (IndexError, KeyError, TypeError):
            return ""
