"""
任务管理器
管理任务列表、状态追踪、筛选和显示
使用 SQLite 数据库进行持久化存储
"""

from typing import List, Dict, Optional
from datetime import datetime
import threading

from src.models import FinancialTaskInput, FinancialTaskResult, TaskStatus
from src.database import DatabaseManager


class TaskManager:
    """任务管理器（使用数据库持久化）"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        初始化任务管理器
        
        Args:
            db_manager: 数据库管理器实例，如果为None则创建新实例
        """
        self._lock = threading.Lock()
        self.db = db_manager or DatabaseManager()
    
    def add_task(self, task: FinancialTaskInput) -> str:
        """添加任务到数据库"""
        with self._lock:
            self.db.add_task(task)
            return task.task_id
    
    def add_tasks(self, tasks: List[FinancialTaskInput]) -> List[str]:
        """批量添加任务"""
        task_ids = []
        for task in tasks:
            task_ids.append(self.add_task(task))
        return task_ids
    
    def update_task_status(self, task_id: str, status: TaskStatus, error_message: str = None):
        """更新任务状态"""
        with self._lock:
            self.db.update_task_status(task_id, status, error_message)
    
    def set_result(self, result: FinancialTaskResult):
        """设置任务结果"""
        with self._lock:
            self.db.save_result(result)
    
    def get_task(self, task_id: str) -> Optional[FinancialTaskInput]:
        """从数据库获取任务"""
        return self.db.get_task(task_id)
    
    def get_result(self, task_id: str) -> Optional[FinancialTaskResult]:
        """从数据库获取结果"""
        return self.db.get_result(task_id)
    
    def get_all_tasks(self, limit: Optional[int] = None) -> List[FinancialTaskInput]:
        """获取所有任务"""
        return self.db.get_all_tasks(limit)
    
    def filter_tasks(
        self,
        status: Optional[TaskStatus] = None,
        证券代码: Optional[str] = None,
        公司名称: Optional[str] = None
    ) -> List[FinancialTaskInput]:
        """
        筛选任务
        
        Args:
            status: 按状态筛选
            证券代码: 按证券代码筛选
            公司名称: 按公司名称筛选
            
        Returns:
            筛选后的任务列表
        """
        return self.db.filter_tasks(status, 证券代码, 公司名称)
    
    def get_task_list_for_display(self) -> List[Dict]:
        """
        获取用于显示的任务列表
        
        Returns:
            包含证券代码、公司名称、评估维度、状态的任务列表
        """
        tasks = self.get_all_tasks()
        display_list = []
        for task in tasks:
            display_list.append({
                "task_id": task.task_id,
                "证券代码": task.证券代码,
                "公司名称": task.公司名称,
                "评估维度": task.评估维度,
                "状态": task.status.value,
                "创建时间": task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else ""
            })
        return display_list
    
    def clear_all_tasks(self):
        """清空所有任务（慎用）"""
        # 这个方法在数据库模式下不推荐使用
        # 保留向后兼容性
        pass
    
    def count_tasks(self, status: Optional[TaskStatus] = None) -> int:
        """统计任务数量"""
        return self.db.count_tasks(status)
