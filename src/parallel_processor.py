"""
并行任务处理器
使用 ThreadPoolExecutor 实现多任务并行处理
"""

import concurrent.futures
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from pathlib import Path
import threading
import json

from config.settings import settings
from src.models import FinancialTaskInput, FinancialTaskResult, TaskStatus
from src.database import DatabaseManager


class ParallelTaskProcessor:
    """并行任务处理器"""
    
    def __init__(self, max_workers: int = None):
        """
        初始化并行处理器
        
        Args:
            max_workers: 最大并行任务数，默认使用 settings.PARALLEL_TASK_COUNT
        """
        self.max_workers = max_workers or settings.PARALLEL_TASK_COUNT
        self.db = DatabaseManager()
        self._lock = threading.Lock()
    
    def process_task(self, task: FinancialTaskInput, process_func: Callable) -> FinancialTaskResult:
        """
        处理单个任务
        
        Args:
            task: 金融任务输入
            process_func: 处理函数，接收 FinancialTaskInput，返回 FinancialTaskResult
            
        Returns:
            任务结果
        """
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()
        
        try:
            result = process_func(task)
            result.status = TaskStatus.COMPLETED
            result.completed_at = datetime.now()
            
            # 保存到数据库
            self.db.save_result(result)
            
            return result
            
        except Exception as e:
            result = FinancialTaskResult(
                task_id=task.task_id,
                证券代码=task.证券代码,
                公司名称=task.公司名称,
                评估维度=task.评估维度,
                status=TaskStatus.FAILED,
                completed_at=datetime.now()
            )
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = str(e)
        
        with self._lock:
            self._results[task.task_id] = result
        
        return result
    
    def process_all(self, process_func: Callable, progress_callback: Optional[Callable] = Dict[str, FinancialTaskResult]):
        """
        并行处理所有任务
        
        Args:
            process_func: 处理函数
            progress_callback: 进度回调函数，接收 (task_id, status, completed_count, total_count)
            
        Returns:
            任务ID到结果的映射
        """
        tasks = list(self._tasks.values())
        total_count = len(tasks)
        completed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {}
            for task in tasks:
                future = executor.submit(self.process_task, task, process_func)
                future_to_task[future] = task
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_task):
                if not self.is_running:
                    break
                
                task = future_to_task[future]
                result = future.result()
                
                # 更新任务状态
                self.db.update_task_status(task.task_id, result.status)
                
                # 更新统计数据
                stats = self._calculate_stats()
                
                # 跻加到日志
                with self.log_lock:
                    log_msg = f"[INFO] 任务完成: {task.公司名称} - {result.status.value}"
                    self.log_queue.put(log_msg)
                
                # 通知回调
                if progress_callback:
                    progress_callback(task.task_id, result.status.value, completed_count, total_count)
                
                completed_count += 1
        
        return self._results
    
    def _calculate_stats(self, tasks: List[FinancialTaskInput]) -> Dict:
        """计算统计信息"""
        stats = {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "processing": len([t for t in tasks if t.status == TaskStatus.PROCESSING]),
            "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in tasks if t.status == TaskStatus.FAILED])
        }
        return stats
