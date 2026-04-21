"""
数据库层 - SQLite 持久化存储
管理任务和结果的数据库操作
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import threading

from src.models import FinancialTaskInput, FinancialTaskResult, TaskStatus


logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 数据库管理器"""
    
    _local = threading.local()
    _lock = threading.Lock()
    
    def __init__(self, db_path: Path = None):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径，默认使用 settings.DATABASE_PATH
        """
        from config.settings import settings
        
        self.db_path = db_path or settings.DATABASE_PATH
        self._ensure_database_exists()
        
        logger.info(f"数据库管理器初始化: {self.db_path}")
    
    def _ensure_database_exists(self):
        """确保数据库文件和表存在"""
        # 确保数据库目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建表结构
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 创建任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    证券代码 TEXT NOT NULL,
                    公司名称 TEXT NOT NULL,
                    统计截止日期 TEXT NOT NULL,
                    评估维度 TEXT NOT NULL,
                    financial_data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT
                )
            """)
            
            # 创建结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    证券代码 TEXT NOT NULL,
                    公司名称 TEXT NOT NULL,
                    评估维度 TEXT NOT NULL,
                    financial_data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    qa_pairs TEXT NOT NULL,
                    total_iterations INTEGER DEFAULT 0,
                    valid_qa_count INTEGER DEFAULT 0,
                    completed_at TEXT,
                    output_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_task_id ON results(task_id)")
            
            conn.commit()
            logger.info("数据库表结构创建完成")
            
        finally:
            cursor.close()
            conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接（线程安全）"""
        return sqlite3.connect(str(self.db_path), check_same_thread=False)
    
    # ============ 任务操作 ============
    
    def add_task(self, task: FinancialTaskInput) -> str:
        """添加任务到数据库"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO tasks (
                    task_id, 证券代码, 公司名称, 统计截止日期, 评估维度,
                    financial_data, status, created_at, started_at, completed_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.证券代码,
                task.公司名称,
                task.统计截止日期,
                task.评估维度,
                json.dumps(task.financial_data, ensure_ascii=False),
                task.status.value,
                task.created_at.isoformat() if task.created_at else None,
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
                task.error_message
            ))
            
            conn.commit()
            logger.debug(f"任务已添加: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"添加任务失败: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_task(self, task_id: str) -> Optional[FinancialTaskInput]:
        """从数据库获取任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT task_id, 证券代码, 公司名称, 统计截止日期, 评估维度,
                       financial_data, status, created_at, started_at, completed_at, error_message
                FROM tasks WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_task_input(row)
            return None
            
        finally:
            cursor.close()
            conn.close()
    
    def get_all_tasks(self, limit: Optional[int] = None) -> List[FinancialTaskInput]:
        """获取所有任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if limit:
                cursor.execute("""
                    SELECT task_id, 证券代码, 公司名称, 统计截止日期, 评估维度,
                           financial_data, status, created_at, started_at, completed_at, error_message
                    FROM tasks ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT task_id, 证券代码, 公司名称, 统计截止日期, 评估维度,
                           financial_data, status, created_at, started_at, completed_at, error_message
                    FROM tasks ORDER BY created_at DESC
                """)
            
            rows = cursor.fetchall()
            return [self._row_to_task_input(row) for row in rows]
            
        finally:
            cursor.close()
            conn.close()
    
    def update_task_status(self, task_id: str, status: TaskStatus, error_message: str = None):
        """更新任务状态"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.now().isoformat()
            
            cursor.execute("""
                UPDATE tasks 
                SET status = ?, 
                    started_at = COALESCE(?, started_at),
                    completed_at = COALESCE(?, completed_at),
                    error_message = COALESCE(?, error_message)
                WHERE task_id = ?
            """, (
                status.value,
                now if status == TaskStatus.PROCESSING else None,
                now if status in [TaskStatus.COMPLETED, TaskStatus.FAILED] else None,
                error_message,
                task_id
            ))
            
            conn.commit()
            logger.debug(f"任务状态已更新: {task_id} -> {status.value}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"更新任务状态失败: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def filter_tasks(
        self,
        status: Optional[TaskStatus] = None,
        证券代码: Optional[str] = None,
        公司名称: Optional[str] = None
    ) -> List[FinancialTaskInput]:
        """筛选任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT task_id, 证券代码, 公司名称, 统计截止日期, 评估维度,
                       financial_data, status, created_at, started_at, completed_at, error_message
                FROM tasks WHERE 1=1
            """
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if 证券代码:
                query += " AND 证券代码 LIKE ?"
                params.append(f"%{证券代码}%")
            
            if 公司名称:
                query += " AND 公司名称 LIKE ?"
                params.append(f"%{公司名称}%")
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_task_input(row) for row in rows]
            
        finally:
            cursor.close()
            conn.close()
    
    def _row_to_task_input(self, row: tuple) -> FinancialTaskInput:
        """将数据库行转换为 FinancialTaskInput"""
        return FinancialTaskInput(
            task_id=row[0],
            证券代码=row[1],
            公司名称=row[2],
            统计截止日期=row[3],
            评估维度=row[4],
            financial_data=json.loads(row[5]),
            status=TaskStatus(row[6]),
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            started_at=datetime.fromisoformat(row[8]) if row[8] else None,
            completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
            error_message=row[10]
        )
    
    # ============ 结果操作 ============
    
    def save_result(self, result: FinancialTaskResult) -> str:
        """保存任务结果到数据库"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 将 QA pairs 转换为 JSON
            qa_pairs_json = json.dumps([qa.model_dump(mode='json') for qa in result.qa_pairs], ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO results (
                    task_id, 证券代码, 公司名称, 评估维度, financial_data, status,
                    qa_pairs, total_iterations, valid_qa_count, completed_at, output_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.task_id,
                result.证券代码,
                result.公司名称,
                result.评估维度,
                result.financial_data,
                result.status.value,
                qa_pairs_json,
                result.total_iterations,
                result.valid_qa_count,
                result.completed_at.isoformat() if result.completed_at else None,
                result.output_path,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            logger.info(f"任务结果已保存: {result.task_id}")
            return result.task_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存任务结果失败: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_result(self, task_id: str) -> Optional[FinancialTaskResult]:
        """从数据库获取任务结果"""
        from src.models import FinancialQAResult
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT task_id, 证券代码, 公司名称, 评估维度, status,
                       qa_pairs, total_iterations, valid_qa_count, completed_at, output_path
                FROM results WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if row:
                # 解析 QA pairs JSON
                qa_pairs_data = json.loads(row[5])
                qa_pairs = [FinancialQAResult(**qa) for qa in qa_pairs_data]
                
                return FinancialTaskResult(
                    task_id=row[0],
                    证券代码=row[1],
                    公司名称=row[2],
                    评估维度=row[3],
                    status=TaskStatus(row[4]),
                    qa_pairs=qa_pairs,
                    total_iterations=row[6],
                    valid_qa_count=row[7],
                    completed_at=datetime.fromisoformat(row[8]) if row[8] else None
                )
            return None
            
        finally:
            cursor.close()
            conn.close()
    
    def task_exists(self, task_id: str) -> bool:
        """检查任务是否存在"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,))
            return cursor.fetchone() is not None
        finally:
            cursor.close()
            conn.close()
    
    def count_tasks(self, status: Optional[TaskStatus] = None) -> int:
        """统计任务数量"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if status:
                cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = ?", (status.value,))
            else:
                cursor.execute("SELECT COUNT(*) FROM tasks")
            
            return cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()
    
    def migrate_from_json_files(self, output_dir: Path):
        """
        从 JSON 文件迁移数据到数据库
        
        Args:
            output_dir: JSON 文件所在目录
            
        Returns:
            (migrated_count, skipped_count)
        """
        migrated = 0
        skipped = 0
        
        json_files = list(output_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 创建 FinancialTaskResult 对象
                result = FinancialTaskResult(**data)
                
                # 检查任务是否已存在
                if not self.task_exists(result.task_id):
                    # 创建对应的任务输入（如果不存在）
                    task_input = FinancialTaskInput(
                        task_id=result.task_id,
                        证券代码=result.证券代码,
                        公司名称=result.公司名称,
                        统计截止日期="2024-12-31",  # 默认值
                        评估维度=result.评估维度,
                        financial_data={},
                        status=result.status
                    )
                    self.add_task(task_input)
                
                # 保存结果
                self.save_result(result)
                
                # 重命名文件
                json_file.rename(json_file.with_suffix('.json.migrated'))
                migrated += 1
                logger.info(f"已迁移: {json_file.name}")
                
            except Exception as e:
                skipped += 1
                logger.warning(f"跳过文件 {json_file.name}: {e}")
        
        return migrated, skipped
