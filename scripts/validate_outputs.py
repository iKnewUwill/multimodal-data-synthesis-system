"""后台检查脚本：排查已完成任务的输出 JSON 是否包含所需数据。
对存在必要字段为空值的任务，输出清单供手动处理。

用法: python scripts/validate_outputs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_ui.handlers import UIHandlers
from src.task_manager import TaskManager
from queue import Queue
import threading


def main():
    tm = TaskManager()
    handlers = UIHandlers(tm, Queue(), threading.Lock())
    report = handlers.validate_completed_tasks()
    print(report)


if __name__ == "__main__":
    main()
