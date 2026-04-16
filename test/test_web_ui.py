"""
Web UI 功能测试套件

针对 web_ui.py 中的功能设计的综合测试用例
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
from queue import Queue
import concurrent.futures

# 导入被测试的类
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_ui import MultimodalSynthesisUI
from src.models import FinancialTaskInput, TaskStatus
from src.task_manager import TaskManager


class TestMultimodalSynthesisUI:
    """MultimodalSynthesisUI 类的测试套件"""

    @pytest.fixture
    def ui_instance(self):
        """创建 UI 实例"""
        return MultimodalSynthesisUI()

    @pytest.fixture
    def sample_data(self):
        """加载测试数据"""
        data_path = Path(__file__).parent.parent / "data" / "uploads" / "samples.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @pytest.fixture
    def temp_json_file(self, sample_data):
        """创建临时 JSON 文件"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(sample_data[:3], temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture
    def mock_graph_result(self):
        """模拟图执行结果"""
        from src.models import FinancialTaskResult
        from datetime import datetime

        result = FinancialTaskResult(
            task_id="test_task_1",
            证券代码="000001",
            公司名称="平安银行",
            评估维度="一、偿债能力",
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(),
            valid_qa_count=5,
            total_iterations=10,
            qa_pairs=[
                {
                    "question": "测试问题",
                    "answer": "测试答案",
                    "metrics": {"relevance": 0.9, "clarity": 0.8}
                }
            ]
        )
        return result


class TestUIInitialization:
    """测试 UI 初始化"""

    def test_ui_creation(self):
        """测试 UI 实例创建"""
        ui = MultimodalSynthesisUI()
        assert ui is not None
        assert ui.graph is None
        assert ui.task_manager is not None
        assert isinstance(ui.task_manager, TaskManager)
        assert ui.is_running is False
        assert isinstance(ui.log_queue, Queue)
        assert isinstance(ui.log_lock, threading.Lock)

    def test_custom_css_exists(self):
        """测试自定义 CSS 存在"""
        ui = MultimodalSynthesisUI()
        assert hasattr(ui, 'CUSTOM_CSS')
        assert isinstance(ui.CUSTOM_CSS, str)
        assert len(ui.CUSTOM_CSS) > 0
        assert '.container' in ui.CUSTOM_CSS
        assert '.header' in ui.CUSTOM_CSS


class TestTaskLoading:
    """测试任务加载功能"""

    def test_load_json_file_success(self, ui_instance, temp_json_file):
        """测试成功加载 JSON 文件"""
        # 模拟加载文件
        result = ui_instance.load_json_file(temp_json_file)

        # 验证返回值
        status, tasks, total, pending, processing, completed, failed, html = result

        assert "成功追加" in status
        assert "个任务" in status
        assert len(tasks) > 0
        assert total == str(len(tasks))
        assert pending == str(len(tasks))
        assert processing == "0"
        assert completed == "0"
        assert failed == "0"
        assert "scrollable-box" in html
        assert "task-item" in html

    def test_load_json_file_no_file(self, ui_instance):
        """测试未提供文件路径"""
        result = ui_instance.load_json_file("")
        status, tasks, total, pending, processing, completed, failed, html = result

        assert "请先上传文件" in status
        assert len(tasks) == 0
        assert total == "0"

    def test_load_json_file_invalid_json(self, ui_instance):
        """测试加载无效的 JSON 文件"""
        # 创建临时无效 JSON 文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write("{ invalid json }")
        temp_file.close()

        try:
            result = ui_instance.load_json_file(temp_file.name)
            status, tasks, total, pending, processing, completed, failed, html = result

            assert "加载失败" in status
            assert len(tasks) == 0
        finally:
            os.unlink(temp_file.name)

    def test_load_json_file_append_mode(self, ui_instance, temp_json_file):
        """测试追加模式加载"""
        # 第一次加载
        result1 = ui_instance.load_json_file(temp_json_file)
        count1 = int(result1[2])  # total tasks after first load

        # 第二次加载（追加）
        result2 = ui_instance.load_json_file(temp_json_file)
        count2 = int(result2[2])  # total tasks after second load

        assert count2 == count1 * 2  # 应该是第一次的两倍


class TestTaskManagement:
    """测试任务管理功能"""

    def test_refresh_task_list_empty(self, ui_instance):
        """测试刷新空任务列表"""
        result = ui_instance.refresh_task_list()
        total, pending, processing, completed, failed, html, msg = result

        assert total == "0"
        assert pending == "0"
        assert processing == "0"
        assert completed == "0"
        assert failed == "0"
        assert "暂无任务" in html
        assert "已刷新" in msg

    def test_refresh_task_list_with_tasks(self, ui_instance, sample_data):
        """测试刷新有任务的任务列表"""
        # 添加一些任务
        for item in sample_data[:3]:
            task = FinancialTaskInput(
                证券代码=item.get("证券代码", ""),
                公司名称=item.get("公司名称", ""),
                统计截止日期=item.get("统计截止日期", ""),
                评估维度=item.get("评估维度", ""),
                关键指标=item.get("关键指标", {})
            )
            ui_instance.task_manager.add_task(task)

        # 刷新任务列表
        result = ui_instance.refresh_task_list()
        total, pending, processing, completed, failed, html, msg = result

        assert total == "3"
        assert pending == "3"
        assert "平安银行" in html or "万科A" in html


class TestBatchProcessing:
    """测试批量处理功能"""

    def test_start_batch_processing_no_tasks(self, ui_instance):
        """测试没有任务时开始批量处理"""
        results = list(ui_instance.start_batch_processing(10, 3))
        assert len(results) > 0

        first_result = results[0]
        assert first_result[7] == 0  # progress
        assert "请先上传任务文件" in first_result[6]  # log_html

    @patch('web_ui.MultimodalSynthesisGraph')
    def test_start_batch_processing_with_tasks(self, mock_graph_class, ui_instance, sample_data, mock_graph_result):
        """测试有任务时开始批量处理"""
        # 添加任务
        for item in sample_data[:2]:
            task = FinancialTaskInput(
                证券代码=item.get("证券代码", ""),
                公司名称=item.get("公司名称", ""),
                统计截止日期=item.get("统计截止日期", ""),
                评估维度=item.get("评估维度", ""),
                关键指标=item.get("关键指标", {})
            )
            ui_instance.task_manager.add_task(task)

        # Mock graph 返回结果
        mock_graph = Mock()
        mock_graph.run = Mock(return_value=mock_graph_result)
        mock_graph_class.return_value = mock_graph

        # 开始批量处理
        results = list(ui_instance.start_batch_processing(5, 2))

        # 验证结果
        assert len(results) > 0
        final_result = results[-1]

        # 最终状态应该是完成
        assert "完成" in final_result[8]  # status_text
        assert final_result[7] == 100  # progress

    @patch('web_ui.MultimodalSynthesisGraph')
    def test_start_batch_processing_parallel_execution(self, mock_graph_class, ui_instance, sample_data, mock_graph_result):
        """测试并行执行"""
        # 添加多个任务
        for item in sample_data[:5]:
            task = FinancialTaskInput(
                证券代码=item.get("证券代码", ""),
                公司名称=item.get("公司名称", ""),
                统计截止日期=item.get("统计截止日期", ""),
                评估维度=item.get("评估维度", ""),
                关键指标=item.get("关键指标", {})
            )
            ui_instance.task_manager.add_task(task)

        # Mock graph 返回结果
        mock_graph = Mock()
        mock_graph.run = Mock(return_value=mock_graph_result)
        mock_graph_class.return_value = mock_graph

        # 使用并行数 3
        results = list(ui_instance.start_batch_processing(3, 3))

        # 验证执行了所有任务
        assert mock_graph.run.call_count == 5

    def test_stop_processing(self, ui_instance):
        """测试停止处理"""
        ui_instance.is_running = True
        result = ui_instance.stop_processing()
        assert "已停止" in result
        assert ui_instance.is_running == False


class TestTaskProcessing:
    """测试单个任务处理"""

    @patch('web_ui.MultimodalSynthesisGraph')
    @patch('web_ui.save_json')
    def test_process_single_task_success(self, mock_save_json, mock_graph_class, ui_instance, sample_data, mock_graph_result):
        """测试成功处理单个任务"""
        # 准备测试数据
        item = sample_data[0]
        task = FinancialTaskInput(
            证券代码=item.get("证券代码", ""),
            公司名称=item.get("公司名称", ""),
            统计截止日期=item.get("统计截止日期", ""),
            评估维度=item.get("评估维度", ""),
            关键指标=item.get("关键指标", {})
        )

        # Mock graph
        mock_graph = Mock()
        mock_graph.run = Mock(return_value=mock_graph_result)
        mock_graph_class.return_value = mock_graph

        # 处理任务
        result = ui_instance.process_single_task(ui_instance, task, 10)

        # 验证结果
        assert result.status == TaskStatus.COMPLETED
        assert result.valid_qa_count == 5
        assert mock_graph.run.called
        assert mock_save_json.called

    @patch('web_ui.MultimodalSynthesisGraph')
    def test_process_single_task_failure(self, mock_graph_class, ui_instance, sample_data):
        """测试任务处理失败"""
        # 准备测试数据
        item = sample_data[0]
        task = FinancialTaskInput(
            证券代码=item.get("证券代码", ""),
            公司名称=item.get("公司名称", ""),
            统计截止日期=item.get("统计截止日期", ""),
            评估维度=item.get("评估维度", ""),
            关键指标=item.get("关键指标", {})
        )

        # Mock graph 抛出异常
        mock_graph = Mock()
        mock_graph.run = Mock(side_effect=Exception("Test error"))
        mock_graph_class.return_value = mock_graph

        # 处理任务
        result = ui_instance.process_single_task(ui_instance, task, 10)

        # 验证结果
        assert result.status == TaskStatus.FAILED


class TestConfigurationManagement:
    """测试配置管理功能"""

    def test_save_llm_config_success(self, ui_instance):
        """测试成功保存 LLM 配置"""
        from config.llm_config import llm_config

        result = ui_instance.save_llm_config_func(
            "new_api_key",
            "https://new-api.example.com",
            "new-model",
            0.8,
            2048
        )

        assert "已保存" in result
        assert llm_config.api_key == "new_api_key"
        assert llm_config.base_url == "https://new-api.example.com"
        assert llm_config.model_name == "new-model"
        assert llm_config.temperature == 0.8
        assert llm_config.max_tokens == 2048

    def test_save_prompts_config_success(self, ui_instance):
        """测试成功保存 Prompt 配置"""
        from config.prompts import prompts_config

        new_prompts = {
            "proposer_system": "新提议者系统提示",
            "proposer_user": "新提议者用户提示",
            "solver_system": "新求解者系统提示",
            "solver_user": "新求解者用户提示",
            "validator_system": "新验证者系统提示",
            "validator_user": "新验证者用户提示"
        }

        result = ui_instance.save_prompts_func(
            new_prompts["proposer_system"],
            new_prompts["proposer_user"],
            new_prompts["solver_system"],
            new_prompts["solver_user"],
            new_prompts["validator_system"],
            new_prompts["validator_user"]
        )

        assert "已保存" in result
        assert prompts_config.proposer_system_prompt == new_prompts["proposer_system"]
        assert prompts_config.proposer_user_prompt == new_prompts["proposer_user"]
        assert prompts_config.solver_system_prompt == new_prompts["solver_system"]
        assert prompts_config.solver_user_prompt == new_prompts["solver_user"]
        assert prompts_config.validator_system_prompt == new_prompts["validator_system"]
        assert prompts_config.validator_user_prompt == new_prompts["validator_user"]


class TestLogManagement:
    """测试日志管理功能"""

    def test_log_queue_thread_safety(self, ui_instance):
        """测试日志队列的线程安全性"""
        import threading

        def write_logs():
            for i in range(100):
                with ui_instance.log_lock:
                    ui_instance.log_queue.put(f"Log message {i}")

        threads = [threading.Thread(target=write_logs) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证所有消息都被写入
        assert ui_instance.log_queue.qsize() == 500

    def test_log_html_generation(self, ui_instance):
        """测试日志 HTML 生成"""
        # 添加不同类型的日志
        with ui_instance.log_lock:
            ui_instance.log_queue.put("[INFO] 信息日志")
            ui_instance.log_queue.put("[SUCCESS] 成功日志")
            ui_instance.log_queue.put("[ERROR] 错误日志")

        # 生成 HTML
        log_html = "<div class='log-box'>"
        with ui_instance.log_lock:
            while not ui_instance.log_queue.empty():
                log_line = ui_instance.log_queue.get()
                if "[ERROR]" in log_line:
                    log_html += f"<span class='log-error'>{log_line}</span>"
                elif "[SUCCESS]" in log_line:
                    log_html += f"<span class='log-success'>{log_line}</span>"
                else:
                    log_html += f"<span class='log-info'>{log_line}</span>"
        log_html += "</div>"

        assert "log-info" in log_html
        assert "log-success" in log_html
        assert "log-error" in log_html


class TestInterfaceCreation:
    """测试界面创建功能"""

    def test_create_interface(self, ui_instance):
        """测试创建 Gradio 界面"""
        interface = ui_instance.create_interface()
        assert interface is not None

    def test_interface_components_exist(self, ui_instance):
        """测试界面组件存在"""
        interface = ui_instance.create_interface()

        # 验证界面有正确的配置
        assert interface.title == "金融财务数据合成系统"
        assert interface.css == ui_instance.CUSTOM_CSS


class TestDataValidation:
    """测试数据验证功能"""

    def test_task_input_validation(self, ui_instance, sample_data):
        """测试任务输入验证"""
        # 测试完整数据
        item = sample_data[0]
        task = FinancialTaskInput(
            证券代码=item.get("证券代码", ""),
            公司名称=item.get("公司名称", ""),
            统计截止日期=item.get("统计截止日期", ""),
            评估维度=item.get("评估维度", ""),
            关键指标=item.get("关键指标", {})
        )

        assert task.证券代码 == "000001"
        assert task.公司名称 == "平安银行"
        assert task.评估维度 == "一、偿债能力"
        assert len(task.关键指标) > 0

    def test_task_input_with_missing_fields(self, ui_instance):
        """测试缺少字段的任务输入"""
        task = FinancialTaskInput(
            证券代码="",
            公司名称="",
            统计截止日期="",
            评估维度="",
            关键指标={}
        )

        # 验证对象可以创建，即使字段为空
        assert task is not None
        assert task.证券代码 == ""


class TestProgressTracking:
    """测试进度跟踪功能"""

    @patch('web_ui.MultimodalSynthesisGraph')
    def test_progress_calculation(self, mock_graph_class, ui_instance, sample_data, mock_graph_result):
        """测试进度计算"""
        # 添加任务
        for item in sample_data[:10]:
            task = FinancialTaskInput(
                证券代码=item.get("证券代码", ""),
                公司名称=item.get("公司名称", ""),
                统计截止日期=item.get("统计截止日期", ""),
                评估维度=item.get("评估维度", ""),
                关键指标=item.get("关键指标", {})
            )
            ui_instance.task_manager.add_task(task)

        # Mock graph
        mock_graph = Mock()
        mock_graph.run = Mock(return_value=mock_graph_result)
        mock_graph_class.return_value = mock_graph

        # 开始处理并收集进度
        results = list(ui_instance.start_batch_processing(5, 3))

        # 验证进度递增
        progress_values = [result[7] for result in results]  # progress is at index 7
        assert len(progress_values) > 0
        assert progress_values[-1] == 100  # 最终进度应该是 100


class TestErrorHandling:
    """测试错误处理功能"""

    def test_handle_corrupted_json_file(self, ui_instance):
        """测试处理损坏的 JSON 文件"""
        # 创建损坏的 JSON 文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"corrupted": [}')
        temp_file.close()

        try:
            result = ui_instance.load_json_file(temp_file.name)
            status = result[0]
            assert "加载失败" in status or "错误" in status
        finally:
            os.unlink(temp_file.name)

    @patch('web_ui.MultimodalSynthesisGraph')
    def test_handle_task_processing_errors(self, mock_graph_class, ui_instance, sample_data):
        """测试处理任务处理错误"""
        # 添加任务
        for item in sample_data[:3]:
            task = FinancialTaskInput(
                证券代码=item.get("证券代码", ""),
                公司名称=item.get("公司名称", ""),
                统计截止日期=item.get("统计截止日期", ""),
                评估维度=item.get("评估维度", ""),
                关键指标=item.get("关键指标", {})
            )
            ui_instance.task_manager.add_task(task)

        # Mock graph - 前两个成功，第三个失败
        mock_graph = Mock()
        call_count = [0]

        def side_effect_func(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                from src.models import FinancialTaskResult
                from datetime import datetime
                return FinancialTaskResult(
                    task_id=f"task_{call_count[0]}",
                    证券代码="000001",
                    公司名称="测试公司",
                    评估维度="测试维度",
                    status=TaskStatus.COMPLETED,
                    completed_at=datetime.now(),
                    valid_qa_count=5
                )
            else:
                raise Exception("模拟错误")

        mock_graph.run = Mock(side_effect=side_effect_func)
        mock_graph_class.return_value = mock_graph

        # 开始处理
        results = list(ui_instance.start_batch_processing(5, 2))

        # 验证有失败的任务
        final_result = results[-1]
        # 失败数应该大于0
        failed_count = int(final_result[4])  # failed is at index 4
        assert failed_count > 0


class TestIntegration:
    """集成测试"""

    @patch('web_ui.MultimodalSynthesisGraph')
    @patch('web_ui.save_json')
    def test_full_workflow(self, mock_save_json, mock_graph_class, ui_instance, sample_data, mock_graph_result):
        """测试完整工作流程"""
        # 1. 加载文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(sample_data[:3], temp_file, ensure_ascii=False, indent=2)
        temp_file.close()

        try:
            load_result = ui_instance.load_json_file(temp_file.name)
            assert "成功追加" in load_result[0]

            # 2. 刷新任务列表
            refresh_result = ui_instance.refresh_task_list()
            assert refresh_result[0] == "3"  # total tasks

            # 3. 开始批量处理
            mock_graph = Mock()
            mock_graph.run = Mock(return_value=mock_graph_result)
            mock_graph_class.return_value = mock_graph

            process_results = list(ui_instance.start_batch_processing(5, 2))
            final_result = process_results[-1]

            # 4. 验证最终状态
            assert "完成" in final_result[8]  # status text
            assert final_result[7] == 100  # progress

        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
