"""
TDD Test Suite for DataFrame Task Selection Refactoring

This test file is written FIRST (TDD Red Phase) and should FAIL initially
because TaskDataConverter has not been implemented yet.

Test Coverage:
- Empty task list handling
- Single task conversion to DataFrame
- Multiple tasks order preservation
- Task status to emoji mapping
- DataFrame row selection
- Empty selection handling
- Full workflow integration test
"""

import pytest
import pandas as pd
from datetime import datetime

# Import existing models
from src.models import FinancialTaskInput, TaskStatus

# Import converter that will be created (EXPECTED TO FAIL)
from web_ui.data_converters import TaskDataConverter


class TestTaskDataConverter:
    """Test suite for TaskDataConverter - TDD Red Phase"""
    
    def test_empty_tasks_returns_custom_message(self):
        """
        Test Case 1: Empty task list should return DataFrame with custom message row
        
        Expected behavior:
        - When task list is empty, DataFrame should contain one row
        - The row should display: "暂无任务,请上传文件" in the company name column
        - Other columns can be empty or have placeholder values
        """
        tasks = []
        
        # Create converter
        converter = TaskDataConverter()
        
        # Convert empty list to DataFrame
        df = converter.tasks_to_dataframe(tasks)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1, "Empty task list should return 1 row with message"
        assert "暂无任务,请上传文件" in df.iloc[0]["状态"], \
            "Empty task list should show custom message"
    
    def test_single_task_converts_correctly(self):
        """
        Test Case 2: Single task should convert to DataFrame with correct data
        
        Expected behavior:
        - DataFrame should have exactly 1 row
        - Columns: ["状态", "公司名称", "证券代码", "评估维度", "任务ID"]
        - Data should match the input task
        """
        # Create single task
        task = FinancialTaskInput(
            证券代码="600519",
            公司名称="贵州茅台",
            统计截止日期="2024-12-31",
            评估维度="财务健康",
            financial_data={"营业收入": 1000000},
            status=TaskStatus.PENDING
        )
        tasks = [task]
        
        # Create converter
        converter = TaskDataConverter()
        
        # Convert to DataFrame
        df = converter.tasks_to_dataframe(tasks)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1, "Single task should result in 1 row"
        
        # Check columns exist
        expected_columns = ["状态", "公司名称", "证券代码", "评估维度", "任务ID"]
        assert list(df.columns) == expected_columns, \
            f"DataFrame columns should be {expected_columns}"
        
        # Check data accuracy
        assert df.iloc[0]["公司名称"] == "贵州茅台"
        assert df.iloc[0]["证券代码"] == "600519"
        assert df.iloc[0]["评估维度"] == "财务健康"
        assert df.iloc[0]["任务ID"] == task.task_id
    
    def test_multiple_tasks_preserve_order(self):
        """
        Test Case 3: Multiple tasks should maintain insertion order
        
        Expected behavior:
        - DataFrame rows should be in the same order as input task list
        - Order preservation is critical for user experience
        """
        # Create multiple tasks
        task1 = FinancialTaskInput(
            证券代码="600519",
            公司名称="贵州茅台",
            统计截止日期="2024-12-31",
            评估维度="财务健康",
            financial_data={}
        )
        
        task2 = FinancialTaskInput(
            证券代码="000858",
            公司名称="五粮液",
            统计截止日期="2024-12-31",
            评估维度="盈利能力",
            financial_data={}
        )
        
        task3 = FinancialTaskInput(
            证券代码="000001",
            公司名称="平安银行",
            统计截止日期="2024-12-31",
            评估维度="偿债能力",
            financial_data={}
        )
        
        tasks = [task1, task2, task3]
        
        # Create converter
        converter = TaskDataConverter()
        
        # Convert to DataFrame
        df = converter.tasks_to_dataframe(tasks)
        
        # Assertions
        assert len(df) == 3, "Three tasks should result in 3 rows"
        
        # Verify order is preserved
        assert df.iloc[0]["公司名称"] == "贵州茅台"
        assert df.iloc[1]["公司名称"] == "五粮液"
        assert df.iloc[2]["公司名称"] == "平安银行"
        
        # Verify task IDs are in correct order
        assert df.iloc[0]["任务ID"] == task1.task_id
        assert df.iloc[1]["任务ID"] == task2.task_id
        assert df.iloc[2]["任务ID"] == task3.task_id
    
    def test_status_emoji_mapping(self):
        """
        Test Case 4: Task status should map to correct emoji
        
        Expected emoji mappings:
        - PENDING (待处理) -> ⏳
        - PROCESSING (处理中) -> 🔄
        - COMPLETED (已完成) -> ✅
        - FAILED (处理失败) -> ❌
        """
        # Create tasks with different statuses
        tasks = [
            FinancialTaskInput(
                证券代码="600519",
                公司名称="任务1",
                统计截止日期="2024-12-31",
                评估维度="测试",
                financial_data={},
                status=TaskStatus.PENDING
            ),
            FinancialTaskInput(
                证券代码="000858",
                公司名称="任务2",
                统计截止日期="2024-12-31",
                评估维度="测试",
                financial_data={},
                status=TaskStatus.PROCESSING
            ),
            FinancialTaskInput(
                证券代码="000001",
                公司名称="任务3",
                统计截止日期="2024-12-31",
                评估维度="测试",
                financial_data={},
                status=TaskStatus.COMPLETED
            ),
            FinancialTaskInput(
                证券代码="000002",
                公司名称="任务4",
                统计截止日期="2024-12-31",
                评估维度="测试",
                financial_data={},
                status=TaskStatus.FAILED
            )
        ]
        
        # Create converter
        converter = TaskDataConverter()
        
        # Convert to DataFrame
        df = converter.tasks_to_dataframe(tasks)
        
        # Verify emoji mappings
        assert "⏳" in df.iloc[0]["状态"], "PENDING status should have ⏳ emoji"
        assert "🔄" in df.iloc[1]["状态"], "PROCESSING status should have 🔄 emoji"
        assert "✅" in df.iloc[2]["状态"], "COMPLETED status should have ✅ emoji"
        assert "❌" in df.iloc[3]["状态"], "FAILED status should have ❌ emoji"
    
    def test_dataframe_selection_returns_correct_task_id(self):
        """
        Test Case 5: Selecting a DataFrame row should return correct task_id
        
        Expected behavior:
        - Given a DataFrame selection (row index)
        - Should return the corresponding task_id
        - Task ID should match the task at that position
        """
        # Create tasks
        task1 = FinancialTaskInput(
            证券代码="600519",
            公司名称="贵州茅台",
            统计截止日期="2024-12-31",
            评估维度="财务健康",
            financial_data={}
        )
        
        task2 = FinancialTaskInput(
            证券代码="000858",
            公司名称="五粮液",
            统计截止日期="2024-12-31",
            评估维度="盈利能力",
            financial_data={}
        )
        
        tasks = [task1, task2]
        
        # Create converter
        converter = TaskDataConverter()
        
        # Convert to DataFrame
        df = converter.tasks_to_dataframe(tasks)
        
        # Simulate row selection (Gradio returns row data as dict or series)
        # Test selecting first row
        selected_row_idx = 0
        task_id = converter.get_task_id_from_selection(df, selected_row_idx)
        
        assert task_id == task1.task_id, \
            f"Row 0 should return task1 ID, got {task_id}"
        
        # Test selecting second row
        selected_row_idx = 1
        task_id = converter.get_task_id_from_selection(df, selected_row_idx)
        
        assert task_id == task2.task_id, \
            f"Row 1 should return task2 ID, got {task_id}"
    
    def test_dataframe_selection_none_returns_default_message(self):
        """
        Test Case 6: No selection should return default message
        
        Expected behavior:
        - When no row is selected (None or empty selection)
        - Should return a default message indicating no selection
        - Default message: "请选择任务查看详情" or similar
        """
        tasks = [
            FinancialTaskInput(
                证券代码="600519",
                公司名称="贵州茅台",
                统计截止日期="2024-12-31",
                评估维度="财务健康",
                financial_data={}
            )
        ]
        
        # Create converter
        converter = TaskDataConverter()
        
        # Convert to DataFrame
        df = converter.tasks_to_dataframe(tasks)
        
        # Test with None selection
        task_id = converter.get_task_id_from_selection(df, None)
        
        assert task_id is None or task_id == "", \
            "None selection should return None or empty string"
        
        # Test with empty selection
        task_id = converter.get_task_id_from_selection(df, pd.NA)
        
        assert task_id is None or task_id == "", \
            "Empty selection should return None or empty string"
    
    def test_full_flow_upload_select_view(self):
        """
        Test Case 7: Integration test - upload JSON -> display in DataFrame -> select -> view details
        
        Expected behavior:
        - Load tasks from JSON-like data
        - Convert to DataFrame for display
        - User selects a row
        - System retrieves task details using task_id
        - Task details match the original task data
        """
        # Simulate JSON upload data
        json_data = [
            {
                "证券代码": "600519",
                "公司名称": "贵州茅台",
                "统计截止日期": "2024-12-31",
                "评估维度": "财务健康",
                "financial_data": {"营业收入": 1000000, "净利润": 500000}
            },
            {
                "证券代码": "000858",
                "公司名称": "五粮液",
                "统计截止日期": "2024-12-31",
                "评估维度": "盈利能力",
                "financial_data": {"营业收入": 800000, "净利润": 400000}
            }
        ]
        
        # Step 1: Convert JSON to tasks
        tasks = [
            FinancialTaskInput(**data) 
            for data in json_data
        ]
        
        # Create converter
        converter = TaskDataConverter()
        
        # Step 2: Convert tasks to DataFrame for display
        df = converter.tasks_to_dataframe(tasks)
        
        # Verify DataFrame has correct structure
        assert len(df) == 2, "Should have 2 tasks in DataFrame"
        assert list(df.columns) == ["状态", "公司名称", "证券代码", "评估维度", "任务ID"]
        
        # Step 3: Simulate user selecting first row (index 0)
        selected_row_idx = 0
        selected_task_id = converter.get_task_id_from_selection(df, selected_row_idx)
        
        # Verify task_id was retrieved
        assert selected_task_id is not None, "Should get a valid task_id"
        assert selected_task_id == tasks[0].task_id, \
            "Selected task_id should match first task"
        
        # Step 4: Verify task details can be retrieved
        # Find the task with selected_task_id
        selected_task = None
        for task in tasks:
            if task.task_id == selected_task_id:
                selected_task = task
                break
        
        assert selected_task is not None, "Should find task by ID"
        assert selected_task.公司名称 == "贵州茅台", \
            "Task details should match original data"
        assert selected_task.证券代码 == "600519", \
            "Task details should match original data"
        assert selected_task.评估维度 == "财务健康", \
            "Task details should match original data"
        
        # Step 5: Simulate selecting second row
        selected_row_idx = 1
        selected_task_id = converter.get_task_id_from_selection(df, selected_row_idx)
        
        assert selected_task_id == tasks[1].task_id, \
            "Second selection should match second task"
        
        # Verify second task details
        selected_task = None
        for task in tasks:
            if task.task_id == selected_task_id:
                selected_task = task
                break
        
        assert selected_task.公司名称 == "五粮液"
        assert selected_task.证券代码 == "000858"


class TestTaskDataConverterEdgeCases:
    """Additional edge case tests for TaskDataConverter"""
    
    def test_dataframe_columns_immutable(self):
        """
        Test that DataFrame columns are always in the correct order
        
        Expected columns: ["状态", "公司名称", "证券代码", "评估维度", "任务ID"]
        """
        task = FinancialTaskInput(
            证券代码="600519",
            公司名称="贵州茅台",
            统计截止日期="2024-12-31",
            评估维度="财务健康",
            financial_data={}
        )
        
        converter = TaskDataConverter()
        df = converter.tasks_to_dataframe([task])
        
        expected_columns = ["状态", "公司名称", "证券代码", "评估维度", "任务ID"]
        assert list(df.columns) == expected_columns, \
            "DataFrame columns must be in exact order"
    
    def test_dataframe_with_special_characters(self):
        """
        Test that special characters in company names are handled correctly
        """
        task = FinancialTaskInput(
            证券代码="600519",
            公司名称="贵州茅台(SH) [A股]",  # Special characters
            统计截止日期="2024-12-31",
            评估维度="财务健康 & 盈利能力",  # Special character
            financial_data={}
        )
        
        converter = TaskDataConverter()
        df = converter.tasks_to_dataframe([task])
        
        assert df.iloc[0]["公司名称"] == "贵州茅台(SH) [A股]"
        assert df.iloc[0]["评估维度"] == "财务健康 & 盈利能力"
    
    def test_dataframe_index_reset(self):
        """
        Test that DataFrame index is reset to 0-based sequential integers
        """
        tasks = [
            FinancialTaskInput(
                证券代码=f"00000{i}",
                公司名称=f"公司{i}",
                统计截止日期="2024-12-31",
                评估维度="测试",
                financial_data={}
            )
            for i in range(5)
        ]
        
        converter = TaskDataConverter()
        df = converter.tasks_to_dataframe(tasks)
        
        # Check index is 0-based sequential
        assert list(df.index) == [0, 1, 2, 3, 4], \
            "DataFrame index should be 0-based sequential integers"


# Pytest configuration and fixtures
@pytest.fixture
def sample_tasks():
    """Fixture providing sample tasks for testing"""
    return [
        FinancialTaskInput(
            证券代码="600519",
            公司名称="贵州茅台",
            统计截止日期="2024-12-31",
            评估维度="财务健康",
            financial_data={"营业收入": 1000000}
        ),
        FinancialTaskInput(
            证券代码="000858",
            公司名称="五粮液",
            统计截止日期="2024-12-31",
            评估维度="盈利能力",
            financial_data={"营业收入": 800000}
        )
    ]


@pytest.fixture
def converter():
    """Fixture providing TaskDataConverter instance"""
    return TaskDataConverter()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
