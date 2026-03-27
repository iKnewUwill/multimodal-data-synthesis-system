@echo off
chcp 65001 >nul

echo ==========================================
echo 多模态数据合成系统
echo ==========================================

:: 检查 Conda 是否安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未检测到 Conda，请先安装 Anaconda 或 Miniconda
    pause
    exit /b 1
)

:: 检查 Python 版本
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set python_version=%%a
echo Python 版本: %python_version%

:: 检查并创建 Conda 环境
echo 检查 Conda 环境...
conda env list | findstr /C:"data_synthesis" >nul
if %errorlevel% neq 0 (
    echo 创建 Conda 环境: data_synthesis
    conda create -n data_synthesis python=3.11 -y
) else (
    echo Conda 环境已存在: data_synthesis
)

:: 激活 Conda 环境
echo 激活 Conda 环境...
call conda activate data_synthesis

:: 安装依赖
echo 检查并安装依赖...
pip install -r requirements.txt

:: 检查环境变量文件
if not exist ".env" (
    echo 警告: 未找到 .env 文件
    echo 请创建 .env 文件并配置以下环境变量：
    echo   OPENAI_API_KEY=your_api_key
    echo   OPENAI_BASE_URL=your_base_url
    echo.
    set /p continue="是否继续启动？(y/n) "
    if /i not "%continue%"=="y" exit /b 1
)

:: 创建必要的目录
echo 创建数据目录...
if not exist "data\uploads" mkdir data\uploads
if not exist "data\outputs" mkdir data\uploads
if not exist "logs" mkdir logs

:: 启动 Web UI
echo ==========================================
echo 启动 Web UI...
echo 访问地址: http://localhost:7860
echo ==========================================

python web_ui.py

pause