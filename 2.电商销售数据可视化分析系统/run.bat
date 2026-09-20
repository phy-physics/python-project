@echo off
chcp 65001 >nul
echo ========================================
echo   电商销售数据可视化分析系统
echo ========================================
echo.
echo 正在启动 Streamlit 服务...
echo.
C:\ProgramData\miniforge3\python.exe -m streamlit run app.py --server.headless true --server.port 8501
pause
