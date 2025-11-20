@echo off
chcp 65001 >nul
echo 启动激活码管理工具...
cd /d %~dp0..
python activation_system/activation_admin.py
pause

