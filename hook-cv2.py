"""
PyInstaller hook for OpenCV
确保OpenCV的所有二进制文件被正确包含
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import cv2

# 收集所有cv2子模块
hiddenimports = collect_submodules('cv2')

# 收集cv2的数据文件（包括DLL等）
datas = collect_data_files('cv2')

