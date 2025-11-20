"""
PyInstaller hook for numpy
确保numpy的所有二进制扩展模块被正确包含
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import numpy

# 收集所有numpy子模块
hiddenimports = collect_submodules('numpy')

# 收集numpy的数据文件
datas = collect_data_files('numpy')

