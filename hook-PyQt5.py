"""
自定义 PyQt5 hook - 避免符号链接冲突
只收集必要的 PyQt5 文件，排除框架符号链接
"""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# 只收集必要的 PyQt5 模块
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
]

# 收集数据文件，但排除框架目录
try:
    all_datas = collect_data_files('PyQt5')
    datas = []
    for data in all_datas:
        src_path = data[0] if isinstance(data, tuple) else data
        # 排除 .framework 目录
        if '.framework' not in src_path:
            datas.append(data)
except:
    datas = []

# 收集动态库，但排除框架目录
try:
    all_binaries = collect_dynamic_libs('PyQt5')
    binaries = []
    for binary in all_binaries:
        src_path = binary[0] if isinstance(binary, tuple) else binary
        # 排除 .framework 目录
        if '.framework' not in src_path:
            binaries.append(binary)
except:
    binaries = []

