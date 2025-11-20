"""
主入口：仅负责启动 GUI，并为 Windows/macOS 提供 HiDPI 适配
"""

import os
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def _configure_high_dpi():
    """
    在创建 QApplication 之前启用 HiDPI 支持，并保持像素渲染一致
    """
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    rounding_policy = getattr(Qt, "HighDpiScaleFactorRoundingPolicy", None)
    if rounding_policy is not None and hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )


def main():
    """应用程序入口"""
    _configure_high_dpi()
    from gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()

