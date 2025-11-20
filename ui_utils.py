"""
UI工具模块
提供DPI感知、字体缩放、主题颜色等工具函数
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtCore import Qt
import platform


class UIScaler:
    """UI缩放工具类，支持DPI感知和字体缩放"""
    
    _instance = None
    _scale_factor = 1.0
    _base_font_size = 12
    _base_dpi = 96
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UIScaler, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        """初始化DPI和缩放因子"""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # 获取屏幕DPI
        screen = app.primaryScreen()
        if screen:
            dpi = screen.logicalDotsPerInch()
            # 计算缩放因子（基于96 DPI作为基准）
            self._scale_factor = max(1.0, dpi / self._base_dpi)
            
            # 对于高分辨率屏幕，进一步调整
            # Retina显示器通常是2x，4K显示器可能是1.5x-2x
            physical_dpi = screen.physicalDotsPerInch()
            if physical_dpi > 200:  # 高分辨率屏幕
                # 根据物理DPI调整，但不要过度放大
                self._scale_factor = min(2.0, max(1.2, physical_dpi / 120))
        else:
            self._scale_factor = 1.0
    
    @classmethod
    def scale(cls, value):
        """缩放像素值"""
        return int(value * cls._instance._scale_factor)
    
    @classmethod
    def font_size(cls, base_size=None):
        """获取缩放后的字体大小"""
        if base_size is None:
            base_size = cls._instance._base_font_size
        return int(base_size * cls._instance._scale_factor)
    
    @classmethod
    def get_scale_factor(cls):
        """获取当前缩放因子"""
        return cls._instance._scale_factor
    
    @classmethod
    def get_font(cls, family=None, size=None, weight=QFont.Normal):
        """获取缩放后的字体"""
        if family is None:
            # 根据系统选择最佳字体
            system = platform.system()
            if system == "Darwin":  # macOS
                family = "PingFang SC"
            elif system == "Windows":
                family = "Microsoft YaHei UI"
            else:  # Linux
                family = "Noto Sans CJK SC"
        
        if size is None:
            size = cls.font_size()
        else:
            size = cls.font_size(size)
        
        font = QFont(family, size, weight)
        font.setHintingPreference(QFont.PreferDefaultHinting)
        return font


class ThemeColors:
    """主题颜色定义，参考Apple设计规范"""
    
    # 浅色主题（Light Mode）- 使用新的品牌色方案
    LIGHT = {
        # 背景色 - 使用新的浅色背景
        "bg_primary": "#FFFFFF",
        "bg_secondary": "#F3F1F0",  # 使用新的浅灰色
        "bg_tertiary": "#E8E6E5",  # 稍深的灰色
        "bg_elevated": "#FFFFFF",
        
        # 文字颜色 - 优化对比度
        "text_primary": "#1A1A1A",  # 深色文字
        "text_secondary": "#666666",  # 次要文字
        "text_tertiary": "#999999",  # 第三级文字
        "text_disabled": "#CCCCCC",
        
        # 边框颜色
        "border_primary": "#CCCCCC",
        "border_secondary": "#E0E0E0",
        "border_tertiary": "#F0F0F0",
        
        # 强调色 - 使用新的品牌色
        "accent": "#3e60a9",  # 主蓝色
        "accent_hover": "#2d4a8a",  # 深蓝色悬停
        "accent_pressed": "#1e3a6b",  # 更深蓝色按下
        
        # 功能色 - 使用新的品牌色
        "success": "#1ea89c",  # 青绿色
        "warning": "#dc2066",  # 粉红色作为警告
        "error": "#cc0000",  # 红色作为错误
        "info": "#3e60a9",  # 蓝色作为信息
        
        # 按钮
        "button_primary": "#3e60a9",  # 主蓝色
        "button_primary_hover": "#2d4a8a",  # 深蓝色悬停
        "button_secondary": "#F3F1F0",  # 浅灰色背景
        "button_secondary_hover": "#E8E6E5",  # 稍深的灰色悬停
        "button_text": "#FFFFFF",
        "button_text_secondary": "#1A1A1A",
        
        # 输入框
        "input_bg": "#FFFFFF",
        "input_border": "#CCCCCC",
        "input_border_focus": "#3e60a9",  # 蓝色焦点
        "input_text": "#1A1A1A",
        
        # 分组框
        "group_bg": "#FFFFFF",
        "group_border": "#E0E0E0",
        "group_title": "#666666",
        
        # 滚动条
        "scrollbar_bg": "#F3F1F0",
        "scrollbar_handle": "#CCCCCC",
        "scrollbar_handle_hover": "#999999",
    }
    
    # 深色主题（Dark Mode）- 使用新的品牌色方案
    DARK = {
        # 背景色 - 深色模式背景
        "bg_primary": "#1A1A1A",  # 深灰背景
        "bg_secondary": "#252525",  # 次要背景
        "bg_tertiary": "#2C2C2C",  # 第三级背景
        "bg_elevated": "#252525",  # 浮起背景
        
        # 文字颜色（高对比度，确保可读性）
        "text_primary": "#FFFFFF",  # 纯白主文字
        "text_secondary": "#CCCCCC",  # 次要文字
        "text_tertiary": "#999999",  # 第三级文字
        "text_disabled": "#666666",  # 禁用文字
        
        # 边框颜色
        "border_primary": "#404040",
        "border_secondary": "#333333",
        "border_tertiary": "#2C2C2C",
        
        # 强调色 - 使用新的品牌色（深色模式下稍亮）
        "accent": "#5a7fc9",  # 更亮的蓝色
        "accent_hover": "#6b8fd9",
        "accent_pressed": "#4a6fb9",
        
        # 功能色 - 使用新的品牌色
        "success": "#2eb8ac",  # 更亮的青绿色
        "warning": "#ec3076",  # 更亮的粉红色
        "error": "#e60000",  # 更亮的红色
        "info": "#5a7fc9",  # 更亮的蓝色
        
        # 按钮
        "button_primary": "#3e60a9",  # 主蓝色
        "button_primary_hover": "#5a7fc9",  # 更亮的蓝色悬停
        "button_secondary": "#2C2C2C",  # 深灰背景
        "button_secondary_hover": "#383838",  # 稍亮的灰色悬停
        "button_text": "#FFFFFF",
        "button_text_secondary": "#FFFFFF",
        
        # 输入框
        "input_bg": "#252525",
        "input_border": "#404040",
        "input_border_focus": "#3e60a9",  # 蓝色焦点
        "input_text": "#FFFFFF",
        
        # 分组框
        "group_bg": "#252525",
        "group_border": "#404040",
        "group_title": "#CCCCCC",
        
        # 滚动条
        "scrollbar_bg": "#252525",
        "scrollbar_handle": "#404040",
        "scrollbar_handle_hover": "#505050",
    }
    
    @classmethod
    def get_colors(cls, theme="light"):
        """获取主题颜色"""
        return cls.DARK if theme == "dark" else cls.LIGHT


def get_stylesheet(theme="light", scale_factor=1.0):
    """生成样式表，支持主题和缩放"""
    colors = ThemeColors.get_colors(theme)
    scaler = UIScaler()
    
    # 基础字体大小（根据缩放因子调整）
    base_font_size = scaler.font_size(12)
    small_font_size = scaler.font_size(11)
    large_font_size = scaler.font_size(14)
    title_font_size = scaler.font_size(13)
    
    # 间距和尺寸（根据缩放因子调整）
    padding_small = scaler.scale(4)
    padding_medium = scaler.scale(6)
    padding_large = scaler.scale(10)
    border_radius = scaler.scale(6)
    border_width = 1
    
    # 字体族
    system = platform.system()
    if system == "Darwin":
        font_family = "'PingFang SC', 'SF Pro Text', -apple-system, sans-serif"
    elif system == "Windows":
        font_family = "'Microsoft YaHei UI', 'Segoe UI', sans-serif"
    else:
        font_family = "'Noto Sans CJK SC', 'WenQuanYi Micro Hei', sans-serif"
    
    stylesheet = f"""
        /* 全局样式 */
        QWidget {{
            background-color: {colors['bg_secondary']};
            color: {colors['text_primary']};
            font-family: {font_family};
            font-size: {base_font_size}px;
        }}
        
        QMainWindow, QDialog {{
            background-color: {colors['bg_secondary']};
        }}
        
        /* 分组框 */
        QGroupBox {{
            border: {border_width}px solid {colors['group_border']};
            margin-top: {scaler.scale(8)}px;
            border-radius: {border_radius}px;
            background-color: {colors['group_bg']};
            padding-top: {scaler.scale(12)}px;
        }}
        
        QGroupBox#highlightGroup {{
            border: {border_width}px solid {colors['error']};
            background-color: {theme == 'dark' and '#2A1A1A' or '#FFF0F0'};
        }}
        
        QGroupBox#secondaryGroup {{
            border: {border_width}px solid {colors['accent']};
            background-color: {theme == 'dark' and '#1A2333' or '#E8EDF5'};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: {scaler.scale(12)}px;
            padding: 0 {scaler.scale(6)}px;
            color: {colors['group_title']};
            font-size: {title_font_size}px;
            font-weight: 500;
        }}
        
        /* 标签 */
        QLabel#fileBadge, QLabel#outputBadge, QLabel#watermarkBadge {{
            padding: {padding_small}px {scaler.scale(10)}px;
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {scaler.scale(4)}px;
            background-color: {colors['bg_primary']};
            font-size: {small_font_size}px;
        }}
        
        QLabel#infoBanner {{
            padding: {padding_medium}px {scaler.scale(12)}px;
            border-radius: {border_radius}px;
            background-color: {theme == 'dark' and '#2A3A4A' or '#E8EDF5'};
            color: {colors['accent']};
            font-weight: 500;
            font-size: {small_font_size}px;
        }}
        
        /* 顶部工具栏样式 */
        QFrame#topBar {{
            background-color: {colors['bg_primary']};
            border-bottom: 1px solid {colors['border_tertiary']};
        }}
        
        QFrame#topBarSeparator {{
            background-color: {colors['border_secondary']};
            max-width: 1px;
            min-width: 1px;
        }}
        
        QLabel#appTitle {{
            color: {colors['text_primary']};
            font-weight: 700;
        }}
        
        QLabel#modeBadge {{
            padding: {scaler.scale(4)}px {scaler.scale(12)}px;
            border-radius: {scaler.scale(12)}px;
            background-color: {theme == 'dark' and '#2A3A4A' or '#E8EDF5'};
            color: {colors['accent']};
        }}
        
        QLabel#modeBadge:hover {{
            background-color: {theme == 'dark' and '#3A4A5A' or '#D8E0F0'};
        }}
        
        QPushButton#modeSwitchButton {{
            background-color: {colors['button_secondary']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_primary']};
            border-radius: {scaler.scale(4)}px;
            font-size: {scaler.font_size(11)}px;
            font-weight: 500;
            padding: {scaler.scale(2)}px {scaler.scale(8)}px;
        }}
        
        QPushButton#modeSwitchButton:hover {{
            background-color: {colors['button_secondary_hover']};
            border-color: {colors['accent']};
            color: {colors['accent']};
        }}
        
        QPushButton#modeSwitchButton:pressed {{
            background-color: {colors['bg_tertiary']};
        }}
        
        QPushButton#homepageButton {{
            background-color: transparent;
            border: none;
            border-radius: {scaler.scale(4)}px;
            padding: {scaler.scale(4)}px;
        }}
        
        QPushButton#homepageButton:hover {{
            background-color: {colors['bg_tertiary']};
        }}
        
        QPushButton#homepageButton:pressed {{
            background-color: {colors['bg_secondary']};
        }}
        
        QLabel#fileLabelText {{
            color: {colors['text_secondary']};
            font-weight: 500;
        }}
        
        QFrame#previewCanvasFrame {{
            border: 1px solid {colors['border_primary']};
            border-radius: {scaler.scale(4)}px;
            background-color: {colors['bg_tertiary']};
        }}
        
        QLabel#fileInfo {{
            color: {colors['text_secondary']};
            padding: {scaler.scale(6)}px {scaler.scale(12)}px;
            border-radius: {scaler.scale(6)}px;
            background-color: transparent;
        }}
        
        QLabel#outputPathLabel {{
            color: {colors['text_secondary']};
            padding: {scaler.scale(6)}px {scaler.scale(12)}px;
            border-radius: {scaler.scale(6)}px;
            background-color: transparent;
        }}
        
        QLabel#previewTitle, QLabel#sizePresetLabel {{
            color: {colors['text_primary']};
            background: transparent;
        }}
        
        QLabel#mainInfoLabel {{
            color: {colors['text_secondary']};
            background: transparent;
        }}
        
        QLabel#videoInfoLabel, QLabel#tipsLabel {{
            color: {theme == 'dark' and '#999999' or colors['text_primary']};
            background-color: {theme == 'dark' and '#F3F1F0' or 'transparent'};
            padding: {scaler.scale(10)}px;
            border-radius: {scaler.scale(6)}px;
        }}
        
        QLabel#accentLabel {{
            color: {colors['accent']};
            font-weight: 500;
        }}
        
        QLabel#warningLabel {{
            color: {colors['error']};
            font-weight: 500;
        }}
        
        QFrame#sizePresetFrame {{
            background-color: {colors['bg_elevated']};
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {border_radius}px;
            padding: {scaler.scale(8)}px;
        }}
        
        /* 输入控件 */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
            background-color: {colors['input_bg']};
            color: {colors['text_primary']};
            border: {border_width}px solid {colors['input_border']};
            border-radius: {scaler.scale(5)}px;
            padding: {scaler.scale(6)}px {scaler.scale(10)}px;
            font-size: {base_font_size}px;
            min-height: {scaler.scale(28)}px;
        }}
        
        QTextEdit {{
            color: {colors['text_primary']};
            selection-background-color: {colors['accent']};
            selection-color: {colors['button_text']};
        }}
        
        QLabel {{
            color: {colors['text_primary']};
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
            border: {border_width}px solid {colors['input_border_focus']};
            outline: none;
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: {scaler.scale(20)}px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: {scaler.scale(4)}px solid transparent;
            border-right: {scaler.scale(4)}px solid transparent;
            border-top: {scaler.scale(5)}px solid {colors['text_secondary']};
            width: 0;
            height: 0;
        }}
        
        /* 按钮 */
        QPushButton {{
            background-color: {colors['button_primary']};
            color: {colors['button_text']};
            border: none;
            padding: {scaler.scale(8)}px {scaler.scale(16)}px;
            border-radius: {scaler.scale(6)}px;
            font-weight: 500;
            font-size: {base_font_size}px;
            min-height: {scaler.scale(32)}px;
        }}
        
        QPushButton:hover {{
            background-color: {colors['button_primary_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {colors['accent_pressed']};
        }}
        
        QPushButton:disabled {{
            background-color: {colors['border_tertiary']};
            color: {colors['text_disabled']};
        }}
        
        QPushButton#primaryButton {{
            background-color: {colors['button_primary']};
            color: {colors['button_text']};
            border: none;
            border-radius: {scaler.scale(8)}px;
            padding: {scaler.scale(10)}px {scaler.scale(20)}px;
            font-weight: 600;
            font-size: {base_font_size}px;
            min-height: {scaler.scale(36)}px;
        }}
        
        QPushButton#primaryButton:hover {{
            background-color: {colors['button_primary_hover']};
        }}
        
        QPushButton#primaryButton:pressed {{
            background-color: {colors['accent_pressed']};
        }}
        
        QPushButton#secondaryButton {{
            background-color: {colors['button_secondary']};
            color: {colors['button_text_secondary']};
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {scaler.scale(8)}px;
            padding: {scaler.scale(10)}px {scaler.scale(20)}px;
            font-weight: 500;
            font-size: {base_font_size}px;
            min-height: {scaler.scale(36)}px;
        }}
        
        QPushButton#secondaryButton:hover {{
            background-color: {colors['button_secondary_hover']};
            border-color: {colors['border_primary']};
        }}
        
        QPushButton#dangerButton {{
            background-color: {colors['error']};
        }}
        
        QPushButton#dangerButton:hover {{
            background-color: {theme == 'dark' and '#FF6B6B' or '#F97066'};
        }}
        
        /* 滚动区域 */
        QScrollArea {{
            background-color: {colors['bg_primary']};
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {scaler.scale(8)}px;
        }}
        
        QScrollBar:vertical {{
            background: {colors['scrollbar_bg']};
            width: {scaler.scale(12)}px;
            border-radius: {scaler.scale(6)}px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background: {colors['scrollbar_handle']};
            border-radius: {scaler.scale(6)}px;
            min-height: {scaler.scale(30)}px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {colors['scrollbar_handle_hover']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        
        QScrollBar:horizontal {{
            background: {colors['scrollbar_bg']};
            height: {scaler.scale(12)}px;
            border-radius: {scaler.scale(6)}px;
            border: none;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {colors['scrollbar_handle']};
            border-radius: {scaler.scale(6)}px;
            min-width: {scaler.scale(30)}px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: {colors['scrollbar_handle_hover']};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
        
        /* 菜单栏 */
        QMenuBar {{
            background-color: {colors['bg_primary']};
            color: {colors['text_primary']};
            border-bottom: {border_width}px solid {colors['border_secondary']};
            padding: {scaler.scale(4)}px;
        }}
        
        QMenuBar::item {{
            padding: {scaler.scale(6)}px {scaler.scale(12)}px;
            border-radius: {scaler.scale(4)}px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {colors['bg_tertiary']};
        }}
        
        QMenu {{
            background-color: {colors['bg_elevated']};
            color: {colors['text_primary']};
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {scaler.scale(6)}px;
            padding: {scaler.scale(4)}px;
        }}
        
        QMenu::item {{
            padding: {scaler.scale(8)}px {scaler.scale(24)}px;
            border-radius: {scaler.scale(4)}px;
        }}
        
        QMenu::item:selected {{
            background-color: {colors['bg_tertiary']};
        }}
        
        /* 状态栏 */
        QStatusBar {{
            background-color: {colors['bg_primary']};
            color: {colors['text_secondary']};
            border-top: {border_width}px solid {colors['border_secondary']};
            font-size: {small_font_size}px;
        }}
        
        /* 消息框 */
        QMessageBox {{
            background-color: {colors['bg_primary']};
            color: {colors['text_primary']};
        }}
        
        QMessageBox QLabel {{
            color: {colors['text_primary']};
            font-size: {base_font_size}px;
        }}
        
        /* 进度条 */
        QProgressBar {{
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {scaler.scale(4)}px;
            background-color: {colors['bg_tertiary']};
            text-align: center;
            font-size: {small_font_size}px;
            color: {colors['text_primary']};
            height: {scaler.scale(20)}px;
        }}
        
        QProgressBar::chunk {{
            background-color: {colors['accent']};
            border-radius: {scaler.scale(3)}px;
        }}
        
        /* 复选框 */
        QCheckBox {{
            color: {colors['text_primary']};
            font-size: {base_font_size}px;
            spacing: {scaler.scale(6)}px;
        }}
        
        QCheckBox::indicator {{
            width: {scaler.scale(18)}px;
            height: {scaler.scale(18)}px;
            border: {border_width}px solid {colors['border_primary']};
            border-radius: {scaler.scale(4)}px;
            background-color: {colors['input_bg']};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {colors['accent']};
            border-color: {colors['accent']};
        }}
        
        QCheckBox::indicator:hover {{
            border-color: {colors['accent']};
        }}
        
        /* 滑块 */
        QSlider::groove:horizontal {{
            border: none;
            height: {scaler.scale(4)}px;
            background: {colors['bg_tertiary']};
            border-radius: {scaler.scale(2)}px;
        }}
        
        QSlider::handle:horizontal {{
            background: {colors['accent']};
            border: none;
            width: {scaler.scale(16)}px;
            height: {scaler.scale(16)}px;
            margin: {scaler.scale(-6)}px 0;
            border-radius: {scaler.scale(8)}px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {colors['accent_hover']};
        }}
        
        /* 列表 */
        QListWidget {{
            background-color: {colors['bg_primary']};
            border: {border_width}px solid {colors['border_secondary']};
            border-radius: {scaler.scale(6)}px;
            color: {colors['text_primary']};
            font-size: {base_font_size}px;
        }}
        
        QListWidget::item {{
            padding: {scaler.scale(8)}px;
            border-bottom: {border_width}px solid {colors['border_tertiary']};
        }}
        
        QListWidget::item:selected {{
            background-color: {colors['bg_tertiary']};
            color: {colors['text_primary']};
        }}
        
        QListWidget::item:hover {{
            background-color: {colors['bg_tertiary']};
        }}
    """
    
    return stylesheet