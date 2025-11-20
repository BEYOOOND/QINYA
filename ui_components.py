"""
UI组件模块
提供可复用的UI组件，如侧边栏导航、步骤指示器等
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFrame, QSizePolicy, QSlider)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush
from ui_utils import UIScaler, ThemeColors


class StepNavigation(QWidget):
    """步骤导航侧边栏"""
    step_changed = pyqtSignal(int)  # 步骤索引变化信号
    
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.steps = steps  # [(step_id, title, description), ...]
        self.current_step = 0
        self.scaler = UIScaler()
        self.colors = ThemeColors.get_colors("light")
        self.setFixedWidth(self.scaler.scale(200))
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.scaler.scale(12), self.scaler.scale(16), 
                                 self.scaler.scale(12), self.scaler.scale(16))
        layout.setSpacing(self.scaler.scale(8))
        
        # 标题
        title = QLabel("工作流程")
        title_font = self.scaler.get_font(size=14, weight=QFont.Bold)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 步骤按钮
        self.step_buttons = []
        for idx, (step_id, title_text, desc) in enumerate(self.steps):
            btn = StepButton(idx, title_text, desc, self)
            btn.clicked.connect(lambda checked, i=idx: self.set_current_step(i))
            self.step_buttons.append(btn)
            layout.addWidget(btn)
        
        layout.addStretch()
    
    def set_current_step(self, step_index):
        """设置当前步骤"""
        if 0 <= step_index < len(self.step_buttons):
            old_step = self.current_step
            self.current_step = step_index
            if old_step < len(self.step_buttons):
                self.step_buttons[old_step].set_active(False)
            self.step_buttons[step_index].set_active(True)
            self.step_changed.emit(step_index)
    
    def set_theme(self, theme):
        """设置主题"""
        self.colors = ThemeColors.get_colors(theme)
        for btn in self.step_buttons:
            btn.set_theme(theme)
        self.update()


class StepButton(QPushButton):
    """步骤按钮"""
    
    def __init__(self, step_index, title, description, parent=None):
        super().__init__(parent)
        self.step_index = step_index
        self.title = title
        self.description = description
        self.is_active = False
        self.scaler = UIScaler()
        self.colors = ThemeColors.get_colors("light")
        self.setup_ui()
    
    def setup_ui(self):
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.scaler.scale(12), self.scaler.scale(10),
                                 self.scaler.scale(12), self.scaler.scale(10))
        layout.setSpacing(self.scaler.scale(4))
        
        # 步骤编号和标题
        title_layout = QHBoxLayout()
        step_num = QLabel(f"{self.step_index + 1}")
        step_num.setFixedWidth(self.scaler.scale(24))
        step_num.setAlignment(Qt.AlignCenter)
        title_label = QLabel(self.title)
        title_font = self.scaler.get_font(size=13, weight=QFont.Medium)
        title_label.setFont(title_font)
        title_layout.addWidget(step_num)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # 描述
        desc_label = QLabel(self.description)
        desc_font = self.scaler.get_font(size=11)
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        self.step_num_label = step_num
        self.title_label = title_label
        self.desc_label = desc_label
        self.update_style()
    
    def set_active(self, active):
        """设置激活状态"""
        self.is_active = active
        self.setChecked(active)
        self.update_style()
    
    def set_theme(self, theme):
        """设置主题"""
        self.colors = ThemeColors.get_colors(theme)
        self.update_style()
    
    def update_style(self):
        """更新样式 - 现代化设计"""
        if self.is_active:
            # 激活状态：使用蓝色背景和白色文字
            bg_color = self.colors['accent']
            text_color = self.colors['button_text']
            desc_color = self.colors['button_text']
            border_color = self.colors['accent']
            num_bg = self.colors['button_text']
            num_text = self.colors['accent']
        else:
            # 非激活状态：使用浅色背景
            bg_color = self.colors['bg_primary']
            text_color = self.colors['text_primary']
            desc_color = self.colors['text_secondary']
            border_color = self.colors['border_secondary']
            num_bg = self.colors['bg_tertiary']
            num_text = self.colors['text_secondary']
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: {self.scaler.scale(10)}px;
                text-align: left;
                padding-left: {self.scaler.scale(12)}px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['bg_tertiary'] if not self.is_active else self.colors['accent_hover']};
                border-color: {self.colors['accent'] if not self.is_active else self.colors['accent']};
            }}
        """)
        
        # 更新标题文字颜色
        self.title_label.setStyleSheet(f"color: {text_color}; font-weight: 600;")
        
        # 更新描述文字颜色
        self.desc_label.setStyleSheet(f"color: {desc_color};")
        
        # 更新步骤编号样式 - 圆形徽章设计
        if self.is_active:
            self.step_num_label.setStyleSheet(f"""
                background-color: {num_bg};
                color: {num_text};
                border-radius: {self.scaler.scale(12)}px;
                font-weight: bold;
                font-size: {self.scaler.font_size(11)}px;
            """)
        else:
            self.step_num_label.setStyleSheet(f"""
                background-color: {num_bg};
                color: {num_text};
                border-radius: {self.scaler.scale(12)}px;
                font-weight: 500;
                font-size: {self.scaler.font_size(11)}px;
            """)


class ThemeToggleWidget(QWidget):
    """胶囊式主题切换组件"""
    theme_changed = pyqtSignal(str)  # 主题变化信号: "light" 或 "dark"
    
    def __init__(self, initial_theme="light", parent=None):
        super().__init__(parent)
        self.current_theme = initial_theme
        self.scaler = UIScaler()
        self.colors_light = ThemeColors.get_colors("light")
        self.colors_dark = ThemeColors.get_colors("dark")
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI - 优化为更小、更扁的设计"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建胶囊容器 - 优化尺寸，更小更扁
        self.container = QFrame()
        self.container.setFixedSize(self.scaler.scale(90), self.scaler.scale(26))  # 减小高度和宽度
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(self.scaler.scale(2), self.scaler.scale(2), 
                                           self.scaler.scale(2), self.scaler.scale(2))  # 减少内边距
        container_layout.setSpacing(0)
        
        # 左侧：白色主题按钮
        self.light_btn = QPushButton("白")
        self.light_btn.setCheckable(True)
        self.light_btn.setCursor(Qt.PointingHandCursor)
        self.light_btn.clicked.connect(lambda: self.set_theme("light"))
        container_layout.addWidget(self.light_btn)
        
        # 右侧：黑色主题按钮
        self.dark_btn = QPushButton("黑")
        self.dark_btn.setCheckable(True)
        self.dark_btn.setCursor(Qt.PointingHandCursor)
        self.dark_btn.clicked.connect(lambda: self.set_theme("dark"))
        container_layout.addWidget(self.dark_btn)
        
        layout.addWidget(self.container)
        # 设置整体组件的最小高度
        self.setFixedHeight(self.scaler.scale(28))
        self.update_style()
    
    def set_theme(self, theme):
        """设置主题"""
        if theme == self.current_theme:
            return
        self.current_theme = theme
        self.light_btn.setChecked(theme == "light")
        self.dark_btn.setChecked(theme == "dark")
        self.update_style()
        self.theme_changed.emit(theme)
    
    def get_theme(self):
        """获取当前主题"""
        return self.current_theme
    
    def update_style(self):
        """更新样式 - 使用图标和文字组合，确保清晰可见"""
        is_light = self.current_theme == "light"
        
        # 容器样式 - 胶囊形状
        container_bg = self.colors_light['bg_tertiary'] if is_light else self.colors_dark['bg_tertiary']
        container_border = self.colors_light['border_secondary'] if is_light else self.colors_dark['border_secondary']
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {container_bg};
                border: 1px solid {container_border};
                border-radius: {self.scaler.scale(16)}px;
            }}
        """)
        
        # 按钮样式 - 使用更明显的对比度
        if is_light:
            # 浅色主题：白色按钮高亮，黑色按钮透明
            light_bg = self.colors_light['accent']
            light_text = self.colors_light['button_text']
            light_border = self.colors_light['accent']
            dark_bg = "transparent"
            dark_text = self.colors_light['text_primary']  # 深色文字，确保在浅色背景上可见
            dark_border = "transparent"
        else:
            # 深色主题：黑色按钮高亮，白色按钮透明
            light_bg = "transparent"
            light_text = self.colors_dark['text_primary']  # 浅色文字，确保在深色背景上可见
            light_border = "transparent"
            dark_bg = self.colors_dark['accent']
            dark_text = self.colors_dark['button_text']
            dark_border = self.colors_dark['accent']
        
        self.light_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {light_bg};
                color: {light_text};
                border: 1px solid {light_border};
                border-radius: {self.scaler.scale(10)}px;
                font-weight: 600;
                font-size: {self.scaler.font_size(11)}px;
                min-width: {self.scaler.scale(42)}px;
                min-height: {self.scaler.scale(22)}px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {self.colors_light['accent_hover'] if is_light else self.colors_dark['bg_tertiary']};
                border-color: {self.colors_light['accent'] if is_light else self.colors_dark['border_primary']};
            }}
        """)
        
        self.dark_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {dark_bg};
                color: {dark_text};
                border: 1px solid {dark_border};
                border-radius: {self.scaler.scale(10)}px;
                font-weight: 600;
                font-size: {self.scaler.font_size(11)}px;
                min-width: {self.scaler.scale(42)}px;
                min-height: {self.scaler.scale(22)}px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {self.colors_dark['accent_hover'] if not is_light else self.colors_light['bg_tertiary']};
                border-color: {self.colors_dark['accent'] if not is_light else self.colors_light['border_primary']};
            }}
        """)
        
        # 设置初始选中状态
        self.light_btn.setChecked(is_light)
        self.dark_btn.setChecked(not is_light)


class CollapsiblePanel(QWidget):
    """可折叠面板"""
    toggled = pyqtSignal(bool)  # 展开/折叠信号
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_expanded = True
        self.scaler = UIScaler()
        self.colors = ThemeColors.get_colors("light")
        self.content_widget = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题栏（可点击）
        self.header = QPushButton(self.title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.clicked.connect(self.toggle)
        self.header.setCursor(Qt.PointingHandCursor)
        self.update_header_style()
        layout.addWidget(self.header)
        
        # 内容区域
        self.content_container = QWidget()
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(self.scaler.scale(12), 
                                              self.scaler.scale(8),
                                              self.scaler.scale(12),
                                              self.scaler.scale(8))
        self.content_layout.setSpacing(self.scaler.scale(8))
        layout.addWidget(self.content_container)
    
    def set_content_widget(self, widget):
        """设置内容组件"""
        if self.content_widget:
            self.content_layout.removeWidget(self.content_widget)
            self.content_widget.deleteLater()
        self.content_widget = widget
        self.content_layout.addWidget(widget)
    
    def toggle(self):
        """切换展开/折叠"""
        self.is_expanded = not self.is_expanded
        self.content_container.setVisible(self.is_expanded)
        self.header.setChecked(self.is_expanded)
        self.update_header_style()
        self.toggled.emit(self.is_expanded)
    
    def update_header_style(self):
        """更新标题栏样式"""
        arrow = "▼" if self.is_expanded else "▶"
        self.header.setText(f"{arrow} {self.title}")
        self.header.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['text_primary']};
                border: 1px solid {self.colors['border_secondary']};
                border-radius: {self.scaler.scale(6)}px;
                padding: {self.scaler.scale(8)}px {self.scaler.scale(12)}px;
                text-align: left;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {self.colors['bg_secondary']};
            }}
        """)
    
    def set_theme(self, theme):
        """设置主题"""
        self.colors = ThemeColors.get_colors(theme)
        self.update_header_style()


class CompactFrameNavigator(QWidget):
    """紧凑的帧导航器"""
    frame_changed = pyqtSignal(int)  # 帧索引变化
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scaler = UIScaler()
        self.max_frames = 0
        self.current_frame = 0
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self.scaler.scale(4), self.scaler.scale(1),
                                 self.scaler.scale(4), self.scaler.scale(1))
        layout.setSpacing(self.scaler.scale(6))
        
        # 上一帧按钮 - 优化尺寸，减少高度占用
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedSize(self.scaler.scale(20), self.scaler.scale(20))
        self.prev_btn.setCursor(Qt.PointingHandCursor)
        self.prev_btn.clicked.connect(self.prev_frame)
        # 优化按钮样式，减少内边距
        self.prev_btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid #D0D7E2;
                border-radius: {self.scaler.scale(4)}px;
                background-color: #FFFFFF;
                font-size: {self.scaler.font_size(10)}px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: #F5F7FA;
                border-color: #3e60a9;
            }}
            QPushButton:disabled {{
                background-color: #F5F5F5;
                color: #CCCCCC;
            }}
        """)
        layout.addWidget(self.prev_btn)
        
        # 帧信息标签（紧凑显示）
        self.frame_label = QLabel("0/0")
        self.frame_label.setFixedWidth(self.scaler.scale(55))
        self.frame_label.setAlignment(Qt.AlignCenter)
        font = self.scaler.get_font(size=10)
        self.frame_label.setFont(font)
        layout.addWidget(self.frame_label)
        
        # 下一帧按钮 - 优化尺寸，减少高度占用
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(self.scaler.scale(20), self.scaler.scale(20))
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.clicked.connect(self.next_frame)
        # 优化按钮样式，减少内边距
        self.next_btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid #D0D7E2;
                border-radius: {self.scaler.scale(4)}px;
                background-color: #FFFFFF;
                font-size: {self.scaler.font_size(10)}px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: #F5F7FA;
                border-color: #3e60a9;
            }}
            QPushButton:disabled {{
                background-color: #F5F5F5;
                color: #CCCCCC;
            }}
        """)
        layout.addWidget(self.next_btn)
        
        # 滑块（更紧凑，减少高度）
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFixedHeight(self.scaler.scale(16))
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider, 1)
        
        # 设置整体组件的最小高度
        self.setFixedHeight(self.scaler.scale(24))
        
        self.update_buttons()
    
    def set_max_frames(self, max_frames):
        """设置最大帧数"""
        self.max_frames = max_frames
        self.slider.setRange(0, max(0, max_frames - 1))
        self.update_display()
    
    def set_current_frame(self, frame_index):
        """设置当前帧"""
        self.current_frame = max(0, min(frame_index, self.max_frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.update_display()
        self.update_buttons()
    
    def prev_frame(self):
        """上一帧"""
        if self.current_frame > 0:
            self.set_current_frame(self.current_frame - 1)
            self.frame_changed.emit(self.current_frame)
    
    def next_frame(self):
        """下一帧"""
        if self.current_frame < self.max_frames - 1:
            self.set_current_frame(self.current_frame + 1)
            self.frame_changed.emit(self.current_frame)
    
    def on_slider_changed(self, value):
        """滑块值变化"""
        if value != self.current_frame:
            self.current_frame = value
            self.update_display()
            self.update_buttons()
            self.frame_changed.emit(self.current_frame)
    
    def update_display(self):
        """更新显示"""
        self.frame_label.setText(f"{self.current_frame + 1}/{self.max_frames}")
    
    def update_buttons(self):
        """更新按钮状态"""
        self.prev_btn.setEnabled(self.current_frame > 0)
        self.next_btn.setEnabled(self.current_frame < self.max_frames - 1)

