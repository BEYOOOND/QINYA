"""
GUI界面模块
提供用户交互界面

开发人员：张诗浩 (Shihao Z)
公司：湖南度尚文化创意有限公司
日期：2025-11-16
"""
import sys
import os
import json
import uuid
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                             QComboBox, QCheckBox, QLineEdit, QProgressBar,
                             QDialog, QDialogButtonBox, QScrollArea, QSizePolicy,
                             QSlider, QListWidget, QListWidgetItem, QStackedWidget,
                             QSplitter, QFrame, QButtonGroup, QToolButton, QTextBrowser,
                             QLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QUrl, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QDesktopServices, QBrush
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QTransform
from PyQt5.QtSvg import QSvgRenderer
import numpy as np
import cv2
import math
from datetime import datetime
from video_processor import VideoProcessor
from typing import Optional
from ui_utils import UIScaler, ThemeColors, get_stylesheet
from ui_components import CollapsiblePanel, CompactFrameNavigator, ThemeToggleWidget
from PIL import Image, ImageOps


WHITE_BACKGROUND_FORMATS = {".jpg", ".jpeg", ".bmp", ".webp"}


def detect_white_background(image: Image.Image, threshold: int = 245, border_ratio: float = 0.12, coverage_ratio: float = 0.65) -> bool:
    """检测图片边缘是否为白色背景"""
    try:
        rgb_image = image.convert("RGB")
        np_image = np.array(rgb_image)
        if np_image.size == 0:
            return False
        height, width, _ = np_image.shape
        border_h = max(1, int(height * border_ratio))
        border_w = max(1, int(width * border_ratio))
        top = np_image[:border_h, :, :]
        bottom = np_image[-border_h:, :, :]
        left = np_image[:, :border_w, :]
        right = np_image[:, -border_w:, :]
        border_pixels = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3),
                                        left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
        if border_pixels.size == 0:
            return False
        white_mask = np.all(border_pixels >= threshold, axis=1)
        return float(np.count_nonzero(white_mask)) / float(border_pixels.shape[0]) >= coverage_ratio
    except Exception:
        return False


def remove_white_background(image: Image.Image, threshold: int = 240, softness: int = 3) -> Image.Image:
    """移除接近白色的背景并转换为RGBA（带羽化处理，减少毛刺）"""
    rgba_image = image.convert("RGBA")
    np_image = np.array(rgba_image)
    if np_image.ndim != 3 or np_image.shape[2] < 3:
        return rgba_image
    rgb = np_image[:, :, :3].astype(np.uint8)
    alpha_channel = np_image[:, :, 3].astype(np.float32)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lightness = lab[:, :, 0]
    saturation = hsv[:, :, 1]

    base_mask = np.where(lightness >= threshold, 255, 0).astype(np.uint8)
    low_sat_mask = np.where(saturation <= 40, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(base_mask, low_sat_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    blur_sigma = max(softness, 1)
    feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    mask_norm = feathered.astype(np.float32) / 255.0

    alpha_channel = alpha_channel * (1.0 - mask_norm)
    np_image[:, :, 3] = np.clip(alpha_channel, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image, "RGBA")


def prepare_overlay_image(file_path: str, auto_detect_white: bool = True):
    """加载素材图片，必要时自动去白底，返回np数组和处理后的PIL图像"""
    path = Path(file_path)
    with Image.open(str(path)) as img:
        image = ImageOps.exif_transpose(img)
        suffix = path.suffix.lower()
        processed = image
        if auto_detect_white and suffix in WHITE_BACKGROUND_FORMATS:
            if detect_white_background(processed):
                processed = remove_white_background(processed)
            else:
                processed = processed.convert("RGBA")
        else:
            processed = processed.convert("RGBA") if processed.mode != "RGBA" else processed
        np_image = np.ascontiguousarray(np.array(processed, dtype=np.uint8))
        return np_image, processed.copy()


def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """将numpy数组转换为PIL图像（RGBA）"""
    if image_array is None:
        raise ValueError("无效的图像数据")
    array = image_array
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        pil_image = Image.fromarray(array, "L").convert("RGBA")
    elif array.ndim == 3 and array.shape[2] == 4:
        pil_image = Image.fromarray(array, "RGBA")
    elif array.ndim == 3 and array.shape[2] >= 3:
        pil_image = Image.fromarray(array[:, :, :3], "RGB").convert("RGBA")
    else:
        raise ValueError("不支持的图像通道数")
    return pil_image


class WelcomeLinkButton(QPushButton):
    """欢迎页外链按钮：左图标右文字 + 轻量动效"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        scaler = UIScaler()
        self.setObjectName("welcomeLinkButton")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.setMinimumWidth(scaler.scale(108))
        self.setMinimumHeight(scaler.scale(34))
        icon_size = scaler.scale(16)
        self.setIconSize(QSize(icon_size, icon_size))

        padding_h = scaler.scale(10)
        padding_v = scaler.scale(5)
        border_radius = scaler.scale(16)
        font = scaler.get_font(size=13, weight=QFont.Medium)
        self.setFont(font)
        self.setStyleSheet(f"""
            QPushButton#welcomeLinkButton {{
                border-radius: {border_radius}px;
                border: 1px solid #E5E7EB;
                background-color: rgba(30, 64, 175, 0.08);
                color: #1F2933;
                padding: {padding_v}px {padding_h}px;
                text-align: left;
            }}
            QPushButton#welcomeLinkButton:hover {{
                background-color: rgba(30, 64, 175, 0.18);
                border-color: #1E40AF;
            }}
            QPushButton#welcomeLinkButton:pressed {{
                background-color: rgba(30, 64, 175, 0.28);
            }}
        """)

        self._base_blur = 6.0
        self._hover_blur = 14.0
        self._shadow_effect = QGraphicsDropShadowEffect(self)
        self._shadow_effect.setBlurRadius(self._base_blur)
        self._shadow_effect.setOffset(0, scaler.scale(1))
        self._shadow_effect.setColor(QColor(30, 64, 175, 60))
        self.setGraphicsEffect(self._shadow_effect)

        self._shadow_anim = QPropertyAnimation(self._shadow_effect, b"blurRadius", self)
        self._shadow_anim.setDuration(200)
        self._shadow_anim.setEasingCurve(QEasingCurve.OutCubic)

    def enterEvent(self, event):
        self._animate_shadow(self._hover_blur)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_shadow(self._base_blur)
        super().leaveEvent(event)

    def _animate_shadow(self, target_value: float):
        self._shadow_anim.stop()
        self._shadow_anim.setStartValue(self._shadow_effect.blurRadius())
        self._shadow_anim.setEndValue(target_value)
        self._shadow_anim.start()


class AssetLibraryManager:
    """素材库管理器：负责素材存储、删除与加载"""

    def __init__(self):
        self.base_dir = self._resolve_base_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "library.json"
        self.assets = self._load_manifest()

    def _resolve_base_dir(self) -> Path:
        custom_dir = os.environ.get("QINGYA_ASSET_DIR")
        if custom_dir:
            return Path(custom_dir)
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support" / "QingyaShenqi"
        elif sys.platform.startswith("win"):
            appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            base = appdata / "QingyaShenqi"
        else:
            base = Path.home() / ".qingya_shenqi"
        return base / "assets"

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return []
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("assets", [])
        except Exception:
            return []

    def _save_manifest(self):
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump({"assets": self.assets}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存素材库清单失败: {e}")

    def list_assets(self):
        return list(self.assets)

    def _store_image(self, pil_image: Image.Image, display_name: str):
        asset_id = uuid.uuid4().hex
        filename = f"{asset_id}.png"
        save_path = self.base_dir / filename
        pil_image.save(str(save_path), "PNG")
        asset = {
            "id": asset_id,
            "name": display_name or f"素材_{asset_id[:6]}",
            "filename": filename,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.assets.append(asset)
        self._save_manifest()
        return asset

    def add_from_path(self, file_path: str, auto_detect_white: bool = True):
        array, pil_image = prepare_overlay_image(file_path, auto_detect_white=auto_detect_white)
        asset = self._store_image(pil_image, Path(file_path).stem)
        return asset, array

    def add_from_array(self, array: np.ndarray, display_name: str = None):
        pil_image = numpy_to_pil(array)
        return self._store_image(pil_image, display_name)

    def delete_asset(self, asset_id: str):
        asset = next((item for item in self.assets if item.get("id") == asset_id), None)
        if not asset:
            return False
        file_path = self.base_dir / asset.get("filename", "")
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        self.assets = [item for item in self.assets if item.get("id") != asset_id]
        self._save_manifest()
        return True

    def load_asset_array(self, asset_id: str):
        asset = next((item for item in self.assets if item.get("id") == asset_id), None)
        if not asset:
            raise FileNotFoundError("素材不存在")
        file_path = self.base_dir / asset.get("filename", "")
        if not file_path.exists():
            raise FileNotFoundError("素材文件缺失")
        with Image.open(str(file_path)) as img:
            rgba = img.convert("RGBA")
            return np.ascontiguousarray(np.array(rgba, dtype=np.uint8))

    def get_asset_path(self, asset_id: str) -> Optional[Path]:
        asset = next((item for item in self.assets if item.get("id") == asset_id), None)
        if not asset:
            return None
        file_path = self.base_dir / asset.get("filename", "")
        return file_path if file_path.exists() else None


class AssetLibraryDialog(QDialog):
    """素材库管理对话框：支持导入、使用和删除素材"""

    def __init__(self, manager: AssetLibraryManager, parent=None, title: str = "素材库"):
        super().__init__(parent)
        self.manager = manager
        self.selected_asset = None
        self.setWindowTitle(title)
        self.setMinimumSize(640, 460)
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_use_clicked)
        layout.addWidget(self.list_widget)
        preview_container = QHBoxLayout()
        self.preview_label = QLabel("选择素材以预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("border: 1px dashed #CBD5F5; color: #94A3B8;")
        preview_container.addWidget(self.preview_label)
        layout.addLayout(preview_container)
        button_row = QHBoxLayout()
        self.import_btn = QPushButton("导入新素材")
        self.import_btn.clicked.connect(self._import_asset)
        button_row.addWidget(self.import_btn)
        self.delete_btn = QPushButton("删除素材")
        self.delete_btn.clicked.connect(self._delete_asset)
        button_row.addWidget(self.delete_btn)
        button_row.addStretch()
        self.use_btn = QPushButton("使用所选素材")
        self.use_btn.setEnabled(False)
        self.use_btn.clicked.connect(self._on_use_clicked)
        button_row.addWidget(self.use_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        self._refresh_asset_list()

    def _refresh_asset_list(self):
        self.list_widget.clear()
        assets = self.manager.list_assets()
        for asset in assets:
            display = f"{asset.get('name','未命名')} · {asset.get('created_at','')}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, asset)
            self.list_widget.addItem(item)
        self.use_btn.setEnabled(bool(assets))
        if assets:
            self.list_widget.setCurrentRow(0)
        else:
            self.preview_label.setText("暂无素材，请先导入。")

    def _current_asset(self):
        item = self.list_widget.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def _on_selection_changed(self, current, previous):
        asset = self._current_asset()
        self.use_btn.setEnabled(asset is not None)
        if asset:
            path = self.manager.get_asset_path(asset.get("id"))
            if path and path.exists():
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.preview_label.setPixmap(scaled)
                    return
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("选择素材以预览")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current = self.list_widget.currentItem()
        if current:
            self._on_selection_changed(current, None)

    def _import_asset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择素材图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)"
        )
        if not file_path:
            return
        try:
            self.manager.add_from_path(file_path, auto_detect_white=True)
            self._refresh_asset_list()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导入素材失败：{str(e)}")

    def _delete_asset(self):
        asset = self._current_asset()
        if not asset:
            return
        confirm = QMessageBox.question(self, "确认删除", f"确定要删除“{asset.get('name', '')}”吗？")
        if confirm != QMessageBox.Yes:
            return
        if self.manager.delete_asset(asset.get("id")):
            self._refresh_asset_list()

    def _on_use_clicked(self, *args):
        asset = self._current_asset()
        if not asset:
            QMessageBox.information(self, "提示", "请先选择素材。")
            return
        self.selected_asset = asset
        self.accept()


class CanvasLabel(QLabel):
    """固定画布标签 - 1053×1746像素，视频可在画布上缩放和移动"""
    
    transform_changed = pyqtSignal(float, float, float)  # scale, offset_x, offset_y
    
    def __init__(self, canvas_width=1053, canvas_height=1746):
        super().__init__()
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # 设置最小尺寸，允许随窗口缩放
        scaler = UIScaler()
        self.setMinimumSize(scaler.scale(400), scaler.scale(500))
        self.setAlignment(Qt.AlignCenter)
        # 样式将在主题应用时更新
        self.setText("")
        
        # 设置尺寸策略，允许扩展以适应窗口
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.show_preview_frame = False  # 是否显示预览指导框
        
        self.original_image = None  # 原始视频首帧（numpy array）
        self.image_pixmap = None  # 原始图像的QPixmap
        
        # 视频在画布上的变换参数
        self.video_scale = 1.0  # 视频缩放比例
        self.video_offset_x = 0.0  # 视频在画布上的X偏移（画布坐标系）
        self.video_offset_y = 0.0  # 视频在画布上的Y偏移（画布坐标系）
        self.min_scale = 0.01
        self.max_scale = 5.0
        
        # 交互状态
        self.is_dragging = False
        self.drag_start_pos = None
        self.last_mouse_pos = None
        
        # 显示缩放（用于将画布缩放到显示区域）
        self.display_scale = 1.0
        self.should_show_guides = True
        # 主题支持
        self.theme = "light"  # 默认主题，会在父窗口应用主题时更新
    
    def set_image(self, image_array):
        """设置原始视频首帧"""
        if image_array is None:
            self.original_image = None
            self.image_pixmap = None
            self.update()
            return
        
        try:
            # 保存原始图像
            self.original_image = image_array.copy()
            h, w = image_array.shape[:2]
            
            # 确保图像数组是正确的格式
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # 确保是RGB格式（3通道）
            if len(image_array.shape) == 2:
                # 灰度图，转换为RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA格式，转换为RGB（参考第二步的实现）
                from PIL import Image
                pil_img = Image.fromarray(image_array, 'RGBA')
                rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                rgb_img.paste(pil_img, mask=pil_img.split()[3])
                image_array = np.array(rgb_img)
            elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
                # 其他格式，只取前3个通道
                image_array = image_array[:, :, :3]
            
            # 确保是连续的数组（某些操作可能产生非连续数组）
            image_array = np.ascontiguousarray(image_array)
            
            # 使用PIL处理图像（参考第二步的实现）
            from PIL import Image
            from io import BytesIO
            
            # 确保图像数组格式正确
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            image_array = np.ascontiguousarray(image_array)
            
            # 确保是RGB格式（3通道）
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
                image_array = image_array[:, :, :3]
            
            img = Image.fromarray(image_array, 'RGB')
            
            # 转换为QPixmap - 使用最可靠的方法：通过字节流（参考第二步的实现）
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            self.image_pixmap = QPixmap()
            if not self.image_pixmap.loadFromData(buffer.getvalue()):
                raise ValueError("无法从字节流加载QPixmap")
                
        except Exception as e:
            print(f"设置图像时出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果出错，创建一个错误提示图像
            self.image_pixmap = QPixmap(w if 'w' in locals() else 100, h if 'h' in locals() else 100)
            self.image_pixmap.fill(QColor(128, 128, 128))
            QMessageBox.warning(None, "警告", f"图片显示异常: {str(e)}")
        
        # 初始化：将视频缩放到能完全覆盖画布
        scale_w = self.canvas_width / w
        scale_h = self.canvas_height / h
        self.video_scale = max(scale_w, scale_h)  # 使用较大的缩放比例，确保覆盖画布
        self.video_scale = max(self.min_scale, min(self.max_scale, self.video_scale))
        
        # 居中显示
        scaled_w = w * self.video_scale
        scaled_h = h * self.video_scale
        self.video_offset_x = (self.canvas_width - scaled_w) / 2
        self.video_offset_y = (self.canvas_height - scaled_h) / 2
        
        self.update()
        self.transform_changed.emit(self.video_scale, self.video_offset_x, self.video_offset_y)
    
    def get_crop_params(self):
        """获取裁切参数：返回(scale, offset_x, offset_y)"""
        return (self.video_scale, self.video_offset_x, self.video_offset_y)
    
    def set_crop_params(self, scale, offset_x, offset_y):
        """设置裁切参数"""
        scale = max(self.min_scale, min(self.max_scale, scale)) if self.image_pixmap is not None else scale
        self.video_scale = scale
        self.video_offset_x = offset_x
        self.video_offset_y = offset_y
        self.update()
        self.transform_changed.emit(self.video_scale, self.video_offset_x, self.video_offset_y)
    
    def set_scale(self, scale, anchor=None):
        """直接设置缩放比例"""
        self._apply_scale(scale, anchor)

    def _apply_scale(self, target_scale, anchor=None):
        if self.image_pixmap is None:
            return
        target_scale = max(self.min_scale, min(self.max_scale, target_scale))
        current_scale = self.video_scale if self.video_scale > 0 else self.min_scale
        if abs(target_scale - current_scale) < 1e-4:
            return
        if anchor is None:
            anchor = (self.canvas_width / 2.0, self.canvas_height / 2.0)
        ax, ay = anchor
        scale_ratio = target_scale / current_scale
        self.video_offset_x = ax - (ax - self.video_offset_x) * scale_ratio
        self.video_offset_y = ay - (ay - self.video_offset_y) * scale_ratio
        self.video_scale = target_scale
        self.transform_changed.emit(self.video_scale, self.video_offset_x, self.video_offset_y)
        self.update()
    
    def _is_point_in_canvas(self, point):
        """判断点是否在画布区域内"""
        label_size = self.size()
        # 计算画布在显示区域中的位置和尺寸
        canvas_display_w = int(self.canvas_width * self.display_scale)
        canvas_display_h = int(self.canvas_height * self.display_scale)
        canvas_display_x = (label_size.width() - canvas_display_w) // 2
        canvas_display_y = (label_size.height() - canvas_display_h) // 2
        
        return (canvas_display_x <= point.x() <= canvas_display_x + canvas_display_w and
                canvas_display_y <= point.y() <= canvas_display_y + canvas_display_h)
    
    def _display_to_canvas_coords(self, display_point):
        """将显示坐标转换为画布坐标"""
        label_size = self.size()
        canvas_display_w = int(self.canvas_width * self.display_scale)
        canvas_display_h = int(self.canvas_height * self.display_scale)
        canvas_display_x = (label_size.width() - canvas_display_w) // 2
        canvas_display_y = (label_size.height() - canvas_display_h) // 2
        
        canvas_x = (display_point.x() - canvas_display_x) / self.display_scale
        canvas_y = (display_point.y() - canvas_display_y) / self.display_scale
        return (canvas_x, canvas_y)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.image_pixmap is None:
            return
        
        if event.button() == Qt.LeftButton and self._is_point_in_canvas(event.pos()):
            self.is_dragging = True
            self.drag_start_pos = event.pos()
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 拖动视频"""
        if self.is_dragging and self.last_mouse_pos and self.image_pixmap:
            # 计算移动距离（画布坐标系）
            dx = (event.pos().x() - self.last_mouse_pos.x()) / self.display_scale
            dy = (event.pos().y() - self.last_mouse_pos.y()) / self.display_scale
            
            # 更新视频偏移
            self.video_offset_x += dx
            self.video_offset_y += dy
            
            self.last_mouse_pos = event.pos()
            self.transform_changed.emit(self.video_scale, self.video_offset_x, self.video_offset_y)
            self.update()
    
    def wheelEvent(self, event):
        """鼠标滚轮事件 - 缩放视频"""
        if self.image_pixmap is None:
            return
        
        # 获取鼠标在画布上的位置
        canvas_x, canvas_y = self._display_to_canvas_coords(event.pos())
        
        delta = event.angleDelta().y()
        if delta == 0:
            return
        steps = delta / 120.0
        scale_factor = math.pow(1.02, steps)
        target_scale = self.video_scale * scale_factor
        self._apply_scale(target_scale, anchor=(canvas_x, canvas_y))
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.is_dragging:
            self.is_dragging = False
            self.drag_start_pos = None
            self.last_mouse_pos = None
    
    def set_theme(self, theme: str):
        """设置主题"""
        self.theme = theme
        self.update()
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 根据主题获取颜色
        colors = ThemeColors.get_colors(self.theme)
        scaler = UIScaler()
        
        # 绘制背景（使用主题背景色）
        bg_color = QColor(colors['bg_tertiary'])
        painter.fillRect(self.rect(), bg_color)
        
        if self.image_pixmap is None:
            return
        
        label_size = self.size()
        
        # 计算画布的显示尺寸（保持宽高比，适应显示区域）
        # 移除1.0的限制，允许画布随窗口等比缩放
        scale_w = (label_size.width() - scaler.scale(20)) / self.canvas_width
        scale_h = (label_size.height() - scaler.scale(20)) / self.canvas_height
        self.display_scale = min(scale_w, scale_h)  # 移除1.0限制，允许缩放
        
        canvas_display_w = int(self.canvas_width * self.display_scale)
        canvas_display_h = int(self.canvas_height * self.display_scale)
        canvas_display_x = (label_size.width() - canvas_display_w) // 2
        canvas_display_y = (label_size.height() - canvas_display_h) // 2
        
        # 绘制画布背景（根据主题使用不同颜色）
        canvas_bg = QColor(colors['bg_primary'] if self.theme == 'light' else colors['bg_elevated'])
        painter.fillRect(canvas_display_x, canvas_display_y, 
                        canvas_display_w, canvas_display_h, canvas_bg)
        
        # 绘制画布边框（使用主题边框色）
        border_color = QColor(colors['border_primary'])
        pen = QPen(border_color, scaler.scale(2))
        painter.setPen(pen)
        painter.drawRect(canvas_display_x, canvas_display_y, 
                        canvas_display_w, canvas_display_h)
        
        # 计算视频在画布坐标系中的显示尺寸和位置
        orig_w = self.image_pixmap.width()
        orig_h = self.image_pixmap.height()
        video_canvas_w = orig_w * self.video_scale
        video_canvas_h = orig_h * self.video_scale
        
        # 计算视频在画布内的可见区域（画布坐标系）
        # 视频在画布坐标系中的位置和尺寸
        video_canvas_x = self.video_offset_x
        video_canvas_y = self.video_offset_y
        
        # 计算视频与画布的交集（可见区域）
        visible_start_x = max(0, video_canvas_x)
        visible_start_y = max(0, video_canvas_y)
        visible_end_x = min(self.canvas_width, video_canvas_x + video_canvas_w)
        visible_end_y = min(self.canvas_height, video_canvas_y + video_canvas_h)
        
        # 可见区域的宽度和高度（画布坐标系）
        visible_w = visible_end_x - visible_start_x
        visible_h = visible_end_y - visible_start_y
        
        if visible_w > 0 and visible_h > 0:
            # 计算在视频图像中的对应区域（原始图像坐标）
            # 视频在画布上的可见区域相对于视频起始位置的偏移
            offset_in_video_x = visible_start_x - video_canvas_x
            offset_in_video_y = visible_start_y - video_canvas_y
            
            # 转换为原始图像坐标
            source_x = int(offset_in_video_x / self.video_scale)
            source_y = int(offset_in_video_y / self.video_scale)
            source_w = int(visible_w / self.video_scale)
            source_h = int(visible_h / self.video_scale)
            
            # 确保不超出原始图像范围
            source_x = max(0, min(source_x, orig_w - 1))
            source_y = max(0, min(source_y, orig_h - 1))
            source_w = min(source_w, orig_w - source_x)
            source_h = min(source_h, orig_h - source_y)
            
            # 计算目标显示区域（显示坐标系）
            target_x = canvas_display_x + int(visible_start_x * self.display_scale)
            target_y = canvas_display_y + int(visible_start_y * self.display_scale)
            target_w = int(visible_w * self.display_scale)
            target_h = int(visible_h * self.display_scale)
            
            # 绘制视频（只绘制画布内的部分）
            source_rect = QRect(source_x, source_y, source_w, source_h)
            target_rect = QRect(target_x, target_y, target_w, target_h)
            painter.drawPixmap(target_rect, self.image_pixmap, source_rect)
        
        # 绘制预览指导框（仅用于1053×1746画布）
        if (self.show_preview_frame and self.canvas_width == 1053 and self.canvas_height == 1746
                and self.should_show_guides):
            guides = [
                {
                    "color": QColor(64, 128, 255),
                    "left": 48,
                    "top": 114,
                    "width": 957,
                    "height": 1584,
                    "caption": "蓝色框为封面挂件编辑区"  # 第一步页面保留文字标注
                },
                {
                    "color": QColor(76, 175, 80),
                    "left": 48,
                    "top": 114,
                    "width": 957,
                    "height": 1278,
                    "caption": "绿色框为封面图尺寸"  # 第一步页面保留文字标注
                }
            ]

            def draw_caption(rect, text, color):
                pad = max(4, int(6 * self.display_scale))
                font = painter.font()
                font.setPointSize(max(9, int(12 * self.display_scale)))
                painter.setFont(font)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(text)
                text_h = metrics.height()
                max_x = canvas_display_x + canvas_display_w
                max_y = canvas_display_y + canvas_display_h

                pos_x = rect.x() + rect.width() + pad
                pos_y = rect.y() + pad

                if pos_x + text_w + pad * 2 > max_x:
                    pos_x = max(canvas_display_x, rect.x() - text_w - pad * 3)
                    pos_y = rect.y() + rect.height() + pad
                if pos_y + text_h + pad * 2 > max_y:
                    pos_y = rect.y() - (text_h + pad * 2 + pad)
                if pos_y < canvas_display_y:
                    pos_y = canvas_display_y + pad

                bg_rect = QRect(pos_x, pos_y, text_w + pad * 2, text_h + pad * 2)
                bg_color = QColor(color)
                bg_color.setAlpha(40)
                painter.setBrush(bg_color)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(bg_rect, pad, pad)
                painter.setPen(color)
                painter.drawText(bg_rect, Qt.AlignCenter, text)
                painter.setBrush(Qt.NoBrush)

            for guide in guides:
                rect_x = canvas_display_x + round(guide["left"] * self.display_scale)
                rect_y = canvas_display_y + round(guide["top"] * self.display_scale)
                rect_w = round(guide["width"] * self.display_scale)
                rect_h = round(guide["height"] * self.display_scale)
                rect = QRect(rect_x, rect_y, rect_w, rect_h)
                pen = QPen(guide["color"], max(2, int(2 * self.display_scale)))
                pen.setStyle(Qt.SolidLine)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
                draw_caption(rect, guide["caption"], guide["color"])
        
        # 绘制画布外的遮罩（半透明黑色）
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.NoPen)
        # 上
        if canvas_display_y > 0:
            painter.drawRect(0, 0, label_size.width(), canvas_display_y)
        # 下
        bottom_y = canvas_display_y + canvas_display_h
        if bottom_y < label_size.height():
            painter.drawRect(0, bottom_y, label_size.width(), label_size.height() - bottom_y)
        # 左
        if canvas_display_x > 0:
            painter.drawRect(0, canvas_display_y, canvas_display_x, canvas_display_h)
        # 右
        right_x = canvas_display_x + canvas_display_w
        if right_x < label_size.width():
            painter.drawRect(right_x, canvas_display_y, label_size.width() - right_x, canvas_display_h)

    def set_show_preview_frame(self, show: bool):
        """设置是否显示预览指导框"""
        self.show_preview_frame = show
        self.update()

    def set_guides_visible(self, visible: bool):
        """控制辅助框显示"""
        self.should_show_guides = visible
        self.update()


class VideoSetupDialog(QDialog):
    """第一步：素材基础设置对话框（时间裁切和整体尺寸裁切）"""
    
    def __init__(self, processor, parent=None, mode="video"):
        super().__init__(parent)
        self.processor = processor
        self.mode = mode if mode in ("video", "image") else "video"
        self.is_image_mode = self.mode == "image"
        self.media_label = "视频" if self.mode == "video" else "图片"
        self.preview_title_prefix = "视频首帧预览" if self.mode == "video" else "图片预览"
        # 标题前缀根据模式动态设置
        self.crop_title_prefix = f"{self.media_label}画面缩放" if self.mode == "video" else f"{self.media_label}尺寸缩放"
        self.position_title = f"{self.media_label}位置调整"
        self.start_time = 0.0
        self.end_time = 3.0
        self.target_fps = 24
        if self.is_image_mode:
            self.end_time = 1.0
            self.target_fps = 1
        self.resize_enabled = True  # 默认开启
        self.resize_x = 0
        self.resize_y = 0
        
        # 画布尺寸选项
        self.canvas_sizes = {
            "1053×1746": (1053, 1746),
            "750×1250": (750, 1250)
        }
        self.current_canvas_size = "1053×1746"  # 默认尺寸
        
        self.setWindowTitle(f"第一步：{self.media_label}基础设置 - 庆雅神器")
        self.setModal(True)
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        
        self.init_ui()
        self.load_video_info()
    
    def init_ui(self):
        """初始化界面 - 优化布局：左侧预览，右侧所有设置"""
        scaler = UIScaler()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(scaler.scale(12), scaler.scale(12), scaler.scale(12), scaler.scale(12))
        layout.setSpacing(scaler.scale(12))
        
        # 素材信息显示（顶部）
        info_group = QGroupBox(f"{self.media_label}信息")
        info_layout = QVBoxLayout()
        self.info_label = QLabel()
        self.info_label.setObjectName("videoInfoLabel")
        self.info_label.setStyleSheet("padding: 10px; background-color: transparent;")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 主内容区：左右布局
        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(scaler.scale(16))
        
        # ========== 左侧：视频首帧预览 ==========
        preview_group = QGroupBox(f"{self.preview_title_prefix}")
        preview_group_layout = QVBoxLayout()
        
        # 画布尺寸选择和启用整体尺寸裁切（合并在一起）
        canvas_size_layout = QHBoxLayout()
        canvas_size_layout.addWidget(QLabel("画布尺寸:"))
        self.canvas_size_combo = QComboBox()
        self.canvas_size_combo.addItems(list(self.canvas_sizes.keys()))
        self.canvas_size_combo.setCurrentText(self.current_canvas_size)
        self.canvas_size_combo.currentTextChanged.connect(self.on_canvas_size_changed)
        canvas_size_layout.addWidget(self.canvas_size_combo)
        
        # 启用整体尺寸裁切（移动到画布尺寸旁边）
        canvas_size_layout.addSpacing(scaler.scale(16))
        self.resize_check = QCheckBox("启用整体尺寸裁切")
        self.resize_check.setChecked(True)  # 默认开启
        self.resize_check.stateChanged.connect(self.on_resize_check_changed)
        canvas_size_layout.addWidget(self.resize_check)
        canvas_size_layout.addStretch()
        preview_group_layout.addLayout(canvas_size_layout)
        
        # 使用QScrollArea包装画布，支持滚动查看
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("QScrollArea { border: 1px solid gray; background-color: #f0f0f0; }")
        
        # 初始化画布（使用默认尺寸）
        canvas_width, canvas_height = self.canvas_sizes[self.current_canvas_size]
        self.canvas_label = CanvasLabel(canvas_width, canvas_height)
        self.canvas_label.set_show_preview_frame(canvas_width == 1053 and canvas_height == 1746)
        self.canvas_label.set_guides_visible(canvas_width == 1053 and canvas_height == 1746)
        self.canvas_label.transform_changed.connect(self.on_transform_changed)
        
        self.scroll_area.setWidget(self.canvas_label)
        preview_group_layout.addWidget(self.scroll_area)
        preview_group.setLayout(preview_group_layout)
        self.preview_group_box = preview_group
        main_content_layout.addWidget(preview_group, 2)  # 左侧占2/3空间
        
        # ========== 右侧：所有设置（视频长度裁切、整体尺寸裁切、位置调整、操作提示） ==========
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(scaler.scale(320))
        right_scroll.setMaximumWidth(scaler.scale(380))
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(scaler.scale(8), scaler.scale(8), scaler.scale(8), scaler.scale(8))
        right_layout.setSpacing(scaler.scale(12))
        
        # 1. 视频长度裁切（仅视频模式显示）
        time_group = QGroupBox(f"{self.media_label}长度裁切")
        time_layout = QVBoxLayout()
        
        start_time_layout = QHBoxLayout()
        start_time_layout.addWidget(QLabel("起始时间（秒）:"))
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0.0, 9999.0)
        self.start_time_spin.setValue(0.0)
        self.start_time_spin.setSingleStep(0.1)
        self.start_time_spin.setDecimals(2)
        start_time_layout.addWidget(self.start_time_spin)
        time_layout.addLayout(start_time_layout)
        
        end_time_layout = QHBoxLayout()
        end_time_layout.addWidget(QLabel("结束时间（秒）:"))
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0.0, 9999.0)
        self.end_time_spin.setValue(self.end_time)
        self.end_time_spin.setSingleStep(0.1)
        self.end_time_spin.setDecimals(2)
        end_time_layout.addWidget(self.end_time_spin)
        time_layout.addLayout(end_time_layout)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("目标帧率:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.target_fps)
        fps_layout.addWidget(self.fps_spin)
        time_layout.addLayout(fps_layout)
        
        time_group.setLayout(time_layout)
        self.time_group = time_group
        if not self.is_image_mode:
            right_layout.addWidget(time_group)
        
        # 2. 视频/图片画面缩放（优化标题和布局）
        scale_title = f"{self.media_label}画面缩放" if self.mode == "video" else f"{self.media_label}尺寸缩放"
        crop_settings_group = QGroupBox(scale_title)
        crop_settings_layout = QVBoxLayout()
        crop_settings_layout.setSpacing(scaler.scale(10))
        
        # 显示当前选择的画布尺寸
        self.canvas_size_label = QLabel()
        self.update_canvas_size_label()
        crop_settings_layout.addWidget(self.canvas_size_label)
        
        scale_control_layout = QHBoxLayout()
        scale_control_layout.addWidget(QLabel("缩放比例:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.00, 5.00)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.01)
        self.scale_spin.setValue(1.00)
        self.scale_spin.valueChanged.connect(self.on_scale_spin_changed)
        scale_control_layout.addWidget(self.scale_spin)
        crop_settings_layout.addLayout(scale_control_layout)
        
        crop_settings_group.setLayout(crop_settings_layout)
        self.crop_group_box = crop_settings_group
        right_layout.addWidget(crop_settings_group)
        
        # 3. 视频位置调整
        position_group = QGroupBox(self.position_title)
        position_layout = QVBoxLayout()
        position_layout.setSpacing(scaler.scale(8))
        
        x_position_layout = QHBoxLayout()
        x_position_layout.addWidget(QLabel("X坐标(像素):"))
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-10000, 10000)
        self.offset_x_spin.setSingleStep(1)
        self.offset_x_spin.setValue(0)
        self.offset_x_spin.valueChanged.connect(self.on_offset_x_changed)
        x_position_layout.addWidget(self.offset_x_spin)
        position_layout.addLayout(x_position_layout)
        
        y_position_layout = QHBoxLayout()
        y_position_layout.addWidget(QLabel("Y坐标(像素):"))
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-10000, 10000)
        self.offset_y_spin.setSingleStep(1)
        self.offset_y_spin.setValue(0)
        self.offset_y_spin.valueChanged.connect(self.on_offset_y_changed)
        y_position_layout.addWidget(self.offset_y_spin)
        position_layout.addLayout(y_position_layout)
        
        position_group.setLayout(position_layout)
        right_layout.addWidget(position_group)
        
        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_content_layout.addWidget(right_scroll, 1)  # 右侧占1/3空间
        
        layout.addLayout(main_content_layout)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        if self.is_image_mode:
            self.time_group.setVisible(False)
            self.start_time_spin.setValue(0.0)
            self.start_time_spin.setEnabled(False)
            self.end_time_spin.setValue(1.0)
            self.end_time_spin.setEnabled(False)
            self.fps_spin.setValue(1)
            self.fps_spin.setEnabled(False)
        
        # 确保默认开启状态正确设置（触发状态更新）
        if hasattr(self, 'resize_check'):
            self.on_resize_check_changed()
    
    def load_video_info(self):
        """加载素材信息"""
        if not self.processor:
            return
        
        try:
            info = self.processor.get_video_info()
            info_text = (f"分辨率: {info['width']}x{info['height']} | "
                        f"时长: {info['duration']:.2f}秒 | "
                        f"帧率: {info['fps']}fps | "
                        f"总帧数: {info['frame_count']} | "
                        f"宽高比: {info['aspect_ratio']:.2f}")
            self.info_label.setText(f"{self.media_label}信息：{info_text}")
            
            # 初始化时间范围
            duration = info['duration']
            if duration <= 0:
                duration = 1.0
            self.start_time_spin.setMaximum(duration)
            self.end_time_spin.setMaximum(duration)
            default_end = min(3.0, duration)
            if self.is_image_mode:
                default_end = 1.0
            self.end_time_spin.setValue(default_end)
            
            # 加载第一帧到画布（原始视频，未裁切）
            first_frame = self.processor.get_first_frame()
            if first_frame is not None:
                # 显示原始视频首帧到画布
                self.canvas_label.set_image(first_frame)
                self.update_manual_scale_display()
                # 更新X、Y坐标显示
                if hasattr(self, 'offset_x_spin'):
                    self.offset_x_spin.blockSignals(True)
                    self.offset_x_spin.setValue(int(round(self.canvas_label.video_offset_x)))
                    self.offset_x_spin.blockSignals(False)
                if hasattr(self, 'offset_y_spin'):
                    self.offset_y_spin.blockSignals(True)
                    self.offset_y_spin.setValue(int(round(self.canvas_label.video_offset_y)))
                    self.offset_y_spin.blockSignals(False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载{self.media_label}信息失败: {str(e)}")
    
    def on_canvas_size_changed(self, size_text):
        """画布尺寸改变"""
        if size_text not in self.canvas_sizes:
            return
        
        self.current_canvas_size = size_text
        canvas_width, canvas_height = self.canvas_sizes[size_text]
        
        # 保存当前的图像和变换参数
        old_image = self.canvas_label.original_image
        
        # 更新画布的尺寸
        self.canvas_label.canvas_width = canvas_width
        self.canvas_label.canvas_height = canvas_height
        self.canvas_label.set_show_preview_frame(canvas_width == 1053 and canvas_height == 1746)
        self.canvas_label.set_guides_visible(canvas_width == 1053 and canvas_height == 1746)
        
        # 如果之前有图像，重新设置到新画布（会自动重新计算缩放和位置）
        if old_image is not None:
            self.canvas_label.set_image(old_image)
            self.update_manual_scale_display()
            # 更新X、Y坐标显示
            if hasattr(self, 'offset_x_spin'):
                self.offset_x_spin.blockSignals(True)
                self.offset_x_spin.setValue(int(round(self.canvas_label.video_offset_x)))
                self.offset_x_spin.blockSignals(False)
            if hasattr(self, 'offset_y_spin'):
                self.offset_y_spin.blockSignals(True)
                self.offset_y_spin.setValue(int(round(self.canvas_label.video_offset_y)))
                self.offset_y_spin.blockSignals(False)
        
        # 更新标签文本
        self.update_canvas_size_label()
        self.update_manual_scale_display()
    
    def update_canvas_size_label(self):
        """更新画布尺寸标签"""
        canvas_width, canvas_height = self.canvas_sizes[self.current_canvas_size]
        self.canvas_size_label.setText(f"当前画布尺寸: {canvas_width}×{canvas_height}")
        
        # 更新预览组标题
        if hasattr(self, 'preview_group_box') and self.preview_group_box:
            self.preview_group_box.setTitle(f"{self.preview_title_prefix}（固定画布 {canvas_width}×{canvas_height}）")
        
        # 更新裁切设置组标题
        if hasattr(self, 'crop_group_box') and self.crop_group_box:
            self.crop_group_box.setTitle(f"{self.crop_title_prefix}（裁切为 {canvas_width}×{canvas_height}）")
    
    def on_resize_check_changed(self):
        """整体尺寸调整选项改变"""
        enabled = self.resize_check.isChecked()
        # 更新复选框文本
        # 不再显示括号内的尺寸信息
        self.resize_check.setText("启用整体尺寸裁切")
    
    def on_transform_changed(self, scale, offset_x, offset_y):
        """视频变换参数改变（从画布拖动/缩放）"""
        if hasattr(self, 'scale_spin'):
            self.update_manual_scale_display()
        # 更新X、Y坐标输入框
        if hasattr(self, 'offset_x_spin'):
            self.offset_x_spin.blockSignals(True)
            self.offset_x_spin.setValue(int(round(offset_x)))
            self.offset_x_spin.blockSignals(False)
        if hasattr(self, 'offset_y_spin'):
            self.offset_y_spin.blockSignals(True)
            self.offset_y_spin.setValue(int(round(offset_y)))
            self.offset_y_spin.blockSignals(False)
    
    def on_scale_spin_changed(self, value):
        """手动调整缩放比例"""
        if not hasattr(self, 'canvas_label') or self.canvas_label.image_pixmap is None:
            return
        adjusted = max(self.canvas_label.min_scale, min(self.canvas_label.max_scale, float(value)))
        self.canvas_label.set_scale(adjusted)
        self.update_manual_scale_display()

    def update_manual_scale_display(self):
        if hasattr(self, 'scale_spin'):
            self.scale_spin.blockSignals(True)
            self.scale_spin.setValue(round(max(0.0, min(5.0, self.canvas_label.video_scale)), 2))
            self.scale_spin.blockSignals(False)
    
    def on_offset_x_changed(self, value):
        """手动调整X坐标"""
        if not hasattr(self, 'canvas_label') or self.canvas_label.image_pixmap is None:
            return
        # 保持当前的scale和offset_y，只更新offset_x
        current_scale = self.canvas_label.video_scale
        current_offset_y = self.canvas_label.video_offset_y
        self.canvas_label.set_crop_params(current_scale, float(value), current_offset_y)
    
    def on_offset_y_changed(self, value):
        """手动调整Y坐标"""
        if not hasattr(self, 'canvas_label') or self.canvas_label.image_pixmap is None:
            return
        # 保持当前的scale和offset_x，只更新offset_y
        current_scale = self.canvas_label.video_scale
        current_offset_x = self.canvas_label.video_offset_x
        self.canvas_label.set_crop_params(current_scale, current_offset_x, float(value))
    
    def accept(self):
        """确认设置"""
        # 检查时间范围
        if self.end_time_spin.value() <= self.start_time_spin.value():
            QMessageBox.warning(self, "警告", "结束时间必须大于起始时间")
            return
        
        # 保存设置
        self.start_time = self.start_time_spin.value()
        self.end_time = self.end_time_spin.value()
        self.target_fps = self.fps_spin.value()
        self.resize_enabled = self.resize_check.isChecked()
        
        # 如果启用了整体尺寸裁切，从画布获取变换参数
        if self.resize_enabled:
            scale, offset_x, offset_y = self.canvas_label.get_crop_params()
            # 保存变换参数，用于后续裁切
            self.video_scale = scale
            self.video_offset_x = offset_x
            self.video_offset_y = offset_y
        else:
            self.video_scale = 1.0
            self.video_offset_x = 0.0
            self.video_offset_y = 0.0
        
        super().accept()
    
    def get_settings(self):
        """获取设置"""
        canvas_width, canvas_height = self.canvas_sizes[self.current_canvas_size]
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'target_fps': self.target_fps,
            'resize_enabled': self.resize_enabled,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height,
            'video_scale': getattr(self, 'video_scale', 1.0),
            'video_offset_x': getattr(self, 'video_offset_x', 0.0),
            'video_offset_y': getattr(self, 'video_offset_y', 0.0)
        }


class ModeSelectionDialog(QDialog):
    """模式选择对话框 - 欢迎界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("庆雅神器")
        self.setModal(True)
        self.selected_mode = None
        scaler = UIScaler()
        
        # 设置窗口尺寸 - 根据屏幕可用空间自适应
        fixed_width = scaler.scale(1000)
        fixed_height = scaler.scale(700)
        screen = QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            safe_width = int(available.width() * 0.9)
            safe_height = int(available.height() * 0.9)
            fixed_width = max(scaler.scale(860), min(fixed_width, safe_width))
            fixed_height = max(scaler.scale(600), min(fixed_height, safe_height))
        self.setFixedSize(int(fixed_width), int(fixed_height))
        # 居中显示窗口
        self._center_window()
        
        # 应用主题样式
        from ui_utils import get_stylesheet
        style = get_stylesheet("light", scaler.get_scale_factor())
        self.setStyleSheet(style)
        
        # 主布局 - 左右布局设计
        main_widget = QWidget()
        main_widget.setObjectName("welcomeMainWidget")
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 设置主窗口布局
        window_layout = QVBoxLayout(self)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(main_widget)
        
        # ========== 左侧：欢迎页照片展示（2:3黄金比例，进一步缩小，等比缩放） ==========
        left_container = QWidget()
        left_container.setObjectName("welcomeLeftPanel")
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(scaler.scale(40), scaler.scale(60), scaler.scale(40), scaler.scale(60))
        left_layout.setSpacing(0)
        
        # 左上角：版本标识（水印效果）
        left_top_widget = QWidget()
        left_top_layout = QVBoxLayout(left_top_widget)
        left_top_layout.setContentsMargins(0, 0, 0, 0)
        left_top_layout.setSpacing(scaler.scale(4))
        
        # 水印文字
        self.activation_status_label = QLabel("开源版·BEYOOOND")
        self.activation_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.activation_status_label.setObjectName("activationStatus")
        activation_font = scaler.get_font(size=11, weight=QFont.Medium)
        self.activation_status_label.setFont(activation_font)
        # 设置为不可选中和复制
        self.activation_status_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.activation_status_label.setCursor(Qt.ArrowCursor)
        # 应用半透明效果（水印）- 提高可见性
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(0.75)  # 75% 透明度，更明显
        self.activation_status_label.setGraphicsEffect(opacity_effect)
        
        left_top_layout.addWidget(self.activation_status_label)
        left_layout.addWidget(left_top_widget)
        
        left_layout.addStretch()
        
        # 加载欢迎页照片 - 根据容器大小自适应缩放，确保完整显示不裁切
        welcome_image_label = QLabel()
        welcome_image_label.setAlignment(Qt.AlignCenter)
        welcome_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许缩放
        # 尝试多个路径查找欢迎页照片
        welcome_image_paths = [
            os.path.join(os.path.dirname(__file__), "resources", "欢迎页照片.png"),
            os.path.join(os.path.dirname(__file__), "欢迎页照片.png"),  # 兼容旧路径
        ]
        if hasattr(sys, '_MEIPASS'):
            welcome_image_paths.insert(0, os.path.join(sys._MEIPASS, "resources", "欢迎页照片.png"))
        
        welcome_image_path = None
        for path in welcome_image_paths:
            if os.path.exists(path):
                welcome_image_path = path
                break
        
        if os.path.exists(welcome_image_path):
            pixmap = QPixmap(welcome_image_path)
            # 设置原始图片，让QLabel自动根据容器大小缩放
            welcome_image_label.setPixmap(pixmap)
            welcome_image_label.setStyleSheet("background-color: transparent;")
            # 启用自动缩放并保持宽高比（不裁切）
            welcome_image_label.setScaledContents(True)
        else:
            # 如果图片不存在，显示占位符
            welcome_image_label.setText("欢迎页照片")
            welcome_image_label.setStyleSheet("background-color: #F5F5F5; color: #999999;")
        
        left_layout.addWidget(welcome_image_label, alignment=Qt.AlignCenter)
        left_layout.addStretch()
        
        main_layout.addWidget(left_container, 2)  # 左侧占2份（2:3黄金比例）
        
        # ========== 右侧：模式选择和相关信息（3份，2:3黄金比例） ==========
        right_container = QWidget()
        right_container.setObjectName("welcomeRightPanel")
        right_layout = QVBoxLayout(right_container)
        # 调整边距，让顶部链接更靠近右上角
        right_layout.setContentsMargins(scaler.scale(40), scaler.scale(32), scaler.scale(40), scaler.scale(46))
        right_layout.setSpacing(scaler.scale(20))
        
        # 顶部外链导航
        links_bar = self._create_links_bar()
        right_layout.addWidget(links_bar, 0, Qt.AlignRight)
        right_layout.addSpacing(scaler.scale(32))
        
        # 引导语前增加适度留白
        right_layout.addSpacing(scaler.scale(12))
        
        # 引导语 - 优化：20px，字重500，颜色#1E40AF，带边框
        guide_label = QLabel("请选择您需要处理的素材类型，开始您的创作之旅")
        guide_label.setAlignment(Qt.AlignCenter)
        guide_label.setObjectName("guideText")
        # 字号20px，字重500
        guide_font = scaler.get_font(size=20, weight=QFont.Medium if hasattr(QFont, 'Medium') else QFont.Normal)
        guide_label.setFont(guide_font)
        guide_label.setWordWrap(False)
        guide_label.setMaximumWidth(scaler.scale(760))
        # 样式：品牌主色#1E40AF，1px圆角边框，上下内边距16px，高度36px
        guide_label.setStyleSheet(f"""
            QLabel#guideText {{
                color: #1E40AF;
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                padding: {scaler.scale(10)}px {scaler.scale(14)}px;
                min-height: {scaler.scale(28)}px;
                line-height: 1.5;
            }}
        """)
        right_layout.addWidget(guide_label, 0, Qt.AlignCenter)
        
        # 引导语后间距20px
        right_layout.addSpacing(scaler.scale(24))
        
        # 模式选择卡片 - 垂直布局（上面动态，下面静态）
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setSpacing(scaler.scale(36))
        # 增加内边距，避免卡片之间及底部贴边
        cards_layout.setContentsMargins(0, scaler.scale(20), 0, scaler.scale(80))
        cards_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        # 动态视频卡片（上面）
        video_card = self._create_mode_card("video")
        cards_layout.addWidget(video_card)
        
        # 静态图片卡片（下面）
        image_card = self._create_mode_card("image")
        cards_layout.addWidget(image_card)
        
        right_layout.addWidget(cards_container, 0, Qt.AlignHCenter)
        right_layout.addSpacing(scaler.scale(16))
        right_layout.addStretch()
        
        # 底部：版权信息
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(scaler.scale(8))
        
        bottom_layout.addStretch()
        
        # 右侧：版权信息（底部右侧，优化：13px，颜色#9CA3AF，使用·分隔符）
        copyright_label = QLabel("湖南度尚文化创意有限公司 · 版本 v1.5")
        copyright_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        copyright_label.setObjectName("copyright")
        # 字号13px，字重400
        copyright_font = scaler.get_font(size=13, weight=QFont.Normal)
        copyright_label.setFont(copyright_font)
        copyright_label.setStyleSheet("color: #9CA3AF;")  # 浅灰
        # 与右边界保持24px间距，顶部预留32px空白
        copyright_label.setContentsMargins(0, scaler.scale(32), scaler.scale(24), 0)
        bottom_layout.addWidget(copyright_label)
        
        bottom_layout.addStretch()
        
        right_layout.addWidget(bottom_container)
        
        main_layout.addWidget(right_container, 3)  # 右侧占3份（2:3黄金比例）
        
        # 应用欢迎界面样式 - 左右布局风格
        self.setStyleSheet(f"""
            QWidget#welcomeMainWidget {{
                background: #FFFFFF;
            }}
            QWidget#welcomeLeftPanel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F8F9FA, stop:1 #F0F2F5);
            }}
            QWidget#welcomeRightPanel {{
                background: #FFFFFF;
            }}
            QLabel#companyLogo {{
                /* 样式已在代码中直接设置，此处保留作为备用 */
                background: transparent;
            }}
            QLabel#subtitle {{
                /* 样式已在代码中直接设置，此处保留作为备用 */
                background: transparent;
            }}
            QLabel#guideText {{
                /* 样式已在代码中直接设置，此处保留作为备用 */
            }}
            QLabel#welcomeNote {{
                color: #666666;
                background: transparent;
                padding: {scaler.scale(8)}px;
            }}
            QLabel#copyright {{
                /* 样式已在代码中直接设置，此处保留作为备用 */
                background: transparent;
            }}
            QLabel#activationStatus {{
                color: #4B5563;
                background: transparent;
                padding: {scaler.scale(4)}px 0;
                font-weight: 500;
            }}
            QLabel#activationCode {{
                color: #4B5563;
                background: transparent;
                padding: {scaler.scale(2)}px 0;
                font-size: {scaler.font_size(10)}px;
                font-weight: 400;
            }}
        """)
        
        # 激活模块已移除，始终展示开源版信息
    
    def _create_links_bar(self):
        scaler = UIScaler()
        container = QWidget()
        container.setObjectName("welcomeLinksBar")
        container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaler.scale(10))

        links = [
            ("Github", "github-icon.svg", "https://github.com/BEYOOOND/QINYA"),
            ("小红书", "xiaohongshu-seeklogo.svg", "https://www.xiaohongshu.com/user/profile/690714830000000030028a93?xsec_token=ABRgCEQkZzDp6ehrunHoNgf23DMOK9dnht_-Hiiq-S5Zw%3D&xsec_source=pc_search"),
            ("服务号", "wechat-seeklogo.svg", "https://mp.weixin.qq.com/s/EfduH7-sTO38uoc7OfNXNQ"),
        ]
        for label, icon_name, url in links:
            button = self._create_link_button(label, icon_name, url)
            layout.addWidget(button)
        doc_button = self._create_doc_button()
        layout.addWidget(doc_button)
        return container

    def _create_link_button(self, label: str, icon_filename: str, url: str):
        button = WelcomeLinkButton(label)
        icon_size = max(16, button.iconSize().width())
        icon = self._load_link_icon(icon_filename, icon_size)
        if icon:
            button.setIcon(icon)
        button.clicked.connect(lambda _, link=url: self._open_external_link(link))
        return button

    def _load_link_icon(self, filename: str, target_size: int):
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "resources", filename),
            os.path.join(os.path.dirname(__file__), filename),
        ]
        if hasattr(sys, "_MEIPASS"):
            candidate_paths.insert(0, os.path.join(sys._MEIPASS, "resources", filename))

        for path in candidate_paths:
            if not os.path.exists(path):
                continue
            pixmap = None
            if path.lower().endswith(".svg"):
                renderer = QSvgRenderer(path)
                if not renderer.isValid():
                    continue
                pixmap = QPixmap(target_size, target_size)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
            else:
                pixmap = QPixmap(path)
                pixmap = pixmap.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if pixmap:
                return QIcon(pixmap)
        return None

    def _open_external_link(self, url: str):
        if not url:
            return
        QDesktopServices.openUrl(QUrl(url))

    def _create_doc_button(self):
        scaler = UIScaler()
        doc_button = QToolButton()
        doc_button.setText("?")
        doc_button.setCursor(Qt.PointingHandCursor)
        doc_button.setObjectName("welcomeDocButton")
        side = scaler.scale(34)
        doc_button.setFixedSize(side, side)
        doc_button.setToolTip("查看说明文件")
        doc_button.setStyleSheet(f"""
            QToolButton#welcomeDocButton {{
                border-radius: {side // 2}px;
                border: 1px solid #E5E7EB;
                background-color: #F8FAFC;
                color: #111827;
                font-size: {scaler.font_size(18)}px;
                font-weight: 700;
            }}
            QToolButton#welcomeDocButton:hover {{
                background-color: #E0E7FF;
                border-color: #4338CA;
                color: #1E3A8A;
            }}
            QToolButton#welcomeDocButton:pressed {{
                background-color: #C7D2FE;
            }}
        """)
        doc_button.clicked.connect(self._show_welcome_docs)
        return doc_button

    def _show_welcome_docs(self):
        scaler = UIScaler()
        dialog = QDialog(self)
        dialog.setWindowTitle("说明文件")
        dialog.setModal(True)
        dialog.resize(scaler.scale(520), scaler.scale(360))

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(scaler.scale(20), scaler.scale(16), scaler.scale(20), scaler.scale(16))
        layout.setSpacing(scaler.scale(12))

        intro = QLabel("快速了解庆雅神器的使用步骤与功能模块，帮助新人 15 分钟内完成导出流程。")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #374151; font-weight: 500;")
        layout.addWidget(intro)

        doc_html = """
        <h3 style="color:#1E3A8A;">核心使用流程</h3>
        <ol>
            <li>欢迎页选择「动态视频」或「静态图片」模式，并完成第一步整体尺寸裁切。</li>
            <li>在主界面设置矩形尺寸、是否反选、输出类型及时间参数。</li>
            <li>在素材面板添加 <strong>[主图]</strong> 或 <strong>[挂件]</strong> 素材，实时查看叠加效果。</li>
            <li>点击「开始处理」导出气泡图、封面图、封面故事或封面外挂等结果。</li>
        </ol>
        <h3 style="color:#1E3A8A;">功能模块概览</h3>
        <ul>
            <li><strong>整体尺寸裁切</strong>：记录 scale/offset，保证多尺寸画布一致。</li>
            <li><strong>双框导出</strong>：封面图模式自动对齐封面框、红包框、挂件框与不可编辑框。</li>
            <li><strong>素材双模块</strong>：公共素材适用于封面/故事/气泡，挂件素材仅作用于封面外挂。</li>
            <li><strong>气泡挂件制作</strong>：独立对话框，480×384 输出，支持底图与素材自由排版。</li>
        </ul>
        """
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setReadOnly(True)
        text_browser.setStyleSheet("""
            QTextBrowser {
                background: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 12px;
                color: #1F2937;
                font-size: 14px;
            }
        """)
        text_browser.setHtml(doc_html)
        layout.addWidget(text_browser)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box, alignment=Qt.AlignRight)

        dialog.exec_()
    
    def _create_mode_card(self, mode):
        """创建模式选择卡片 - 简化设计：图标+标题"""
        scaler = UIScaler()
        card = QFrame()
        card.setObjectName(f"modeCard_{mode}")
        card.setCursor(Qt.PointingHandCursor)
        # 卡片尺寸 - 优化为更合适的高度，设置合适的最大宽度避免裁切
        card.setMinimumHeight(scaler.scale(120))
        card.setFixedWidth(scaler.scale(420))
        card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        
        # 卡片布局 - 水平布局
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(scaler.scale(28), scaler.scale(24), scaler.scale(28), scaler.scale(24))
        card_layout.setSpacing(scaler.scale(24))
        
        # 左侧：大尺寸图标（64x64）
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setObjectName("cardIcon")
        icon_label.setFixedSize(scaler.scale(64), scaler.scale(64))
        
        # 加载图标文件
        icon_filename = "videofolder_99361.png" if mode == "video" else "picture_photo_image_icon_131252.png"
        # 尝试多个路径查找图标
        icon_paths = [
            os.path.join(os.path.dirname(__file__), "resources", icon_filename),
            os.path.join(os.path.dirname(__file__), icon_filename),  # 兼容旧路径
        ]
        if hasattr(sys, '_MEIPASS'):
            icon_paths.insert(0, os.path.join(sys._MEIPASS, "resources", icon_filename))
        
        icon_path = None
        for path in icon_paths:
            if os.path.exists(path):
                icon_path = path
                break
        
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            # 统一图标尺寸为64x64
            icon_size = scaler.scale(64)
            # 保持宽高比缩放
            if pixmap.width() > pixmap.height():
                pixmap = pixmap.scaledToWidth(icon_size, Qt.SmoothTransformation)
            else:
                pixmap = pixmap.scaledToHeight(icon_size, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(Qt.AlignCenter)
        else:
            # 如果图标文件不存在，使用emoji作为后备
            icon_label.setText("🎬" if mode == "video" else "🖼️")
            icon_font = scaler.get_font(size=40, weight=QFont.Bold)
            icon_label.setFont(icon_font)
        
        card_layout.addWidget(icon_label)
        
        # 右侧：标题
        title = "动态视频模式" if mode == "video" else "静态图片模式"
        title_label = QLabel(title)
        title_font = scaler.get_font(size=18, weight=QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setObjectName("cardTitle")
        card_layout.addWidget(title_label, 1)
        
        # 点击事件（整个卡片可点击）
        def on_click():
            self._choose_mode(mode)
        
        card.mousePressEvent = lambda e: on_click()
        
        # 优化后的卡片样式（符合专家建议）
        if mode == "video":
            # 动态视频卡片 - 品牌蓝色系
            card.setStyleSheet(f"""
                QFrame#modeCard_video {{
                    background-color: #FFFFFF;
                    border: 1px solid #E5E7EB;
                    border-radius: {scaler.scale(8)}px;
                }}
                QFrame#modeCard_video:hover {{
                    background-color: #FFFFFF;
                    border: 1px solid #2563EB;
                    border-radius: {scaler.scale(8)}px;
                }}
                QPushButton#ctaButton {{
                    background-color: #2563EB;
                    color: #FFFFFF;
                    border: none;
                    border-radius: {scaler.scale(6)}px;
                    font-weight: 500;
                }}
                QPushButton#ctaButton:hover {{
                    background-color: #1D4ED8;
                }}
                QPushButton#ctaButton:pressed {{
                    background-color: #1E40AF;
                }}
                QLabel#cardTitle {{
                    color: #1A1A1A;
                    background: transparent;
                }}
            """)
        else:
            # 静态图片卡片 - 品牌蓝色系（统一风格）
            card.setStyleSheet(f"""
                QFrame#modeCard_image {{
                    background-color: #FFFFFF;
                    border: 1px solid #E5E7EB;
                    border-radius: {scaler.scale(8)}px;
                }}
                QFrame#modeCard_image:hover {{
                    background-color: #FFFFFF;
                    border: 1px solid #2563EB;
                    border-radius: {scaler.scale(8)}px;
                }}
                QPushButton#ctaButton {{
                    background-color: #2563EB;
                    color: #FFFFFF;
                    border: none;
                    border-radius: {scaler.scale(6)}px;
                    font-weight: 500;
                }}
                QPushButton#ctaButton:hover {{
                    background-color: #1D4ED8;
                }}
                QPushButton#ctaButton:pressed {{
                    background-color: #1E40AF;
                }}
                QLabel#cardTitle {{
                    color: #1A1A1A;
                    background: transparent;
                }}
            """)
        
        # 添加精美阴影效果（符合专家建议：4px, 0.16 opacity）
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(scaler.scale(24))
        shadow.setXOffset(0)
        shadow.setYOffset(scaler.scale(6))
        shadow.setColor(QColor(0, 0, 0, 15))  # 更柔和的阴影
        card.setGraphicsEffect(shadow)
        
        return card
    
    def _center_window(self):
        """将窗口居中显示在屏幕上"""
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)
    
    def _choose_mode(self, mode: str):
        self.selected_mode = mode
        self.accept()


class ImageLabel(QLabel):
    """可绘制矩形的图像标签"""
    
    selection_changed = pyqtSignal(int, int, int, int)  # x, y, width, height
    watermark_position_changed = pyqtSignal(int, int, int)  # index, x, y
    watermark_scale_changed = pyqtSignal(int, float)  # index, scale
    watermark_selected = pyqtSignal(int)  # index
    color_picked = pyqtSignal(int, int, int)  # r, g, b
    
    def __init__(self):
        super().__init__()
        scaler = UIScaler()
        self.setMinimumSize(scaler.scale(560), scaler.scale(420))
        self.setAlignment(Qt.AlignCenter)
        # 样式将在主题应用时更新
        self.theme = "light"  # 默认主题
        
        self.original_image = None
        self.scaled_image = None
        self.scale_factor = 1.0
        self.start_pos = None
        self.end_pos = None
        self.is_drawing = False
        self.rect = None  # (x, y, width, height) 在原始图像坐标系中
        self.watermarks = []  # 多个素材
        self.active_watermark_index = -1
        self.is_dragging_rect = False  # 是否正在拖动矩形
        self.drag_start_pos = None  # 拖动开始位置（相对于矩形的偏移）
        self.is_dragging_watermark = False  # 是否正在拖动素材
        self.dragging_watermark_index = None  # 当前拖动的素材索引
        self.watermark_drag_start_pos = None  # 素材拖动开始位置（显示坐标）
        self.is_resizing_watermark = False  # 是否正在调整素材大小
        self.resizing_watermark_index = None  # 当前调整大小的素材索引
        self.watermark_resize_start_pos = None  # 素材调整大小开始位置（显示坐标）
        self.watermark_initial_scale = 1.0
        self.watermark_initial_width = 0.0
        self.watermark_initial_height = 0.0
        self.handle_size = 12  # 素材缩放手柄大小
        self.rect_locked = False  # 是否锁定矩形
        self.color_pick_mode = False  # 是否处于颜色拾取模式
        self.show_guides = True  # 是否显示辅助框
        self.should_show_guides = True  # 辅助框显示开关
        self.display_override = None  # 覆盖显示的图像（用于实时预览）
        self.display_pixmap = None  # 覆盖显示的缩放图像
        self.display_override_has_transparency = False  # 覆盖图像是否含透明区域
        self._checker_pixmap = None
        self._checker_tile_size = 0
        self.overlay_image = None  # 附件图片（12345.png）
        self.show_overlay = False  # 是否显示附件图片
        self.overlay_offset_adjustment = (0, 0)  # 附件图片额外偏移
        self.dual_frame_mode = False  # 是否启用封面+挂件双框模式
        self.dual_cover_size = None
        self.dual_pendant_size = None
        self.dual_cover_offset = (0, 0)
        self.dual_non_editable_offset = (0, 0)
        self.dual_non_editable_size = None
        self.dual_red_packet_offset = (0, 0)
        self.dual_red_packet_size = None
        self.inverse_overlay_active = False
        self._load_overlay_image()
    
    def set_theme(self, theme: str):
        """设置主题"""
        self.theme = theme
        self.update()
    
    def _load_overlay_image(self):
        """加载附件图片 12345.png"""
        try:
            # 尝试多个路径来查找资源文件（支持打包后的情况）
            overlay_paths = []
            
            # 1. 如果是PyInstaller打包后的情况，尝试从临时目录加载
            if hasattr(sys, '_MEIPASS'):
                overlay_paths.append(os.path.join(sys._MEIPASS, "resources", "12345.png"))
                overlay_paths.append(os.path.join(sys._MEIPASS, "12345.png"))  # 兼容旧路径
            
            # 2. 尝试从可执行文件所在目录加载（打包后的情况）
            if getattr(sys, 'frozen', False):
                # 如果是打包后的可执行文件
                exe_dir = os.path.dirname(sys.executable)
                overlay_paths.append(os.path.join(exe_dir, "resources", "12345.png"))
                overlay_paths.append(os.path.join(exe_dir, "12345.png"))  # 兼容旧路径
            
            # 3. 尝试从当前脚本所在目录加载（开发环境）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            overlay_paths.append(os.path.join(current_dir, "resources", "12345.png"))
            overlay_paths.append(os.path.join(current_dir, "12345.png"))  # 兼容旧路径
            
            # 4. 尝试从工作目录加载
            overlay_paths.append(os.path.join(os.getcwd(), "resources", "12345.png"))
            overlay_paths.append(os.path.join(os.getcwd(), "12345.png"))  # 兼容旧路径
            
            # 尝试每个路径
            overlay_path = None
            for path in overlay_paths:
                if os.path.exists(path):
                    overlay_path = path
                    break
            
            if overlay_path and os.path.exists(overlay_path):
                self.overlay_image = QPixmap(overlay_path)
                if self.overlay_image.isNull():
                    print(f"警告：无法加载附件图片 {overlay_path}")
                    self.overlay_image = None
                else:
                    print(f"成功加载附件图片: {overlay_path}")
                # 附件图片的原始像素尺寸：957*1353
                self.overlay_width = 957
                self.overlay_height = 1353
            else:
                print(f"警告：未找到附件图片，尝试的路径: {overlay_paths}")
                self.overlay_image = None
                self.overlay_width = 957
                self.overlay_height = 1353
        except Exception as e:
            print(f"加载附件图片失败: {e}")
            import traceback
            traceback.print_exc()
            self.overlay_image = None
            self.overlay_width = 957
            self.overlay_height = 1353
    
    def set_overlay_visible(self, visible):
        """设置是否显示附件图片"""
        self.show_overlay = visible
        self.update()

    def set_overlay_offset_adjustment(self, offset_x: int = 0, offset_y: int = 0):
        """额外控制附件图片在原图坐标系中的偏移"""
        self.overlay_offset_adjustment = (int(offset_x), int(offset_y))
        self.update()
    
    def configure_dual_frame_mode(self, enabled: bool, cover_size=None, pendant_size=None, cover_offset=None,
                                  red_packet_size=None, red_packet_offset=None,
                                  non_editable_offset=None, non_editable_size=None):
        """配置封面/挂件双框模式"""
        self.dual_frame_mode = bool(enabled)
        if self.dual_frame_mode:
            self.dual_cover_size = cover_size if cover_size else self.dual_cover_size
            self.dual_pendant_size = pendant_size if pendant_size else self.dual_pendant_size
            self.dual_cover_offset = cover_offset if cover_offset else self.dual_cover_offset
            if red_packet_size:
                self.dual_red_packet_size = red_packet_size
            if red_packet_offset:
                self.dual_red_packet_offset = red_packet_offset
            if non_editable_offset:
                self.dual_non_editable_offset = non_editable_offset
            if non_editable_size:
                self.dual_non_editable_size = non_editable_size
            if self.dual_cover_size and self.original_image is not None:
                width, height = self.dual_cover_size
                base_x = self.rect[0] if self.rect else 0
                base_y = self.rect[1] if self.rect else 0
                clamped_x, clamped_y = self._clamp_dual_position(base_x, base_y)
                self.rect = (clamped_x, clamped_y, width, height)
        else:
            self.dual_cover_size = None
            self.dual_pendant_size = None
            self.dual_cover_offset = (0, 0)
            self.dual_non_editable_offset = (0, 0)
            self.dual_non_editable_size = None
            self.dual_red_packet_offset = (0, 0)
            self.dual_red_packet_size = None
        self.update()
    
    def set_inverse_overlay_active(self, active: bool):
        """设置是否显示挂件遮罩高亮"""
        self.inverse_overlay_active = bool(active)
        self.update()
        
    def set_image(self, image_array):
        """设置显示的图像"""
        if image_array is None:
            return
        
        # 确保图像数组是正确的格式
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # 确保是RGB格式（3通道）
        if len(image_array.shape) == 2:
            # 灰度图，转换为RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA格式，转换为RGB
            from PIL import Image
            pil_img = Image.fromarray(image_array, 'RGBA')
            rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
            rgb_img.paste(pil_img, mask=pil_img.split()[3])
            image_array = np.array(rgb_img)
        elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
            # 其他格式，只取前3个通道
            image_array = image_array[:, :, :3]
        
        # 确保是连续的数组（某些操作可能产生非连续数组）
        image_array = np.ascontiguousarray(image_array)
        
        self.original_image = image_array
        h, w = image_array.shape[:2]
        
        # 计算缩放比例以适应显示区域
        label_size = self.size()
        scale_w = (label_size.width() - 20) / w
        scale_h = (label_size.height() - 20) / h
        self.scale_factor = max(0.01, min(scale_w, scale_h))
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        # 缩放图像用于显示
        from PIL import Image
        try:
            # 确保图像数组格式正确
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            image_array = np.ascontiguousarray(image_array)
            
            # 确保是RGB格式（3通道）
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
                image_array = image_array[:, :, :3]
            
            img = Image.fromarray(image_array, 'RGB')
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 转换为QPixmap - 使用最可靠的方法：通过字节流
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            qimage = QImage()
            if not qimage.loadFromData(buffer.getvalue()):
                # 如果PNG加载失败，尝试使用原始RGB数据
                img_array_resized = np.array(img, dtype=np.uint8)
                img_array_resized = np.ascontiguousarray(img_array_resized)
                height, width = img_array_resized.shape[:2]
                # 使用tobytes确保数据是连续的
                img_bytes = img_array_resized.tobytes()
                qimage = QImage(img_bytes, width, height, QImage.Format_RGB888)
            
            if not qimage.isNull():
                self.scaled_image = QPixmap.fromImage(qimage)
            else:
                # 如果QImage创建失败，尝试使用临时文件方法
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    img.save(tmp_file.name, 'PNG')
                    self.scaled_image = QPixmap(tmp_file.name)
                    os.unlink(tmp_file.name)
                self.display_override = None
                self.display_pixmap = None
            self.update()
            if self.dual_frame_mode and self.dual_cover_size:
                base_x, base_y = self._clamp_dual_position(0, 0)
                self.rect = (base_x, base_y, self.dual_cover_size[0], self.dual_cover_size[1])
        except Exception as e:
            # 如果出错，使用最安全的方法：保存到临时文件
            import traceback
            import tempfile
            traceback.print_exc()
            try:
                # 备用方法：通过临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    img.save(tmp_file.name, 'PNG')
                    self.scaled_image = QPixmap(tmp_file.name)
                    import os
                    os.unlink(tmp_file.name)
                self.update()
            except Exception as e2:
                print(f"Failed to load image: {e2}")
                # 如果还是失败，显示错误
                self.scaled_image = None
                self.update()
    
    def set_watermarks(self, watermark_items=None, active_index=-1):
        """批量设置要显示的素材"""
        self.watermarks = []
        if watermark_items:
            for item in watermark_items:
                image = item.get("image")
                if image is None:
                    continue
                self.watermarks.append({
                    "image": image,
                    "x": int(item.get("x", 0)),
                    "y": int(item.get("y", 0)),
                    "scale": max(0.01, min(5.0, float(item.get("scale", 1.0)))),
                    "name": item.get("name", ""),
                    "angle": float(item.get("angle", 0.0))
                })
        if self.watermarks:
            self.active_watermark_index = active_index if 0 <= active_index < len(self.watermarks) else 0
        else:
            self.active_watermark_index = -1
        self.update()
    
    def clear_watermarks(self):
        """清除所有素材"""
        self.watermarks = []
        self.active_watermark_index = -1
        self.update()
    
    def _is_point_in_rect(self, point, rect_display):
        """判断点是否在矩形内"""
        x, y, w, h = rect_display
        return x <= point.x() <= x + w and y <= point.y() <= y + h
    
    def _get_watermark_base_geometry(self, index):
        """获取素材未旋转时的显示矩形，用于后续计算"""
        if (index < 0 or index >= len(self.watermarks) or 
                self.scaled_image is None or self.original_image is None or self.scale_factor <= 0):
            return None
        wm = self.watermarks[index]
        image = wm.get("image")
        if image is None:
            return None
        label_size = self.size()
        pixmap_size = self.scaled_image.size()
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2
        display_x = int(wm.get("x", 0) * self.scale_factor) + offset_x
        display_y = int(wm.get("y", 0) * self.scale_factor) + offset_y
        scale = max(0.01, float(wm.get("scale", 1.0)))
        display_w = max(1, int(image.shape[1] * self.scale_factor * scale))
        display_h = max(1, int(image.shape[0] * self.scale_factor * scale))
        return (display_x, display_y, display_w, display_h)

    def _get_watermark_display_rect(self, index):
        """获取考虑旋转后的素材矩形（用于命中测试与描边）"""
        base = self._get_watermark_base_geometry(index)
        if base is None:
            return None
        display_x, display_y, display_w, display_h = base
        angle = float(self.watermarks[index].get("angle", 0.0))
        if abs(angle) < 1e-3:
            return (display_x, display_y, display_w, display_h)
        angle_rad = math.radians(angle)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        rotated_w = display_w * cos_a + display_h * sin_a
        rotated_h = display_w * sin_a + display_h * cos_a
        center_x = display_x + display_w / 2
        center_y = display_y + display_h / 2
        rect_x = int(center_x - rotated_w / 2)
        rect_y = int(center_y - rotated_h / 2)
        return (rect_x, rect_y, int(rotated_w), int(rotated_h))

    def _watermark_at_point(self, point):
        """返回在指定点上的素材索引"""
        for idx in reversed(range(len(self.watermarks))):
            rect = self._get_watermark_display_rect(idx)
            if rect and self._is_point_in_rect(point, rect):
                return idx
        return None
    
    def _get_watermark_handle_rect(self, index):
        """获取素材右下角缩放手柄的显示区域"""
        rect = self._get_watermark_display_rect(index)
        if rect is None:
            return None
        handle_size = max(8, int(self.handle_size * max(1.0, self.scale_factor)))
        x = rect[0] + rect[2] - handle_size
        y = rect[1] + rect[3] - handle_size
        return QRect(x, y, handle_size, handle_size)

    def _clamp_dual_position(self, x: int, y: int):
        """约束双框模式下的位置，确保外挂矩形保持在图像范围内"""
        if self.original_image is None:
            return max(0, x), max(0, y)
        image_h, image_w = self.original_image.shape[:2]
        pend_w = self.dual_pendant_size[0] if (self.dual_pendant_size and self.dual_pendant_size[0]) else \
            (self.dual_cover_size[0] if self.dual_cover_size else 0)
        pend_h = self.dual_pendant_size[1] if (self.dual_pendant_size and self.dual_pendant_size[1]) else \
            (self.dual_cover_size[1] if self.dual_cover_size else 0)
        offset_x = self.dual_cover_offset[0] if self.dual_cover_offset else 0
        offset_y = self.dual_cover_offset[1] if self.dual_cover_offset else 0
        pendant_x = int(round(x - offset_x))
        pendant_y = int(round(y - offset_y))
        max_pendant_x = max(0, image_w - pend_w)
        max_pendant_y = max(0, image_h - pend_h)
        pendant_x = max(0, min(pendant_x, max_pendant_x))
        pendant_y = max(0, min(pendant_y, max_pendant_y))
        clamped_x = pendant_x + offset_x
        clamped_y = pendant_y + offset_y
        return clamped_x, clamped_y

    def _get_dual_rects(self):
        """返回封面、红包框、外挂与不可编辑矩形（图像坐标）"""
        if not self.dual_frame_mode or self.rect is None or self.dual_cover_size is None:
            return None, None, None, None
        cover_w, cover_h = self.dual_cover_size
        red_w, red_h = self.dual_red_packet_size if self.dual_red_packet_size else (cover_w, cover_h)
        pend_w, pend_h = self.dual_pendant_size if self.dual_pendant_size else self.dual_cover_size
        base_x, base_y = self.rect[0], self.rect[1]
        offset_x = self.dual_cover_offset[0] if self.dual_cover_offset else 0
        offset_y = self.dual_cover_offset[1] if self.dual_cover_offset else 0
        pendant_x = base_x - offset_x
        pendant_y = base_y - offset_y
        if self.original_image is not None:
            image_h, image_w = self.original_image.shape[:2]
            max_pendant_x = max(0, image_w - pend_w)
            max_pendant_y = max(0, image_h - pend_h)
            pendant_x = max(0, min(int(round(pendant_x)), max_pendant_x))
            pendant_y = max(0, min(int(round(pendant_y)), max_pendant_y))
        else:
            pendant_x = int(round(pendant_x))
            pendant_y = int(round(pendant_y))
        red_offset = self.dual_red_packet_offset if self.dual_red_packet_offset else (0, 0)
        red_rect = (
            pendant_x + red_offset[0],
            pendant_y + red_offset[1],
            red_w,
            red_h
        )
        non_edit_offset = self.dual_non_editable_offset if self.dual_non_editable_offset else (0, 0)
        non_edit_size = self.dual_non_editable_size if self.dual_non_editable_size else (0, 0)
        non_edit_rect = (
            pendant_x + non_edit_offset[0],
            pendant_y + non_edit_offset[1],
            non_edit_size[0],
            non_edit_size[1]
        )
        return (
            (int(round(base_x)), int(round(base_y)), cover_w, cover_h),
            red_rect,
            (pendant_x, pendant_y, pend_w, pend_h),
            non_edit_rect
        )

    def _image_rect_to_display(self, rect, offset_x, offset_y):
        """将图像坐标矩形转换为显示坐标矩形"""
        if rect is None or self.scale_factor <= 0:
            return None
        display_x = int(round(rect[0] * self.scale_factor)) + offset_x
        display_y = int(round(rect[1] * self.scale_factor)) + offset_y
        display_w = int(round(rect[2] * self.scale_factor))
        display_h = int(round(rect[3] * self.scale_factor))
        return QRect(display_x, display_y, display_w, display_h)

    def _paint_dual_rectangles(self, painter: QPainter, offset_x: int, offset_y: int):
        """绘制封面/外挂双矩形"""
        cover_rect, red_rect, pendant_rect, non_edit_rect = self._get_dual_rects()
        if cover_rect is None or pendant_rect is None or red_rect is None:
            return
        cover_display = self._image_rect_to_display(cover_rect, offset_x, offset_y)
        red_display = self._image_rect_to_display(red_rect, offset_x, offset_y)
        pendant_display = self._image_rect_to_display(pendant_rect, offset_x, offset_y)
        if cover_display is None or pendant_display is None or red_display is None:
            return
        cover_pen = QPen(QColor(255, 214, 102), max(2, int(round(2 * self.scale_factor))))
        cover_pen.setStyle(Qt.SolidLine)
        painter.setPen(cover_pen)
        painter.setBrush(QColor(255, 214, 102, 40) if not self.inverse_overlay_active else Qt.NoBrush)
        painter.drawRect(cover_display)
        red_pen = QPen(QColor(250, 151, 112), max(2, int(round(2 * self.scale_factor))))
        red_pen.setStyle(Qt.SolidLine)
        painter.setPen(red_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(red_display)
        pendant_pen = QPen(QColor(255, 138, 76), max(2, int(round(2 * self.scale_factor))))
        pendant_pen.setStyle(Qt.DashLine)
        painter.setPen(pendant_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(pendant_display)
        if non_edit_rect:
            non_edit_display = self._image_rect_to_display(non_edit_rect, offset_x, offset_y)
            if non_edit_display:
                non_edit_pen = QPen(QColor(148, 163, 184), max(2, int(round(2 * self.scale_factor))))
                non_edit_pen.setStyle(Qt.DotLine)
                painter.setPen(non_edit_pen)
                painter.drawRect(non_edit_display)
        if self.inverse_overlay_active:
            self._paint_inverse_ring_preview(painter, pendant_display, red_display)

    def _paint_inverse_ring_preview(self, painter: QPainter, outer_rect: QRect, inner_rect: QRect):
        """绘制反选抠除区域的实时指示（仅描边，无亮度改动）"""
        if not outer_rect or not inner_rect:
            return
        painter.save()
        highlight_pen = QPen(QColor(56, 189, 248), max(2, int(round(2 * self.scale_factor))))
        highlight_pen.setStyle(Qt.DashDotLine)
        highlight_pen.setCapStyle(Qt.RoundCap)
        highlight_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(highlight_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(outer_rect)
        painter.drawRect(inner_rect)

        corner_pen = QPen(QColor(56, 189, 248, 220), max(1, int(round(self.scale_factor))))
        corner_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(corner_pen)
        marker = max(10, int(round(12 * self.scale_factor)))

        def _draw_corners(rect: QRect):
            points = [
                rect.topLeft(),
                rect.topRight(),
                rect.bottomLeft(),
                rect.bottomRight(),
            ]
            for point in points:
                painter.drawLine(point, QPoint(point.x() + marker, point.y()))
                painter.drawLine(point, QPoint(point.x(), point.y() + marker))

        _draw_corners(outer_rect)
        _draw_corners(inner_rect)
        painter.restore()

    def _hit_watermark_handle(self, point):
        """判断是否点击在当前选中素材的缩放手柄上"""
        idx = getattr(self, "active_watermark_index", -1)
        if idx is None or idx < 0 or idx >= len(self.watermarks):
            return None
        handle_rect = self._get_watermark_handle_rect(idx)
        if handle_rect and handle_rect.contains(point):
            return idx
        return None

    def _begin_watermark_resize(self, index, pos):
        if index < 0 or index >= len(self.watermarks):
            return
        wm = self.watermarks[index]
        self.is_resizing_watermark = True
        self.resizing_watermark_index = index
        self.watermark_resize_start_pos = pos
        self.watermark_initial_scale = float(wm.get("scale", 1.0))
        self.watermark_initial_width = wm["image"].shape[1] * self.watermark_initial_scale
        self.watermark_initial_height = wm["image"].shape[0] * self.watermark_initial_scale
        self.setCursor(Qt.SizeFDiagCursor)

    def _handle_watermark_resize(self, pos):
        if self.resizing_watermark_index is None or self.resizing_watermark_index >= len(self.watermarks):
            return
        wm = self.watermarks[self.resizing_watermark_index]
        denom = max(self.scale_factor, 1e-6)
        delta_x = (pos.x() - self.watermark_resize_start_pos.x()) / denom
        delta_y = (pos.y() - self.watermark_resize_start_pos.y()) / denom
        primary_delta = delta_x if abs(delta_x) >= abs(delta_y) else delta_y
        new_width = max(1.0, self.watermark_initial_width + primary_delta)
        orig_w = wm["image"].shape[1]
        new_scale = max(0.01, min(5.0, new_width / max(1.0, float(orig_w))))
        if abs(new_scale - wm.get("scale", 1.0)) > 1e-4:
            wm["scale"] = new_scale
            self.watermark_scale_changed.emit(self.resizing_watermark_index, new_scale)
            self.update()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.scaled_image is None:
            return
        
        if event.button() == Qt.LeftButton:
            if self.color_pick_mode:
                color = self._get_color_at_point(event.pos())
                if color is not None:
                    self.color_pick_mode = False
                    self.setCursor(Qt.ArrowCursor)
                    self.color_picked.emit(*color)
                return
            
            handle_index = self._hit_watermark_handle(event.pos())
            if handle_index is not None:
                self._begin_watermark_resize(handle_index, event.pos())
                return
            
            # 优先检查是否点击在素材上
            hit_index = self._watermark_at_point(event.pos())
            if hit_index is not None:
                # 在素材上，开始拖动素材
                self.is_dragging_watermark = True
                self.dragging_watermark_index = hit_index
                self.watermark_drag_start_pos = event.pos()
                self.active_watermark_index = hit_index
                self.watermark_selected.emit(hit_index)
                self.setCursor(Qt.ClosedHandCursor)  # 设置拖动光标
                return
            
            if self.rect_locked:
                return
            if self.dual_frame_mode:
                if self.rect and not self.is_drawing:
                    label_size = self.size()
                    pixmap_size = self.scaled_image.size()
                    offset_x = (label_size.width() - pixmap_size.width()) // 2
                    offset_y = (label_size.height() - pixmap_size.height()) // 2
                    display_x = int(self.rect[0] * self.scale_factor) + offset_x
                    display_y = int(self.rect[1] * self.scale_factor) + offset_y
                    display_w = int(self.rect[2] * self.scale_factor)
                    display_h = int(self.rect[3] * self.scale_factor)
                    rect_display = (display_x, display_y, display_w, display_h)
                    if self._is_point_in_rect(event.pos(), rect_display):
                        self.is_dragging_rect = True
                        self.drag_start_pos = event.pos()
                return
            
            # 检查是否点击在现有矩形内
            if self.rect and not self.is_drawing:
                label_size = self.size()
                pixmap_size = self.scaled_image.size()
                offset_x = (label_size.width() - pixmap_size.width()) // 2
                offset_y = (label_size.height() - pixmap_size.height()) // 2
                
                display_x = int(self.rect[0] * self.scale_factor) + offset_x
                display_y = int(self.rect[1] * self.scale_factor) + offset_y
                display_w = int(self.rect[2] * self.scale_factor)
                display_h = int(self.rect[3] * self.scale_factor)
                
                rect_display = (display_x, display_y, display_w, display_h)
                
                if self._is_point_in_rect(event.pos(), rect_display):
                    # 在矩形内，开始拖动
                    self.is_dragging_rect = True
                    self.drag_start_pos = event.pos()
                    return
            
            # 不在矩形内，开始绘制新矩形
            self.is_drawing = True
            self.start_pos = event.pos()
            self.end_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.color_pick_mode:
            self.setCursor(Qt.CrossCursor)
            return
        if self.is_resizing_watermark and self.resizing_watermark_index is not None:
            self._handle_watermark_resize(event.pos())
            return
        # 检查鼠标是否在素材或手柄上（用于改变光标）
        if not self.is_dragging_watermark and not self.is_dragging_rect and not self.is_drawing:
            if self._hit_watermark_handle(event.pos()) is not None:
                self.setCursor(Qt.SizeFDiagCursor)
            elif self._watermark_at_point(event.pos()) is not None:
                self.setCursor(Qt.OpenHandCursor)  # 悬停在素材上时显示手型光标
            else:
                self.setCursor(Qt.ArrowCursor)  # 其他情况恢复默认光标
        
        if (self.is_dragging_watermark and self.watermark_drag_start_pos is not None 
                and self.dragging_watermark_index is not None 
                and 0 <= self.dragging_watermark_index < len(self.watermarks)):
            # 拖动素材
            label_size = self.size()
            pixmap_size = self.scaled_image.size()
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # 计算移动距离（原始图像坐标）
            dx = (event.pos().x() - self.watermark_drag_start_pos.x()) / self.scale_factor
            dy = (event.pos().y() - self.watermark_drag_start_pos.y()) / self.scale_factor
            
            wm = self.watermarks[self.dragging_watermark_index]
            # 允许素材移动到画布四周外面，不限制在图像范围内
            new_wm_x = int(wm.get("x", 0) + dx)
            new_wm_y = int(wm.get("y", 0) + dy)
            wm["x"] = new_wm_x
            wm["y"] = new_wm_y
            self.watermark_drag_start_pos = event.pos()
            self.watermark_position_changed.emit(self.dragging_watermark_index, new_wm_x, new_wm_y)
            self.update()
        elif self.is_dragging_rect and self.rect and self.drag_start_pos and not self.rect_locked:
            # 拖动矩形
            label_size = self.size()
            pixmap_size = self.scaled_image.size()
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # 计算移动距离（原始图像坐标）
            dx = (event.pos().x() - self.drag_start_pos.x()) / self.scale_factor
            dy = (event.pos().y() - self.drag_start_pos.y()) / self.scale_factor
            
            if self.dual_frame_mode and self.dual_cover_size:
                new_x, new_y = self._clamp_dual_position(self.rect[0] + dx, self.rect[1] + dy)
                rect_w, rect_h = self.dual_cover_size
            else:
                max_x = int(self.original_image.shape[1] - self.rect[2]) if self.original_image is not None else int(self.rect[0] + dx)
                max_y = int(self.original_image.shape[0] - self.rect[3]) if self.original_image is not None else int(self.rect[1] + dy)
                new_x = max(0, min(int(self.rect[0] + dx), max_x))
                new_y = max(0, min(int(self.rect[1] + dy), max_y))
                rect_w, rect_h = self.rect[2], self.rect[3]
            
            self.rect = (new_x, new_y, rect_w, rect_h)
            self.drag_start_pos = event.pos()
            self.selection_changed.emit(*self.rect)
            self.update()
        elif self.is_drawing and self.start_pos and not self.rect_locked:
            self.end_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.is_dragging_watermark:
            self.is_dragging_watermark = False
            self.watermark_drag_start_pos = None
            self.dragging_watermark_index = None
            self.setCursor(Qt.ArrowCursor)  # 恢复默认光标
        elif self.is_resizing_watermark:
            self.is_resizing_watermark = False
            self.resizing_watermark_index = None
            self.watermark_resize_start_pos = None
            self.setCursor(Qt.ArrowCursor)
        elif self.is_dragging_rect:
            self.is_dragging_rect = False
            self.drag_start_pos = None
        elif self.is_drawing and self.start_pos and self.end_pos:
            self.is_drawing = False
            # 计算矩形区域（在原始图像坐标系中）
            self._update_rect()
            if self.rect:
                self.selection_changed.emit(*self.rect)
    
    def _update_rect(self):
        """更新矩形区域（转换为原始图像坐标）"""
        if not self.start_pos or not self.end_pos or self.scaled_image is None:
            return
        
        # 获取图像在标签中的位置（居中显示）
        label_size = self.size()
        pixmap_size = self.scaled_image.size()
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2
        
        # 转换为相对于图像的坐标
        x1 = max(0, min(self.start_pos.x(), self.end_pos.x()) - offset_x)
        y1 = max(0, min(self.start_pos.y(), self.end_pos.y()) - offset_y)
        x2 = min(pixmap_size.width(), max(self.start_pos.x(), self.end_pos.x()) - offset_x)
        y2 = min(pixmap_size.height(), max(self.start_pos.y(), self.end_pos.y()) - offset_y)
        
        # 转换为原始图像坐标
        if self.scale_factor > 0:
            orig_x = int(x1 / self.scale_factor)
            orig_y = int(y1 / self.scale_factor)
            orig_w = int((x2 - x1) / self.scale_factor)
            orig_h = int((y2 - y1) / self.scale_factor)
            
            if orig_w > 0 and orig_h > 0:
                if self.dual_frame_mode and self.dual_cover_size:
                    orig_w, orig_h = self.dual_cover_size
                    orig_x, orig_y = self._clamp_dual_position(orig_x, orig_y)
                self.rect = (orig_x, orig_y, orig_w, orig_h)
    
    def paintEvent(self, event):
        """绘制事件"""
        super().paintEvent(event)
        
        pixmap_to_draw = self.display_pixmap if self.display_pixmap is not None else self.scaled_image
        if pixmap_to_draw is None or pixmap_to_draw.isNull():
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制图像（居中）
        label_size = self.size()
        pixmap_size = pixmap_to_draw.size()
        if pixmap_size.width() <= 0 or pixmap_size.height() <= 0:
            return
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2

        if (
            self.display_override_has_transparency
            and self.display_pixmap is not None
            and not self.display_pixmap.isNull()
        ):
            tile = self._ensure_checker_pattern(max(6, int(round(12 * self.scale_factor))))
            if tile:
                painter.drawTiledPixmap(
                    QRect(offset_x, offset_y, pixmap_size.width(), pixmap_size.height()),
                    tile
                )
        
        painter.drawPixmap(offset_x, offset_y, pixmap_to_draw)
        
        # 绘制辅助框
        if self.show_guides and self.should_show_guides:
            self._draw_guides(painter, offset_x, offset_y)
        
        # 绘制附件图片（如果启用）
        if self.show_overlay and self.overlay_image and not self.overlay_image.isNull():
            # 获取原始图像尺寸
            if self.original_image is not None:
                orig_h, orig_w = self.original_image.shape[:2]
                if self.dual_frame_mode and self.rect is not None:
                    base_x = self.rect[0]
                    base_y = self.rect[1]
                    pend_h = self.dual_pendant_size[1] if self.dual_pendant_size else orig_h
                    overlay_x_orig = max(0, base_x)
                    overlay_y_orig = base_y + max(0, pend_h - self.overlay_height - 48 - 96)
                else:
                    # 附件图片位置：距离底部48px，左右各48px；整体左移48px，并上移96px
                    overlay_x_orig = 0
                    overlay_y_orig = max(0, orig_h - self.overlay_height - 48 - 96)

                offset_dx, offset_dy = self.overlay_offset_adjustment
                overlay_x_orig = max(0, overlay_x_orig + offset_dx)
                overlay_y_orig = max(0, overlay_y_orig + offset_dy)
                
                # 转换为显示坐标
                overlay_x_display = int(overlay_x_orig * self.scale_factor) + offset_x
                overlay_y_display = int(overlay_y_orig * self.scale_factor) + offset_y
                
                # 按照预览图像的缩放比例缩放附件图片
                overlay_display_w = int(self.overlay_width * self.scale_factor)
                overlay_display_h = int(self.overlay_height * self.scale_factor)
                
                # 缩放并绘制附件图片
                scaled_overlay = self.overlay_image.scaled(
                    overlay_display_w, 
                    overlay_display_h, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                painter.drawPixmap(overlay_x_display, overlay_y_display, scaled_overlay)
        
        # 绘制素材（如果存在）
        if self.watermarks:
            from PIL import Image
            for idx, wm in enumerate(self.watermarks):
                image = wm.get("image")
                if image is None:
                    continue
                base_geometry = self._get_watermark_base_geometry(idx)
                if base_geometry is None:
                    continue
                base_x, base_y, base_w, base_h = base_geometry
                if len(image.shape) == 3 and image.shape[2] == 4:
                    wm_img = Image.fromarray(image, 'RGBA')
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    wm_img = Image.fromarray(image, 'RGB')
                else:
                    wm_img = Image.fromarray(image)
                wm_img = wm_img.resize((base_w, base_h), Image.Resampling.LANCZOS)
                if wm_img.mode == 'RGBA':
                    wm_qimage = QImage(wm_img.tobytes(), base_w, base_h, QImage.Format_RGBA8888)
                else:
                    wm_qimage = QImage(wm_img.tobytes(), base_w, base_h, QImage.Format_RGB888)
                wm_pixmap = QPixmap.fromImage(wm_qimage)
                angle = float(wm.get("angle", 0.0))
                if abs(angle) >= 1e-3:
                    transform = QTransform()
                    transform.translate(base_w / 2, base_h / 2)
                    transform.rotate(angle)
                    transform.translate(-base_w / 2, -base_h / 2)
                    wm_pixmap = wm_pixmap.transformed(transform, Qt.SmoothTransformation)
                center_x = base_x + base_w / 2
                center_y = base_y + base_h / 2
                draw_x = int(center_x - wm_pixmap.width() / 2)
                draw_y = int(center_y - wm_pixmap.height() / 2)
                painter.drawPixmap(draw_x, draw_y, wm_pixmap)
                rect = self._get_watermark_display_rect(idx)
                if rect and (idx == self.active_watermark_index or (self.is_dragging_watermark and idx == self.dragging_watermark_index)):
                    pen = QPen(QColor(0, 150, 255), 2, Qt.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect[0] - 2, rect[1] - 2, rect[2] + 4, rect[3] + 4)
                    handle_rect = self._get_watermark_handle_rect(idx)
                    if handle_rect:
                        painter.fillRect(handle_rect, QColor(22, 119, 255))
        
        # 绘制矩形选择框
        if self.dual_frame_mode and self.rect:
            self._paint_dual_rectangles(painter, offset_x, offset_y)
        else:
            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 0, 0, 30))
            if self.is_drawing and self.start_pos and self.end_pos:
                x1 = min(self.start_pos.x(), self.end_pos.x())
                y1 = min(self.start_pos.y(), self.end_pos.y())
                x2 = max(self.start_pos.x(), self.end_pos.x())
                y2 = max(self.start_pos.y(), self.end_pos.y())
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            elif self.rect and not self.is_drawing:
                display_x = int(self.rect[0] * self.scale_factor) + offset_x
                display_y = int(self.rect[1] * self.scale_factor) + offset_y
                display_w = int(self.rect[2] * self.scale_factor)
                display_h = int(self.rect[3] * self.scale_factor)
                painter.drawRect(display_x, display_y, display_w, display_h)
        if self.inverse_overlay_active and self.dual_frame_mode:
            cover_rect, red_rect, pendant_rect, non_edit_rect = self._get_dual_rects()
            if non_edit_rect:
                shade_rect = self._image_rect_to_display(non_edit_rect, offset_x, offset_y)
                if shade_rect:
                    painter.fillRect(shade_rect, QColor(115, 120, 130, 210))
    
    def set_rect_from_params(self, x, y, width, height):
        """根据参数设置矩形区域"""
        if self.original_image is None:
            return
        if self.dual_frame_mode and self.dual_cover_size:
            width, height = self.dual_cover_size
            x, y = self._clamp_dual_position(x, y)
        else:
            img_h, img_w = self.original_image.shape[:2]
            max_x = max(0, img_w - max(1, width))
            max_y = max(0, img_h - max(1, height))
            x = max(0, min(int(x), max_x))
            y = max(0, min(int(y), max_y))
        
        # 直接设置rect（原始图像坐标）
        self.rect = (x, y, width, height)
        self.update()
    
    
    def set_rect_lock(self, locked: bool):
        """设置矩形是否锁定"""
        self.rect_locked = locked
        if locked:
            self.is_drawing = False
            self.is_dragging_rect = False
        self.update()
    
    def enable_color_pick(self, enabled: bool):
        """启用或禁用颜色拾取模式"""
        self.color_pick_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()
    
    def set_display_override(self, image_array):
        """设置用于显示覆盖的图像，不影响原始数据用于拾色等逻辑"""
        if image_array is None:
            self.display_override = None
            self.display_pixmap = None
            self.display_override_has_transparency = False
            self.update()
            return

        if self.original_image is None or self.scale_factor <= 0:
            return

        array = image_array
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        array = np.ascontiguousarray(array)

        # 根据通道数确定模式
        mode = None
        has_alpha = False
        if len(array.shape) == 2:
            mode = 'L'
        elif len(array.shape) == 3:
            channels = array.shape[2]
            if channels == 1:
                array = array[:, :, 0]
                mode = 'L'
            elif channels == 3:
                mode = 'RGB'
            elif channels == 4:
                mode = 'RGBA'
                has_alpha = True
            else:
                array = array[:, :, :3]
                mode = 'RGB'
        else:
            return

        from PIL import Image
        try:
            img = Image.fromarray(array, mode)
            new_w = max(1, int(array.shape[1] * self.scale_factor))
            new_h = max(1, int(array.shape[0] * self.scale_factor))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            qimage = QImage()
            if qimage.loadFromData(buffer.getvalue()):
                self.display_override = array
                self.display_pixmap = QPixmap.fromImage(qimage)
                if has_alpha and array.shape[2] == 4:
                    self.display_override_has_transparency = bool(np.any(array[:, :, 3] < 255))
                else:
                    self.display_override_has_transparency = False
            else:
                self.display_override = None
                self.display_pixmap = None
                self.display_override_has_transparency = False
        except Exception:
            self.display_override = None
            self.display_pixmap = None
            self.display_override_has_transparency = False
        self.update()

    def _ensure_checker_pattern(self, tile_size: int):
        """确保棋盘格纹理已生成"""
        tile_size = max(2, int(tile_size))
        if self._checker_pixmap is not None and self._checker_tile_size == tile_size:
            return self._checker_pixmap
        pix = QPixmap(tile_size * 2, tile_size * 2)
        pix.fill(QColor(248, 250, 252))
        painter = QPainter(pix)
        color_dark = QColor(226, 232, 240)
        painter.fillRect(0, 0, tile_size, tile_size, color_dark)
        painter.fillRect(tile_size, tile_size, tile_size, tile_size, color_dark)
        painter.end()
        self._checker_pixmap = pix
        self._checker_tile_size = tile_size
        return self._checker_pixmap
    
    def _display_to_image_coords(self, point):
        """将显示坐标转换为原始图像坐标"""
        if self.original_image is None or self.scaled_image is None:
            return None
        pixmap_size = self.scaled_image.size()
        if pixmap_size.width() <= 0 or pixmap_size.height() <= 0 or self.scale_factor == 0:
            return None
        label_size = self.size()
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2
        x = point.x() - offset_x
        y = point.y() - offset_y
        if x < 0 or y < 0 or x >= pixmap_size.width() or y >= pixmap_size.height():
            return None
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        orig_x = max(0, min(orig_x, self.original_image.shape[1] - 1))
        orig_y = max(0, min(orig_y, self.original_image.shape[0] - 1))
        return orig_x, orig_y
    
    def _get_color_at_point(self, point):
        """获取指定显示点的原始图像颜色"""
        coords = self._display_to_image_coords(point)
        if coords is None:
            return None
        x, y = coords
        if y < 0 or y >= self.original_image.shape[0] or x < 0 or x >= self.original_image.shape[1]:
            return None
        color = self.original_image[y, x]
        return tuple(int(c) for c in color[:3])

    def _draw_guides(self, painter, offset_x, offset_y):
        """绘制封面图/外挂辅助框"""
        if self.dual_frame_mode:
            return
        if self.original_image is None or self.scaled_image is None or self.scale_factor <= 0:
            return

        h, w = self.original_image.shape[:2]
        # 仅在标准画布尺寸下显示辅助框
        if w != 1053 or h != 1746:
            return

        guides = [
            {
                "color": QColor(64, 128, 255),
                "left": 0,
                "top": 0,
                "width": 1053,
                "height": 1746,
                "caption": ""
            },
            {
                "color": QColor(76, 175, 80),
                "left": 48,
                "top": 114,
                "width": 957,
                "height": 1278,
                "caption": ""
            },
            {
                "color": QColor(250, 151, 112),
                "left": 48,
                "top": 114,
                "width": 957,
                "height": 1584,
                "caption": ""
            },
            {
                "color": QColor(148, 163, 184),
                "left": 96,
                "top": 342,
                "width": 861,
                "height": 1308,
                "caption": ""
            }
        ]

        pixmap_size = self.scaled_image.size()
        max_x = offset_x + pixmap_size.width()
        max_y = offset_y + pixmap_size.height()

        for guide in guides:
            rect_x = offset_x + int(round(guide["left"] * self.scale_factor))
            rect_y = offset_y + int(round(guide["top"] * self.scale_factor))
            rect_w = int(round(guide["width"] * self.scale_factor))
            rect_h = int(round(guide["height"] * self.scale_factor))
            rect = QRect(rect_x, rect_y, rect_w, rect_h)

            pen = QPen(guide["color"], max(2, int(round(2 * self.scale_factor))))
            pen.setStyle(Qt.SolidLine)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            # 只在有文字时才绘制文字标注
            if guide["caption"]:
                pad = max(4, int(round(6 * self.scale_factor)))
                font = painter.font()
                font.setPointSize(max(8, int(round(11 * max(self.scale_factor, 0.8)))))
                painter.setFont(font)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(guide["caption"])
                text_h = metrics.height()

                pos_x = rect.x() + rect.width() + pad
                pos_y = rect.y() + pad

                if pos_x + text_w + pad * 2 > max_x:
                    pos_x = max(offset_x, rect.x() - text_w - pad * 3)
                    pos_y = rect.y() + rect.height() + pad
                if pos_y + text_h + pad * 2 > max_y:
                    pos_y = rect.y() - (text_h + pad * 2 + pad)
                if pos_y < offset_y:
                    pos_y = offset_y + pad

                bg_rect = QRect(pos_x, pos_y, text_w + pad * 2, text_h + pad * 2)
                bg_color = QColor(guide["color"])
                bg_color.setAlpha(40)
                painter.setBrush(bg_color)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(bg_rect, pad, pad)
                painter.setPen(guide["color"])
                painter.drawText(bg_rect, Qt.AlignCenter, guide["caption"])
                painter.setBrush(Qt.NoBrush)

    def set_guides_visible(self, visible: bool):
        """设置辅助框显示"""
        self.should_show_guides = visible
        self.update()


class BubbleCanvas(QWidget):
    """气泡挂件画布"""
    
    overlay_position_changed = pyqtSignal(int, int, int)  # index, x, y
    overlay_selected = pyqtSignal(int)
    overlay_scale_changed = pyqtSignal(int, float)
    
    CANVAS_WIDTH = 720
    CANVAS_HEIGHT = 384
    EDITABLE_X = 240
    EDITABLE_WIDTH = 480
    TOP_MARGIN = 60
    TOP_SECTION_HEIGHT = 96
    MIDDLE_SECTION_HEIGHT = 216
    BOTTOM_SECTION_HEIGHT = 72
    
    def __init__(self, parent=None, interactive=True):
        super().__init__(parent)
        self.setFixedSize(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        self.setMouseTracking(True)
        self.base_pixmap = None
        self.overlays = []
        self.active_index = -1
        self.dragging_index = None
        self.drag_start_pos = None
        self.interactive = interactive
        self.setCursor(Qt.ArrowCursor)
        self.handle_size = 12
        self.is_resizing_overlay = False
        self.resizing_overlay_index = None
        self.overlay_resize_start_pos = None
        self.overlay_initial_width = 0.0
        self.overlay_initial_scale = 1.0
    
    def set_base_pixmap(self, pixmap: QPixmap):
        self.base_pixmap = pixmap
        self.update()
    
    def set_overlays(self, overlays):
        self.overlays = overlays if overlays is not None else []
        self.update()
    
    def set_active_index(self, index: int):
        self.active_index = index
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        
        # 绘制气泡底图
        if self.base_pixmap and not self.base_pixmap.isNull():
            target_height = self.CANVAS_HEIGHT - self.TOP_MARGIN * 2
            scaled_base = self.base_pixmap.scaled(
                self.CANVAS_WIDTH, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            base_x = (self.CANVAS_WIDTH - scaled_base.width()) // 2
            base_y = self.TOP_MARGIN
            painter.drawPixmap(base_x, base_y, scaled_base)
        else:
            painter.setPen(QPen(QColor("#1D4ED8"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 12, 12)
        
        # 绘制可编辑区域
        edit_rect = QRect(self.EDITABLE_X, 0, self.EDITABLE_WIDTH, self.CANVAS_HEIGHT)
        painter.setBrush(QColor(56, 189, 248, 25))
        painter.setPen(QPen(QColor("#0EA5E9"), 2))
        painter.drawRect(edit_rect)
        
        # 绘制上下分区
        top_rect = QRect(self.EDITABLE_X, 0, self.EDITABLE_WIDTH, self.TOP_SECTION_HEIGHT)
        middle_rect = QRect(self.EDITABLE_X, self.TOP_SECTION_HEIGHT, self.EDITABLE_WIDTH, self.MIDDLE_SECTION_HEIGHT)
        bottom_rect = QRect(self.EDITABLE_X, self.TOP_SECTION_HEIGHT + self.MIDDLE_SECTION_HEIGHT, self.EDITABLE_WIDTH, self.BOTTOM_SECTION_HEIGHT)
        painter.fillRect(middle_rect, QColor(248, 113, 113, 50))
        painter.setPen(QPen(QColor("#94A3B8"), 1, Qt.DashLine))
        painter.drawLine(self.EDITABLE_X, self.TOP_SECTION_HEIGHT, self.CANVAS_WIDTH, self.TOP_SECTION_HEIGHT)
        painter.drawLine(self.EDITABLE_X, self.TOP_SECTION_HEIGHT + self.MIDDLE_SECTION_HEIGHT, self.CANVAS_WIDTH, self.TOP_SECTION_HEIGHT + self.MIDDLE_SECTION_HEIGHT)
        
        # 绘制提示文字
        painter.setPen(QColor("#CBD5F5"))
        painter.drawText(top_rect.adjusted(8, 4, -8, -4), Qt.AlignLeft | Qt.AlignVCenter, "上区 480×96")
        painter.drawText(middle_rect.adjusted(8, 4, -8, -4), Qt.AlignLeft | Qt.AlignVCenter, "中区 480×216（禁止摆放素材）")
        painter.drawText(bottom_rect.adjusted(8, 4, -8, -4), Qt.AlignLeft | Qt.AlignVCenter, "下区 480×72")
        
        # 绘制素材
        for idx, overlay in enumerate(self.overlays):
            pixmap, rect = self._get_overlay_pixmap_and_rect(overlay)
            if pixmap and rect:
                painter.drawPixmap(rect, pixmap)
                border_color = QColor("#38BDF8") if idx == self.active_index else QColor("#F8FAFC")
                pen = QPen(border_color, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect.adjusted(-2, -2, 2, 2))
                if self.interactive and idx == self.active_index:
                    handle_rect = self._get_overlay_handle_rect(idx)
                    if handle_rect:
                        painter.fillRect(handle_rect, QColor("#1677FF"))
    
    def _get_overlay_pixmap_and_rect(self, overlay):
        image = overlay.get("image")
        if image is None:
            return None, None
        scale = max(0.01, float(overlay.get("scale", 1.0)))
        base_h, base_w = image.shape[:2]
        disp_w = max(1, int(base_w * scale))
        disp_h = max(1, int(base_h * scale))
        qimage = self._np_to_qimage(image)
        if qimage is None:
            return None, None
        pixmap = QPixmap.fromImage(qimage).scaled(disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        angle = float(overlay.get("angle", 0.0))
        if abs(angle) >= 1e-3:
            transform = QTransform()
            transform.translate(pixmap.width() / 2, pixmap.height() / 2)
            transform.rotate(angle)
            transform.translate(-pixmap.width() / 2, -pixmap.height() / 2)
            pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
        center_x = float(overlay.get("x", 0)) + disp_w / 2
        center_y = float(overlay.get("y", 0)) + disp_h / 2
        rect_left = int(round(center_x - pixmap.width() / 2))
        rect_top = int(round(center_y - pixmap.height() / 2))
        rect = QRect(rect_left, rect_top, pixmap.width(), pixmap.height())
        return pixmap, rect
    
    def _np_to_qimage(self, array):
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        if array.ndim != 3:
            return None
        height, width, channels = array.shape
        if channels == 3:
            fmt = QImage.Format_RGB888
        elif channels == 4:
            fmt = QImage.Format_RGBA8888
        else:
            return None
        bytes_per_line = channels * width
        qimage = QImage(array.data, width, height, bytes_per_line, fmt)
        return qimage.copy()
    
    def _overlay_hit_test(self, point):
        for idx in reversed(range(len(self.overlays))):
            _, rect = self._get_overlay_pixmap_and_rect(self.overlays[idx])
            if rect and rect.contains(point):
                return idx
        return None
    
    def _get_overlay_handle_rect(self, index):
        if not self.interactive or index < 0 or index >= len(self.overlays):
            return None
        _, rect = self._get_overlay_pixmap_and_rect(self.overlays[index])
        if rect is None:
            return None
        size = self.handle_size
        return QRect(rect.right() - size + 1, rect.bottom() - size + 1, size, size)

    def _hit_overlay_handle(self, point):
        if not self.interactive:
            return None
        if self.active_index is None or self.active_index < 0 or self.active_index >= len(self.overlays):
            return None
        handle_rect = self._get_overlay_handle_rect(self.active_index)
        if handle_rect and handle_rect.contains(point):
            return self.active_index
        return None

    def _begin_overlay_resize(self, index, pos):
        overlay = self.overlays[index]
        self.is_resizing_overlay = True
        self.resizing_overlay_index = index
        self.overlay_resize_start_pos = pos
        self.overlay_initial_scale = float(overlay.get("scale", 1.0))
        self.overlay_initial_width = overlay["image"].shape[1] * self.overlay_initial_scale
        self.setCursor(Qt.SizeFDiagCursor)

    def _handle_overlay_resize(self, pos):
        if self.resizing_overlay_index is None or self.resizing_overlay_index >= len(self.overlays):
            return
        overlay = self.overlays[self.resizing_overlay_index]
        delta_x = pos.x() - self.overlay_resize_start_pos.x()
        delta_y = pos.y() - self.overlay_resize_start_pos.y()
        primary_delta = delta_x if abs(delta_x) >= abs(delta_y) else delta_y
        new_width = max(1.0, self.overlay_initial_width + primary_delta)
        orig_w = overlay["image"].shape[1]
        new_scale = max(0.01, min(5.0, new_width / max(1.0, float(orig_w))))
        if abs(new_scale - overlay.get("scale", 1.0)) > 1e-4:
            overlay["scale"] = new_scale
            self.overlay_scale_changed.emit(self.resizing_overlay_index, new_scale)
            self.update()
    
    def mousePressEvent(self, event):
        if not self.interactive:
            return super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            handle_idx = self._hit_overlay_handle(event.pos())
            if handle_idx is not None:
                self._begin_overlay_resize(handle_idx, event.pos())
                return
            hit_index = self._overlay_hit_test(event.pos())
            if hit_index is not None:
                self.dragging_index = hit_index
                self.active_index = hit_index
                self.drag_start_pos = event.pos()
                self.overlay_selected.emit(hit_index)
                self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if not self.interactive:
            return super().mouseMoveEvent(event)
        if self.is_resizing_overlay and self.resizing_overlay_index is not None:
            self._handle_overlay_resize(event.pos())
            return
        if self.dragging_index is None:
            if self._hit_overlay_handle(event.pos()) is not None:
                self.setCursor(Qt.SizeFDiagCursor)
            elif self._overlay_hit_test(event.pos()) is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        if self.dragging_index is not None and self.drag_start_pos is not None:
            dx = event.pos().x() - self.drag_start_pos.x()
            dy = event.pos().y() - self.drag_start_pos.y()
            overlay = self.overlays[self.dragging_index]
            _, rect = self._get_overlay_pixmap_and_rect(overlay)
            if rect:
                new_x = overlay.get("x", 0) + dx
                new_y = overlay.get("y", 0) + dy
                max_x = self.CANVAS_WIDTH - rect.width()
                max_y = self.CANVAS_HEIGHT - rect.height()
                overlay["x"] = max(0, min(int(round(new_x)), max_x))
                overlay["y"] = max(0, min(int(round(new_y)), max_y))
                self.drag_start_pos = event.pos()
                self.overlay_position_changed.emit(self.dragging_index, overlay["x"], overlay["y"])
                self.update()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if not self.interactive:
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            if self.is_resizing_overlay:
                self.is_resizing_overlay = False
                self.resizing_overlay_index = None
                self.overlay_resize_start_pos = None
                self.setCursor(Qt.ArrowCursor)
            elif self.dragging_index is not None:
                self.dragging_index = None
                self.drag_start_pos = None
                self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class BubblePendantWidget(QGroupBox):
    """静态图片模式下的气泡挂件制作模块"""
    
    MAX_OVERLAYS = 5
    
    def __init__(self, parent_gui, asset_manager=None):
        super().__init__("气泡挂件制作")
        self.parent_gui = parent_gui
        self.asset_manager = asset_manager or getattr(parent_gui, "asset_manager", None)
        self.overlays = []
        self.active_overlay_index = -1
        self.base_image_path = None
        self.base_pixmap = None
        
        self.canvas = BubbleCanvas()
        self.canvas.set_overlays(self.overlays)
        self.canvas.overlay_selected.connect(self.on_canvas_overlay_selected)
        self.canvas.overlay_position_changed.connect(self.on_canvas_overlay_moved)
        self.canvas.overlay_scale_changed.connect(self.on_canvas_overlay_scaled)
        
        self._build_ui()
        self._refresh_overlay_list()
        self._sync_overlay_controls()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        hint = QLabel("说明：请先选择气泡底图（720×264px），再添加相关PNG素材进行排版。中区禁止放置素材。")
        hint.setStyleSheet("color:#94A3B8;")
        layout.addWidget(hint)
        
        base_row = QHBoxLayout()
        self.base_status_label = QLabel("气泡底图：未选择")
        self.base_status_label.setObjectName("outputBadge")
        base_row.addWidget(self.base_status_label, 1)
        self.select_base_btn = QPushButton("选择气泡底图")
        self.select_base_btn.setObjectName("secondaryButton")
        self.select_base_btn.clicked.connect(self.select_base_image)
        base_row.addWidget(self.select_base_btn)
        layout.addLayout(base_row)
        
        content_row = QHBoxLayout()
        content_row.addWidget(self.canvas, 3)
        
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        
        list_label = QLabel(f"素材列表（最多{self.MAX_OVERLAYS}张PNG）")
        control_panel.addWidget(list_label)
        self.overlay_list = QListWidget()
        self.overlay_list.setFixedHeight(120)
        self.overlay_list.currentRowChanged.connect(self.on_overlay_list_changed)
        control_panel.addWidget(self.overlay_list)
        
        overlay_btn_row = QHBoxLayout()
        self.add_overlay_btn = QPushButton("添加素材")
        self.add_overlay_btn.setObjectName("secondaryButton")
        self.add_overlay_btn.clicked.connect(self.add_overlay)
        overlay_btn_row.addWidget(self.add_overlay_btn)
        self.remove_overlay_btn = QPushButton("删除素材")
        self.remove_overlay_btn.setObjectName("dangerButton")
        self.remove_overlay_btn.clicked.connect(self.remove_overlay)
        overlay_btn_row.addWidget(self.remove_overlay_btn)
        control_panel.addLayout(overlay_btn_row)
        overlay_library_row = QHBoxLayout()
        self.overlay_save_btn = QPushButton("保存到素材库")
        self.overlay_save_btn.setObjectName("secondaryButton")
        self.overlay_save_btn.setEnabled(False)
        self.overlay_save_btn.clicked.connect(self.save_selected_overlay_to_library)
        overlay_library_row.addWidget(self.overlay_save_btn)
        self.overlay_library_btn = QPushButton("素材库…")
        self.overlay_library_btn.clicked.connect(self.add_overlay_from_library)
        overlay_library_row.addWidget(self.overlay_library_btn)
        control_panel.addLayout(overlay_library_row)
        
        param_group = QGroupBox("素材参数（单位：像素）")
        param_layout = QVBoxLayout()
        self.overlay_x_spin = QSpinBox()
        self.overlay_x_spin.setRange(0, BubbleCanvas.CANVAS_WIDTH)
        self.overlay_x_spin.valueChanged.connect(self.on_overlay_param_changed)
        self.overlay_y_spin = QSpinBox()
        self.overlay_y_spin.setRange(0, BubbleCanvas.CANVAS_HEIGHT)
        self.overlay_y_spin.valueChanged.connect(self.on_overlay_param_changed)
        self.overlay_scale_spin = QDoubleSpinBox()
        self.overlay_scale_spin.setRange(0.01, 5.0)
        self.overlay_scale_spin.setDecimals(2)
        self.overlay_scale_spin.setSingleStep(0.01)
        self.overlay_scale_spin.setValue(1.0)
        self.overlay_scale_spin.setKeyboardTracking(False)
        # 重写keyPressEvent以支持直接输入小数点（包括以小数点开头）
        original_keyPressEvent_overlay = self.overlay_scale_spin.keyPressEvent
        def custom_keyPressEvent_overlay(event):
            if event.key() == Qt.Key_Period or event.key() == Qt.Key_Comma:
                line_edit = self.overlay_scale_spin.lineEdit()
                current_text = line_edit.text()
                cursor_pos = line_edit.cursorPosition()
                # 如果当前文本中没有小数点
                if '.' not in current_text:
                    # 如果当前文本为空或只有负号，插入"0."
                    if current_text == "" or current_text == "-":
                        line_edit.setText("0.")
                        line_edit.setCursorPosition(2)
                        return
                    # 如果光标在开头，插入"0."
                    elif cursor_pos == 0:
                        line_edit.setText("0." + current_text)
                        line_edit.setCursorPosition(2)
                        return
                    # 否则在当前位置插入小数点
                    else:
                        new_text = current_text[:cursor_pos] + "." + current_text[cursor_pos:]
                        line_edit.setText(new_text)
                        line_edit.setCursorPosition(cursor_pos + 1)
                        return
            original_keyPressEvent_overlay(event)
        self.overlay_scale_spin.keyPressEvent = custom_keyPressEvent_overlay
        self.overlay_scale_spin.valueChanged.connect(self.on_overlay_param_changed)
        
        for text, spin in [("X坐标", self.overlay_x_spin), ("Y坐标", self.overlay_y_spin)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{text}:"))
            row.addWidget(spin)
            param_layout.addLayout(row)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("缩放倍数:"))
        scale_row.addWidget(self.overlay_scale_spin)
        param_layout.addLayout(scale_row)
        rotation_row = QHBoxLayout()
        rotation_row.addWidget(QLabel("旋转 (°):"))
        self.overlay_rotation_spin = QSpinBox()
        self.overlay_rotation_spin.setRange(-180, 180)
        self.overlay_rotation_spin.valueChanged.connect(self.on_overlay_param_changed)
        rotation_row.addWidget(self.overlay_rotation_spin)
        param_layout.addLayout(rotation_row)
        param_group.setLayout(param_layout)
        control_panel.addWidget(param_group)
        
        control_panel.addStretch()
        self.export_btn = QPushButton("导出气泡挂件（480×384，PNG）")
        self.export_btn.clicked.connect(self.export_pendant)
        control_panel.addWidget(self.export_btn)
        
        content_row.addLayout(control_panel, 2)
        layout.addLayout(content_row)
        
        self.overlay_param_spins = [self.overlay_x_spin, self.overlay_y_spin, self.overlay_scale_spin, self.overlay_rotation_spin]
        self._update_overlay_controls_state()
    
    def select_base_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择气泡底图", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;PNG图片 (*.png);;JPEG图片 (*.jpg *.jpeg);;BMP图片 (*.bmp);;WEBP图片 (*.webp);;所有文件 (*.*)"
            )
            if not file_path:
                return
            
            # 使用PIL加载图片，支持多种格式
            try:
                from PIL import Image
                img = Image.open(file_path)
                if img.mode == 'RGBA':
                    img_array = np.array(img)
                    array = np.ascontiguousarray(img_array)
                    channels = array.shape[2]
                    if channels == 4:
                        fmt = QImage.Format_RGBA8888
                    else:
                        fmt = QImage.Format_RGB888
                    qimage = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt).copy()
                    pixmap = QPixmap.fromImage(qimage)
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.uint8)
                    array = np.ascontiguousarray(img_array)
                    qimage = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format_RGB888).copy()
                    pixmap = QPixmap.fromImage(qimage)
                
                if pixmap.isNull():
                    QMessageBox.warning(self, "错误", "无法加载所选的图片。")
                    return
                self._set_base_pixmap(pixmap, os.path.basename(file_path))
                self._update_overlay_controls_state()
            except Exception as e:
                print(f"加载图片失败: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "错误", f"无法加载所选的图片: {str(e)}")
        finally:
            self._refocus_editor_window()
    
    def set_base_image_from_array(self, array: np.ndarray, label: str = "预设气泡图"):
        if array is None:
            return
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3:
            return
        array = np.ascontiguousarray(array)
        channels = array.shape[2]
        if channels == 3:
            fmt = QImage.Format_RGB888
        elif channels == 4:
            fmt = QImage.Format_RGBA8888
        else:
            return
        qimage = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt).copy()
        pixmap = QPixmap.fromImage(qimage)
        self._set_base_pixmap(pixmap, label)
        self._update_overlay_controls_state()
    
    def _set_base_pixmap(self, pixmap: QPixmap, label: str):
        self.base_pixmap = pixmap
        self.base_image_path = label
        display_label = label if label else "未命名"
        self.base_status_label.setText(f"气泡底图：{display_label}")
        self.canvas.set_base_pixmap(pixmap)
    
    def _load_png_rgba(self, file_path):
        try:
            from PIL import Image
            img = Image.open(file_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return np.ascontiguousarray(np.array(img, dtype=np.uint8))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载PNG失败: {str(e)}")
            return None
    
    def _trim_transparent_pixels(self, image_array):
        """自动裁切PNG图片周围的透明像素（类似Photoshop的像素裁切）"""
        if image_array is None or len(image_array.shape) != 3:
            return image_array
        
        # 确保是RGBA格式
        if image_array.shape[2] != 4:
            return image_array
        
        # 获取alpha通道
        alpha = image_array[:, :, 3]
        
        # 找到非透明像素的位置
        non_transparent = np.where(alpha > 0)
        
        if len(non_transparent[0]) == 0:
            # 如果全部透明，返回原图
            return image_array
        
        # 计算边界框
        top = np.min(non_transparent[0])
        bottom = np.max(non_transparent[0]) + 1
        left = np.min(non_transparent[1])
        right = np.max(non_transparent[1]) + 1
        
        # 裁切图像
        trimmed = image_array[top:bottom, left:right, :]
        
        return np.ascontiguousarray(trimmed)
    
    def add_overlay(self):
        try:
            if len(self.overlays) >= self.MAX_OVERLAYS:
                QMessageBox.information(self, "提示", f"最多只能添加{self.MAX_OVERLAYS}张素材。")
                return
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择素材图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)"
            )
            if not file_path:
                return
            try:
                suffix = Path(file_path).suffix.lower()
                auto_detect = False
                if suffix in WHITE_BACKGROUND_FORMATS:
                    reply = QMessageBox.question(
                        self,
                        "白底处理",
                        "检测到非 PNG 图片。\n是否去除白色背景？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    auto_detect = (reply == QMessageBox.Yes)
                image_array, _ = prepare_overlay_image(file_path, auto_detect_white=auto_detect)
                self._append_overlay_from_array(image_array, os.path.basename(file_path), source_path=file_path)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载素材失败: {str(e)}")
        finally:
            self._refocus_editor_window()
    
    def _append_overlay_from_array(self, image_array, display_name: str = None, source_path: str = None):
        if image_array is None:
            return
        array = np.ascontiguousarray(image_array)
        array = self._trim_transparent_pixels(array)
        default_x = BubbleCanvas.EDITABLE_X + max(0, (BubbleCanvas.EDITABLE_WIDTH - array.shape[1]) // 2)
        default_y = max(0, (BubbleCanvas.CANVAS_HEIGHT - array.shape[0]) // 2)
        overlay = {
            "path": source_path,
            "name": display_name or (os.path.basename(source_path) if source_path else "素材"),
            "image": array,
            "x": default_x,
            "y": default_y,
            "scale": 1.0,
            "angle": 0.0
        }
        self.overlays.append(overlay)
        self.canvas.set_overlays(self.overlays)
        self.set_active_overlay(len(self.overlays) - 1)
    
    def add_overlay_from_library(self):
        try:
            asset = None
            if hasattr(self.parent_gui, "open_asset_library_dialog"):
                asset = self.parent_gui.open_asset_library_dialog("素材库")
            elif self.asset_manager:
                dialog = AssetLibraryDialog(self.asset_manager, self, "素材库")
                if dialog.exec_() == QDialog.Accepted:
                    asset = dialog.selected_asset
            if not asset or not self.asset_manager:
                return
            try:
                array = self.asset_manager.load_asset_array(asset.get("id"))
                self._append_overlay_from_array(array, asset.get("name"))
            except Exception as e:
                QMessageBox.warning(self, "错误", f"载入素材失败：{str(e)}")
        finally:
            self._refocus_editor_window()
    
    def save_selected_overlay_to_library(self):
        if not self.asset_manager:
            QMessageBox.warning(self, "提示", "素材库不可用，请检查权限后重试。")
            return
        if self.active_overlay_index < 0 or self.active_overlay_index >= len(self.overlays):
            QMessageBox.information(self, "提示", "请选择需要保存的素材。")
            return
        overlay = self.overlays[self.active_overlay_index]
        try:
            asset = self.asset_manager.add_from_array(overlay.get("image"), overlay.get("name"))
            QMessageBox.information(self, "完成", f"已保存至素材库：{asset.get('name')}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存素材失败：{str(e)}")
    
    def remove_overlay(self):
        if self.active_overlay_index < 0 or self.active_overlay_index >= len(self.overlays):
            return
        del self.overlays[self.active_overlay_index]
        if self.active_overlay_index >= len(self.overlays):
            new_index = len(self.overlays) - 1
        else:
            new_index = self.active_overlay_index
        self.canvas.update()
        self.set_active_overlay(new_index)
    
    def _refresh_overlay_list(self):
        self.overlay_list.blockSignals(True)
        self.overlay_list.clear()
        for idx, overlay in enumerate(self.overlays):
            item = QListWidgetItem(f"{idx + 1}. {overlay.get('name', '素材')}")
            self.overlay_list.addItem(item)
        if 0 <= self.active_overlay_index < len(self.overlays):
            self.overlay_list.setCurrentRow(self.active_overlay_index)
        else:
            self.overlay_list.clearSelection()
        self.overlay_list.blockSignals(False)
    
    def _sync_overlay_controls(self):
        if 0 <= self.active_overlay_index < len(self.overlays):
            overlay = self.overlays[self.active_overlay_index]
            values = (
                int(overlay.get("x", 0)),
                int(overlay.get("y", 0)),
                float(overlay.get("scale", 1.0)),
                int(round(float(overlay.get("angle", 0.0))))
            )
        else:
            values = (0, 0, 1.0, 0)
        for spin, value in zip(self.overlay_param_spins, values):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.canvas.set_active_index(self.active_overlay_index)
        self._update_overlay_controls_state()
    
    def _update_overlay_controls_state(self):
        has_selection = 0 <= self.active_overlay_index < len(self.overlays)
        self.add_overlay_btn.setEnabled(len(self.overlays) < self.MAX_OVERLAYS)
        self.remove_overlay_btn.setEnabled(has_selection)
        self.export_btn.setEnabled(bool(self.base_pixmap) and bool(self.overlays))
        if hasattr(self, 'overlay_save_btn'):
            self.overlay_save_btn.setEnabled(has_selection)
        for spin in self.overlay_param_spins:
            spin.setEnabled(has_selection)

    def _refocus_editor_window(self):
        window = self.window()
        if window:
            window.raise_()
            window.activateWindow()
    
    def on_overlay_list_changed(self, row):
        if row is None:
            row = -1
        self.set_active_overlay(row)
    
    def set_active_overlay(self, index):
        if index is None or index < 0 or index >= len(self.overlays):
            self.active_overlay_index = -1
        else:
            self.active_overlay_index = index
        self._refresh_overlay_list()
        self._sync_overlay_controls()
        self.canvas.update()
    
    def on_overlay_param_changed(self):
        if self.active_overlay_index < 0 or self.active_overlay_index >= len(self.overlays):
            return
        overlay = self.overlays[self.active_overlay_index]
        overlay["x"] = self.overlay_x_spin.value()
        overlay["y"] = self.overlay_y_spin.value()
        overlay["scale"] = max(0.01, min(5.0, self.overlay_scale_spin.value()))
        overlay["angle"] = self.overlay_rotation_spin.value()
        self._clamp_overlay_position(overlay)
        self.canvas.update()
    
    def on_canvas_overlay_selected(self, index):
        self.overlay_list.blockSignals(True)
        self.overlay_list.setCurrentRow(index)
        self.overlay_list.blockSignals(False)
        self.set_active_overlay(index)
    
    def on_canvas_overlay_moved(self, index, x, y):
        if index < 0 or index >= len(self.overlays):
            return
        overlay = self.overlays[index]
        overlay["x"] = x
        overlay["y"] = y
        if index == self.active_overlay_index:
            self.overlay_x_spin.blockSignals(True)
            self.overlay_y_spin.blockSignals(True)
            self.overlay_x_spin.setValue(x)
            self.overlay_y_spin.setValue(y)
            self.overlay_x_spin.blockSignals(False)
            self.overlay_y_spin.blockSignals(False)
        self.canvas.update()
    
    def on_canvas_overlay_scaled(self, index, scale):
        if index < 0 or index >= len(self.overlays):
            return
        self.overlays[index]["scale"] = scale
        if index == self.active_overlay_index:
            self.overlay_scale_spin.blockSignals(True)
            self.overlay_scale_spin.setValue(scale)
            self.overlay_scale_spin.blockSignals(False)
        self.canvas.update()
    
    def _clamp_overlay_position(self, overlay):
        image = overlay.get("image")
        if image is None:
            return
        scale = max(0.05, float(overlay.get("scale", 1.0)))
        base_w = max(1, int(image.shape[1] * scale))
        base_h = max(1, int(image.shape[0] * scale))
        angle = float(overlay.get("angle", 0.0))
        if abs(angle) < 1e-3:
            overlay["x"] = max(0, min(int(round(overlay.get("x", 0))), BubbleCanvas.CANVAS_WIDTH - base_w))
            overlay["y"] = max(0, min(int(round(overlay.get("y", 0))), BubbleCanvas.CANVAS_HEIGHT - base_h))
        else:
            angle_rad = math.radians(angle)
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))
            rotated_w = max(1.0, base_w * cos_a + base_h * sin_a)
            rotated_h = max(1.0, base_w * sin_a + base_h * cos_a)
            min_center_x = rotated_w / 2.0
            max_center_x = BubbleCanvas.CANVAS_WIDTH - rotated_w / 2.0
            min_center_y = rotated_h / 2.0
            max_center_y = BubbleCanvas.CANVAS_HEIGHT - rotated_h / 2.0
            center_x = float(overlay.get("x", 0)) + base_w / 2.0
            center_y = float(overlay.get("y", 0)) + base_h / 2.0
            center_x = max(min_center_x, min(max_center_x, center_x))
            center_y = max(min_center_y, min(max_center_y, center_y))
            overlay["x"] = int(round(center_x - base_w / 2.0))
            overlay["y"] = int(round(center_y - base_h / 2.0))
        if 0 <= self.active_overlay_index < len(self.overlays) and overlay is self.overlays[self.active_overlay_index]:
            self.overlay_x_spin.blockSignals(True)
            self.overlay_y_spin.blockSignals(True)
            self.overlay_x_spin.setValue(overlay["x"])
            self.overlay_y_spin.setValue(overlay["y"])
            self.overlay_x_spin.blockSignals(False)
            self.overlay_y_spin.blockSignals(False)
    
    def _overlay_dimensions(self, overlay):
        _, _, width, height = self._overlay_bounding_rect(overlay)
        return width, height

    def _overlay_bounding_rect(self, overlay):
        image = overlay.get("image")
        if image is None:
            return (overlay.get("x", 0), overlay.get("y", 0), 0, 0)
        scale = max(0.05, float(overlay.get("scale", 1.0)))
        width = max(1, int(image.shape[1] * scale))
        height = max(1, int(image.shape[0] * scale))
        angle = float(overlay.get("angle", 0.0))
        if abs(angle) < 1e-3:
            return (
                int(round(overlay.get("x", 0))),
                int(round(overlay.get("y", 0))),
                width,
                height
            )
        angle_rad = math.radians(angle)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        rotated_w = int(width * cos_a + height * sin_a)
        rotated_h = int(width * sin_a + height * cos_a)
        center_x = float(overlay.get("x", 0)) + width / 2.0
        center_y = float(overlay.get("y", 0)) + height / 2.0
        left = int(round(center_x - rotated_w / 2.0))
        top = int(round(center_y - rotated_h / 2.0))
        return (left, top, max(1, rotated_w), max(1, rotated_h))
    
    def _middle_region_intersection(self, overlay):
        left, top, width, height = self._overlay_bounding_rect(overlay)
        right = left + width
        bottom = top + height
        middle_left = BubbleCanvas.EDITABLE_X
        middle_top = BubbleCanvas.TOP_SECTION_HEIGHT
        middle_right = BubbleCanvas.EDITABLE_X + BubbleCanvas.EDITABLE_WIDTH
        middle_bottom = BubbleCanvas.TOP_SECTION_HEIGHT + BubbleCanvas.MIDDLE_SECTION_HEIGHT
        overlap_x = min(right, middle_right) - max(left, middle_left)
        overlap_y = min(bottom, middle_bottom) - max(top, middle_top)
        return overlap_x > 0 and overlap_y > 0
    
    def export_pendant(self):
        if not self.base_pixmap:
            QMessageBox.warning(self, "提示", "请先选择气泡底图。")
            return
        if not self.overlays:
            QMessageBox.warning(self, "提示", "请至少添加一张PNG素材。")
            return
        if any(self._middle_region_intersection(ov) for ov in self.overlays):
            QMessageBox.warning(self, "提示", "存在素材覆盖在中区，无法导出。请调整素材位置。")
            return
        # 静态图片模式：所有图片放在同一文件夹，不需要二级文件夹
        if getattr(self.parent_gui, "is_image_mode", False):
            # 静态图片模式：使用与 start_processing 相同的逻辑
            # 获取上传的图片名称（不含扩展名）作为文件夹名
            if hasattr(self.parent_gui, "media_path") and self.parent_gui.media_path:
                media_name = os.path.splitext(os.path.basename(self.parent_gui.media_path))[0]
            else:
                media_name = "未命名图片"
            # 使用 output_base_dir，然后加上 media_name，与 start_processing 保持一致
            output_base_dir = getattr(self.parent_gui, "output_base_dir", None)
            if not output_base_dir:
                # 允许独立使用，如果没有设置输出目录，让用户选择
                output_base_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
                if not output_base_dir:
                    return
            target_dir = os.path.join(output_base_dir, media_name)
            os.makedirs(target_dir, exist_ok=True)
            filename = "气泡挂件.png"
        else:
            # 动态视频模式：使用原来的二级文件夹结构
            output_dir = getattr(self.parent_gui, "output_base_dir", None)
            if not output_dir:
                # 允许独立使用，如果没有设置输出目录，让用户选择
                output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
                if not output_dir:
                    return
            target_dir = os.path.join(output_dir, "气泡挂件")
            os.makedirs(target_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_index = 1
            filename = f"bubble_{timestamp}.png"
        
        # 允许独立使用，如果没有processor，创建一个临时的
        processor = getattr(self.parent_gui, "processor", None)
        temp_processor = False
        if processor is None:
            try:
                from video_processor import VideoProcessor
                # 创建不需要video_path的临时processor（仅用于save_frame）
                processor = VideoProcessor(video_path=None, mode="image")
                temp_processor = True
            except Exception as e:
                import traceback
                error_msg = f"无法创建处理器：{str(e)}"
                print(error_msg)
                traceback.print_exc()
                QMessageBox.warning(self, "提示", error_msg)
                return
        
        try:
            # 创建RGBA画布，尺寸为480*384（EDITABLE_WIDTH * CANVAS_HEIGHT）
            # 注意：数组形状是(height, width, channels)，所以是(CANVAS_HEIGHT, EDITABLE_WIDTH)
            # 初始化为完全透明（alpha=0）
            canvas_rgba = np.zeros((BubbleCanvas.CANVAS_HEIGHT, BubbleCanvas.EDITABLE_WIDTH, 4), dtype=np.uint8)
            print(f"创建canvas_rgba: 尺寸={canvas_rgba.shape}, 应该是(384, 480, 4)")
            print(f"overlays数量: {len(self.overlays)}")
            
            # 合成所有overlay
            print(f"开始合成 {len(self.overlays)} 个overlay到可编辑区域（480*384）...")
            for idx, overlay in enumerate(self.overlays):
                try:
                    print(f"\n=== 开始合成第{idx+1}个overlay ===")
                    print(f"Overlay信息: x={overlay.get('x')}, y={overlay.get('y')}, scale={overlay.get('scale')}")
                    self._composite_overlay(canvas_rgba, overlay)
                    print(f"第{idx+1}个overlay合成完成")
                except Exception as overlay_error:
                    print(f"合成overlay时出错: {overlay_error}")
                    import traceback
                    traceback.print_exc()
                    continue
            print(f"\n所有overlay合成完成")
            
            # 确保canvas_rgba格式正确
            if canvas_rgba.shape[2] != 4:
                # 如果不是4通道，转换为RGBA
                if canvas_rgba.shape[2] == 3:
                    new_canvas = np.zeros((canvas_rgba.shape[0], canvas_rgba.shape[1], 4), dtype=np.uint8)
                    new_canvas[:, :, :3] = canvas_rgba
                    new_canvas[:, :, 3] = 255
                    canvas_rgba = new_canvas
                else:
                    raise ValueError(f"不支持的图像通道数: {canvas_rgba.shape[2]}")
            
            # 确保数据类型和连续性
            canvas_rgba = np.ascontiguousarray(canvas_rgba.astype(np.uint8))
            
            # 验证数组形状：应该是(384, 480, 4)
            if len(canvas_rgba.shape) != 3 or canvas_rgba.shape[2] != 4:
                raise ValueError(f"canvas_rgba形状不正确: {canvas_rgba.shape}")
            if canvas_rgba.shape[0] != BubbleCanvas.CANVAS_HEIGHT or canvas_rgba.shape[1] != BubbleCanvas.EDITABLE_WIDTH:
                raise ValueError(f"canvas_rgba尺寸不正确: 期望({BubbleCanvas.CANVAS_HEIGHT}, {BubbleCanvas.EDITABLE_WIDTH}, 4), 实际{canvas_rgba.shape}")
            
            # 处理文件名冲突（仅静态模式需要）
            if getattr(self.parent_gui, "is_image_mode", False):
                save_path = os.path.join(target_dir, filename)
                if os.path.exists(save_path):
                    file_index = 1
                    base_name = "气泡挂件"
                    while os.path.exists(save_path):
                        filename = f"{base_name}_{file_index}.png"
                        save_path = os.path.join(target_dir, filename)
                        file_index += 1
            else:
                save_path = os.path.join(target_dir, filename)
                while os.path.exists(save_path):
                    file_index += 1
                    filename = f"bubble_{timestamp}_{file_index}.png"
                    save_path = os.path.join(target_dir, filename)
            
            # 使用更安全的方式保存
            print(f"准备保存气泡挂件图，尺寸: {canvas_rgba.shape} (应该是384x480x4), 路径: {save_path}")
            # 验证是否有内容（至少有一个像素的alpha > 0）
            has_content = np.any(canvas_rgba[:, :, 3] > 0)
            if has_content:
                # 统计非透明像素数量
                non_transparent_count = np.sum(canvas_rgba[:, :, 3] > 0)
                total_pixels = canvas_rgba.shape[0] * canvas_rgba.shape[1]
                print(f"✓ 画布包含 {non_transparent_count}/{total_pixels} 个非透明像素 ({non_transparent_count/total_pixels*100:.2f}%)")
                # 检查RGB通道是否有内容
                has_rgb_content = np.any(canvas_rgba[:, :, :3] > 0)
                print(f"✓ RGB通道是否有内容: {has_rgb_content}")
                if not has_rgb_content:
                    print("警告：虽然alpha>0，但RGB通道全为0，可能是合成逻辑有问题")
            else:
                print("❌ 错误：画布完全是空白的（所有像素alpha=0）！")
                print(f"overlays数量: {len(self.overlays)}")
                for idx, ov in enumerate(self.overlays):
                    ov_x = ov.get('x', 0)
                    ov_y = ov.get('y', 0)
                    ov_img = ov.get('image')
                    print(f"  Overlay {idx+1}:")
                    print(f"    - 坐标: x={ov_x}, y={ov_y} (整个画布坐标系)")
                    print(f"    - 可编辑区域x范围: [{BubbleCanvas.EDITABLE_X}, {BubbleCanvas.EDITABLE_X + BubbleCanvas.EDITABLE_WIDTH}]")
                    print(f"    - 可编辑区域y范围: [0, {BubbleCanvas.CANVAS_HEIGHT}]")
                    if ov_img is not None:
                        print(f"    - 图像尺寸: {ov_img.shape}")
                        if len(ov_img.shape) >= 3 and ov_img.shape[2] >= 4:
                            alpha_count = np.sum(ov_img[:, :, 3] > 0)
                            print(f"    - 图像非透明像素数: {alpha_count}")
                        else:
                            print(f"    - 图像通道数: {ov_img.shape[2] if len(ov_img.shape) >= 3 else 'N/A'}")
                    else:
                        print(f"    - 图像: None")
                    print(f"    - 缩放: {ov.get('scale', 1.0)}")
                QMessageBox.warning(self, "警告", "导出的画布是空白的，请检查素材是否正确添加到可编辑区域（x坐标应在240-720之间）。\n\n详细信息请查看控制台输出。")
                return
            
            # 确保保存的图片尺寸正确：480x384 (width x height)
            # 注意：numpy数组是(height, width)，但保存时需要确保是480x384
            if canvas_rgba.shape[0] != 384 or canvas_rgba.shape[1] != 480:
                print(f"错误：canvas_rgba尺寸不正确！期望(384, 480, 4)，实际{canvas_rgba.shape}")
                raise ValueError(f"canvas_rgba尺寸不正确: {canvas_rgba.shape}")
            
            success = processor.save_frame(canvas_rgba, save_path, max_size_bytes=300 * 1024)
            if success:
                QMessageBox.information(self, "完成", f"已导出：{save_path}")
            else:
                QMessageBox.warning(self, "错误", "导出失败，请检查素材文件。")
        except Exception as e:
            import traceback
            error_msg = f"导出时发生错误：{str(e)}"
            print(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"{error_msg}\n\n详细信息请查看控制台输出。")
        finally:
            # 如果是临时创建的processor，清理它
            if temp_processor and processor is not None:
                try:
                    processor.close()
                except Exception as cleanup_error:
                    print(f"清理processor时出错: {cleanup_error}")
    
    def _composite_overlay(self, canvas_rgba, overlay):
        """
        将overlay合成到canvas_rgba上
        参考主页面中add_watermark的逻辑
        canvas_rgba: 480×384的RGBA画布（可编辑区域）
        overlay: 包含image、x、y、scale的字典，坐标是相对于整个画布的（720×384）
        """
        image = overlay.get("image")
        if image is None:
            print("警告：overlay没有image，跳过")
            return
        
        try:
            # 获取overlay的坐标（相对于整个画布720×384）
            overlay_x_full = float(overlay.get("x", 0))  # 整个画布上的x坐标
            overlay_y = float(overlay.get("y", 0))  # y坐标（可编辑区域从y=0开始）
            scale = max(0.05, float(overlay.get("scale", 1.0)))
            
            print(f"\n=== 合成overlay ===")
            print(f"原始坐标（整个画布）: x={overlay_x_full}, y={overlay_y}, scale={scale}")
            
            # 转换为可编辑区域坐标（480×384）
            # 可编辑区域从x=240开始，所以需要减去240
            overlay_x = int(round(overlay_x_full - BubbleCanvas.EDITABLE_X))
            overlay_y = int(round(overlay_y))
            
            print(f"转换后坐标（可编辑区域）: x={overlay_x}, y={overlay_y}")
            print(f"可编辑区域范围: x=[0, {BubbleCanvas.EDITABLE_WIDTH}], y=[0, {BubbleCanvas.CANVAS_HEIGHT}]")
            
            # 确保媒介是合规的 numpy 数组
            if not isinstance(image, np.ndarray):
                raise ValueError("overlay image必须是numpy数组")
            if len(image.shape) < 2 or len(image.shape) > 3:
                raise ValueError(f"不支持的图像形状: {image.shape}")
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = np.ascontiguousarray(image)

            from PIL import Image as PILImage
            if image.shape[2] == 4:
                wm_img = PILImage.fromarray(image, 'RGBA')
            elif image.shape[2] == 3:
                wm_img = PILImage.fromarray(image, 'RGB').convert('RGBA')
            else:
                raise ValueError(f"不支持的通道数: {image.shape[2]}")
            if scale != 1.0:
                new_w = max(1, int(image.shape[1] * scale))
                new_h = max(1, int(image.shape[0] * scale))
                wm_img = wm_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            orig_w, orig_h = wm_img.width, wm_img.height
            angle = float(overlay.get("angle", 0.0))
            if abs(angle) >= 1e-3:
                wm_img = wm_img.rotate(angle, expand=True, resample=PILImage.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
                delta_x = (orig_w - wm_img.width) / 2.0
                delta_y = (orig_h - wm_img.height) / 2.0
                overlay_x_full += delta_x
                overlay_y += delta_y
                watermark = np.array(wm_img, dtype=np.uint8)
            
            # 确保水印是RGBA格式（参考add_watermark的逻辑）
            if watermark.shape[2] == 3:
                wm_rgba = np.zeros((watermark.shape[0], watermark.shape[1], 4), dtype=np.uint8)
                wm_rgba[:, :, :3] = watermark
                wm_rgba[:, :, 3] = 255
            else:
                wm_rgba = watermark.copy()
            
            wm_rgba = np.ascontiguousarray(wm_rgba.astype(np.uint8))
            
            # 验证图像
            if wm_rgba.shape[0] == 0 or wm_rgba.shape[1] == 0:
                print(f"警告：overlay图像尺寸为0，跳过")
                return
            
            has_alpha = np.any(wm_rgba[:, :, 3] > 0)
            if not has_alpha:
                print(f"警告：overlay图像完全透明，跳过")
                return
            
            print(f"overlay图像尺寸（缩放后）: {wm_rgba.shape[1]}x{wm_rgba.shape[0]}")
            print(f"非透明像素数: {np.sum(wm_rgba[:, :, 3] > 0)}")
            
            # 参考add_watermark的逻辑：计算粘贴区域
            h, w = canvas_rgba.shape[:2]  # canvas_rgba是480×384
            wm_h, wm_w = wm_rgba.shape[:2]
            
            # 确保坐标在有效范围内（相对于可编辑区域）
            x = max(0, min(overlay_x, w - 1))
            y = max(0, min(overlay_y, h - 1))
            
            # 计算实际粘贴区域
            x_end = min(x + wm_w, w)
            y_end = min(y + wm_h, h)
            wm_x_end = x_end - x
            wm_y_end = y_end - y
            
            print(f"粘贴区域: canvas坐标 x=[{x}, {x_end}], y=[{y}, {y_end}]")
            print(f"watermark使用区域: x=[0, {wm_x_end}], y=[0, {wm_y_end}]")
            
            if wm_x_end > 0 and wm_y_end > 0:
                # 提取水印的alpha通道（参考add_watermark的逻辑）
                wm_alpha = wm_rgba[:wm_y_end, :wm_x_end, 3:4] / 255.0
                wm_rgb = wm_rgba[:wm_y_end, :wm_x_end, :3]
                
                # 混合水印和原图（参考add_watermark的逻辑）
                # canvas_rgba初始是透明的，所以直接使用水印的RGB和alpha
                canvas_rgba[y:y_end, x:x_end, :3] = (
                    canvas_rgba[y:y_end, x:x_end, :3] * (1 - wm_alpha) + 
                    wm_rgb * wm_alpha
                ).astype(np.uint8)
                # 更新alpha通道
                canvas_rgba[y:y_end, x:x_end, 3:4] = np.maximum(
                    canvas_rgba[y:y_end, x:x_end, 3:4],
                    (wm_alpha * 255).astype(np.uint8)
                )
                
                # 验证合成结果
                final_alpha_count = np.sum(canvas_rgba[y:y_end, x:x_end, 3] > 0)
                print(f"✓ 合成完成！在canvas_rgba[{y}:{y_end}, {x}:{x_end}]区域中有 {final_alpha_count} 个非透明像素")
            else:
                print(f"警告：粘贴区域无效，跳过")
                
        except Exception as e:
            print(f"处理overlay时出错: {e}")
            import traceback
            traceback.print_exc()
            return



class ProcessingThread(QThread):
    """处理线程"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, processor, output_dir, rect, inverse, target_size, 
                 start_time, end_time, target_fps, resize_crop, 
                 watermarks=None, canvas_size=None, background_colors=None, max_size_bytes: int = 300 * 1024,
                 background_tolerance: int = 30, inverse_inner_rect=None,
                 inverse_overlay_outer_rect=None, inverse_overlay_inner_rect=None,
                 filename_base: str = None, default_white_enabled: bool = True):
        super().__init__()
        self.processor = processor
        self.output_dir = output_dir
        self.rect = rect
        self.inverse = inverse
        self.target_size = target_size
        self.start_time = start_time
        self.end_time = end_time
        self.target_fps = target_fps
        self.resize_crop = resize_crop  # (canvas_width, canvas_height, scale, offset_x, offset_y) 或 None
        self.watermarks = watermarks or []
        self.canvas_size = canvas_size  # (canvas_width, canvas_height) 或 None
        self.inverse_background_colors = [tuple(int(c) for c in color) for color in (background_colors or [])]  # 反选挂件背景颜色列表
        self.max_size_bytes = max_size_bytes
        self.background_tolerance = background_tolerance
        self.inverse_inner_rect = inverse_inner_rect
        self.inverse_overlay_outer_rect = inverse_overlay_outer_rect
        self.inverse_overlay_inner_rect = inverse_overlay_inner_rect
        self.filename_base = filename_base  # 静态图片模式的文件名基础（如"封面图"、"气泡图"等）
        self.default_white_enabled = bool(default_white_enabled)
        self._stop_requested = False
    
    def request_stop(self):
        """请求停止处理"""
        self._stop_requested = True
    
    @staticmethod
    def _ensure_rgba(image):
        if image is None:
            return None
        if image.ndim < 3:
            return None
        if image.shape[2] == 4:
            return np.ascontiguousarray(image)
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate([image, alpha], axis=2))

    @staticmethod
    def _build_ring_mask_local(width, height, outer_rect, inner_rect):
        mask = np.zeros((height, width), dtype=np.uint8)
        if not outer_rect:
            return mask
        ox = int(round(outer_rect[0]))
        oy = int(round(outer_rect[1]))
        ow = int(round(outer_rect[2]))
        oh = int(round(outer_rect[3]))
        ox = max(0, ox)
        oy = max(0, oy)
        ex = min(width, ox + max(0, ow))
        ey = min(height, oy + max(0, oh))
        if ex > ox and ey > oy:
            mask[oy:ey, ox:ex] = 1
        if inner_rect:
            ix = int(round(inner_rect[0]))
            iy = int(round(inner_rect[1]))
            iw = int(round(inner_rect[2]))
            ih = int(round(inner_rect[3]))
            ix = max(0, ix)
            iy = max(0, iy)
            ix2 = min(width, ix + max(0, iw))
            iy2 = min(height, iy + max(0, ih))
            if ix2 > ix and iy2 > iy:
                mask[iy:iy2, ix:ix2] = 0
        return mask
    
    def run(self):
        """执行处理"""
        try:
            # 提取帧（使用起始和结束时间）
            effective_duration = self.end_time - self.start_time
            frames = self.processor.extract_frames(
                max_duration=effective_duration if effective_duration > 0 else 0,
                target_fps=self.target_fps,
                start_time=self.start_time,
                end_time=self.end_time
            )
            total = len(frames)
            
            if total == 0:
                self.finished.emit(False, "未能提取任何帧")
                return
            
            # 处理每一帧
            for i, frame in enumerate(frames):
                if self._stop_requested:
                    self.finished.emit(False, "用户已停止处理")
                    return
                
                # 第一步：整体尺寸裁切（如果启用）- 使用画布裁切
                if self.resize_crop:
                    # resize_crop 现在包含 (canvas_width, canvas_height, scale, offset_x, offset_y)
                    canvas_width, canvas_height, scale, offset_x, offset_y = self.resize_crop
                    frame = self.processor.crop_frame_canvas(
                        frame, canvas_width, canvas_height, scale, offset_x, offset_y
                    )
                
                # 第二步：矩形区域裁切
                ring_mask_a = None
                if self.inverse:
                    cropped = self.processor.crop_frame(
                        frame, self.rect[0], self.rect[1], self.rect[2], self.rect[3]
                    )
                else:
                    cropped = self.processor.crop_frame(
                        frame, self.rect[0], self.rect[1], self.rect[2], self.rect[3]
                    )
                
                # 第三步：调整到目标输出尺寸
                # 记录裁切后的尺寸，用于素材坐标缩放
                cropped_h, cropped_w = cropped.shape[:2]
                if self.target_size:
                    cropped = self.processor.resize_frame(
                        cropped, self.target_size[0], self.target_size[1]
                    )
                    # 计算缩放比例（用于调整素材坐标）
                    scale_x = self.target_size[0] / cropped_w
                    scale_y = self.target_size[1] / cropped_h
                else:
                    # 如果没有调整尺寸，缩放比例为1
                    scale_x = 1.0
                    scale_y = 1.0
                
                def scale_rect(rect):
                    if not rect:
                        return None
                    return (
                        rect[0] * scale_x,
                        rect[1] * scale_y,
                        rect[2] * scale_x,
                        rect[3] * scale_y
                    )

                if self.inverse:
                    full_rect = (0, 0, cropped.shape[1], cropped.shape[0])
                    scaled_red = scale_rect(self.inverse_inner_rect)
                    ring_mask_a = self._build_ring_mask_local(
                        cropped.shape[1], cropped.shape[0], full_rect, scaled_red
                    )

                export_rgba = self._ensure_rgba(cropped)
                base_rgba = export_rgba.copy()
                
                # 第五步：移除挂件外侧背景颜色（仅反选时，且只作用于主画面）
                if self.inverse:
                    colors_to_remove = []
                    if self.default_white_enabled:
                        colors_to_remove.append((255, 255, 255))
                    for sel in self.inverse_background_colors:
                        if isinstance(sel, dict):
                            color = sel["color"]
                        else:
                            color = sel
                        if color and tuple(color) not in colors_to_remove:
                            colors_to_remove.append(tuple(color))
                    if ring_mask_a is not None:
                        valid_mask = np.ascontiguousarray((ring_mask_a > 0).astype(np.uint8) * 255)
                    else:
                        rect_x = int(round(self.rect[0] * scale_x))
                        rect_y = int(round(self.rect[1] * scale_y))
                        rect_w = int(round(self.rect[2] * scale_x))
                        rect_h = int(round(self.rect[3] * scale_y))
                        valid_mask = np.ones((base_rgba.shape[0], base_rgba.shape[1]), dtype=np.uint8) * 255
                        x_end = min(valid_mask.shape[1], rect_x + rect_w)
                        y_end = min(valid_mask.shape[0], rect_y + rect_h)
                        rect_x = max(0, rect_x)
                        rect_y = max(0, rect_y)
                        if rect_x < x_end and rect_y < y_end:
                            valid_mask[rect_y:y_end, rect_x:x_end] = 0
                        valid_mask = np.ascontiguousarray(valid_mask)
                    for sel in colors_to_remove:
                        tolerance = self.background_tolerance
                        for color_item in self.inverse_background_colors:
                            if isinstance(color_item, dict) and tuple(color_item["color"]) == tuple(sel):
                                    tolerance = color_item.get("tolerance", self.background_tolerance)
                                    break
                        base_rgba = self.processor.remove_background_color(
                            base_rgba, sel, tolerance=tolerance, valid_mask=valid_mask
                        )
                
                result_rgba = base_rgba
                
                # 第四步：添加素材（如果存在）
                if self.watermarks:
                    for wm_data in self.watermarks:
                        watermark_img = wm_data.get("image")
                        if watermark_img is None:
                            continue
                        wm_scale = float(wm_data.get("scale", 1.0))
                        if self.inverse:
                            wm_x = int(wm_data.get("x", 0)) - int(self.rect[0])
                            wm_y = int(wm_data.get("y", 0)) - int(self.rect[1])
                            base_w, base_h = self.rect[2], self.rect[3]
                        else:
                            wm_x = int(wm_data.get("x", 0)) - self.rect[0]
                            wm_y = int(wm_data.get("y", 0)) - self.rect[1]
                            base_w, base_h = cropped_w, cropped_h
                        if base_w <= 0 or base_h <= 0:
                            continue
                        if self.target_size:
                            scale_x = self.target_size[0] / base_w
                            scale_y = self.target_size[1] / base_h
                            adjusted_wm_x = int(wm_x * scale_x)
                            adjusted_wm_y = int(wm_y * scale_y)
                        else:
                            adjusted_wm_x = int(wm_x)
                            adjusted_wm_y = int(wm_y)
                        result_rgba = self.processor.add_watermark(
                            result_rgba, watermark_img,
                            adjusted_wm_x, adjusted_wm_y,
                            wm_scale,
                            float(wm_data.get("angle", 0.0))
                        )
                
                if self.inverse:
                    if ring_mask_a is not None and np.any(ring_mask_a):
                        mask_a = ring_mask_a > 0
                        masked = np.zeros_like(result_rgba)
                        masked[mask_a, :3] = result_rgba[mask_a, :3]
                        masked[mask_a, 3] = result_rgba[mask_a, 3]
                        result_rgba = masked
                    else:
                        result_rgba = np.zeros_like(result_rgba)
                cropped = result_rgba
                
                # 保存
                if self.filename_base:
                    # 静态图片模式：使用文件名基础（如"封面图.png"）
                    filename = f"{self.filename_base}.png"
                    save_path = os.path.join(self.output_dir, filename)
                    # 如果文件已存在，添加序号
                    if os.path.exists(save_path):
                        file_index = 1
                        base_name = self.filename_base
                        while os.path.exists(save_path):
                            filename = f"{base_name}_{file_index}.png"
                            save_path = os.path.join(self.output_dir, filename)
                            file_index += 1
                else:
                    # 动态视频模式：使用帧序号
                    frame_num = i + 1
                    filename = f"{frame_num:05d}.png"
                    save_path = os.path.join(self.output_dir, filename)
                self.processor.save_frame(cropped, save_path, self.max_size_bytes)
                
                # 更新进度
                progress = int((i + 1) / total * 100)
                self.progress.emit(progress)
            
            self.finished.emit(True, f"成功处理 {total} 帧")
        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}"
            print(f"处理线程异常: {error_msg}")
            traceback.print_exc()
            try:
                self.finished.emit(False, error_msg)
            except Exception as emit_error:
                print(f"发送完成信号失败: {emit_error}")
        finally:
            # 确保资源清理
            try:
                # 给一点时间让信号发送完成
                import time
                time.sleep(0.1)
            except Exception:
                pass


class VideoProcessorGUI(QMainWindow):
    """主窗口"""
    
    # 默认像素尺寸
    DEFAULT_SIZES = {
        "气泡图": (720, 264),
        "封面图": (957, 1278),
        "封面故事": (750, 1250),
        "自定义": None
    }
    
    COVER_OFFSET = (48, 114)
    COVER_RECT = (48, 114, 957, 1278)
    RED_PACKET_OFFSET = (48, 114)
    RED_PACKET_SIZE = (957, 1584)
    PENDANT_RECT = (0, 0, 1053, 1746)
    PENDANT_FULL_SIZE = (1053, 1746)
    NON_EDITABLE_OFFSET = (96, 342)  # 相对挂件框左上角的偏移
    NON_EDITABLE_SIZE = (861, 1308)
    WATERMARK_SCOPES = {
        "universal": "封面图 + 挂件（自动沿挂件图框分割）",
        "common": "封面图 / 封面故事 / 气泡图",
        "pendant": "封面图外挂（反选导出）"
    }
    
    def __init__(self, mode="video"):
        super().__init__()
        try:
            self.asset_manager = AssetLibraryManager()
        except Exception as e:
            print(f"初始化素材库失败: {e}")
            self.asset_manager = None
        self.theme = "light"  # 默认使用白色主题
        # 初始化UI缩放器
        self.scaler = UIScaler()
        # 标记是否是第一次进入（用于窗口定位）
        self._is_first_entry = True
        # 使用新的样式表生成函数
        self._initialize_state(mode)
        self._init_main_ui()
        self.apply_theme(self.theme)
        self.should_show_guides = True
        self._updating_rect_from_preset = False
        # 第一次进入时居中显示窗口
        if self._is_first_entry:
            self._center_window()
            self._is_first_entry = False

    def _initialize_state(self, mode):
        """初始化或重置与模式相关的状态数据"""
        self.mode = mode if mode in ("video", "image") else "video"
        self.is_image_mode = self.mode == "image"
        self.media_label = "视频" if self.mode == "video" else "图片"
        self.media_path = None
        self.processor = None
        self.output_base_dir = None
        self.processing_thread = None
        self.rect_locked = False
        # 存储额外背景颜色，每个元素是字典：{"color": (r, g, b), "tolerance": int}
        self.inverse_background_colors = []
        self.default_white_enabled = True
        self.color_pick_active = False
        self.background_tolerance = 30  # 默认阈值，用于新添加的颜色
        self.preview_frames = []
        self.current_preview_index = 0
        self.video_settings = {
            'start_time': 0.0,
            'end_time': 3.0,
            'target_fps': 24,
            'resize_enabled': True,  # 默认开启
            'canvas_width': 1053,
            'canvas_height': 1746,
            'video_scale': 1.0,
            'video_offset_x': 0.0,
            'video_offset_y': 0.0
        }
        self.watermarks = []
        self.active_watermark_index = -1  # 全局索引
        self.max_watermarks = 5  # 每个素材模块最多5张
        self.watermark_scope_mode = "universal"
        self._list_index_map = []
        self._canvas_index_map = []
        self.scope_buttons = {}
        self.bubble_widget = None
        self.bubble_editor_dialog = None
        self.statusBar().showMessage("准备就绪")
        self.should_show_guides = True
        self._updating_rect_from_preset = False
        self._layout_metrics = self._responsive_layout_metrics()
        self.dual_frame_mode = False

    def _responsive_layout_metrics(self):
        """根据可用分辨率动态计算窗口和侧栏尺寸"""
        scaler = self.scaler
        width = scaler.scale(1480)
        height = scaler.scale(920)
        min_width = scaler.scale(1150)
        min_height = scaler.scale(720)
        right_panel = scaler.scale(360)
        screen = QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            safe_width = int(available.width() * 0.92)
            safe_height = int(available.height() * 0.9)
            width = max(scaler.scale(1080), min(width, safe_width))
            height = max(scaler.scale(680), min(height, safe_height))
            min_width = min(max(scaler.scale(960), int(available.width() * 0.7)), width)
            min_height = min(max(scaler.scale(600), int(available.height() * 0.65)), height)
            right_panel = max(
                scaler.scale(320),
                min(int(width * 0.32), scaler.scale(400))
            )
        return {
            "width": int(width),
            "height": int(height),
            "min_width": int(min_width),
            "min_height": int(min_height),
            "right_panel_width": int(right_panel)
        }

    def _cleanup_processing_resources(self):
        """停止线程并释放处理器等资源"""
        if self.processing_thread:
            try:
                if self.processing_thread.isRunning():
                    self.processing_thread.request_stop()
                    self.processing_thread.wait(5000)
            except Exception:
                pass
            try:
                self.processing_thread.deleteLater()
            except Exception:
                pass
            self.processing_thread = None
        if self.processor:
            try:
                self.processor.close()
            except Exception:
                pass
            self.processor = None

    def switch_mode(self, mode):
        """切换制作模式"""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先停止正在进行的处理任务，再切换模式。")
            return
        # 保存当前窗口位置（用于后续切换时保持位置）
        current_pos = self.pos()
        current_theme = self.theme
        self._cleanup_processing_resources()
        self._initialize_state(mode)
        self._init_main_ui()
        self.theme = current_theme
        self.apply_theme(self.theme)
        # 如果不是第一次进入，保持当前位置；否则居中显示
        if self._is_first_entry:
            self._center_window()
            self._is_first_entry = False
        else:
            # 保持当前窗口位置
            self.move(current_pos)

    def _center_window(self):
        """将窗口居中显示在屏幕上"""
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)

    def return_to_mode_selection(self):
        """返回到欢迎页（模式选择对话框）- 简单跳转，无任何检测"""
        # 关闭当前窗口
        self.close()
        
        # 创建并显示欢迎页对话框
        mode_dialog = ModeSelectionDialog()
        if mode_dialog.exec_() == QDialog.Accepted and mode_dialog.selected_mode:
            # 用户选择了模式，创建新的主窗口
            new_window = VideoProcessorGUI(mode=mode_dialog.selected_mode)
            new_window.show()
        else:
            # 用户取消，退出应用
            QApplication.instance().quit()
    
    def switch_mode_direct(self):
        """直接切换模式（在video和image之间切换）"""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(self, "提示", "处理过程中无法切换模式，请先停止处理。")
            return
        # 直接切换模式：如果当前是video，切换到image；如果当前是image，切换到video
        new_mode = "image" if self.mode == "video" else "video"
        if new_mode == self.mode:
            return
        self.switch_mode(new_mode)
    
    def _init_main_ui(self):
        """初始化主界面布局 - 采用侧边栏导航和步骤式工作流"""
        self.setWindowTitle("庆雅神器")
        scaler = self.scaler
        metrics = getattr(self, "_layout_metrics", self._responsive_layout_metrics())
        if not hasattr(self, '_is_first_entry') or self._is_first_entry:
            self.resize(metrics["width"], metrics["height"])
        self.setMinimumSize(metrics["min_width"], metrics["min_height"])
        old_widget = self.takeCentralWidget()
        if old_widget:
            old_widget.deleteLater()

        # 主布局：垂直布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(scaler.scale(8), scaler.scale(8), scaler.scale(8), scaler.scale(8))
        main_layout.setSpacing(scaler.scale(8))
        main_widget.setLayout(main_layout)

        # ========== 顶部工具栏（现代化设计） ==========
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(scaler.scale(56))
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(scaler.scale(20), scaler.scale(12), scaler.scale(20), scaler.scale(12))
        top_bar_layout.setSpacing(scaler.scale(16))
        
        # 左侧：返回欢迎页按钮（使用图标，根据主题加载不同图标）
        self.homepage_btn = QPushButton()
        self.homepage_btn.setObjectName("homepageButton")
        self.homepage_btn.setCursor(Qt.PointingHandCursor)
        self.homepage_btn.setFixedSize(scaler.scale(32), scaler.scale(32))
        self.homepage_btn.setToolTip("返回欢迎页")
        
        # 保存图标尺寸，用于主题切换时更新
        self.homepage_icon_size = scaler.scale(24)
        
        # 加载并设置图标（根据当前主题）
        self._update_homepage_icon()
        
        self.homepage_btn.clicked.connect(self.return_to_mode_selection)
        top_bar_layout.addWidget(self.homepage_btn)
        
        # 添加小间距
        top_bar_layout.addSpacing(scaler.scale(8))
        
        # 模式标签（可点击切换模式）
        mode_badge = QLabel(f"{'动态视频' if self.mode == 'video' else '静态图片'}模式")
        mode_badge.setObjectName("modeBadge")
        mode_badge_font = scaler.get_font(size=13, weight=QFont.Medium)
        mode_badge.setFont(mode_badge_font)
        mode_badge.setCursor(Qt.PointingHandCursor)  # 添加手型光标，提示可点击
        # 添加点击事件，点击模式标签直接切换模式（不返回欢迎页）
        mode_badge.mousePressEvent = lambda e: self.switch_mode_direct()
        top_bar_layout.addWidget(mode_badge)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setObjectName("topBarSeparator")
        top_bar_layout.addWidget(separator)
        
        top_bar_layout.addStretch()
        
        # 右侧：按钮组 - 重新布局，更美观合理
        button_container = QHBoxLayout()
        button_container.setSpacing(scaler.scale(8))
        
        # 基础设置按钮
        self.setup_btn = QPushButton("基础设置")
        self.setup_btn.setObjectName("secondaryButton")
        self.setup_btn.setEnabled(False)
        self.setup_btn.clicked.connect(self.open_setup_dialog)
        button_container.addWidget(self.setup_btn)
        
        # 分隔线
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setObjectName("topBarSeparator")
        button_container.addWidget(separator2)
        
        # 选择文件按钮
        self.select_file_btn = QPushButton(f"选择{self.media_label}")
        self.select_file_btn.setObjectName("primaryButton")
        self.select_file_btn.clicked.connect(self.select_media_file)
        button_container.addWidget(self.select_file_btn)
        
        top_bar_layout.addLayout(button_container)
        
        main_layout.addWidget(top_bar)
        
        # ========== 主内容区：使用QSplitter分为左侧预览、右侧参数 ==========
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(scaler.scale(6))
        main_splitter.setChildrenCollapsible(False)
        
        # 左侧：预览区域
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(scaler.scale(8), scaler.scale(8), scaler.scale(8), scaler.scale(8))
        center_layout.setSpacing(scaler.scale(8))
        
        # 预览标题和信息（简化，基础设置按钮已移到顶部菜单栏）
        preview_header = QHBoxLayout()
        preview_title_label = QLabel("预览区域")
        preview_title_font = scaler.get_font(size=16, weight=QFont.Medium)
        preview_title_label.setFont(preview_title_font)
        preview_title_label.setObjectName("previewTitle")
        preview_header.addWidget(preview_title_label)
        preview_header.addStretch()
        center_layout.addLayout(preview_header)
        
        # 图片/视频信息显示区域（包含文件路径）
        self.info_label = QLabel(f"{self.media_label}信息：未加载")
        self.info_label.setObjectName("mainInfoLabel")
        self.info_label.setWordWrap(True)  # 允许换行显示
        center_layout.addWidget(self.info_label)
        
        # ========== 输出尺寸预设（单独放在预览区域上方，更显眼） ==========
        size_preset_frame = QFrame()
        size_preset_frame.setObjectName("sizePresetFrame")
        size_preset_layout = QHBoxLayout(size_preset_frame)
        size_preset_layout.setContentsMargins(scaler.scale(12), scaler.scale(10), scaler.scale(12), scaler.scale(10))
        size_preset_layout.setSpacing(scaler.scale(12))
        
        size_label = QLabel("输出尺寸预设:")
        size_label_font = scaler.get_font(size=15, weight=QFont.Medium)  # 从13加大到15
        size_label.setFont(size_label_font)
        size_label.setObjectName("sizePresetLabel")
        size_preset_layout.addWidget(size_label)
        
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(self.DEFAULT_SIZES.keys()))
        self.size_combo.setCurrentText("封面图")
        self.size_combo.currentTextChanged.connect(self.on_size_type_changed)
        self.size_combo.setMinimumWidth(scaler.scale(150))
        size_preset_layout.addWidget(self.size_combo)
        
        if self.is_image_mode:
            self.bubble_confirm_btn = QPushButton("同步气泡底图")
            self.bubble_confirm_btn.setObjectName("secondaryButton")
            self.bubble_confirm_btn.clicked.connect(self.confirm_bubble_base_image)
            self.bubble_confirm_btn.setVisible(False)
            size_preset_layout.addWidget(self.bubble_confirm_btn)
            self.bubble_editor_btn = QPushButton("气泡挂件制作")
            self.bubble_editor_btn.setObjectName("secondaryButton")
            self.bubble_editor_btn.clicked.connect(self.open_bubble_editor)
            self.bubble_editor_btn.setVisible(False)
            size_preset_layout.addWidget(self.bubble_editor_btn)
        else:
            self.bubble_confirm_btn = None
            self.bubble_editor_btn = None
        
        size_preset_layout.addStretch()
        center_layout.addWidget(size_preset_frame)
        
        # 预览区域（使用ScrollArea支持滚动）- 优化尺寸，确保画布与线框一致
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_min = min(scaler.scale(420), int(metrics["height"] * 0.55))
        preview_scroll.setMinimumHeight(preview_min)
        preview_scroll.setFrameShape(QFrame.NoFrame)  # 移除ScrollArea的边框
        preview_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(scaler.scale(4))
        preview_layout.setAlignment(Qt.AlignTop)  # 保持与上方白框同宽
        
        # 创建带边框的画布容器
        canvas_frame = QFrame()
        canvas_frame.setObjectName("previewCanvasFrame")
        canvas_frame.setFrameShape(QFrame.Box)
        canvas_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_frame_layout = QVBoxLayout(canvas_frame)
        canvas_frame_layout.setContentsMargins(0, 0, 0, 0)
        canvas_frame_layout.setSpacing(0)
        
        self.image_label = ImageLabel()
        # 移除image_label的尺寸限制，让它填充整个canvas_frame
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.selection_changed.connect(self.on_selection_changed)
        self.image_label.watermark_position_changed.connect(self.on_watermark_position_changed)
        self.image_label.watermark_scale_changed.connect(self.on_watermark_scale_changed)
        self.image_label.watermark_selected.connect(self.on_canvas_watermark_selected)
        self.image_label.color_picked.connect(self.on_background_color_picked)
        canvas_frame_layout.addWidget(self.image_label)
        
        preview_layout.addWidget(canvas_frame, 1)
        
        # 帧导航器（紧凑版）
        if self.mode == "video":
            self.frame_navigator = CompactFrameNavigator()
            self.frame_navigator.frame_changed.connect(self.on_frame_navigator_changed)
            preview_layout.addWidget(self.frame_navigator)
            self.frame_slider = self.frame_navigator.slider
            self.frame_index_label = self.frame_navigator.frame_label
        else:
            self.frame_navigator = None
            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setEnabled(False)
            self.frame_slider.setVisible(False)
            self.frame_index_label = QLabel("")
            self.frame_index_label.setVisible(False)
        
        preview_scroll.setWidget(preview_container)
        center_layout.addWidget(preview_scroll, 1)
        
        main_splitter.addWidget(center_widget)
        
        # 右侧：可折叠的参数面板 - 改为双列卡片式布局
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_panel_width = metrics["right_panel_width"]
        max_right_width = right_panel_width + scaler.scale(24)
        right_scroll.setMinimumWidth(right_panel_width)
        right_scroll.setMaximumWidth(max_right_width)
        right_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(scaler.scale(8), scaler.scale(8), scaler.scale(8), scaler.scale(8))
        right_layout.setSpacing(scaler.scale(10))  # 增加间距，使卡片更清晰
        
        # ========== 使用可折叠面板组织参数 ==========
        
        # 1. 裁切矩形面板 - 优化为双列布局，减少间距
        rect_panel = CollapsiblePanel("裁切矩形")
        rect_widget = QWidget()
        rect_layout = QVBoxLayout(rect_widget)
        rect_layout.setContentsMargins(scaler.scale(4), scaler.scale(4), scaler.scale(4), scaler.scale(4))
        rect_layout.setSpacing(scaler.scale(8))  # 从6增加到8，但使用双列布局
        
        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        self.width_spin = QSpinBox()
        self.height_spin = QSpinBox()
        
        # 双列布局：X和宽度一行，Y和高度一行
        row1 = QHBoxLayout()
        row1.setSpacing(scaler.scale(8))
        x_label = QLabel("X:")
        x_label.setMinimumWidth(scaler.scale(35))  # 固定标签宽度，减少间距
        x_label.setMaximumWidth(scaler.scale(35))
        row1.addWidget(x_label)
        self.x_spin.setRange(0, 99999)
        self.x_spin.valueChanged.connect(self.on_params_changed)
        row1.addWidget(self.x_spin, 1)
        
        width_label = QLabel("宽度:")
        width_label.setMinimumWidth(scaler.scale(40))
        width_label.setMaximumWidth(scaler.scale(40))
        row1.addWidget(width_label)
        self.width_spin.setRange(1, 99999)
        self.width_spin.valueChanged.connect(self.on_params_changed)
        row1.addWidget(self.width_spin, 1)
        rect_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.setSpacing(scaler.scale(8))
        y_label = QLabel("Y:")
        y_label.setMinimumWidth(scaler.scale(35))
        y_label.setMaximumWidth(scaler.scale(35))
        row2.addWidget(y_label)
        self.y_spin.setRange(0, 99999)
        self.y_spin.valueChanged.connect(self.on_params_changed)
        row2.addWidget(self.y_spin, 1)
        
        height_label = QLabel("高度:")
        height_label.setMinimumWidth(scaler.scale(40))
        height_label.setMaximumWidth(scaler.scale(40))
        row2.addWidget(height_label)
        self.height_spin.setRange(1, 99999)
        self.height_spin.valueChanged.connect(self.on_params_changed)
        row2.addWidget(self.height_spin, 1)
        rect_layout.addLayout(row2)
        
        rect_panel.set_content_widget(rect_widget)
        right_layout.addWidget(rect_panel)
        
        # 2. 反选和背景色面板 - 优化布局和位置（移到矩形区域面板之后，更符合使用逻辑）
        # 保存inverse_panel引用，以便后续控制显示/隐藏
        self.inverse_panel = CollapsiblePanel("封面图挂件")
        inverse_widget = QWidget()
        inverse_layout = QVBoxLayout(inverse_widget)
        inverse_layout.setContentsMargins(scaler.scale(4), scaler.scale(4), scaler.scale(4), scaler.scale(4))
        inverse_layout.setSpacing(scaler.scale(8))
        
        self.inverse_check = QCheckBox("反选（保留矩形外区域）")
        self.inverse_check.stateChanged.connect(self.on_inverse_changed)
        inverse_layout.addWidget(self.inverse_check)
        
        color_btn_row = QHBoxLayout()
        self.pick_color_btn = QPushButton("拾取颜色")
        self.pick_color_btn.setObjectName("secondaryButton")
        self.pick_color_btn.setEnabled(False)
        self.pick_color_btn.clicked.connect(self.start_pick_background_color)
        color_btn_row.addWidget(self.pick_color_btn)
        inverse_layout.addLayout(color_btn_row)
        
        # 颜色色块显示区域 - 显示所有颜色（最多5个），均匀分布
        color_swatches_container = QWidget()
        color_swatches_layout = QHBoxLayout(color_swatches_container)
        color_swatches_layout.setContentsMargins(0, 0, 0, 0)
        color_swatches_layout.setSpacing(scaler.scale(8))
        
        # 存储色块按钮的引用
        self.color_swatch_buttons = []
        self.selected_color_index = -1  # -1表示默认白色，>=0表示额外颜色索引
        
        # 创建默认白色色块
        default_swatch = QPushButton()
        default_swatch.setFixedSize(scaler.scale(60), scaler.scale(60))
        default_swatch.setStyleSheet(f"""
            QPushButton {{
                background-color: #FFFFFF;
                border: 3px solid #3e60a9;
                border-radius: {scaler.scale(8)}px;
            }}
            QPushButton:hover {{
                border-color: #5a7fc7;
            }}
        """)
        default_swatch.setCursor(Qt.PointingHandCursor)
        default_swatch.clicked.connect(lambda: self._on_color_swatch_clicked(-1))
        color_swatches_layout.addWidget(default_swatch)
        self.color_swatch_buttons.append(default_swatch)
        
        # 为额外颜色预留位置（最多4个）
        for i in range(4):
            swatch_btn = QPushButton()
            swatch_btn.setFixedSize(scaler.scale(60), scaler.scale(60))
            swatch_btn.setVisible(False)
            swatch_btn.setCursor(Qt.PointingHandCursor)
            swatch_btn.clicked.connect(lambda checked, idx=i: self._on_color_swatch_clicked(idx))
            color_swatches_layout.addWidget(swatch_btn)
            self.color_swatch_buttons.append(swatch_btn)
        
        color_swatches_layout.addStretch()
        inverse_layout.addWidget(color_swatches_container)
        
        # 颜色详细信息显示区域（点击色块后显示）
        self.color_detail_widget = QWidget()
        self.color_detail_layout = QVBoxLayout(self.color_detail_widget)
        self.color_detail_layout.setContentsMargins(0, scaler.scale(8), 0, 0)
        self.color_detail_layout.setSpacing(scaler.scale(8))
        self.color_detail_widget.setVisible(False)
        inverse_layout.addWidget(self.color_detail_widget)
        
        self.inverse_panel.set_content_widget(inverse_widget)
        # 默认隐藏，只有在"封面图"预设时才显示
        self.inverse_panel.setVisible(False)
        right_layout.addWidget(self.inverse_panel)
        
        # 初始化颜色显示（默认选中白色）
        self._update_color_display()
        
        # 3. 素材设置面板 - 优化布局
        watermark_panel = CollapsiblePanel("素材设置")
        watermark_widget = QWidget()
        watermark_layout = QVBoxLayout(watermark_widget)
        watermark_layout.setContentsMargins(scaler.scale(4), scaler.scale(4), scaler.scale(4), scaler.scale(4))
        watermark_layout.setSpacing(scaler.scale(6))
        
        list_label = QLabel(f"已添加素材（每个模块最多{self.max_watermarks}张）")
        list_label.setObjectName("watermarkBadge")
        watermark_layout.addWidget(list_label)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(scaler.scale(8))
        scope_label = QLabel("素材模块：")
        scope_row.addWidget(scope_label)
        self.scope_button_group = QButtonGroup(self)
        self.scope_buttons = {}

        def _build_scope_button(scope_key: str, text: str, default_checked: bool = False):
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setMinimumWidth(scaler.scale(86))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setChecked(default_checked)
            btn.toggled.connect(lambda checked, key=scope_key: checked and self._set_watermark_scope(key))
            self.scope_button_group.addButton(btn)
            self.scope_buttons[scope_key] = btn
            scope_row.addWidget(btn)

        scope_definitions = [
            ("universal", "通用"),
            ("common", "主图"),
            ("pendant", "挂件"),
        ]
        for scope_key, label in scope_definitions:
            _build_scope_button(scope_key, label, scope_key == self.watermark_scope_mode)
        scope_row.addStretch()
        watermark_layout.addLayout(scope_row)

        self.watermark_scope_hint = QLabel("")
        self.watermark_scope_hint.setObjectName("watermarkScopeHint")
        self.watermark_scope_hint.setStyleSheet("color:#64748B;")
        watermark_layout.addWidget(self.watermark_scope_hint)
        
        self.watermark_list = QListWidget()
        self.watermark_list.setFixedHeight(scaler.scale(90))
        self.watermark_list.currentRowChanged.connect(self.on_watermark_list_changed)
        watermark_layout.addWidget(self.watermark_list)
        
        watermark_btn_row = QHBoxLayout()
        self.add_watermark_btn = QPushButton("添加素材")
        self.add_watermark_btn.setObjectName("secondaryButton")
        self.add_watermark_btn.clicked.connect(self.add_watermark)
        watermark_btn_row.addWidget(self.add_watermark_btn)
        self.remove_watermark_btn = QPushButton("删除")
        self.remove_watermark_btn.setObjectName("dangerButton")
        self.remove_watermark_btn.setEnabled(False)
        self.remove_watermark_btn.clicked.connect(self.remove_selected_watermark)
        watermark_btn_row.addWidget(self.remove_watermark_btn)
        for btn in [self.add_watermark_btn, self.remove_watermark_btn]:
            btn.setMinimumHeight(scaler.scale(32))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        watermark_layout.addLayout(watermark_btn_row)
        library_btn_row = QHBoxLayout()
        self.save_to_library_btn = QPushButton("保存到素材库")
        self.save_to_library_btn.setObjectName("secondaryButton")
        self.save_to_library_btn.setEnabled(False)
        self.save_to_library_btn.clicked.connect(self.save_active_watermark_to_library)
        library_btn_row.addWidget(self.save_to_library_btn)
        self.open_library_btn = QPushButton("素材库…")
        self.open_library_btn.clicked.connect(self.open_watermark_library)
        library_btn_row.addWidget(self.open_library_btn)
        for btn in [self.save_to_library_btn, self.open_library_btn]:
            btn.setMinimumHeight(scaler.scale(32))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        watermark_layout.addLayout(library_btn_row)
        self.open_asset_dir_btn = QPushButton("打开素材文件夹")
        self.open_asset_dir_btn.clicked.connect(self.open_asset_directory)
        self.open_asset_dir_btn.setMinimumHeight(scaler.scale(30))
        self.open_asset_dir_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        watermark_layout.addWidget(self.open_asset_dir_btn)
        
        watermark_params = QVBoxLayout()
        watermark_params.setSpacing(scaler.scale(6))
        self.watermark_x_spin = QSpinBox()
        self.watermark_y_spin = QSpinBox()
        self.watermark_scale_spin = QDoubleSpinBox()
        self.watermark_rotation_spin = QSpinBox()
        # 允许素材移动到画布四周外面，使用负数范围
        self.watermark_x_spin.setRange(-9999, 9999)
        self.watermark_y_spin.setRange(-9999, 9999)
        self.watermark_x_spin.valueChanged.connect(self.on_watermark_param_changed)
        self.watermark_y_spin.valueChanged.connect(self.on_watermark_param_changed)
        self.watermark_rotation_spin.setRange(-180, 180)
        self.watermark_rotation_spin.setSingleStep(1)
        self.watermark_rotation_spin.valueChanged.connect(self.on_watermark_param_changed)
        spin_width = scaler.scale(90)
        for spin in [self.watermark_x_spin, self.watermark_y_spin, self.watermark_scale_spin, self.watermark_rotation_spin]:
            spin.setFixedWidth(spin_width)
        self.watermark_scale_spin.setRange(0.01, 5.0)
        self.watermark_scale_spin.setSingleStep(0.01)
        self.watermark_scale_spin.setDecimals(2)
        self.watermark_scale_spin.setValue(1.0)
        self.watermark_scale_spin.setKeyboardTracking(False)
        self.watermark_scale_spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        self.watermark_scale_spin.setCorrectionMode(QDoubleSpinBox.CorrectToNearestValue)
        # 重写keyPressEvent以支持直接输入小数点
        original_keyPressEvent = self.watermark_scale_spin.keyPressEvent
        def custom_keyPressEvent(event):
            if event.key() == Qt.Key_Period or event.key() == Qt.Key_Comma:
                line_edit = self.watermark_scale_spin.lineEdit()
                current_text = line_edit.text()
                cursor_pos = line_edit.cursorPosition()
                if '.' not in current_text:
                    if current_text == "" or current_text == "-":
                        line_edit.setText("0.")
                        line_edit.setCursorPosition(2)
                        return
                    elif cursor_pos == 0:
                        line_edit.setText("0." + current_text)
                        line_edit.setCursorPosition(2)
                        return
                    else:
                        new_text = current_text[:cursor_pos] + "." + current_text[cursor_pos:]
                        line_edit.setText(new_text)
                        line_edit.setCursorPosition(cursor_pos + 1)
                        return
            original_keyPressEvent(event)
        self.watermark_scale_spin.keyPressEvent = custom_keyPressEvent
        self.watermark_scale_spin.valueChanged.connect(self.on_watermark_param_changed)
        coords_layout = QVBoxLayout()
        coords_layout.setSpacing(scaler.scale(4))

        row1 = QHBoxLayout()
        row1.setSpacing(scaler.scale(6))
        x_label = QLabel("X:")
        x_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scale_label = QLabel("缩放:")
        scale_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row1.addWidget(x_label)
        row1.addWidget(self.watermark_x_spin, 1)
        row1.addWidget(scale_label)
        row1.addWidget(self.watermark_scale_spin, 1)

        row2 = QHBoxLayout()
        row2.setSpacing(scaler.scale(6))
        y_label = QLabel("Y:")
        y_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rotation_label = QLabel("旋转:")
        rotation_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row2.addWidget(y_label)
        row2.addWidget(self.watermark_y_spin, 1)
        row2.addWidget(rotation_label)
        row2.addWidget(self.watermark_rotation_spin, 1)

        coords_layout.addLayout(row1)
        coords_layout.addLayout(row2)

        watermark_params.addLayout(coords_layout)
        self.watermark_param_spins = [
            self.watermark_x_spin,
            self.watermark_y_spin,
            self.watermark_scale_spin,
            self.watermark_rotation_spin
        ]
        watermark_layout.addLayout(watermark_params)
        
        watermark_panel.set_content_widget(watermark_widget)
        right_layout.addWidget(watermark_panel)
        
        # 5. 输出设置面板 - 优化布局
        output_panel = CollapsiblePanel("输出设置")
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(scaler.scale(4), scaler.scale(4), scaler.scale(4), scaler.scale(4))
        output_layout.setSpacing(scaler.scale(8))
        
        self.output_dir_label = QLabel("未设置")
        self.output_dir_label.setObjectName("outputBadge")
        self.output_dir_label.setWordWrap(True)
        self.output_dir_label.setTextFormat(Qt.PlainText)
        output_layout.addWidget(self.output_dir_label)
        
        self.select_output_btn = QPushButton("选择输出目录")
        self.select_output_btn.setObjectName("secondaryButton")
        self.select_output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.select_output_btn)
        
        output_panel.set_content_widget(output_widget)
        right_layout.addWidget(output_panel)
        
        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_splitter.addWidget(right_scroll)
        
        # 设置分割器比例
        main_splitter.setStretchFactor(0, 5)  # 左侧区域获得主要伸缩空间
        main_splitter.setStretchFactor(1, 1)
        default_right_width = right_panel_width
        left_width = max(int(metrics["width"] * 0.68), metrics["width"] - default_right_width - scaler.scale(24))
        main_splitter.setSizes([left_width, default_right_width])
        
        main_layout.addWidget(main_splitter, 1)
        
        # ========== 底部操作栏 ==========
        bottom_bar = QFrame()
        bottom_bar.setFixedHeight(scaler.scale(60))
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(scaler.scale(12), scaler.scale(8), scaler.scale(12), scaler.scale(8))
        bottom_layout.setSpacing(scaler.scale(12))
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(scaler.scale(24))
        bottom_layout.addWidget(self.progress_bar)
        
        bottom_layout.addStretch()
        
        # 操作按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        bottom_layout.addWidget(self.process_btn)
        
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        bottom_layout.addWidget(self.stop_btn)
        
        main_layout.addWidget(bottom_bar)
        
        # 页脚和主题切换（左下角）- 修复显示不全问题
        footer_container = QWidget()
        footer_container.setMinimumHeight(scaler.scale(50))  # 确保有足够高度
        footer_layout = QHBoxLayout(footer_container)
        footer_layout.setContentsMargins(scaler.scale(12), scaler.scale(10), scaler.scale(12), scaler.scale(10))  # 增加上下边距
        
        # 左下角：胶囊式主题切换 - 确保完整显示
        self.theme_toggle = ThemeToggleWidget(initial_theme="light" if self.theme == "light" else "dark")
        self.theme_toggle.theme_changed.connect(lambda theme: self.on_theme_changed("白色" if theme == "light" else "黑色"))
        footer_layout.addWidget(self.theme_toggle)
        
        footer_layout.addStretch()
        
        # 页脚文字（居中）
        footer = QLabel("湖南度尚文化创意有限公司  |  版本 v1.5")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFixedHeight(scaler.scale(24))
        footer_layout.addWidget(footer)
        
        footer_layout.addStretch()
        
        main_layout.addWidget(footer_container)

        # 初始化状态
        self.statusBar().showMessage("准备就绪")
        self._refresh_watermark_list()
        self._sync_watermark_param_controls()
        self._update_watermark_scope_hint()
        self._update_color_display()
        self.on_inverse_changed(self.inverse_check.checkState())
        # 确保"反选与背景色"面板的显示状态与当前选择的尺寸预设一致
        if hasattr(self, 'inverse_panel') and hasattr(self, 'size_combo'):
            current_preset = self.size_combo.currentText()
            self.inverse_panel.setVisible(current_preset == "封面图")
        self.apply_size_lock_if_needed()
        self._update_bubble_confirm_visibility()
        

    def _set_watermark_scope(self, scope: str):
        """切换素材模块"""
        if scope not in self.WATERMARK_SCOPES:
            return
        if self.watermark_scope_mode == scope:
            return
        self.watermark_scope_mode = scope
        if getattr(self, "scope_buttons", None):
            for key, btn in self.scope_buttons.items():
                if not btn:
                    continue
                desired = (key == scope)
                if btn.isChecked() != desired:
                    btn.blockSignals(True)
                    btn.setChecked(desired)
                    btn.blockSignals(False)
        self.active_watermark_index = -1
        self._refresh_watermark_list()
        self._sync_watermark_param_controls()
        self.update_watermark_display()
        self._update_watermark_controls_state()
        self._update_watermark_scope_hint()

    def _update_watermark_scope_hint(self):
        """更新模块说明与按钮文案"""
        desc = self.WATERMARK_SCOPES.get(self.watermark_scope_mode, "")
        if hasattr(self, "watermark_scope_hint") and self.watermark_scope_hint:
            self.watermark_scope_hint.setText(f"作用范围：{desc}")
        if hasattr(self, "add_watermark_btn") and self.add_watermark_btn:
            if self.watermark_scope_mode == "pendant":
                self.add_watermark_btn.setText("添加挂件素材")
            else:
                self.add_watermark_btn.setText("添加素材")

    def _normalize_scope_value(self, scope_value: Optional[str]) -> str:
        """保证scope合法，并兼容旧数据"""
        if scope_value in self.WATERMARK_SCOPES:
            return scope_value
        return "common"

    def _get_scope_filtered_overlays(self, scope: Optional[str] = None):
        """返回指定模块下的素材索引与数据"""
        scope = scope or self.watermark_scope_mode
        indices = []
        overlays = []
        for idx, wm in enumerate(self.watermarks):
            if self._normalize_scope_value(wm.get("scope")) == scope:
                indices.append(idx)
                overlays.append(wm)
        return indices, overlays

    def _collect_watermark_items(self, scopes):
        """根据模块收集素材参数"""
        if isinstance(scopes, str):
            target_scopes = {scopes}
        else:
            target_scopes = set(scopes or [])
        if not target_scopes:
            target_scopes = {"common"}
        normalized_targets = set()
        for scope in target_scopes:
            if scope in self.WATERMARK_SCOPES:
                normalized_targets.add(scope)
        if not normalized_targets:
            normalized_targets = {"common"}
        items = []
        for wm in self.watermarks:
            scope_value = self._normalize_scope_value(wm.get("scope"))
            if scope_value not in normalized_targets:
                continue
            image = wm.get("image")
            if image is None:
                continue
            items.append({
                "image": image,
                "x": int(wm.get("x", 0)),
                "y": int(wm.get("y", 0)),
                "scale": float(wm.get("scale", 1.0)),
                "angle": float(wm.get("angle", 0.0))
            })
        return items

    def _get_scope_counts(self):
        counts = {key: 0 for key in self.WATERMARK_SCOPES.keys()}
        for wm in self.watermarks:
            scope = self._normalize_scope_value(wm.get("scope"))
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def _refresh_watermark_list(self):
        """刷新素材列表显示"""
        if not hasattr(self, 'watermark_list'):
            return
        self.watermark_list.blockSignals(True)
        self.watermark_list.clear()
        self._list_index_map = list(range(len(self.watermarks)))
        scope_label_map = {
            "universal": "通用",
            "common": "主图",
            "pendant": "挂件"
        }
        for display_idx, wm in enumerate(self.watermarks):
            name = wm.get("name") or f"素材{display_idx + 1}"
            scope_value = self._normalize_scope_value(wm.get("scope"))
            scope_tag = scope_label_map.get(scope_value, "主图")
            item = QListWidgetItem(f"{display_idx + 1}. [{scope_tag}] {name}")
            item.setData(Qt.UserRole, display_idx)
            self.watermark_list.addItem(item)
        if 0 <= self.active_watermark_index < len(self.watermarks):
            self.watermark_list.setCurrentRow(self.active_watermark_index)
        else:
            self.watermark_list.clearSelection()
        self.watermark_list.blockSignals(False)

    def _sync_watermark_param_controls(self):
        """根据当前选中的素材同步参数输入框"""
        if not hasattr(self, 'watermark_x_spin'):
            return
        if 0 <= self.active_watermark_index < len(self.watermarks):
            wm = self.watermarks[self.active_watermark_index]
            values = (
                int(wm.get("x", 0)),
                int(wm.get("y", 0)),
                float(wm.get("scale", 1.0)),
                int(round(float(wm.get("angle", 0.0))))
            )
        else:
            values = (0, 0, 1.0, 0)
        spins = [
            (self.watermark_x_spin, values[0]),
            (self.watermark_y_spin, values[1]),
            (self.watermark_scale_spin, values[2]),
            (self.watermark_rotation_spin, values[3])
        ]
        for spin, value in spins:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_watermark_controls_state()

    def _update_watermark_controls_state(self):
        """更新素材相关控件启用状态"""
        counts = self._get_scope_counts()
        current_scope_count = counts.get(self.watermark_scope_mode, 0)
        has_selection = 0 <= self.active_watermark_index < len(self.watermarks)
        max_reached = current_scope_count >= self.max_watermarks
        if hasattr(self, 'add_watermark_btn'):
            self.add_watermark_btn.setEnabled(not max_reached)
        if hasattr(self, 'remove_watermark_btn'):
            self.remove_watermark_btn.setEnabled(has_selection)
        if hasattr(self, 'save_to_library_btn'):
            self.save_to_library_btn.setEnabled(has_selection)
        if hasattr(self, 'watermark_param_spins'):
            for spin in self.watermark_param_spins:
                spin.setEnabled(has_selection)

    def on_watermark_list_changed(self, row):
        """列表选中项变化"""
        if row is None or row < 0:
            self.set_active_watermark(-1)
            return
        if row >= len(getattr(self, "_list_index_map", [])):
            self.set_active_watermark(-1)
            return
        global_index = self._list_index_map[row]
        self.set_active_watermark(global_index)

    def on_canvas_watermark_selected(self, index):
        """画布中通过点击选择某个素材"""
        if not hasattr(self, 'watermark_list'):
            return
        if index is None or index < 0:
            return
        if index >= len(getattr(self, "_canvas_index_map", [])):
            return
        global_index = self._canvas_index_map[index]
        if self.watermark_list:
            self.watermark_list.blockSignals(True)
            if global_index < len(self._list_index_map):
                self.watermark_list.setCurrentRow(global_index)
            else:
                self.watermark_list.clearSelection()
            self.watermark_list.blockSignals(False)
        self.set_active_watermark(global_index)

    def set_active_watermark(self, index):
        """设置当前活动水印"""
        if index is None or index < 0 or index >= len(self.watermarks):
            self.active_watermark_index = -1
        else:
            self.active_watermark_index = index
        self._refresh_watermark_list()
        self._sync_watermark_param_controls()
        self.update_watermark_display()

    def clear_all_watermarks(self):
        """清空所有水印"""
        self.watermarks = []
        self.set_active_watermark(-1)
        if hasattr(self.image_label, 'clear_watermarks'):
            self.image_label.clear_watermarks()

    def add_watermark(self):
        """添加新的素材图片"""
        counts = self._get_scope_counts()
        if counts.get(self.watermark_scope_mode, 0) >= self.max_watermarks:
            QMessageBox.information(self, "提示", f"当前素材模块最多只能添加{self.max_watermarks}张图片。")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择素材图片", "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)"
        )
        if not file_path:
            return
        try:
            suffix = Path(file_path).suffix.lower()
            auto_detect = False
            if suffix in WHITE_BACKGROUND_FORMATS:
                reply = QMessageBox.question(
                    self,
                    "白底处理",
                    "检测到非 PNG 图片。\n是否去除白色背景？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                auto_detect = (reply == QMessageBox.Yes)
            np_img, _ = prepare_overlay_image(file_path, auto_detect_white=auto_detect)
            self._add_watermark_from_array(np_img, os.path.basename(file_path), source_path=file_path)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载素材图片失败: {str(e)}")

    def _add_watermark_from_array(self, image_array, display_name: str = None, source_path: str = None):
        if image_array is None:
            return
        counts = self._get_scope_counts()
        if counts.get(self.watermark_scope_mode, 0) >= self.max_watermarks:
            QMessageBox.information(self, "提示", f"当前素材模块最多只能添加{self.max_watermarks}张图片。")
            return
        np_img = np.ascontiguousarray(image_array)
        default_x = 0
        default_y = 0
        if self.processor:
            preview_frame = self.processor.get_first_frame()
            if preview_frame is not None:
                if self.video_settings.get('resize_enabled', False):
                    canvas_height = self.video_settings.get('canvas_height', 1746)
                    canvas_width = self.video_settings.get('canvas_width', 1053)
                    h, w = canvas_height, canvas_width
                else:
                    h, w = preview_frame.shape[:2]
                wm_w = np_img.shape[1]
                default_x = max(0, (w - wm_w) // 2)
                default_y = 0
        watermark = {
            "path": source_path,
            "name": display_name or (os.path.basename(source_path) if source_path else "素材"),
            "image": np_img,
            "x": default_x,
            "y": default_y,
            "scale": 1.0,
            "angle": 0.0,
            "scope": self.watermark_scope_mode
        }
        self.watermarks.append(watermark)
        self.set_active_watermark(len(self.watermarks) - 1)
        self.update_watermark_display()

    def save_active_watermark_to_library(self):
        if not self.asset_manager:
            QMessageBox.warning(self, "提示", "素材库不可用，请检查权限后重试。")
            return
        if self.active_watermark_index < 0 or self.active_watermark_index >= len(self.watermarks):
            QMessageBox.information(self, "提示", "请选择需要保存的素材。")
            return
        wm = self.watermarks[self.active_watermark_index]
        try:
            asset = self.asset_manager.add_from_array(wm.get("image"), wm.get("name"))
            QMessageBox.information(self, "完成", f"已保存到素材库：{asset.get('name')}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存素材失败：{str(e)}")

    def open_asset_library_dialog(self, title: str = "素材库"):
        if not self.asset_manager:
            QMessageBox.warning(self, "提示", "素材库不可用，请检查权限后重试。")
            return None
        dialog = AssetLibraryDialog(self.asset_manager, self, title=title)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_asset
        return None

    def open_watermark_library(self):
        asset = self.open_asset_library_dialog("素材库")
        if not asset:
            return
        try:
            array = self.asset_manager.load_asset_array(asset.get("id"))
            self._add_watermark_from_array(array, asset.get("name"))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"载入素材失败：{str(e)}")

    def open_asset_directory(self):
        if not self.asset_manager:
            QMessageBox.information(self, "提示", "素材库功能不可用。")
            return
        try:
            asset_dir = self.asset_manager.base_dir
            asset_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(asset_dir)))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开素材文件夹：{str(e)}")

    def remove_selected_watermark(self):
        """删除当前选择的素材"""
        if self.active_watermark_index < 0 or self.active_watermark_index >= len(self.watermarks):
            return
        del self.watermarks[self.active_watermark_index]
        if self.watermarks:
            next_index = min(self.active_watermark_index, len(self.watermarks) - 1)
        else:
            next_index = -1
        self.set_active_watermark(next_index)

    def on_watermark_param_changed(self):
        """参数控件变更"""
        if self.active_watermark_index < 0 or self.active_watermark_index >= len(self.watermarks):
            return
        self.watermarks[self.active_watermark_index]["x"] = self.watermark_x_spin.value()
        self.watermarks[self.active_watermark_index]["y"] = self.watermark_y_spin.value()
        new_scale = max(0.01, min(5.0, self.watermark_scale_spin.value()))
        if abs(new_scale - self.watermark_scale_spin.value()) > 1e-4:
            self.watermark_scale_spin.blockSignals(True)
            self.watermark_scale_spin.setValue(new_scale)
            self.watermark_scale_spin.blockSignals(False)
        self.watermarks[self.active_watermark_index]["scale"] = new_scale
        self.watermarks[self.active_watermark_index]["angle"] = self.watermark_rotation_spin.value()
        self.update_watermark_display()
    
    def select_media_file(self):
        """选择素材文件"""
        if self.is_image_mode:
            filters = "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
        else:
            filters = "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"选择{self.media_label}文件", "", filters
        )
        
        if file_path:
            self.media_path = file_path
            # 不再在顶部菜单栏显示文件名，文件路径将显示在info_label中
            self.clear_all_watermarks()
            self.process_btn.setEnabled(False)
            self.setup_btn.setEnabled(False)
            self.load_media()
    
    def load_media(self):
        """加载素材"""
        if not self.media_path:
            return
        
        try:
            self.processor = VideoProcessor(self.media_path, mode=self.mode)
            if not self.processor.load_video():
                QMessageBox.warning(self, "错误", f"无法加载{self.media_label}文件")
                self.processor = None
                self.media_path = None
                self.info_label.setText(f"{self.media_label}信息：未加载")
                return
            
            # 显示素材信息（包含文件路径）
            info = self.processor.get_video_info()
            info_text = (f"分辨率: {info['width']}x{info['height']} | "
                        f"时长: {info['duration']:.2f}秒 | "
                        f"帧率: {info['fps']}fps | "
                        f"总帧数: {info['frame_count']} | "
                        f"宽高比: {info['aspect_ratio']:.2f}")
            # 添加文件路径，换行显示
            file_path_text = f"文件路径: {self.media_path}"
            full_text = f"{self.media_label}信息：{info_text}\n{file_path_text}"
            self.info_label.setText(full_text)
            
            # 弹出第一步设置对话框
            dialog = VideoSetupDialog(self.processor, self, mode=self.mode)
            if dialog.exec_() == QDialog.Accepted:
                # 保存第一步设置
                self.video_settings = dialog.get_settings()
                if self.is_image_mode:
                    self.video_settings['target_fps'] = 1
                # 显示第一帧（应用第一步设置）
                self.update_preview_frame()
                
                # 自动设置输出目录
                media_name = os.path.splitext(os.path.basename(self.media_path))[0]
                default_dir = os.path.join(os.path.dirname(self.media_path), media_name)
                self.output_base_dir = default_dir
                display_text = default_dir
                if len(display_text) > 40:
                    display_text = "..." + display_text[-37:]
                # 更新右侧面板的标签
                if hasattr(self, 'output_dir_label'):
                    self.output_dir_label.setText(display_text)
                    self.output_dir_label.setToolTip(default_dir)
                # 更新顶部菜单栏的标签
                if hasattr(self, 'output_path_label'):
                    self.output_path_label.setText(display_text)
                    self.output_path_label.setToolTip(default_dir)
                
                self.process_btn.setEnabled(True)
                self.setup_btn.setEnabled(True)
            else:
                # 用户取消，重置
                self.processor = None
                self.media_path = None
                self.info_label.setText(f"{self.media_label}信息：未加载")
                self.setup_btn.setEnabled(False)
                self.process_btn.setEnabled(False)
                self.output_base_dir = None
                # 更新右侧面板的标签
                if hasattr(self, 'output_dir_label'):
                    self.output_dir_label.setText("未设置")
                # 更新顶部菜单栏的标签
                if hasattr(self, 'output_path_label'):
                    self.output_path_label.setText("未设置")
                    self.output_path_label.setToolTip("未设置")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载{self.media_label}失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def open_setup_dialog(self):
        """打开第一步设置对话框"""
        if not self.processor:
            return
        
        dialog = VideoSetupDialog(self.processor, self, mode=self.mode)
        # 设置当前值
        dialog.start_time_spin.setValue(self.video_settings['start_time'])
        dialog.end_time_spin.setValue(self.video_settings['end_time'])
        dialog.fps_spin.setValue(self.video_settings['target_fps'])
        dialog.resize_check.setChecked(self.video_settings['resize_enabled'])
        
        # 恢复画布尺寸
        canvas_width = self.video_settings.get('canvas_width', 1053)
        canvas_height = self.video_settings.get('canvas_height', 1746)
        # 查找对应的尺寸文本
        for size_text, (w, h) in dialog.canvas_sizes.items():
            if w == canvas_width and h == canvas_height:
                dialog.canvas_size_combo.setCurrentText(size_text)
                break
        
        dialog.on_resize_check_changed()  # 先启用/禁用控件
        if self.video_settings['resize_enabled']:
            # 恢复画布的变换参数
            scale = self.video_settings.get('video_scale', 1.0)
            offset_x = self.video_settings.get('video_offset_x', 0.0)
            offset_y = self.video_settings.get('video_offset_y', 0.0)
            dialog.canvas_label.set_crop_params(scale, offset_x, offset_y)
            # 更新X、Y坐标输入框
            if hasattr(dialog, 'offset_x_spin'):
                dialog.offset_x_spin.blockSignals(True)
                dialog.offset_x_spin.setValue(int(round(offset_x)))
                dialog.offset_x_spin.blockSignals(False)
            if hasattr(dialog, 'offset_y_spin'):
                dialog.offset_y_spin.blockSignals(True)
                dialog.offset_y_spin.setValue(int(round(offset_y)))
                dialog.offset_y_spin.blockSignals(False)
        
        if dialog.exec_() == QDialog.Accepted:
            # 保存第一步设置
            self.video_settings = dialog.get_settings()
            # 更新预览
            self.update_preview_frame()
    
    def on_selection_changed(self, x, y, width, height):
        """选择区域改变"""
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        if not self.rect_locked:
            self._ensure_custom_size_selection()
        if getattr(self, "inverse_check", None) and self.inverse_check.isChecked():
            self.refresh_preview_display()
    
    def on_watermark_position_changed(self, index, x, y):
        """素材位置改变（从拖动）"""
        if index is None or index < 0:
            return
        if index >= len(getattr(self, "_canvas_index_map", [])):
            return
        global_index = self._canvas_index_map[index]
        if global_index < 0 or global_index >= len(self.watermarks):
            return
        self.watermarks[global_index]["x"] = x
        self.watermarks[global_index]["y"] = y
        if global_index == self.active_watermark_index:
            self.watermark_x_spin.blockSignals(True)
            self.watermark_y_spin.blockSignals(True)
            self.watermark_x_spin.setValue(x)
            self.watermark_y_spin.setValue(y)
            self.watermark_x_spin.blockSignals(False)
            self.watermark_y_spin.blockSignals(False)
    
    def on_watermark_scale_changed(self, index, scale):
        """素材缩放比例通过画布交互改变"""
        if index is None or index < 0:
            return
        if index >= len(getattr(self, "_canvas_index_map", [])):
            return
        global_index = self._canvas_index_map[index]
        if global_index < 0 or global_index >= len(self.watermarks):
            return
        clamped = max(0.01, min(5.0, float(scale)))
        self.watermarks[global_index]["scale"] = clamped
        if global_index == self.active_watermark_index:
            self.watermark_scale_spin.blockSignals(True)
            self.watermark_scale_spin.setValue(round(clamped, 2))
            self.watermark_scale_spin.blockSignals(False)
        self.update_watermark_display()
    
    def on_params_changed(self):
        """参数改变"""
        if self.rect_locked:
            return
        x = self.x_spin.value()
        y = self.y_spin.value()
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        # 更新图像标签的矩形显示
        self.image_label.set_rect_from_params(x, y, width, height)
        self._ensure_custom_size_selection()
        if getattr(self, "inverse_check", None) and self.inverse_check.isChecked():
            self.refresh_preview_display()
    
    def _ensure_custom_size_selection(self):
        """根据当前矩形尺寸自动切换到自定义预设"""
        if not hasattr(self, 'size_combo'):
            return
        if self._updating_rect_from_preset:
            return
        if self.inverse_check.isChecked():
            return
        width = self.width_spin.value()
        height = self.height_spin.value()
        if width <= 0 or height <= 0:
            return
        for name, size in self.DEFAULT_SIZES.items():
            if not size:
                continue
            if width == size[0] and height == size[1]:
                return
        if self.size_combo.currentText() != "自定义":
            self.size_combo.blockSignals(True)
            self.size_combo.setCurrentText("自定义")
            self.size_combo.blockSignals(False)
            self.apply_size_lock_if_needed()
        self._update_bubble_confirm_visibility()
    
    def on_size_type_changed(self, text):
        """尺寸类型改变"""
        size = self.DEFAULT_SIZES.get(text)
        self._updating_rect_from_preset = True
        try:
            if size:
                self.width_spin.blockSignals(True)
                self.height_spin.blockSignals(True)
                self.x_spin.blockSignals(True)
                self.y_spin.blockSignals(True)
                self.width_spin.setValue(size[0])
                self.height_spin.setValue(size[1])
                # 重置位置到左上角
                self.x_spin.setValue(0)
                self.y_spin.setValue(0)
                self.width_spin.blockSignals(False)
                self.height_spin.blockSignals(False)
                self.x_spin.blockSignals(False)
                self.y_spin.blockSignals(False)
                # 直接更新image_label的rect，确保立即显示
                if hasattr(self.image_label, 'set_rect_from_params'):
                    self.image_label.set_rect_from_params(0, 0, size[0], size[1])
                self.on_params_changed()
        finally:
            self._updating_rect_from_preset = False
        
        # 控制"反选与背景色"面板的显示：仅在"封面图"预设时显示
        if hasattr(self, 'inverse_panel'):
            is_cover_preset = text == "封面图"
            self.inverse_panel.setVisible(is_cover_preset)
            # 如果不是"封面图"预设，重置反选状态和相关设置
            if not is_cover_preset:
                if hasattr(self, 'inverse_check') and self.inverse_check.isChecked():
                    self.inverse_check.setChecked(False)
                if hasattr(self, 'inverse_background_colors'):
                    self.inverse_background_colors.clear()
                if hasattr(self, '_update_color_display'):
                    self._update_color_display()
        
        # 强制刷新预览显示
        self.apply_size_lock_if_needed()
        # 使用QTimer延迟刷新，确保所有更新完成
        QTimer.singleShot(50, self.refresh_preview_display)
        self._update_bubble_confirm_visibility()

    def _disable_dimension_inputs(self, size_hint=None):
        """禁用宽高输入框并可选同步尺寸"""
        if not hasattr(self, "width_spin") or not hasattr(self, "height_spin"):
            return
        if size_hint:
            self.width_spin.blockSignals(True)
            self.height_spin.blockSignals(True)
            self.width_spin.setValue(size_hint[0])
            self.height_spin.setValue(size_hint[1])
            self.width_spin.blockSignals(False)
            self.height_spin.blockSignals(False)
        for spin in [self.width_spin, self.height_spin]:
            spin.setEnabled(False)
            spin.setStyleSheet("color: #94A3B8; background-color: #F5F6FA;")

    def _enable_dimension_inputs(self):
        """恢复宽高输入框"""
        if not hasattr(self, "width_spin") or not hasattr(self, "height_spin"):
            return
        for spin in [self.width_spin, self.height_spin]:
            if not self.rect_locked:
                spin.setEnabled(True)
                spin.setStyleSheet("")

    def _prepare_dual_frame_mode(self):
        """设置封面/外挂双框交互"""
        cover_size = self.DEFAULT_SIZES.get("封面图")
        if not cover_size or not hasattr(self, "image_label"):
            return
        self.rect_locked = False
        self.image_label.set_rect_lock(False)
        self._disable_dimension_inputs(cover_size)
        for spin in [self.x_spin, self.y_spin]:
            spin.setEnabled(True)
            spin.setStyleSheet("")
        x = self.x_spin.value() if hasattr(self, "x_spin") else 0
        y = self.y_spin.value() if hasattr(self, "y_spin") else 0
        self.image_label.set_rect_from_params(x, y, cover_size[0], cover_size[1])

    def apply_size_lock_if_needed(self):
        """根据尺寸类型应用矩形锁定"""
        size_type = self.size_combo.currentText()
        show_guides = size_type == "封面图"
        if self.inverse_check.isChecked():
            show_guides = True

        self.should_show_guides = show_guides
        if hasattr(self.image_label, "show_guides"):
            self.image_label.show_guides = show_guides
        if hasattr(self.image_label, "set_guides_visible"):
            self.image_label.set_guides_visible(show_guides)

        # 检查第一步是否启用了整体尺寸裁切
        resize_enabled = getattr(self, 'video_settings', {}).get('resize_enabled', False)
        canvas_width = getattr(self, 'video_settings', {}).get('canvas_width', self.PENDANT_FULL_SIZE[0])
        canvas_height = getattr(self, 'video_settings', {}).get('canvas_height', self.PENDANT_FULL_SIZE[1])
        dual_mode = size_type == "封面图" and not resize_enabled
        self.dual_frame_mode = dual_mode
        cover_size = self.DEFAULT_SIZES.get("封面图", (957, 1278))
        pendant_full = self.PENDANT_FULL_SIZE
        cover_story_size = self.DEFAULT_SIZES.get("封面故事", (750, 1250))
        should_lock_cover_story_canvas = (
            resize_enabled
            and size_type == "封面故事"
            and cover_story_size
            and canvas_width == cover_story_size[0]
            and canvas_height == cover_story_size[1]
        )
        
        if hasattr(self.image_label, "configure_dual_frame_mode"):
            self.image_label.configure_dual_frame_mode(
                dual_mode,
                cover_size=cover_size,
                pendant_size=pendant_full,
                cover_offset=self.COVER_OFFSET,
                red_packet_size=self.RED_PACKET_SIZE,
                red_packet_offset=self.RED_PACKET_OFFSET,
                non_editable_offset=self.NON_EDITABLE_OFFSET,
                non_editable_size=self.NON_EDITABLE_SIZE
            )
        if hasattr(self.image_label, "set_inverse_overlay_active"):
            self.image_label.set_inverse_overlay_active(self.inverse_check.isChecked())
        
        # 控制附件图片显示：第一步启用整体尺寸裁切 + 第二步选择"封面图"
        should_show_overlay = size_type == "封面图"
        if hasattr(self.image_label, "set_overlay_visible"):
            self.image_label.set_overlay_visible(should_show_overlay)
        if hasattr(self.image_label, "set_overlay_offset_adjustment"):
            if should_show_overlay and resize_enabled:
                self.image_label.set_overlay_offset_adjustment(48, 96)
            else:
                self.image_label.set_overlay_offset_adjustment(0, 0)

        if dual_mode:
            self.set_selection_lock(False)
            self._prepare_dual_frame_mode()
            return
        elif self.inverse_check.isChecked():
            cover_w, cover_h = cover_size
            base_x = self.x_spin.value() if hasattr(self, "x_spin") else self.COVER_RECT[0]
            base_y = self.y_spin.value() if hasattr(self, "y_spin") else self.COVER_RECT[1]
            lock_rect = (base_x, base_y, cover_w, cover_h)
            self.set_selection_lock(True, lock_rect)
            for spin in [self.x_spin, self.y_spin, self.width_spin, self.height_spin]:
                spin.setStyleSheet("color: #94A3B8; background-color: #F5F6FA;")
        elif should_lock_cover_story_canvas:
            lock_rect = (0, 0, canvas_width, canvas_height)
            self.set_selection_lock(True, lock_rect)
            for spin in [self.x_spin, self.y_spin, self.width_spin, self.height_spin]:
                spin.setStyleSheet("color: #94A3B8; background-color: #F5F6FA;")
        elif show_guides:
            if resize_enabled:
                self.set_selection_lock(True, self.COVER_RECT)
                for spin in [self.x_spin, self.y_spin, self.width_spin, self.height_spin]:
                    spin.setStyleSheet("color: #94A3B8; background-color: #F5F6FA;")
            else:
                self.set_selection_lock(False)
                self._enable_dimension_inputs()
                for spin in [self.x_spin, self.y_spin]:
                    spin.setStyleSheet("")
                if hasattr(self, 'x_spin') and hasattr(self, 'y_spin') and hasattr(self, 'width_spin') and hasattr(self, 'height_spin'):
                    x = self.x_spin.value()
                    y = self.y_spin.value()
                    width = self.width_spin.value()
                    height = self.height_spin.value()
                    if hasattr(self.image_label, 'set_rect_from_params'):
                        self.image_label.set_rect_from_params(x, y, width, height)
        else:
            self.set_selection_lock(False)
            self._enable_dimension_inputs()
            for spin in [self.x_spin, self.y_spin, self.width_spin, self.height_spin]:
                spin.setStyleSheet("")
            if hasattr(self, 'x_spin') and hasattr(self, 'y_spin') and hasattr(self, 'width_spin') and hasattr(self, 'height_spin'):
                x = self.x_spin.value()
                y = self.y_spin.value()
                width = self.width_spin.value()
                height = self.height_spin.value()
                if hasattr(self.image_label, 'set_rect_from_params'):
                    self.image_label.set_rect_from_params(x, y, width, height)

    def _compute_layout_rects(self):
        """计算封面、红包框、挂件及不可编辑框矩形"""
        if self.dual_frame_mode and self.size_combo.currentText() == "封面图":
            cover_x = self.x_spin.value() if hasattr(self, "x_spin") else 0
            cover_y = self.y_spin.value() if hasattr(self, "y_spin") else 0
            cover_rect = (cover_x, cover_y, self.COVER_RECT[2], self.COVER_RECT[3])
            outer_x = max(0, cover_x - self.COVER_OFFSET[0])
            outer_y = max(0, cover_y - self.COVER_OFFSET[1])
            pendant_rect = (outer_x, outer_y, self.PENDANT_FULL_SIZE[0], self.PENDANT_FULL_SIZE[1])
            red_rect = (
                outer_x + self.RED_PACKET_OFFSET[0],
                outer_y + self.RED_PACKET_OFFSET[1],
                self.RED_PACKET_SIZE[0],
                self.RED_PACKET_SIZE[1]
            )
        else:
            video_settings = getattr(self, "video_settings", {}) or {}
            canvas_w = video_settings.get('canvas_width', self.PENDANT_FULL_SIZE[0])
            canvas_h = video_settings.get('canvas_height', self.PENDANT_FULL_SIZE[1])
            cover_x = self.x_spin.value() if hasattr(self, "x_spin") else self.COVER_RECT[0]
            cover_y = self.y_spin.value() if hasattr(self, "y_spin") else self.COVER_RECT[1]
            cover_w = max(1, self.width_spin.value()) if hasattr(self, "width_spin") else self.COVER_RECT[2]
            cover_h = max(1, self.height_spin.value()) if hasattr(self, "height_spin") else self.COVER_RECT[3]
            max_cover_x = max(0, canvas_w - cover_w)
            max_cover_y = max(0, canvas_h - cover_h)
            cover_x = max(0, min(cover_x, max_cover_x))
            cover_y = max(0, min(cover_y, max_cover_y))
            cover_rect = (cover_x, cover_y, cover_w, cover_h)
            pendant_w, pendant_h = self.PENDANT_FULL_SIZE
            pendant_x = cover_x - self.COVER_OFFSET[0]
            pendant_y = cover_y - self.COVER_OFFSET[1]
            max_pendant_x = max(0, canvas_w - pendant_w)
            max_pendant_y = max(0, canvas_h - pendant_h)
            pendant_x = max(0, min(pendant_x, max_pendant_x))
            pendant_y = max(0, min(pendant_y, max_pendant_y))
            pendant_rect = (pendant_x, pendant_y, pendant_w, pendant_h)
            red_rect = (
                pendant_x + self.RED_PACKET_OFFSET[0],
                pendant_y + self.RED_PACKET_OFFSET[1],
                self.RED_PACKET_SIZE[0],
                self.RED_PACKET_SIZE[1]
            )
        non_editable_rect = (
            pendant_rect[0] + self.NON_EDITABLE_OFFSET[0],
            pendant_rect[1] + self.NON_EDITABLE_OFFSET[1],
            self.NON_EDITABLE_SIZE[0],
            self.NON_EDITABLE_SIZE[1]
        )
        return cover_rect, red_rect, pendant_rect, non_editable_rect
    
    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_base_dir = dir_path
            # 如果路径太长，显示省略号
            display_text = dir_path
            if len(display_text) > 40:
                display_text = "..." + display_text[-37:]
            # 更新右侧面板的标签
            if hasattr(self, 'output_dir_label'):
                self.output_dir_label.setText(display_text)
                self.output_dir_label.setToolTip(dir_path)
            # 更新顶部菜单栏的标签
            if hasattr(self, 'output_path_label'):
                self.output_path_label.setText(display_text)
                self.output_path_label.setToolTip(dir_path)
    
    def update_watermark_display(self):
        """更新素材在画布中的显示"""
        if not hasattr(self, 'image_label'):
            return
        self._canvas_index_map = list(range(len(self.watermarks)))
        if not self.watermarks:
            self.image_label.clear_watermarks()
            return
        items = []
        for wm in self.watermarks:
            items.append({
                "image": wm.get("image"),
                "x": wm.get("x", 0),
                "y": wm.get("y", 0),
                "scale": wm.get("scale", 1.0),
                "name": wm.get("name", ""),
                "angle": wm.get("angle", 0.0)
            })
        active_row = self.active_watermark_index if 0 <= self.active_watermark_index < len(self.watermarks) else -1
        self.image_label.set_watermarks(items, active_row)
    
    def update_preview_frame(self):
        """更新预览帧（应用第一步设置：整体尺寸裁切）"""
        if not self.processor:
            return
        
        try:
            self.load_preview_frames()
            if not self.preview_frames:
                return
            self.display_preview_frame(self.current_preview_index, reconfigure_limits=True)
            self.apply_size_lock_if_needed()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新预览失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_preview_frames(self):
        """加载用于预览的全部关键帧"""
        if not self.processor or not getattr(self, 'video_settings', None):
            self.preview_frames = []
            self.frame_slider.setEnabled(False)
            # 更新帧导航器
            if hasattr(self, 'frame_navigator') and self.frame_navigator:
                self.frame_navigator.set_max_frames(0)
            elif hasattr(self, 'frame_slider') and self.frame_slider:
                self.frame_slider.setRange(0, 0)
            self.frame_index_label.setText("0 / 0")
            return

        start_time = self.video_settings['start_time']
        end_time = self.video_settings['end_time']
        target_fps = self.video_settings['target_fps']
        duration = max(0.0, end_time - start_time)

        effective_duration = duration if duration > 0 else (1.0 / target_fps if target_fps > 0 else 1.0)

        frames = self.processor.extract_frames(
            max_duration=effective_duration,
            target_fps=target_fps,
            start_time=start_time,
            end_time=end_time
        )

        processed_frames = []
        resize_enabled = self.video_settings.get('resize_enabled', False)
        canvas_width = self.video_settings.get('canvas_width', 1053)
        canvas_height = self.video_settings.get('canvas_height', 1746)
        scale = self.video_settings.get('video_scale', 1.0)
        offset_x = self.video_settings.get('video_offset_x', 0.0)
        offset_y = self.video_settings.get('video_offset_y', 0.0)
        
        if frames:
            if resize_enabled:
                for frame in frames:
                    canvas_frame = self.processor.crop_frame_canvas(
                        frame, canvas_width, canvas_height, scale, offset_x, offset_y
                    )
                    processed_frames.append(np.ascontiguousarray(canvas_frame))
            else:
                processed_frames = [np.ascontiguousarray(frame) for frame in frames]
        else:
            fallback = self.processor.get_first_frame()
            if fallback is not None:
                if resize_enabled:
                    fallback = self.processor.crop_frame_canvas(
                        fallback, canvas_width, canvas_height, scale, offset_x, offset_y
                    )
                processed_frames = [np.ascontiguousarray(fallback)]

        self.preview_frames = processed_frames
        total = len(self.preview_frames)
        self.current_preview_index = 0 if total else 0

        self.frame_slider.blockSignals(True)
        if total:
            self.frame_slider.setEnabled(True)
            # 更新帧导航器
            if hasattr(self, 'frame_navigator') and self.frame_navigator:
                self.frame_navigator.set_max_frames(total)
                self.frame_navigator.set_current_frame(0)
            elif hasattr(self, 'frame_slider') and self.frame_slider:
                self.frame_slider.setRange(0, total - 1)
                self.frame_slider.setValue(0)
        else:
            self.frame_slider.setEnabled(False)
            # 更新帧导航器
            if hasattr(self, 'frame_navigator') and self.frame_navigator:
                self.frame_navigator.set_max_frames(0)
            elif hasattr(self, 'frame_slider') and self.frame_slider:
                self.frame_slider.setRange(0, 0)
            self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.update_frame_indicator()
    
    def display_preview_frame(self, index: int, reconfigure_limits: bool = False):
        """显示指定索引的预览帧"""
        if not self.preview_frames:
            return
        index = max(0, min(index, len(self.preview_frames) - 1))
        # 更新帧导航器（如果使用新的紧凑导航器）
        if hasattr(self, 'frame_navigator') and self.frame_navigator:
            self.frame_navigator.set_current_frame(index)
        else:
            # 兼容旧的frame_slider
            if hasattr(self, 'frame_slider') and self.frame_slider:
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(index)
                self.frame_slider.blockSignals(False)
        self.current_preview_index = index
        frame = self.preview_frames[index]

        if reconfigure_limits:
            self._configure_parameter_limits(frame)

        self.image_label.set_image(frame)

        h, w = frame.shape[:2]
        # 允许素材移动到画布四周外面，不限制最大值
        # 移除最大值限制，允许负数坐标

        self.update_watermark_display()
        self.refresh_preview_display()
        self.update_frame_indicator()
    
    def _configure_parameter_limits(self, frame):
        """根据当前帧尺寸更新相关输入范围"""
        h, w = frame.shape[:2]
        self.width_spin.setMaximum(max(1, w))
        self.height_spin.setMaximum(max(1, h))
        self.x_spin.setMaximum(max(0, w - 1))
        self.y_spin.setMaximum(max(0, h - 1))

    def update_frame_indicator(self):
        total = len(self.preview_frames)
        # 更新帧导航器（如果使用新的紧凑导航器）
        if hasattr(self, 'frame_navigator') and self.frame_navigator:
            self.frame_navigator.set_max_frames(total)
            self.frame_navigator.set_current_frame(self.current_preview_index)
        else:
            # 兼容旧的frame_index_label
            if hasattr(self, 'frame_index_label') and self.frame_index_label:
                if total == 0:
                    self.frame_index_label.setText("0 / 0")
                else:
                    self.frame_index_label.setText(f"{self.current_preview_index + 1} / {total}")

    def refresh_preview_display(self):
        """根据反选和背景设置刷新预览显示"""
        if not self.processor or not self.preview_frames or self.image_label.original_image is None:
            return

        if not self.inverse_check.isChecked():
            self.image_label.set_display_override(None)
            self.image_label.update()
            return

        frame = self.preview_frames[self.current_preview_index]
        colors_to_remove = []
        if getattr(self, "default_white_enabled", True):
            colors_to_remove.append((255, 255, 255))
        color_tolerances = {}  # 存储每种颜色对应的阈值
        
        # 兼容新旧格式
        for sel in self.inverse_background_colors:
            if isinstance(sel, dict):
                color = sel["color"]
                tolerance = sel.get("tolerance", self.background_tolerance)
            else:
                color = sel
                tolerance = self.background_tolerance
            if color and tuple(color) not in colors_to_remove:
                colors_to_remove.append(tuple(color))
                color_tolerances[tuple(color)] = tolerance

        try:
            resize_enabled = self.video_settings.get('resize_enabled', False) if self.video_settings else False
            canvas_w = self.video_settings.get('canvas_width', frame.shape[1]) if self.video_settings else frame.shape[1]
            canvas_h = self.video_settings.get('canvas_height', frame.shape[0]) if self.video_settings else frame.shape[0]
            if not resize_enabled:
                canvas_w = frame.shape[1]
                canvas_h = frame.shape[0]
            scale_x = frame.shape[1] / canvas_w if canvas_w else 1.0
            scale_y = frame.shape[0] / canvas_h if canvas_h else 1.0
            cover_rect, red_rect, pendant_rect, non_edit_rect = self._compute_layout_rects()

            def scale_rect(rect):
                if rect is None:
                    return None
                if resize_enabled:
                    return (
                        int(round(rect[0] * scale_x)),
                        int(round(rect[1] * scale_y)),
                        int(round(rect[2] * scale_x)),
                        int(round(rect[3] * scale_y)),
                    )
                return rect

            red_scaled = scale_rect(red_rect)
            pendant_scaled = scale_rect(pendant_rect)
            non_edit_scaled = scale_rect(non_edit_rect)

            h, w = frame.shape[:2]
            processed = np.zeros((h, w, 4), dtype=np.uint8)
            base_rgb = frame if frame.dtype == np.uint8 else frame.astype(np.uint8)
            processed[:, :, :3] = base_rgb
            processed[:, :, 3] = 255

            a_ring_mask = np.zeros((h, w), dtype=np.uint8)
            b_ring_mask = np.zeros((h, w), dtype=np.uint8)

            def apply_ring(mask_arr, outer, inner, value=1):
                if not outer:
                    return
                x1 = max(0, outer[0])
                y1 = max(0, outer[1])
                x2 = min(w, x1 + outer[2])
                y2 = min(h, y1 + outer[3])
                if x1 >= x2 or y1 >= y2:
                    return
                mask_arr[y1:y2, x1:x2] = value
                if inner:
                    ix1 = max(0, inner[0])
                    iy1 = max(0, inner[1])
                    ix2 = min(w, ix1 + inner[2])
                    iy2 = min(h, iy1 + inner[3])
                    if ix1 < ix2 and iy1 < iy2:
                        mask_arr[iy1:iy2, ix1:ix2] = 0

            apply_ring(a_ring_mask, pendant_scaled, red_scaled, value=1)
            apply_ring(b_ring_mask, red_scaled, non_edit_scaled, value=1)

            if pendant_scaled:
                ox = max(0, pendant_scaled[0])
                oy = max(0, pendant_scaled[1])
                ox2 = min(w, ox + pendant_scaled[2])
                oy2 = min(h, oy + pendant_scaled[3])
                if ox < ox2 and oy < oy2:
                    roi = processed[oy:oy2, ox:ox2].copy()
                    roi_mask = np.ones((oy2 - oy, ox2 - ox), dtype=np.uint8) * 255
                    if red_scaled:
                        ix = max(0, red_scaled[0] - ox)
                        iy = max(0, red_scaled[1] - oy)
                        ix2 = min(roi_mask.shape[1], ix + red_scaled[2])
                        iy2 = min(roi_mask.shape[0], iy + red_scaled[3])
                        if ix < ix2 and iy < iy2:
                            roi_mask[iy:iy2, ix:ix2] = 0
                    for color in colors_to_remove:
                        tolerance = color_tolerances.get(tuple(color), self.background_tolerance)
                        roi = self.processor.remove_background_color(
                            roi, color, tolerance=tolerance, valid_mask=roi_mask
                        )
                    processed[oy:oy2, ox:ox2] = roi
            else:
                mask = ((a_ring_mask + b_ring_mask) > 0).astype(np.uint8) * 255
                for color in colors_to_remove:
                    tolerance = color_tolerances.get(tuple(color), self.background_tolerance)
                    processed = self.processor.remove_background_color(
                        processed, color, tolerance=tolerance, valid_mask=mask
                    )
            self.image_label.set_display_override(np.ascontiguousarray(processed))
        except Exception:
            self.image_label.set_display_override(None)

    def on_frame_slider_changed(self, value):
        if not self.preview_frames:
            return
        if value < 0 or value >= len(self.preview_frames):
            return
        self.display_preview_frame(value)
        self.apply_size_lock_if_needed()
    
    def on_frame_navigator_changed(self, frame_index):
        """紧凑帧导航器的帧变化回调"""
        if not self.preview_frames:
            return
        if frame_index < 0 or frame_index >= len(self.preview_frames):
            return
        self.display_preview_frame(frame_index)
        self.apply_size_lock_if_needed()
    
    def start_processing(self):
        """开始处理"""
        try:
            if not self.processor or not self.output_base_dir:
                QMessageBox.warning(self, "警告", f"请先选择{self.media_label}文件和输出目录")
                return
            
            if self.color_pick_active:
                self.image_label.enable_color_pick(False)
                self.color_pick_active = False
                self.pick_color_btn.setText("拾取挂件背景颜色")
                self.pick_color_btn.setEnabled(self.inverse_check.isChecked())
                self.statusBar().clearMessage()
            
            # 获取参数
            x = self.x_spin.value()
            y = self.y_spin.value()
            width = self.width_spin.value()
            height = self.height_spin.value()
            
            if width <= 0 or height <= 0:
                QMessageBox.warning(self, "警告", "请设置有效的裁切区域")
                return
            
            # 反选选项
            inverse = self.inverse_check.isChecked()
            
            # 获取目标尺寸，根据尺寸确定文件夹名称和文件名
            size_type = self.size_combo.currentText()
            selection_rect = (x, y, width, height)
            inverse_inner_rect = None
            inverse_overlay_outer_rect = None
            inverse_overlay_inner_rect = None
            rect_for_processing = selection_rect
            if inverse:
                cover_rect, red_rect, pendant_rect, non_editable_rect = self._compute_layout_rects()
                rect_for_processing = pendant_rect
                width, height = pendant_rect[2], pendant_rect[3]

                def to_local(rect):
                    if not rect:
                        return None
                    return (
                        rect[0] - pendant_rect[0],
                        rect[1] - pendant_rect[1],
                        rect[2],
                        rect[3]
                    )

                local_red = to_local(red_rect)
                local_non_edit = to_local(non_editable_rect)
                inverse_inner_rect = local_red
                inverse_overlay_outer_rect = local_red
                inverse_overlay_inner_rect = local_non_edit
            elif self.dual_frame_mode and size_type == "封面图":
                cover_size = self.DEFAULT_SIZES.get("封面图", (957, 1278))
                width, height = cover_size
            if size_type == "自定义":
                target_size = (width, height)
            else:
                target_size = self.DEFAULT_SIZES.get(size_type, self.DEFAULT_SIZES.get("封面图"))
            
            # 如果反选，使用"封面图外挂"文件名
            if inverse:
                filename_base = "封面图外挂"
                target_size = (1053, 1746)
            else:
                filename_base = size_type
            
            # 静态图片模式：所有图片放在同一文件夹，不需要二级文件夹
            if self.is_image_mode:
                # 获取上传的图片名称（不含扩展名）作为文件夹名
                if self.media_path:
                    media_name = os.path.splitext(os.path.basename(self.media_path))[0]
                else:
                    media_name = "未命名图片"
                output_dir = os.path.join(self.output_base_dir, media_name)
                os.makedirs(output_dir, exist_ok=True)
            else:
                # 动态视频模式：按原来的逻辑，每个尺寸类型一个文件夹
                folder_name = filename_base
                output_dir = os.path.join(self.output_base_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
                filename_base = None  # 动态模式使用时间戳命名
            
            # 计算大小限制
            if inverse:
                max_size_bytes = 300 * 1024
            elif size_type == "封面图":
                max_size_bytes = 500 * 1024
            elif size_type == "封面故事":
                max_size_bytes = 300 * 1024
            else:
                max_size_bytes = 300 * 1024
            
            # 获取时间参数（从第一步设置）
            start_time = self.video_settings['start_time']
            end_time = self.video_settings['end_time']
            target_fps = self.video_settings['target_fps']
            
            # 获取整体尺寸裁切选项（从第一步设置）
            resize_crop = None
            if self.video_settings['resize_enabled']:
                canvas_width = self.video_settings.get('canvas_width', 1053)
                canvas_height = self.video_settings.get('canvas_height', 1746)
                scale = self.video_settings.get('video_scale', 1.0)
                offset_x = self.video_settings.get('video_offset_x', 0.0)
                offset_y = self.video_settings.get('video_offset_y', 0.0)
                resize_crop = (canvas_width, canvas_height, scale, offset_x, offset_y)
            
            # 获取素材参数
            if inverse:
                watermark_scopes = ("pendant", "universal")
            else:
                watermark_scopes = ("common",)
                if size_type == "封面图":
                    watermark_scopes = ("common", "universal")
            watermark_items = self._collect_watermark_items(watermark_scopes)
            
            # 禁用按钮
            self.process_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 获取画布尺寸（用于素材处理）
            canvas_size = None
            if inverse:
                canvas_size = (rect_for_processing[2], rect_for_processing[3])
            elif self.video_settings['resize_enabled']:
                canvas_width = self.video_settings.get('canvas_width', 1053)
                canvas_height = self.video_settings.get('canvas_height', 1746)
                canvas_size = (canvas_width, canvas_height)
            # 提取颜色值（兼容新旧格式）
            background_colors = []
            if inverse:
                for item in self.inverse_background_colors:
                    if isinstance(item, dict):
                        background_colors.append(item["color"])
                    else:
                        background_colors.append(item)
            default_white_enabled = getattr(self, "default_white_enabled", True)
            
            # 先确保之前的线程已清理（必须在创建新线程之前）
            if self.processing_thread is not None:
                try:
                    if self.processing_thread.isRunning():
                        self.processing_thread.request_stop()
                        self.processing_thread.wait(3000)  # 等待最多3秒
                    try:
                        self.processing_thread.progress.disconnect()
                    except Exception:
                        pass
                    try:
                        self.processing_thread.finished.disconnect()
                    except Exception:
                        pass
                    self.processing_thread.deleteLater()
                except Exception as e:
                    print(f"清理旧线程时出错: {e}")
                finally:
                    self.processing_thread = None
            
            # 创建处理线程
            try:
                self.processing_thread = ProcessingThread(
                    self.processor, output_dir, rect_for_processing, 
                    inverse, target_size, start_time, end_time, target_fps,
                    resize_crop, watermark_items, canvas_size,
                    background_colors, max_size_bytes, self.background_tolerance,
                    inverse_inner_rect=inverse_inner_rect,
                    inverse_overlay_outer_rect=inverse_overlay_outer_rect,
                    inverse_overlay_inner_rect=inverse_overlay_inner_rect,
                    filename_base=filename_base if self.is_image_mode else None,
                    default_white_enabled=default_white_enabled
                )
                
                # 连接信号
                self.processing_thread.progress.connect(self.progress_bar.setValue)
                self.processing_thread.finished.connect(self.on_processing_finished)
                
                # 启动线程
                self.processing_thread.start()
            except Exception as e:
                import traceback
                error_msg = f"启动处理线程失败: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                # 恢复UI状态
                self.process_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "错误", error_msg)
                self.processing_thread = None
        except Exception as e:
            import traceback
            error_msg = f"启动处理时发生未知错误: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            # 恢复UI状态
            try:
                self.process_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "错误", error_msg)
            except Exception:
                pass
            self.processing_thread = None
    
    def stop_processing(self):
        """请求停止处理"""
        if self.processing_thread is not None:
            try:
                self.processing_thread.request_stop()
                self.statusBar().showMessage("正在停止...", 2000)
            except Exception:
                pass
    
    def on_processing_finished(self, success, message):
        """处理完成"""
        try:
            # 先等待线程完全结束
            if self.processing_thread is not None:
                try:
                    # 等待线程结束，最多等待5秒
                    if self.processing_thread.isRunning():
                        self.processing_thread.wait(5000)
                except Exception as e:
                    print(f"等待线程结束时的错误: {e}")
            
            # 更新UI
            self.progress_bar.setVisible(False)
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            # 断开信号连接，避免重复调用
            if self.processing_thread is not None:
                try:
                    self.processing_thread.progress.disconnect()
                    self.processing_thread.finished.disconnect()
                except Exception:
                    pass  # 如果已经断开，忽略错误
            
            # 清理线程对象
            old_thread = self.processing_thread
            self.processing_thread = None
            
            # 删除线程对象（延迟删除，确保信号处理完成）
            if old_thread is not None:
                try:
                    old_thread.deleteLater()
                except Exception:
                    pass
            
            # 显示消息（使用QTimer延迟，避免在信号处理中直接显示对话框导致问题）
            def show_message():
                try:
                    if success:
                        QMessageBox.information(self, "完成", message)
                    else:
                        QMessageBox.warning(self, "错误", message)
                except Exception as e:
                    print(f"显示消息框时出错: {e}")
                    # 如果消息框显示失败，至少更新状态栏
                    self.statusBar().showMessage(message, 5000)
            
            # 延迟显示消息，确保线程完全清理
            QTimer.singleShot(100, show_message)
            
        except Exception as e:
            import traceback
            print(f"处理完成回调中发生错误: {e}")
            traceback.print_exc()
            # 即使出错也要恢复UI状态
            try:
                self.progress_bar.setVisible(False)
                self.process_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.statusBar().showMessage(f"处理完成，但发生错误: {str(e)}", 5000)
            except Exception:
                pass

    def _update_color_display(self):
        """更新背景颜色显示 - 使用色块点击式布局"""
        scaler = self.scaler
        
        # 更新默认白色色块的选中状态
        if self.selected_color_index == -1:
            self.color_swatch_buttons[0].setStyleSheet(f"""
                QPushButton {{
                    background-color: #FFFFFF;
                    border: 3px solid #3e60a9;
                    border-radius: {scaler.scale(8)}px;
                }}
                QPushButton:hover {{
                    border-color: #5a7fc7;
                }}
            """)
        else:
            self.color_swatch_buttons[0].setStyleSheet(f"""
                QPushButton {{
                    background-color: #FFFFFF;
                    border: 2px solid #D0D7E2;
                    border-radius: {scaler.scale(8)}px;
                }}
                QPushButton:hover {{
                    border-color: #3e60a9;
                }}
            """)
        
        # 更新额外颜色色块
        num_colors = len(self.inverse_background_colors)
        for i in range(4):  # 最多4个额外颜色
            swatch_btn = self.color_swatch_buttons[i + 1]
            
            if i < num_colors:
                # 显示这个色块
                swatch_btn.setVisible(True)
                # 兼容旧格式（如果是元组）和新格式（如果是字典）
                color_data = self.inverse_background_colors[i]
                if isinstance(color_data, dict):
                    r, g, b = color_data["color"]
                else:
                    # 旧格式，转换为新格式
                    r, g, b = color_data
                    tolerance = self.background_tolerance
                    self.inverse_background_colors[i] = {"color": (r, g, b), "tolerance": tolerance}
                
                # 设置颜色和选中状态
                if self.selected_color_index == i:
                    swatch_btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: rgb({r}, {g}, {b});
                            border: 3px solid #3e60a9;
                            border-radius: {scaler.scale(8)}px;
                        }}
                        QPushButton:hover {{
                            border-color: #5a7fc7;
                        }}
                    """)
                else:
                    swatch_btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: rgb({r}, {g}, {b});
                            border: 2px solid #D0D7E2;
                            border-radius: {scaler.scale(8)}px;
                        }}
                        QPushButton:hover {{
                            border-color: #3e60a9;
                        }}
                    """)
            else:
                # 隐藏这个色块
                swatch_btn.setVisible(False)
        
        # 更新详细信息显示
        self._update_color_detail_display()
        
        # 更新按钮状态
        inverse_enabled = self.inverse_check.isChecked() if hasattr(self, 'inverse_check') else False
        if inverse_enabled and not self.color_pick_active:
            self.pick_color_btn.setEnabled(len(self.inverse_background_colors) < 4)  # 最多4个额外颜色
    
    def _on_color_swatch_clicked(self, index):
        """色块被点击"""
        self.selected_color_index = index
        self._update_color_display()
    
    def _handle_default_white_toggle(self, state):
        enabled = state == Qt.Checked
        if self.default_white_enabled != enabled:
            self.default_white_enabled = enabled
            # 记录状态后无需额外处理，导出时自动应用
    
    def _clear_layout(self, layout: QLayout):
        """递归清空布局内容，避免遗留控件遮挡"""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget:
                widget.deleteLater()
            if child_layout:
                self._clear_layout(child_layout)

    def _update_color_detail_display(self):
        """更新颜色详细信息显示"""
        scaler = self.scaler
        # 清除现有内容
        self._clear_layout(self.color_detail_layout)
        
        if self.selected_color_index == -1:
            # 显示默认白色的详细信息
            self.color_detail_widget.setVisible(True)
            
            note_label = QLabel("默认白色将自动抠除，您可随时关闭该效果。")
            note_label.setStyleSheet("font-size: 11px; color: #666666; padding: 4px;")
            self.color_detail_layout.addWidget(note_label)
            
            checkbox = QCheckBox("启用默认白色抠除")
            checkbox.setChecked(self.default_white_enabled)
            checkbox.stateChanged.connect(self._handle_default_white_toggle)
            self.color_detail_layout.addWidget(checkbox)
            
        elif 0 <= self.selected_color_index < len(self.inverse_background_colors):
            # 显示额外颜色的详细信息
            self.color_detail_widget.setVisible(True)
            
            color_data = self.inverse_background_colors[self.selected_color_index]
            if isinstance(color_data, dict):
                r, g, b = color_data["color"]
                tolerance = color_data.get("tolerance", self.background_tolerance)
            else:
                # 旧格式，转换为新格式
                r, g, b = color_data
                tolerance = self.background_tolerance
                self.inverse_background_colors[self.selected_color_index] = {"color": (r, g, b), "tolerance": tolerance}
            
            # 阈值控制
            tolerance_layout = QHBoxLayout()
            tolerance_layout.setSpacing(scaler.scale(8))
            tolerance_label = QLabel("阈值:")
            tolerance_label.setStyleSheet("font-size: 12px; color: #666666;")
            tolerance_layout.addWidget(tolerance_label)
            
            tolerance_spin = QSpinBox()
            tolerance_spin.setRange(0, 255)
            tolerance_spin.setValue(tolerance)
            tolerance_spin.setMinimumWidth(scaler.scale(80))
            tolerance_spin.valueChanged.connect(lambda v: self._on_color_tolerance_changed(self.selected_color_index, v))
            tolerance_layout.addWidget(tolerance_spin)
            tolerance_layout.addStretch()
            self.color_detail_layout.addLayout(tolerance_layout)
            
            # 删除按钮
            delete_btn = QPushButton("删除此颜色")
            delete_btn.setObjectName("dangerButton")
            delete_btn.setFixedHeight(scaler.scale(32))
            delete_btn.clicked.connect(lambda: self._remove_color_item(self.selected_color_index))
            self.color_detail_layout.addWidget(delete_btn)
        else:
            # 没有选中任何颜色，隐藏详细信息
            self.color_detail_widget.setVisible(False)
    
    
    def _on_color_tolerance_changed(self, index, value):
        """颜色阈值改变"""
        if 0 <= index < len(self.inverse_background_colors):
            if isinstance(self.inverse_background_colors[index], dict):
                self.inverse_background_colors[index]["tolerance"] = value
            else:
                # 旧格式，转换为新格式
                r, g, b = self.inverse_background_colors[index]
                self.inverse_background_colors[index] = {"color": (r, g, b), "tolerance": value}
            self.refresh_preview_display()
    
    def _remove_color_item(self, index):
        """删除单个颜色项"""
        if 0 <= index < len(self.inverse_background_colors):
            # 如果删除的是当前选中的颜色，重置选中状态
            if self.selected_color_index == index:
                self.selected_color_index = -1  # 默认选中白色
            del self.inverse_background_colors[index]
            self._update_color_display()
            self.refresh_preview_display()

    def _update_bubble_confirm_visibility(self):
        visible = self.is_image_mode and self.size_combo.currentText() == "气泡图"
        if hasattr(self, "bubble_confirm_btn") and self.bubble_confirm_btn is not None:
            self.bubble_confirm_btn.setVisible(visible)
        if hasattr(self, "bubble_editor_btn") and self.bubble_editor_btn is not None:
            self.bubble_editor_btn.setVisible(visible)

    def confirm_bubble_base_image(self):
        """将当前预览图同步为气泡挂件底图"""
        if not self.is_image_mode or self.size_combo.currentText() != "气泡图":
            QMessageBox.information(self, "提示", "仅在静态图片模式且输出尺寸为“气泡图”时可使用此功能。")
            return
        bubble_widget = self._ensure_bubble_widget()
        if self.processor is None or getattr(self.image_label, "original_image", None) is None:
            QMessageBox.warning(self, "提示", "请先加载素材并完成预览。")
            return
        frame = np.ascontiguousarray(self.image_label.original_image)
        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            QMessageBox.warning(self, "提示", "当前预览图无效，无法生成气泡底图。")
            return
        x = max(0, min(self.x_spin.value(), frame_w - 1))
        y = max(0, min(self.y_spin.value(), frame_h - 1))
        width = max(1, self.width_spin.value())
        height = max(1, self.height_spin.value())
        width = min(width, frame_w - x)
        height = min(height, frame_h - y)
        if width <= 0 or height <= 0:
            QMessageBox.warning(self, "提示", "裁切范围超出可用区域，无法同步气泡图。")
            return
        cropped = frame[y:y+height, x:x+width].copy()
        target_size = self.DEFAULT_SIZES.get("气泡图")
        if target_size and (cropped.shape[1] != target_size[0] or cropped.shape[0] != target_size[1]):
            if hasattr(self.processor, "resize_frame"):
                cropped = self.processor.resize_frame(cropped, target_size[0], target_size[1])
            else:
                from PIL import Image
                cropped = np.array(Image.fromarray(cropped).resize((target_size[0], target_size[1]), Image.Resampling.LANCZOS))
        bubble_widget.set_base_image_from_array(cropped, "当前预览气泡图")
        QMessageBox.information(self, "完成", "已将当前预览气泡图同步至气泡挂件底图。")

    def _ensure_bubble_widget(self):
        if self.bubble_widget is None:
            self.bubble_widget = BubblePendantWidget(self, getattr(self, "asset_manager", None))
        return self.bubble_widget

    def _ensure_bubble_editor_dialog(self):
        widget = self._ensure_bubble_widget()
        if self.bubble_editor_dialog is None:
            self.bubble_editor_dialog = QDialog(self)
            self.bubble_editor_dialog.setWindowTitle("气泡挂件制作")
            self.bubble_editor_dialog.setMinimumSize(1120, 720)
            # 禁用回车键关闭对话框
            original_keyPressEvent = self.bubble_editor_dialog.keyPressEvent
            def custom_keyPressEvent(event):
                if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                    # 忽略回车键，不关闭对话框
                    event.ignore()
                    return
                original_keyPressEvent(event)
            self.bubble_editor_dialog.keyPressEvent = custom_keyPressEvent
            layout = QVBoxLayout(self.bubble_editor_dialog)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.addWidget(widget)
        return self.bubble_editor_dialog

    def open_bubble_editor(self):
        if not self.is_image_mode:
            QMessageBox.information(self, "提示", "仅静态图片模式支持气泡挂件制作。")
            return
        dialog = self._ensure_bubble_editor_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def on_inverse_changed(self, state):
        """反选选项改变"""
        enabled = state == Qt.Checked
        self.pick_color_btn.setEnabled(enabled and len(self.inverse_background_colors) < 5)
        if not enabled:
            if self.color_pick_active:
                self.image_label.enable_color_pick(False)
                self.color_pick_active = False
            self.inverse_background_colors = []
            self.pick_color_btn.setText("拾取挂件背景颜色")
            self._update_color_display()
            self.image_label.set_display_override(None)
        else:
            self._update_color_display()
            self.pick_color_btn.setEnabled(len(self.inverse_background_colors) < 5)
            if self.size_combo.currentText() != "封面图":
                self.size_combo.blockSignals(True)
                self.size_combo.setCurrentText("封面图")
                self.size_combo.blockSignals(False)
        self.apply_size_lock_if_needed()
        self.refresh_preview_display()
    
    def start_pick_background_color(self):
        """启动背景颜色拾取"""
        if self.image_label.original_image is None:
            QMessageBox.information(self, "提示", f"请先加载{self.media_label}并显示预览图像。")
            return
        if len(self.inverse_background_colors) >= 5:  # 增加到5种颜色
            QMessageBox.information(self, "提示", "最多选择5种额外背景颜色。")
            return
        self.color_pick_active = True
        self.image_label.enable_color_pick(True)
        self.pick_color_btn.setEnabled(False)
        self.pick_color_btn.setText("请在预览中点击颜色")
        self.statusBar().showMessage("请在预览图中点击要扣除的背景颜色", 3000)
    
    def on_background_color_picked(self, r, g, b):
        """颜色拾取完成"""
        picked = (int(r), int(g), int(b))
        
        # 检查是否已存在该颜色
        existing_colors = []
        for item in self.inverse_background_colors:
            if isinstance(item, dict):
                existing_colors.append(item["color"])
            else:
                existing_colors.append(item)
        
        if picked == (255, 255, 255):
            QMessageBox.information(self, "提示", "白色已默认扣除，无需重复选择。")
            # 选中默认白色
            self.selected_color_index = -1
        elif picked not in existing_colors:
            if len(self.inverse_background_colors) < 4:  # 最多4个额外颜色
                # 使用新格式存储
                self.inverse_background_colors.append({
                    "color": picked,
                    "tolerance": self.background_tolerance
                })
                # 自动选中新拾取的颜色
                self.selected_color_index = len(self.inverse_background_colors) - 1
            else:
                QMessageBox.information(self, "提示", "最多选择4种额外背景颜色（加上默认白色共5种）。")
        else:
            QMessageBox.information(self, "提示", "该颜色已存在。")
            # 如果颜色已存在，选中该颜色
            for idx, color_data in enumerate(self.inverse_background_colors):
                if isinstance(color_data, dict):
                    if color_data["color"] == picked:
                        self.selected_color_index = idx
                        break
                elif color_data == picked:
                    self.selected_color_index = idx
                    break
        
        self._update_color_display()
        self.color_pick_active = False
        if self.inverse_check.isChecked() and len(self.inverse_background_colors) < 4:
            self.pick_color_btn.setEnabled(True)
        else:
            self.pick_color_btn.setEnabled(False)
        self.pick_color_btn.setText("拾取挂件背景颜色")
        self.image_label.enable_color_pick(False)
        self.refresh_preview_display()

    def set_selection_lock(self, locked: bool, rect=None):
        """设置矩形选择是否锁定"""
        self.rect_locked = locked
        self.image_label.set_rect_lock(locked)
        for spin in [self.x_spin, self.y_spin, self.width_spin, self.height_spin]:
            spin.setEnabled(not locked)
        if locked and rect:
            x, y, width, height = rect
            self.x_spin.blockSignals(True)
            self.y_spin.blockSignals(True)
            self.width_spin.blockSignals(True)
            self.height_spin.blockSignals(True)
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
            self.x_spin.blockSignals(False)
            self.y_spin.blockSignals(False)
            self.width_spin.blockSignals(False)
            self.height_spin.blockSignals(False)
            self.image_label.set_rect_from_params(x, y, width, height)
        if not locked:
            # 解锁时允许重新绘制
            self.image_label.update()
            for spin in [self.x_spin, self.y_spin, self.width_spin, self.height_spin]:
                if not spin.isEnabled():
                    spin.setEnabled(True)
                    spin.setStyleSheet("")

    def apply_theme(self, theme: str):
        """应用界面主题"""
        theme = "light" if theme not in ("light", "dark") else theme
        self.theme = theme
        # 使用新的样式表生成函数，支持DPI缩放
        style = get_stylesheet(theme, self.scaler.get_scale_factor())
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(style)
        # 更新主题下拉框文本
        if hasattr(self, 'theme_combo'):
            block = self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText("白色" if theme == "light" else "黑色")
            self.theme_combo.blockSignals(block)
        self._update_color_display()
        # 更新CanvasLabel等组件的样式
        self._update_canvas_styles()
        # 更新返回欢迎页图标（根据主题加载不同图标文件）
        self._update_homepage_icon()
    
    def _update_homepage_icon(self):
        """更新返回欢迎页图标（根据主题加载不同的图标文件）"""
        if not hasattr(self, 'homepage_btn') or not self.homepage_btn:
            return
        
        # 根据主题选择对应的图标文件
        if self.theme == "dark":
            # 黑色主题：使用黑.png
            icon_filename = "黑.png"
        else:
            # 白色主题：使用白.png
            icon_filename = "白.png"
        
        # 尝试多个路径查找图标
        icon_paths = [
            os.path.join(os.path.dirname(__file__), "resources", icon_filename),
            os.path.join(os.path.dirname(__file__), icon_filename),  # 兼容旧路径
        ]
        if hasattr(sys, '_MEIPASS'):
            icon_paths.insert(0, os.path.join(sys._MEIPASS, "resources", icon_filename))
            icon_paths.insert(1, os.path.join(sys._MEIPASS, icon_filename))  # 兼容旧路径
        
        icon_path = None
        for path in icon_paths:
            if os.path.exists(path):
                icon_path = path
                break
        
        # 如果图标文件不存在，使用文字按钮作为后备
        if not icon_path or not os.path.exists(icon_path):
            self.homepage_btn.setText("返回")
            self.homepage_btn.setIcon(QIcon())  # 清除图标
            return
        
        # 加载图标
        pixmap = QPixmap(icon_path)
        icon_size = self.homepage_icon_size if hasattr(self, 'homepage_icon_size') else self.scaler.scale(24)
        
        # 缩放图标
        if pixmap.width() > pixmap.height():
            pixmap = pixmap.scaledToWidth(icon_size, Qt.SmoothTransformation)
        else:
            pixmap = pixmap.scaledToHeight(icon_size, Qt.SmoothTransformation)
        
        # 设置图标
        icon = QIcon(pixmap)
        self.homepage_btn.setIcon(icon)
        self.homepage_btn.setText("")  # 清除文字
        self.homepage_btn.setIconSize(QSize(icon_size, icon_size))
    
    def _update_canvas_styles(self):
        """更新CanvasLabel等组件的样式以支持主题"""
        colors = ThemeColors.get_colors(self.theme)
        scaler = self.scaler
        
        # 更新CanvasLabel样式和主题
        if hasattr(self, 'canvas_label') and self.canvas_label:
            border_color = colors['border_primary']
            bg_color = colors['bg_tertiary']
            border_width = scaler.scale(2)
            self.canvas_label.setStyleSheet(
                f"border: {border_width}px solid {border_color}; "
                f"background-color: {bg_color}; "
                f"border-radius: {scaler.scale(4)}px;"
            )
            if hasattr(self.canvas_label, 'set_theme'):
                self.canvas_label.set_theme(self.theme)
        
        # 更新ImageLabel样式
        if hasattr(self, 'image_label') and self.image_label:
            border_color = colors['border_primary']
            bg_color = colors['bg_tertiary']
            border_width = scaler.scale(2)
            self.image_label.setStyleSheet(
                f"border: {border_width}px solid {border_color}; "
                f"background-color: {bg_color}; "
                f"border-radius: {scaler.scale(4)}px;"
            )
            if hasattr(self.image_label, 'set_theme'):
                self.image_label.set_theme(self.theme)
        
        # 更新CompactFrameNavigator主题
        if hasattr(self, 'frame_navigator') and self.frame_navigator:
            if hasattr(self.frame_navigator, 'set_theme'):
                self.frame_navigator.set_theme(self.theme)
        
        # 更新ThemeToggleWidget主题
        if hasattr(self, 'theme_toggle'):
            self.theme_toggle.set_theme(self.theme)
        
        # 更新VideoSetupDialog中的canvas_label
        # 这个会在对话框打开时自动应用主题
    
    def on_theme_changed(self, text: str):
        """主题选择变化"""
        if text == "白色":
            theme = "light"
        else:
            theme = "dark"
        self.apply_theme(theme)
        # 更新主题切换组件状态
        if hasattr(self, 'theme_toggle'):
            self.theme_toggle.set_theme(theme)


def main():
    """主函数"""
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # 设置应用程序字体（使用缩放后的字体）
    scaler = UIScaler()
    app_font = scaler.get_font()
    app.setFont(app_font)
    
    mode_dialog = ModeSelectionDialog()
    if mode_dialog.exec_() != QDialog.Accepted or not mode_dialog.selected_mode:
        sys.exit(0)
    window = VideoProcessorGUI(mode=mode_dialog.selected_mode)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

