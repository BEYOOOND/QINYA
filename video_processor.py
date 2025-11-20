"""
视频处理核心模块
负责视频读取、转图片、裁切等功能
"""
import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, List
from io import BytesIO
from collections import deque


class VideoProcessor:
    """媒体处理器类（支持视频与图片）"""
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(self, video_path: str = None, mode: str = "video"):
        """
        初始化视频处理器
        
        Args:
            video_path: 视频文件路径（可选，用于独立使用save_frame时可以为None）
            mode: 处理模式，"video" 或 "image"
        """
        self.video_path = video_path
        self.mode = mode if mode in ("video", "image") else "video"
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        self.single_image = None
    
    def _should_treat_as_image(self) -> bool:
        if self.mode == "image":
            return True
        if self.video_path is None:
            return False
        _, ext = os.path.splitext(self.video_path.lower())
        return ext in self.IMAGE_EXTENSIONS
    
    def _load_image_source(self) -> bool:
        """加载静态图片作为单帧源"""
        try:
            img = Image.open(self.video_path)
            # 确保转换为RGB格式（处理RGBA、P模式等）
            if img.mode != 'RGB':
                img = img.convert("RGB")
            # 确保图像数据是连续的，并且数据类型正确
            img_array = np.array(img, dtype=np.uint8)
            # 确保是连续数组，避免内存对齐问题
            self.single_image = np.ascontiguousarray(img_array)
            # 验证数组形状
            if len(self.single_image.shape) != 3 or self.single_image.shape[2] != 3:
                raise ValueError(f"图片格式不正确: shape={self.single_image.shape}, 期望(height, width, 3)")
            self.height, self.width = self.single_image.shape[:2]
            self.fps = 1
            self.frame_count = 1
            self.duration = 1.0
            print(f"成功加载图片: {self.width}x{self.height}, shape={self.single_image.shape}, dtype={self.single_image.dtype}")
            return True
        except Exception as e:
            print(f"加载图片失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def load_video(self) -> bool:
        """
        加载视频文件并获取基本信息
        
        Returns:
            bool: 是否成功加载
        """
        try:
            if self._should_treat_as_image():
                return self._load_image_source()
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                return False
            
            # 获取视频属性
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            
            return True
        except Exception as e:
            print(f"加载视频失败: {e}")
            return False
    
    def get_video_info(self) -> dict:
        """
        获取视频信息
        
        Returns:
            dict: 包含视频信息的字典
        """
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'aspect_ratio': self.width / self.height if self.height > 0 else 0
        }
    
    def get_first_frame(self) -> Optional[np.ndarray]:
        """
        获取视频第一帧
        
        Returns:
            np.ndarray: 第一帧图像，失败返回None
        """
        if self.single_image is not None:
            return self.single_image.copy()
        
        if self.cap is None:
            if not self.load_video():
                return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
    
    def extract_frames(self, max_duration: float = 3.0, target_fps: int = 24, 
                      start_time: float = 0.0, end_time: Optional[float] = None) -> List[np.ndarray]:
        """
        提取视频帧（限制时长和帧率，支持指定时间范围）
        
        Args:
            max_duration: 最大时长（秒），如果end_time未指定则使用此参数
            target_fps: 目标帧率
            start_time: 起始时间（秒）
            end_time: 结束时间（秒），如果为None则使用start_time + max_duration
            
        Returns:
            List[np.ndarray]: 提取的帧列表
        """
        if self.single_image is not None:
            return [self.single_image.copy()]
        
        if self.cap is None:
            if not self.load_video():
                return []
        
        # 计算实际结束时间
        if end_time is None:
            end_time = start_time + max_duration
        end_time = min(end_time, self.duration)
        
        # 计算起始和结束帧号
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        total_frames = end_frame - start_frame
        
        # 计算目标帧数和帧间隔
        duration = end_time - start_time
        max_frames = int(duration * target_fps)
        frame_interval = max(1, int(self.fps / target_fps)) if self.fps > 0 else 1
        
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        extracted_count = 0
        
        while frame_index < end_frame and extracted_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if (frame_index - start_frame) % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
            
            frame_index += 1
        
        return frames
    
    def crop_frame(self, frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        裁切单帧图像
        
        Args:
            frame: 原始帧
            x: 起始x坐标
            y: 起始y坐标
            width: 裁切宽度
            height: 裁切高度
            
        Returns:
            np.ndarray: 裁切后的图像
        """
        h, w = frame.shape[:2]
        # 确保坐标在有效范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return frame[y:y+height, x:x+width]
    
    def crop_frame_inverse(self, frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        反选裁切：保留矩形区域外的部分，矩形内区域删除形成透明底
        
        Args:
            frame: 原始帧（RGB格式）
            x: 矩形起始x坐标
            y: 矩形起始y坐标
            width: 矩形宽度
            height: 矩形高度
            
        Returns:
            np.ndarray: 反选后的图像（RGBA格式，矩形内透明）
        """
        h, w = frame.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = min(width, w - x)
        height = min(height, h - y)
        
        # 转换为RGBA格式
        if frame.shape[2] == 3:
            result = np.zeros((h, w, 4), dtype=np.uint8)
            result[:, :, :3] = frame
            result[:, :, 3] = 255  # 完全不透明
        else:
            result = frame.copy()
        
        # 将矩形区域设为透明
        if width > 0 and height > 0:
            result[y:y+height, x:x+width, 3] = 0  # Alpha通道设为0（透明）
        
        return result
    
    def crop_frame_ring(self, frame: np.ndarray, outer_rect, inner_rect):
        """
        生成封面挂件环形区域（保留外框与内框之间的区域）
        返回带透明通道的图像以及环形区域掩码
        """
        h, w = frame.shape[:2]
        outer_x = max(0, min(int(outer_rect[0]), w - 1))
        outer_y = max(0, min(int(outer_rect[1]), h - 1))
        outer_w = max(1, min(int(outer_rect[2]), w - outer_x))
        outer_h = max(1, min(int(outer_rect[3]), h - outer_y))
        outer_crop = frame[outer_y:outer_y + outer_h, outer_x:outer_x + outer_w]
        result = np.zeros((outer_h, outer_w, 4), dtype=np.uint8)
        result[:, :, :3] = outer_crop
        result[:, :, 3] = 255
        mask = np.ones((outer_h, outer_w), dtype=np.uint8)
        if inner_rect:
            inner_x = int(inner_rect[0] - outer_x)
            inner_y = int(inner_rect[1] - outer_y)
            inner_w = int(inner_rect[2])
            inner_h = int(inner_rect[3])
            inner_x = max(0, min(inner_x, outer_w))
            inner_y = max(0, min(inner_y, outer_h))
            inner_end_x = max(inner_x, min(inner_x + inner_w, outer_w))
            inner_end_y = max(inner_y, min(inner_y + inner_h, outer_h))
            if inner_x < inner_end_x and inner_y < inner_end_y:
                mask[inner_y:inner_end_y, inner_x:inner_end_x] = 0
                result[inner_y:inner_end_y, inner_x:inner_end_x, 3] = 0
        return np.ascontiguousarray(result), np.ascontiguousarray(mask)
    
    def crop_frame_canvas(self, frame: np.ndarray, canvas_width: int, canvas_height: int,
                         scale: float, offset_x: float, offset_y: float) -> np.ndarray:
        """
        基于画布的裁切：将视频缩放到指定比例，放置在画布上，然后裁切出画布区域
        
        Args:
            frame: 原始帧（RGB格式）
            canvas_width: 画布宽度（1053）
            canvas_height: 画布高度（1746）
            scale: 视频缩放比例
            offset_x: 视频在画布上的X偏移（画布坐标系）
            offset_y: 视频在画布上的Y偏移（画布坐标系）
            
        Returns:
            np.ndarray: 裁切后的图像（RGB格式，尺寸为canvas_width×canvas_height）
        """
        from PIL import Image
        
        # 获取原始图像尺寸
        orig_h, orig_w = frame.shape[:2]
        
        # 缩放视频
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        
        # 使用PIL缩放
        # 确保输入图像格式正确
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        
        pil_img = Image.fromarray(frame, 'RGB')
        scaled_img = pil_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        scaled_frame = np.array(scaled_img, dtype=np.uint8)
        scaled_frame = np.ascontiguousarray(scaled_frame)
        
        # 创建画布（白色背景）
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # 计算视频在画布上的位置
        video_canvas_x = int(offset_x)
        video_canvas_y = int(offset_y)
        
        # 计算视频在画布内的可见区域
        visible_start_x = max(0, video_canvas_x)
        visible_start_y = max(0, video_canvas_y)
        visible_end_x = min(canvas_width, video_canvas_x + scaled_w)
        visible_end_y = min(canvas_height, video_canvas_y + scaled_h)
        
        # 计算视频图像中对应的区域
        offset_in_video_x = visible_start_x - video_canvas_x
        offset_in_video_y = visible_start_y - video_canvas_y
        visible_w = visible_end_x - visible_start_x
        visible_h = visible_end_y - visible_start_y
        
        if visible_w > 0 and visible_h > 0:
            # 确保不超出缩放后的视频范围
            offset_in_video_x = max(0, min(offset_in_video_x, scaled_w - 1))
            offset_in_video_y = max(0, min(offset_in_video_y, scaled_h - 1))
            visible_w = min(visible_w, scaled_w - offset_in_video_x)
            visible_h = min(visible_h, scaled_h - offset_in_video_y)
            
            # 将视频的可见部分复制到画布上
            video_region = scaled_frame[
                offset_in_video_y:offset_in_video_y + visible_h,
                offset_in_video_x:offset_in_video_x + visible_w
            ]
            canvas[
                visible_start_y:visible_start_y + visible_h,
                visible_start_x:visible_start_x + visible_w
            ] = video_region
        
        # 确保返回的数组格式正确（RGB格式，uint8类型，连续数组）
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        return canvas
    
    def resize_frame(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        调整帧大小到目标尺寸
        
        Args:
            frame: 原始帧
            target_width: 目标宽度
            target_height: 目标高度
            
        Returns:
            np.ndarray: 调整后的图像
        """
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    def save_frame(self, frame: np.ndarray, save_path: str, max_size_bytes: int = 300 * 1024) -> bool:
        """
        保存帧为PNG图片（支持RGBA透明通道）
        自动压缩图片，确保单张图片大小不超过指定阈值，但保持原始像素尺寸不变
        设置DPI为300
        
        Args:
            frame: 要保存的帧（RGB或RGBA格式）
            save_path: 保存路径
            max_size_bytes: 文件大小上限（字节）
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 构造初始图像对象
            has_alpha = frame.shape[2] == 4
            if has_alpha:
                img = Image.fromarray(frame, 'RGBA')
            else:
                img = Image.fromarray(frame, 'RGB')

            def save_png_bytes(pil_img: Image.Image) -> bytes:
                buffer = BytesIO()
                save_kwargs = {
                    'format': 'PNG',
                    'optimize': True,
                    'compress_level': 9,
                    'dpi': (300, 300)
                }
                if pil_img.mode == 'P':
                    transparency = pil_img.info.get('transparency')
                    if transparency is not None:
                        save_kwargs['transparency'] = transparency
                pil_img.save(buffer, **save_kwargs)
                return buffer.getvalue()

            # 初始保存（原始尺寸）
            best_image = img
            best_data = save_png_bytes(img)

            # 若超过限制，尝试进行颜色量化压缩
            if len(best_data) > max_size_bytes:
                if has_alpha:
                    base_image = img.convert('RGBA')
                else:
                    base_image = img.convert('RGB')

                palette_sizes = [256, 224, 192, 160, 144, 128, 112, 96, 80, 72, 64, 56, 48, 40, 32, 24, 20, 16, 12, 8, 6, 4, 3, 2]
                quant_methods = [Image.Quantize.FASTOCTREE, Image.Quantize.MEDIANCUT]

                for method in quant_methods:
                    for colors in palette_sizes:
                        try:
                            if has_alpha:
                                quantized = base_image.quantize(colors=colors, method=method, dither=Image.Dither.NONE)
                            else:
                                quantized = base_image.convert('P', palette=Image.ADAPTIVE, colors=colors, dither=Image.Dither.NONE)

                            candidate_data = save_png_bytes(quantized)
                            if len(candidate_data) < len(best_data):
                                best_data = candidate_data
                                best_image = quantized
                            if len(candidate_data) <= max_size_bytes:
                                break
                        except Exception:
                            continue
                    if len(best_data) <= max_size_bytes:
                        break

            # 写入最终文件
            with open(save_path, 'wb') as f:
                f.write(best_data)
            return True
        except Exception as e:
            print(f"保存图片失败: {e}")
            return False
    
    def add_watermark(self, frame: np.ndarray, watermark: np.ndarray, 
                     x: int, y: int, scale: float = 1.0, angle: float = 0.0) -> np.ndarray:
        """
        在帧上添加水印
        
        Args:
            frame: 原始帧（RGB或RGBA格式）
            watermark: 水印图像（RGB或RGBA格式）
            x: 水印X坐标
            y: 水印Y坐标
            scale: 水印缩放比例
            
        Returns:
            np.ndarray: 添加水印后的图像
        """
        if frame is None or watermark is None:
            return frame

        # 确保frame是RGBA格式
        if frame.shape[2] == 3:
            result = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = frame
            result[:, :, 3] = 255
        else:
            result = frame.copy()
        
        watermark = np.ascontiguousarray(watermark)
        if watermark.ndim != 3 or watermark.shape[2] < 3:
            raise ValueError("watermark图像格式不正确，必须至少包含RGB通道")
        
        if watermark.shape[2] == 3:
            alpha_channel = np.full((watermark.shape[0], watermark.shape[1], 1), 255, dtype=np.uint8)
            wm_rgba = np.concatenate([watermark, alpha_channel], axis=2)
        else:
            wm_rgba = watermark.copy()
        wm_rgba = np.ascontiguousarray(wm_rgba)

        from PIL import Image
        if scale != 1.0 or abs(angle) >= 1e-3:
            wm_img = Image.fromarray(wm_rgba, 'RGBA')
            if scale != 1.0:
                new_w = max(1, int(round(wm_img.width * scale)))
                new_h = max(1, int(round(wm_img.height * scale)))
                wm_img = wm_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if abs(angle) >= 1e-3:
                orig_w, orig_h = wm_img.width, wm_img.height
                wm_img = wm_img.rotate(
                    angle,
                    expand=True,
                    resample=Image.Resampling.BICUBIC,
                    fillcolor=(0, 0, 0, 0)
                )
                delta_x = (orig_w - wm_img.width) / 2.0
                delta_y = (orig_h - wm_img.height) / 2.0
                x += int(round(delta_x))
                y += int(round(delta_y))
            wm_rgba = np.array(wm_img, dtype=np.uint8)
            wm_rgba = np.ascontiguousarray(wm_rgba)
        
        # 计算水印位置
        h, w = result.shape[:2]
        wm_h, wm_w = wm_rgba.shape[:2]
        
        # 处理素材位于裁切区域外的情况
        # 如果坐标是负数，说明素材的一部分在裁切区域外，需要先裁切素材图像
        wm_x_start = 0  # 素材图像中使用的起始x坐标
        wm_y_start = 0  # 素材图像中使用的起始y坐标
        canvas_x = x    # 在画布上的x坐标
        canvas_y = y    # 在画布上的y坐标
        
        # 如果素材在画布左侧外，需要裁切素材的左部分
        if canvas_x < 0:
            wm_x_start = -canvas_x
            canvas_x = 0
        
        # 如果素材在画布顶部外，需要裁切素材的上部分
        if canvas_y < 0:
            wm_y_start = -canvas_y
            canvas_y = 0
        
        # 确保坐标在有效范围内
        canvas_x = max(0, min(canvas_x, w - 1))
        canvas_y = max(0, min(canvas_y, h - 1))
        
        # 计算实际粘贴区域
        # 计算画布上可以粘贴的区域
        x_end = min(canvas_x + (wm_w - wm_x_start), w)
        y_end = min(canvas_y + (wm_h - wm_y_start), h)
        # 计算素材图像中实际使用的区域
        wm_x_end = min(wm_x_start + (x_end - canvas_x), wm_w)
        wm_y_end = min(wm_y_start + (y_end - canvas_y), wm_h)
        
        # 确保素材区域有效
        if wm_x_end > wm_x_start and wm_y_end > wm_y_start and x_end > canvas_x and y_end > canvas_y:
            wm_slice = wm_rgba[wm_y_start:wm_y_end, wm_x_start:wm_x_end]
            wm_alpha = wm_slice[:, :, 3:4].astype(np.float32) / 255.0
            if np.any(wm_alpha > 0):
                dst_region = result[canvas_y:y_end, canvas_x:x_end]
                dst_rgb = dst_region[:, :, :3].astype(np.float32) / 255.0
                dst_alpha = dst_region[:, :, 3:4].astype(np.float32) / 255.0
                wm_rgb = wm_slice[:, :, :3].astype(np.float32) / 255.0

                # 预乘alpha进行混合
                src_rgb_premul = wm_rgb * wm_alpha
                dst_rgb_premul = dst_rgb * dst_alpha
                out_alpha = wm_alpha + dst_alpha * (1.0 - wm_alpha)
                out_rgb_premul = src_rgb_premul + dst_rgb_premul * (1.0 - wm_alpha)

                safe_alpha = np.where(out_alpha > 1e-6, out_alpha, 1.0)
                out_rgb = np.where(
                    out_alpha > 1e-6,
                    out_rgb_premul / safe_alpha,
                    0.0
                )

                dst_region[:, :, :3] = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
                dst_region[:, :, 3:4] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8)
                result[canvas_y:y_end, canvas_x:x_end] = dst_region
        
        return np.ascontiguousarray(result)
    
    def remove_background_color(self, frame: np.ndarray, color: Tuple[int, int, int], tolerance: int = 30, expand_px: int = 2, edge_clear_px: int = 6,
                                valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        移除图片外侧的指定背景颜色（将匹配的边缘颜色设为透明）
        
        Args:
            frame: 输入图像（RGB或RGBA）
            color: 需要移除的背景颜色 (r, g, b)
            tolerance: 颜色匹配的容差
            valid_mask: 可选的有效区域掩膜（非零区域参与扣除，零区域保留）
        
        Returns:
            np.ndarray: 处理后的图像（RGBA）
        """
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        has_alpha = frame.shape[2] == 4
        if frame.shape[2] == 3:
            alpha_channel = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
            result = np.concatenate([frame, alpha_channel], axis=2)
        elif has_alpha:
            result = frame.copy()
        else:
            return frame

        h, w = result.shape[:2]
        if h == 0 or w == 0:
            return result

        rgb = result[:, :, :3]
        target = np.array(color, dtype=np.int16)
        tol = max(0, tolerance)
        lower = np.clip(target - tol, 0, 255).astype(np.uint8)
        upper = np.clip(target + tol, 0, 255).astype(np.uint8)
        match_mask = cv2.inRange(rgb, lower, upper)
        mask_limiter = None
        if valid_mask is not None:
            mask_limiter = np.ascontiguousarray(valid_mask)
            if mask_limiter.dtype != np.uint8:
                mask_limiter = (mask_limiter > 0).astype(np.uint8)
            mask_limiter = np.where(mask_limiter > 0, 255, 0).astype(np.uint8)
            if mask_limiter.shape != match_mask.shape:
                mask_limiter = cv2.resize(mask_limiter, (match_mask.shape[1], match_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            match_mask = cv2.bitwise_and(match_mask, mask_limiter)
        if not np.any(match_mask):
            return result

        # 平滑并关闭小孔，保证掩膜连续
        match_mask = cv2.medianBlur(match_mask, 3)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        match_mask = cv2.morphologyEx(match_mask, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)

        # 基于边缘检测，定位主体边缘
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)

        # 找到与图像边界相连的匹配区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(match_mask, connectivity=4)
        if num_labels <= 1:
            return result

        removal_mask = np.zeros_like(match_mask)
        for label_idx in range(1, num_labels):
            x, y, w_comp, h_comp, _ = stats[label_idx]
            if x == 0 or y == 0 or (x + w_comp) >= w or (y + h_comp) >= h:
                removal_mask[labels == label_idx] = 255

        if not np.any(removal_mask):
            return result

        # 结合边缘信息扩展靠近主体边缘的背景区域
        if edge_clear_px > 0:
            edge_band = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
            edge_band = (edge_band <= max(1, edge_clear_px)).astype(np.uint8) * 255
            removal_mask = cv2.bitwise_or(removal_mask, cv2.bitwise_and(edge_band, match_mask))

        if mask_limiter is not None:
            removal_mask[mask_limiter == 0] = 0

        # 扩展并平滑透明区，减少锯齿
        if expand_px > 0:
            expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
            removal_mask = cv2.dilate(removal_mask, expand_kernel, iterations=1)

        removal_mask = cv2.morphologyEx(removal_mask, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)
        if mask_limiter is not None:
            removal_mask[mask_limiter == 0] = 0

        alpha = result[:, :, 3]
        alpha[removal_mask > 0] = 0

        # 羽化边缘，进一步减轻毛刺
        feather_radius = max(1, expand_px * 2)
        distance = cv2.distanceTransform(cv2.bitwise_not(removal_mask), cv2.DIST_L2, 3)
        feather_zone = (removal_mask == 0) & (distance <= feather_radius) & (alpha > 0)
        if mask_limiter is not None:
            feather_zone = feather_zone & (mask_limiter > 0)
        if np.any(feather_zone):
            fade = (distance[feather_zone] / float(feather_radius)).clip(0.0, 1.0)
            alpha_vals = alpha[feather_zone].astype(np.float32)
            alpha[feather_zone] = (alpha_vals * fade).astype(np.uint8)

        result[:, :, 3] = alpha
        return np.ascontiguousarray(result, dtype=np.uint8)
    
    def close(self):
        """关闭视频文件"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.single_image = None

