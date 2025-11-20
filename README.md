# 庆雅神器 · 开发文档概览

跨平台红包封面/气泡挂件编辑器，支持视频帧裁切、封面图挂件反选、独立气泡挂件排版。本文件为开发入口，汇总核心能力、模块分布与常用流程。

---

## 1. 核心能力

| 能力 | 说明 |
| --- | --- |
| 画布裁切 | 第一步对话框可对视频进行 1053×1746 或 750×1250 画布裁切，记录 scale/offset，后续导出共享。 |
| 多层框架 | 封面图模式提供封面框、红包带开字框、挂件框、不可编辑框四层助线，所有偏移自动计算。 |
| 反选挂件 | “封面图挂件”自动生成环形 mask（A/B 两区），A 区保留主画面+所有素材，B 区仅保留挂件模块素材。 |
| 素材双模块 | 素材面板同时显示 `[主图]` 与 `[挂件]` 两类素材，导出阶段根据 `scope` 字段自动区分职责。 |
| 气泡挂件制作 | BubblePendantWidget 支持底图导入、素材排列、区块限制，导出 480×384 RGBA PNG。 |
| 打包脚本 | `build_exe.py`、`build_macos_app.py` 一键生成 Win/mac 产物，内置 PyQt5/numpy/cv2 hook 与资源拷贝。 |

详见 `gui.py` 内的 `VideoProcessorGUI`、`ImageLabel`、`BubblePendantWidget` 以及 `video_processor.py` 内的 `VideoProcessor`/`ProcessingThread`。

---

## 2. 目录速览

```
├── gui.py                # 主界面 + 业务逻辑
├── video_processor.py    # 帧提取、裁切、透明度、背景扣除
├── ui_components.py      # 折叠面板、帧导航、主题切换
├── ui_utils.py           # DPI/主题工具
├── build_exe.py          # Windows 打包脚本
├── build_macos_app.py    # macOS 打包脚本（含 Info.plist/xattr）
├── create_icns.py        # ico/png -> icns 转换
├── resources/            # 图标、预置素材、主题图片
├── scripts/              # 启动/激活相关脚本
├── docs/                 # 当前保留的核心文档
└── requirements.txt      # Python 依赖
```

更多细节见 `docs/项目结构说明.md`。

---

## 3. 快速上手

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
python main.py             # 或 python gui.py
```

默认流程：
1. 在欢迎页选择模式（动态/静态）
2. 完成第一步“整体尺寸裁切”
3. 进入主界面 → 设置矩形、素材、反选/背景色 → 导出

---

## 4. 典型工作流

### 4.1 通用导出
1. 选择媒体与输出目录
2. 预览区拖动矩形或在右侧输入坐标
3. 切换尺寸预设（气泡图/封面图/封面故事/自定义）
4. 在素材面板追加 `[主图]` 素材（默认模块）
5. 点击“开始处理”，在输出目录中检查对应文件夹

### 4.2 封面图挂件（反选）
1. 封面图尺寸下勾选“反选与背景色”
2. 按需在面板中添加最多 4 个背景颜色样本
3. 在素材面板切换到“封面图挂件”模块，添加挂件素材
4. A 区（挂件→红包）保留主画面与所有素材，B 区（红包→不可编辑）仅保留挂件素材
5. 导出文件命名为“封面图挂件”并放入专属子目录

### 4.3 气泡挂件排版
1. 切换到“静态图片模式”
2. 点击“气泡挂件制作”打开对话框
3. 选择底图、添加素材，注意中区禁放限制
4. 取消文件选择时窗口会自动 `raise_`，避免被主界面遮挡
5. 导出 480×384 PNG 用于聊天气泡挂件

---

## 5. 打包摘要

| 平台 | 命令 | 产物 |
| --- | --- | --- |
| Windows | `python build_exe.py` | `dist/庆雅神器/庆雅神器.exe` + 依赖 onedir |
| macOS | `python build_macos_app.py` | `dist/庆雅神器.app` + CLI 版本 |

详见 `docs/打包说明.md`（包含签名、Gatekeeper、常见问题）。

---

## 6. 文档索引

| 文件 | 说明 |
| --- | --- |
| `docs/README.md` | 本概览，提供功能/流程入口 |
| `docs/安装说明.md` | 推荐 Python 版本、依赖安装、常见报错 |
| `docs/使用说明.md` | GUI & 气泡挂件详细操作步骤 |
| `docs/打包说明.md` | Win/mac 打包、验证、常见问题 |
| `docs/项目结构说明.md` | 目录结构、常用命令速查 |

---

## 7. 贡献/排障建议

1. **资源统一**：新增 PNG/字体 → 放入 `resources/`，并在两个打包脚本的 `datas` 中登记。
2. **素材作用域**：`self.watermarks` 中的 `scope` 字段决定素材在哪个导出流程生效，请勿丢失。
3. **长耗时操作**：放入 `ProcessingThread`，通过信号回主线程刷新 UI。
4. **提交前自检**：运行至少一次 `build_exe.py` 或 `build_macos_app.py`，确保资源、hook、依赖完整。

若文档无法覆盖的问题，请在工单或 PR 中附：系统信息、Python 版本、复现步骤与日志，便于定位。EOF