"""
打包脚本 - 用于将程序打包为可执行文件
支持 Windows 和 macOS 系统
"""
import PyInstaller.__main__
import os
import sys
import platform
import tempfile

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
system = platform.system()

print(f"当前系统: {system}")
print(f"工作目录: {current_dir}")

# 清理旧的构建文件
print("\n清理旧的构建文件...")
import shutil
app_name = "庆雅神器"
old_app_name = "庆雅神器"

# 清理 build 目录
build_dir = os.path.join(current_dir, 'build')
if os.path.exists(build_dir):
    try:
        shutil.rmtree(build_dir)
        print(f"✓ 已清理 build 目录")
    except Exception as e:
        print(f"⚠ 清理 build 目录时出错: {e}")

# 清理 dist 目录
dist_dir = os.path.join(current_dir, 'dist')
if os.path.exists(dist_dir):
    try:
        shutil.rmtree(dist_dir)
        print(f"✓ 已清理 dist 目录")
    except Exception as e:
        print(f"⚠ 清理 dist 目录时出错: {e}")

# 删除旧的 spec 文件
old_spec_files = [
    os.path.join(current_dir, f"{old_app_name}.spec"),
    os.path.join(current_dir, f"{app_name}.spec")
]
for spec_file in old_spec_files:
    if os.path.exists(spec_file):
        try:
            os.remove(spec_file)
            print(f"✓ 已删除旧的 spec 文件: {os.path.basename(spec_file)}")
        except Exception as e:
            print(f"⚠ 删除 spec 文件时出错: {e}")

print("清理完成！\n")

staging_paths = []

# PyInstaller参数
args = [
    'main.py',
    '--name=庆雅神器',
    '--windowed',  # 不显示控制台
    '--onefile',   # 打包为单个文件
    '--clean',     # 清理临时文件
    '--noconfirm', # 覆盖输出目录
    # 使用自定义hook文件
    f'--additional-hooks-dir={current_dir}',  # 使用当前目录的hook文件
    # PyQt5相关
    '--hidden-import=PyQt5',
    '--hidden-import=PyQt5.QtCore',
    '--hidden-import=PyQt5.QtGui',
    '--hidden-import=PyQt5.QtWidgets',
    '--hidden-import=PyQt5.QtSvg',  # SVG 图标渲染
    '--collect-all=PyQt5',
    '--collect-submodules=PyQt5.QtSvg',
    # numpy相关（解决numpy.core.multiarray导入错误）
    '--hidden-import=numpy',
    '--hidden-import=numpy.core',
    '--hidden-import=numpy.core.multiarray',
    '--hidden-import=numpy.core._multiarray_umath',
    '--hidden-import=numpy.core._multiarray_tests',
    '--collect-submodules=numpy',
    '--collect-all=numpy',  # 收集numpy的所有文件（包括二进制扩展）
    # OpenCV相关
    '--hidden-import=cv2',
    '--collect-submodules=cv2',
    '--collect-all=cv2',  # 收集cv2的所有文件（包括DLL）
    # Pillow相关
    '--hidden-import=PIL',
    '--hidden-import=PIL.Image',
    '--collect-submodules=PIL',
    # 其他可能需要的模块
    '--hidden-import=scipy',  # 如果numpy依赖scipy
]

# 根据操作系统选择图标文件和数据分隔符
if system == 'Windows':
    # Windows系统：使用.ico格式
    icon_files = [
        'App icon_128x128.ico',
        'icon.ico',
        'app.ico'
    ]
    data_separator = ';'  # Windows使用分号
elif system == 'Darwin':  # macOS
    # macOS系统：使用.icns格式
    icon_files = [
        'App icon_128x128.icns',
        'icon.icns',
        'app.icns',
        'App icon_128x128.ico',  # 如果没有.icns，尝试.ico
        'icon.ico',
        'app.ico'
    ]
    data_separator = ':'  # macOS/Linux使用冒号
else:  # Linux
    icon_files = [
        'icon.png',
        'icon.ico',
        'app.ico'
    ]
    data_separator = ':'

# 自动检测并添加图标文件
icon_path = None
icon_staging_path = None
icon_search_dirs = [current_dir, os.path.join(current_dir, 'resources')]
for icon_file in icon_files:
    for icon_dir in icon_search_dirs:
        icon_full_path = os.path.join(icon_dir, icon_file)
        if os.path.exists(icon_full_path):
            icon_path = icon_full_path
            # Windows 在包含中文路径时偶发无法嵌入图标，统一复制到ASCII临时目录
            staging_dir = tempfile.mkdtemp(prefix="qingya_icon_")
            staging_paths.append(staging_dir)
            icon_staging_path = os.path.join(staging_dir, os.path.basename(icon_full_path))
            shutil.copy2(icon_full_path, icon_staging_path)
            args.append(f'--icon={icon_staging_path}')
            print(f"✓ 找到图标文件: {icon_path} -> 已复制到 {icon_staging_path}")
            break
    if icon_path:
        break

if icon_path is None:
    print("⚠ 警告：未找到图标文件，将使用默认图标")
    print(f"  请将图标文件放在以下位置之一:")
    if system == 'Windows':
        print("    - App icon_128x128.ico")
        print("    - icon.ico")
        print("    - app.ico")
    elif system == 'Darwin':
        print("    - App icon_128x128.icns (推荐)")
        print("    - icon.icns")
        print("    - app.icns")
    else:
        print("    - icon.png")
        print("    - icon.ico")

# 包含资源文件
resource_files = [
    ('resources/12345.png', 'resources'),  # 气泡挂件附件图片
    ('resources/欢迎页照片.png', 'resources'),  # 欢迎页左侧展示图片
    ('resources/videofolder_99361.png', 'resources'),  # 动态视频模式图标
    ('resources/picture_photo_image_icon_131252.png', 'resources'),  # 静态图片模式图标
    ('resources/白.png', 'resources'),  # 白色主题返回按钮图标
    ('resources/黑.png', 'resources'),  # 黑色主题返回按钮图标
    ('resources/github-icon.svg', 'resources'),  # Github 链接图标
    ('resources/xiaohongshu-seeklogo.svg', 'resources'),  # 小红书链接图标
    ('resources/wechat-seeklogo.svg', 'resources'),  # 服务号链接图标
]

for resource_name, target_name in resource_files:
    resource_path = os.path.join(current_dir, resource_name)
    if os.path.exists(resource_path):
        args.append(f'--add-data={resource_path}{data_separator}{target_name}')
        print(f"✓ 包含资源文件: {resource_path}")
    else:
        print(f"⚠ 警告：未找到资源文件 {resource_name}")

# macOS特定设置
if system == 'Darwin':
    # macOS需要设置应用信息
    args.extend([
        '--osx-bundle-identifier=com.wechat.redpacket.editor',
    ])
    # 提示：macOS 建议使用 build_macos_app.py 创建 .app 包
    print("\n提示：macOS 系统建议使用 build_macos_app.py 脚本打包")
    print("   该脚本会创建 .app 应用包并处理 Gatekeeper 问题")
    print("   如果继续使用此脚本，打包后需要手动处理 Gatekeeper 限制\n")

print("\n开始打包...")
print("=" * 50)

try:
    # 执行打包
    PyInstaller.__main__.run(args)
    print("\n" + "=" * 50)
    print("✓ 打包完成！")
    if system == 'Windows':
        print("可执行文件位于: dist/庆雅神器.exe")
    elif system == 'Darwin':
        print("可执行文件位于: dist/庆雅神器")
        print("\n⚠ 重要提示：")
        print("1. 如果应用无法运行（被 Gatekeeper 阻止），请使用 build_macos_app.py 重新打包")
        print("2. 或者手动移除隔离属性：")
        print(f"   xattr -d com.apple.quarantine dist/{app_name}")
        print("3. 或者右键点击应用，选择'打开'（而不是双击）")
        print("\n如果拥有开发者证书，可以签名应用：")
        print(f"  codesign --deep --force --verify --verbose --sign \"Developer ID Application: Your Name\" dist/{app_name}")
    else:
        print("可执行文件位于: dist/庆雅神器")
except Exception as e:
    print(f"\n✗ 打包失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # 清理临时资源
    for path in staging_paths:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
