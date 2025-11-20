"""
macOS 专用打包脚本 - 创建 .app 应用包
解决 Gatekeeper 阻止运行的问题
"""
import PyInstaller.__main__
import os
import sys
import platform
import shutil
import subprocess

# 检查是否为 macOS
if platform.system() != 'Darwin':
    print("错误：此脚本仅适用于 macOS 系统")
    sys.exit(1)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
app_name = "庆雅神器"

print(f"当前系统: macOS")
print(f"工作目录: {current_dir}")
print(f"应用名称: {app_name}")

# 清理旧的构建文件
print("\n清理旧的构建文件...")
build_dir = os.path.join(current_dir, 'build')
dist_dir = os.path.join(current_dir, 'dist')
app_bundle = os.path.join(dist_dir, f"{app_name}.app")
spec_file = os.path.join(current_dir, f"{app_name}.spec")

# 强制清理，处理符号链接问题
def force_remove(path):
    """强制删除路径，处理符号链接"""
    if not os.path.exists(path):
        return True
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            # 先尝试正常删除
            try:
                shutil.rmtree(path)
            except OSError:
                # 如果失败，可能是符号链接，尝试单独处理
                if os.path.islink(path):
                    os.remove(path)
                else:
                    # 递归删除，忽略错误
                    import stat
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    shutil.rmtree(path, onerror=handle_remove_readonly)
        return True
    except Exception as e:
        print(f"⚠ 清理 {path} 时出错: {e}")
        return False

if os.path.exists(build_dir):
    if force_remove(build_dir):
        print(f"✓ 已清理 build 目录")
    else:
        print(f"⚠ 清理 build 目录失败，请手动删除: {build_dir}")

if os.path.exists(dist_dir):
    if force_remove(dist_dir):
        print(f"✓ 已清理 dist 目录")
    else:
        print(f"⚠ 清理 dist 目录失败，请手动删除: {dist_dir}")

if os.path.exists(spec_file):
    try:
        os.remove(spec_file)
        print(f"✓ 已删除旧的 spec 文件")
    except Exception as e:
        print(f"⚠ 删除 spec 文件时出错: {e}")

print("清理完成！\n")

# PyInstaller参数 - 使用 --onedir 模式创建 .app 包
args = [
    'main.py',
    '--name=' + app_name,
    '--windowed',  # 不显示控制台
    '--onedir',    # 使用目录模式（不是单文件），这样才能创建 .app 包
    '--clean',     # 清理临时文件
    '--noconfirm', # 覆盖输出目录
    # 使用自定义hook文件
    '--additional-hooks-dir=.',
    # PyQt5相关（避免符号链接冲突，不收集框架文件）
    '--hidden-import=PyQt5',
    '--hidden-import=PyQt5.QtCore',
    '--hidden-import=PyQt5.QtGui',
    '--hidden-import=PyQt5.QtWidgets',
    '--hidden-import=PyQt5.QtSvg',
    # 不收集 PyQt5 框架，避免符号链接冲突
    # 框架文件会在运行时从系统路径加载
    # numpy相关
    '--hidden-import=numpy',
    '--hidden-import=numpy.core',
    '--hidden-import=numpy.core.multiarray',
    '--hidden-import=numpy.core._multiarray_umath',
    '--hidden-import=numpy.core._multiarray_tests',
    '--collect-submodules=numpy',
    '--collect-all=numpy',
    # OpenCV相关
    '--hidden-import=cv2',
    '--collect-submodules=cv2',
    '--collect-all=cv2',
    # Pillow相关
    '--hidden-import=PIL',
    '--hidden-import=PIL.Image',
    '--collect-submodules=PIL',
    '--collect-submodules=PyQt5.QtSvg',
    # 其他可能需要的模块
    '--hidden-import=scipy',
    # macOS特定设置
    '--osx-bundle-identifier=com.wechat.redpacket.editor',
    # 排除所有不需要的 PyQt5 模块，避免符号链接冲突
    '--exclude-module=PyQt5.QtBluetooth',
    '--exclude-module=PyQt5.QtDBus',
    '--exclude-module=PyQt5.QtDesigner',
    '--exclude-module=PyQt5.QtHelp',
    '--exclude-module=PyQt5.QtLocation',
    '--exclude-module=PyQt5.QtMultimedia',
    '--exclude-module=PyQt5.QtMultimediaWidgets',
    '--exclude-module=PyQt5.QtNfc',
    '--exclude-module=PyQt5.QtOpenGL',
    '--exclude-module=PyQt5.QtPositioning',
    '--exclude-module=PyQt5.QtQml',
    '--exclude-module=PyQt5.QtQuick',
    '--exclude-module=PyQt5.QtQuickWidgets',
    '--exclude-module=PyQt5.QtSensors',
    '--exclude-module=PyQt5.QtSerialPort',
    '--exclude-module=PyQt5.QtSql',
    '--exclude-module=PyQt5.QtTest',
    '--exclude-module=PyQt5.QtWebChannel',
    '--exclude-module=PyQt5.QtWebKit',
    '--exclude-module=PyQt5.QtWebKitWidgets',
    '--exclude-module=PyQt5.QtWebSockets',
    '--exclude-module=PyQt5.QtXml',
    '--exclude-module=PyQt5.QtXmlPatterns',
    # 使用 hook 文件来精确控制 PyQt5 的收集
    # 创建自定义 hook 来避免框架符号链接问题
]

# 查找图标文件（优先查找 resources 目录）
icon_files = [
    'resources/App icon_128x128.icns',
    'resources/icon.icns',
    'resources/app.icns',
    'App icon_128x128.icns',
    'icon.icns',
    'app.icns',
    'resources/App icon_128x128.ico',
    'App icon_128x128.ico',
    'icon.ico',
    'app.ico'
]

icon_path = None
for icon_file in icon_files:
    icon_full_path = os.path.join(current_dir, icon_file)
    if os.path.exists(icon_full_path):
        icon_path = icon_full_path
        args.append(f'--icon={icon_path}')
        print(f"✓ 找到图标文件: {icon_full_path}")
        break

if icon_path is None:
    print("⚠ 警告：未找到图标文件，将使用默认图标")

# 包含资源文件
resource_files = [
    ('resources/12345.png', 'resources'),  # 气泡挂件附件图片
    ('resources/App icon.png', 'resources'),  # 应用图标
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
        args.append(f'--add-data={resource_name}:{target_name}')
        print(f"✓ 包含资源文件: {resource_name}")
    else:
        print(f"⚠ 警告：未找到资源文件 {resource_name}")

def create_app_bundle(dist_dir, app_name, icon_path=None):
    """手动创建 .app 应用包"""
    app_bundle = os.path.join(dist_dir, f"{app_name}.app")
    contents_dir = os.path.join(app_bundle, 'Contents')
    macos_dir = os.path.join(contents_dir, 'MacOS')
    resources_dir = os.path.join(contents_dir, 'Resources')
    
    # 创建目录结构
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    
    # 查找可执行文件
    executable_name = app_name
    executable_source = os.path.join(dist_dir, app_name, executable_name)
    if not os.path.exists(executable_source):
        # 尝试查找其他可能的可执行文件
        for item in os.listdir(dist_dir):
            item_path = os.path.join(dist_dir, item)
            if os.path.isdir(item_path) and item != f"{app_name}.app":
                executable_source = os.path.join(item_path, item)
                if os.path.exists(executable_source):
                    break
    
    if os.path.exists(executable_source):
        executable_dest = os.path.join(macos_dir, executable_name)
        shutil.copy2(executable_source, executable_dest)
        os.chmod(executable_dest, 0o755)
        print(f"✓ 已复制可执行文件到: {executable_dest}")
    else:
        print(f"✗ 未找到可执行文件: {executable_source}")
        return False
    
    # 复制资源文件（从应用目录复制到 Resources 目录）
    # PyInstaller 会将 --add-data 的文件放在应用目录中
    app_dir = os.path.dirname(executable_source)
    if os.path.exists(app_dir):
        # 需要复制的资源文件列表（包含目录结构）
        resource_files_to_copy = [
            ('resources/12345.png', 'resources'),
            ('resources/欢迎页照片.png', 'resources'),
            ('resources/videofolder_99361.png', 'resources'),
            ('resources/picture_photo_image_icon_131252.png', 'resources'),
            ('resources/白.png', 'resources'),
            ('resources/黑.png', 'resources'),
            ('resources/github-icon.svg', 'resources'),
            ('resources/xiaohongshu-seeklogo.svg', 'resources'),
            ('resources/wechat-seeklogo.svg', 'resources'),
        ]
        
        for resource_path, target_dir in resource_files_to_copy:
            # 源文件路径（在应用目录中）
            resource_source = os.path.join(app_dir, resource_path)
            # 如果不在应用目录，尝试在 resources 子目录中查找
            if not os.path.exists(resource_source):
                resource_source = os.path.join(app_dir, os.path.basename(resource_path))
            
            if os.path.exists(resource_source):
                target_resources_dir = os.path.join(resources_dir, target_dir)
                os.makedirs(target_resources_dir, exist_ok=True)
                if os.path.isdir(resource_source):
                    dest_dir = os.path.join(resources_dir, target_dir)
                    if os.path.exists(dest_dir):
                        shutil.rmtree(dest_dir, ignore_errors=True)
                    shutil.copytree(resource_source, dest_dir)
                    print(f"✓ 已复制资源目录: {resource_path}")
                else:
                    resource_dest = os.path.join(target_resources_dir, os.path.basename(resource_path))
                    shutil.copy2(resource_source, resource_dest)
                    print(f"✓ 已复制资源文件: {resource_path}")
            else:
                # 尝试在 resources 目录中查找
                alt_source = os.path.join(app_dir, 'resources', os.path.basename(resource_path))
                if os.path.exists(alt_source):
                    target_resources_dir = os.path.join(resources_dir, target_dir)
                    os.makedirs(target_resources_dir, exist_ok=True)
                    if os.path.isdir(alt_source):
                        dest_dir = os.path.join(resources_dir, target_dir)
                        if os.path.exists(dest_dir):
                            shutil.rmtree(dest_dir, ignore_errors=True)
                        shutil.copytree(alt_source, dest_dir)
                        print(f"✓ 已复制资源目录: {resource_path} (从备用路径)")
                    else:
                        resource_dest = os.path.join(target_resources_dir, os.path.basename(resource_path))
                        shutil.copy2(alt_source, resource_dest)
                        print(f"✓ 已复制资源文件: {resource_path} (从备用路径)")
                else:
                    print(f"⚠ 警告：未找到资源文件 {resource_path}")
    
    # 复制图标（确保图标文件在 Resources 目录中）
    icon_copied = False
    if icon_path and os.path.exists(icon_path):
        icon_dest = os.path.join(resources_dir, 'icon.icns')
        try:
            if icon_path.endswith('.icns'):
                # 直接复制 .icns 文件
                shutil.copy2(icon_path, icon_dest)
                icon_copied = True
                print(f"✓ 已复制图标到: {icon_dest}")
            elif icon_path.endswith(('.ico', '.png', '.jpg', '.jpeg')):
                # 尝试使用 sips 命令转换图标（macOS 内置工具）
                try:
                    subprocess.run(
                        ['sips', '-s', 'format', 'icns', icon_path, '--out', icon_dest],
                        check=True,
                        capture_output=True
                    )
                    icon_copied = True
                    print(f"✓ 已转换并复制图标: {icon_path} -> {icon_dest}")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # 如果 sips 不可用，尝试直接复制（虽然可能不工作）
                    print(f"⚠ 无法转换图标格式，尝试直接复制...")
                    shutil.copy2(icon_path, icon_dest)
                    icon_copied = True
            else:
                # 其他格式，尝试直接复制
                shutil.copy2(icon_path, icon_dest)
                icon_copied = True
                print(f"✓ 已复制图标到: {icon_dest}")
        except Exception as e:
            print(f"⚠ 复制图标时出错: {e}")
            icon_copied = False
    else:
        # 如果未找到图标文件，尝试从 resources 目录查找
        alt_icon_paths = [
            os.path.join(current_dir, 'resources', 'App icon_128x128.icns'),
            os.path.join(current_dir, 'resources', 'icon.icns'),
        ]
        for alt_path in alt_icon_paths:
            if os.path.exists(alt_path):
                icon_dest = os.path.join(resources_dir, 'icon.icns')
                try:
                    shutil.copy2(alt_path, icon_dest)
                    icon_copied = True
                    print(f"✓ 已从备用路径复制图标: {alt_path} -> {icon_dest}")
                    break
                except Exception as e:
                    print(f"⚠ 从备用路径复制图标失败: {e}")
    
    # 创建 Info.plist
    # 注意：CFBundleIconFile 不应该包含 .icns 扩展名
    icon_file_name = 'icon' if icon_copied else ''
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{app_name}</string>"""
    
    if icon_file_name:
        info_plist += f"""
    <key>CFBundleIconFile</key>
    <string>{icon_file_name}</string>"""
    
    info_plist += f"""
    <key>CFBundleIdentifier</key>
    <string>com.wechat.redpacket.editor</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>"""
    
    plist_path = os.path.join(contents_dir, 'Info.plist')
    with open(plist_path, 'w', encoding='utf-8') as f:
        f.write(info_plist)
    print(f"✓ 已创建 Info.plist")
    
    # 移除隔离属性
    try:
        subprocess.run(['xattr', '-d', 'com.apple.quarantine', app_bundle], 
                      capture_output=True)
        print("✓ 已移除隔离属性")
    except:
        pass
    
    print(f"\n✓ 应用包创建完成: {app_bundle}")
    return True


print("\n开始打包...")
print("=" * 50)

# 执行打包
try:
    PyInstaller.__main__.run(args)
    print("\n" + "=" * 50)
    print("✓ PyInstaller 打包完成")
    
    # 检查是否生成了应用目录（--onedir 模式）
    app_dir = os.path.join(dist_dir, app_name)
    app_bundle = os.path.join(dist_dir, f"{app_name}.app")
    
    # 如果 PyInstaller 没有自动创建 .app，我们需要手动创建
    if os.path.exists(app_dir) and not os.path.exists(app_bundle):
        print(f"\n检测到应用目录: {app_dir}")
        print("正在创建 .app 应用包...")
        if create_app_bundle(dist_dir, app_name, icon_path):
            print("✓ 已创建 .app 应用包")
            app_bundle = os.path.join(dist_dir, f"{app_name}.app")
        else:
            print("⚠ 创建 .app 应用包失败，但应用目录已存在")
            app_bundle = None
    
    if os.path.exists(app_bundle):
        print(f"✓ 已创建应用包: {app_bundle}")
        
        # 移除隔离属性（解决 Gatekeeper 阻止问题）
        print("\n正在移除隔离属性...")
        try:
            result = subprocess.run(
                ['xattr', '-d', 'com.apple.quarantine', app_bundle],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✓ 已移除隔离属性")
            else:
                # 如果属性不存在，这不是错误
                if 'No such xattr' not in result.stderr:
                    print(f"⚠ 移除隔离属性时出现警告: {result.stderr}")
                else:
                    print("✓ 隔离属性不存在（可能已被移除）")
        except Exception as e:
            print(f"⚠ 移除隔离属性时出错: {e}")
            print("   可以手动运行: xattr -d com.apple.quarantine " + app_bundle)
        
        # 添加执行权限
        print("\n正在设置执行权限...")
        executable_path = os.path.join(app_bundle, 'Contents', 'MacOS', app_name)
        if os.path.exists(executable_path):
            try:
                os.chmod(executable_path, 0o755)
                print("✓ 已设置执行权限")
            except Exception as e:
                print(f"⚠ 设置执行权限时出错: {e}")
        
        print("\n" + "=" * 50)
        print("✓ 打包完成！")
        print(f"\n应用包位于: {app_bundle}")
        print("\n如果仍然无法运行，请尝试以下方法：")
        print("\n方法 1: 右键点击应用，选择'打开'（而不是双击）")
        print("方法 2: 在终端中运行以下命令：")
        print(f"  xattr -d com.apple.quarantine \"{app_bundle}\"")
        print("方法 3: 在'系统偏好设置 > 安全性与隐私'中允许运行")
        print("\n如果拥有开发者证书，可以签名应用：")
        print(f"  codesign --deep --force --verify --verbose --sign \"Developer ID Application: Your Name\" \"{app_bundle}\"")
        
    elif os.path.exists(app_dir):
        # 如果只有应用目录，手动创建 .app 包
        print("\n正在创建 .app 应用包...")
        if create_app_bundle(dist_dir, app_name, icon_path):
            app_bundle = os.path.join(dist_dir, f"{app_name}.app")
            # 重新处理隔离属性等
            if os.path.exists(app_bundle):
                print("\n正在移除隔离属性...")
                try:
                    result = subprocess.run(
                        ['xattr', '-d', 'com.apple.quarantine', app_bundle],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0 or 'No such xattr' in result.stderr:
                        print("✓ 已处理隔离属性")
                except:
                    pass
                
                executable_path = os.path.join(app_bundle, 'Contents', 'MacOS', app_name)
                if os.path.exists(executable_path):
                    os.chmod(executable_path, 0o755)
                    print("✓ 已设置执行权限")
                
                print(f"\n✓ 应用包创建完成: {app_bundle}")
        else:
            print(f"⚠ 创建 .app 失败，但应用目录可用: {app_dir}")
    else:
        print(f"\n✗ 未找到应用目录或应用包: {app_dir} 或 {app_bundle}")
        print("   请检查打包过程中的错误信息")
        
except Exception as e:
    print(f"\n✗ 打包失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

