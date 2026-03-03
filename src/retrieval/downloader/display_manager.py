#!/usr/bin/env python3
"""
虚拟显示器管理模块

支持在无显示器的 Linux 服务器上运行有头模式浏览器。
macOS 不需要此模块（图形界面始终可用或使用 headless 模式）。

使用方法：
    from display_manager import ensure_display, get_display_mode
    
    # 自动检测并设置显示器
    display = ensure_display()
    
    # 获取当前显示模式
    mode = get_display_mode()  # "real", "virtual", "headless"

环境变量：
    DISPLAY_MODE: 控制显示模式
        - "auto": 自动检测（默认）
        - "virtual": 强制使用虚拟显示器
        - "headless": 强制使用无头模式
        - "real": 强制使用真实显示器（需要有显示器）
    
    VIRTUAL_DISPLAY_SIZE: 虚拟显示器分辨率，默认 "1920x1080"

Linux 服务器安装依赖：
    sudo apt-get install -y xvfb
    pip install pyvirtualdisplay
"""

import os
import sys
import platform
from typing import Optional, Tuple
from .log_utils import logger

# 全局虚拟显示器实例
_virtual_display = None
_display_mode = None


def is_display_available() -> bool:
    """
    检测是否有可用的显示器
    
    Returns:
        bool: 是否有显示器可用
    """
    # macOS 始终有显示器（即使是远程也可以用虚拟显示）
    if platform.system() == "Darwin":
        return True
    
    # Windows 始终有显示器
    if platform.system() == "Windows":
        return True
    
    # Linux: 检查 DISPLAY 环境变量
    display = os.environ.get("DISPLAY")
    if display:
        # 尝试连接 X 服务器验证
        try:
            import subprocess
            result = subprocess.run(
                ["xdpyinfo"], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            # xdpyinfo 不可用，假设 DISPLAY 变量有效
            return True
    
    return False


def start_virtual_display(
    width: int = 1920, 
    height: int = 1080,
    color_depth: int = 24
) -> Optional[object]:
    """
    启动虚拟显示器（仅 Linux）
    
    Args:
        width: 显示器宽度
        height: 显示器高度
        color_depth: 颜色深度
        
    Returns:
        Display 对象或 None（如果启动失败或不需要）
    """
    global _virtual_display
    
    # 如果已经有虚拟显示器在运行
    if _virtual_display is not None:
        return _virtual_display
    
    # 非 Linux 系统不需要虚拟显示器
    if platform.system() != "Linux":
        logger.info(f"当前系统 {platform.system()} 不需要虚拟显示器")
        return None
    
    try:
        from pyvirtualdisplay import Display
        
        logger.info(f"正在启动虚拟显示器 ({width}x{height}, {color_depth}bit)...")
        
        _virtual_display = Display(
            visible=False,  # 不可见
            size=(width, height),
            color_depth=color_depth,
            backend="xvfb"  # 使用 Xvfb
        )
        _virtual_display.start()
        
        logger.info(f"虚拟显示器已启动: DISPLAY={os.environ.get('DISPLAY')}")
        return _virtual_display
        
    except ImportError:
        logger.warning("pyvirtualdisplay 未安装，无法启动虚拟显示器")
        logger.warning("请运行: pip install pyvirtualdisplay")
        return None
    except Exception as e:
        logger.error(f"启动虚拟显示器失败: {e}")
        logger.warning("请确保已安装 xvfb: sudo apt-get install xvfb")
        return None


def stop_virtual_display():
    """停止虚拟显示器"""
    global _virtual_display
    
    if _virtual_display is not None:
        try:
            _virtual_display.stop()
            logger.info("虚拟显示器已停止")
        except Exception as e:
            logger.error(f"停止虚拟显示器失败: {e}")
        finally:
            _virtual_display = None


def get_display_mode() -> str:
    """
    获取当前显示模式
    
    Returns:
        str: "real", "virtual", "headless"
    """
    global _display_mode
    return _display_mode or "unknown"


def ensure_display(force_mode: Optional[str] = None) -> Tuple[bool, str]:
    """
    确保有可用的显示器（自动检测或强制模式）
    
    Args:
        force_mode: 强制使用的模式，可选：
            - None: 使用环境变量 DISPLAY_MODE 或自动检测
            - "auto": 自动检测
            - "virtual": 强制虚拟显示器
            - "headless": 强制无头模式
            - "real": 强制真实显示器
            
    Returns:
        Tuple[bool, str]: (是否应该使用有头模式, 显示模式)
    """
    global _display_mode
    
    # 确定模式
    mode = force_mode or os.environ.get("DISPLAY_MODE", "auto").lower()
    
    # 解析虚拟显示器尺寸
    size_str = os.environ.get("VIRTUAL_DISPLAY_SIZE", "1920x1080")
    try:
        width, height = map(int, size_str.split("x"))
    except ValueError:
        width, height = 1920, 1080
    
    logger.info(f"显示模式配置: {mode}")
    
    # 强制无头模式
    if mode == "headless":
        _display_mode = "headless"
        logger.info("使用无头模式（注意：capsolver 扩展可能无法工作）")
        return False, "headless"
    
    # macOS/Windows: 始终有显示器，使用有头模式
    if platform.system() in ("Darwin", "Windows"):
        _display_mode = "real"
        logger.info(f"{platform.system()} 系统使用真实显示器")
        return True, "real"
    
    # Linux: 检测或启动虚拟显示器
    if mode == "real":
        # 强制真实显示器
        if is_display_available():
            _display_mode = "real"
            logger.info("使用真实显示器")
            return True, "real"
        else:
            logger.error("要求使用真实显示器，但未检测到显示器")
            _display_mode = "headless"
            return False, "headless"
    
    if mode == "virtual":
        # 强制虚拟显示器
        display = start_virtual_display(width, height)
        if display:
            _display_mode = "virtual"
            return True, "virtual"
        else:
            _display_mode = "headless"
            logger.warning("虚拟显示器启动失败，回退到无头模式")
            return False, "headless"
    
    # 自动模式
    if is_display_available():
        _display_mode = "real"
        logger.info("检测到真实显示器")
        return True, "real"
    else:
        # 尝试启动虚拟显示器
        logger.info("未检测到显示器，尝试启动虚拟显示器...")
        display = start_virtual_display(width, height)
        if display:
            _display_mode = "virtual"
            return True, "virtual"
        else:
            _display_mode = "headless"
            logger.warning("无可用显示器，使用无头模式（capsolver 扩展可能无法工作）")
            return False, "headless"


def should_use_headed_mode() -> bool:
    """
    便捷函数：是否应该使用有头模式
    
    Returns:
        bool: True=有头模式, False=无头模式
    """
    headed, _ = ensure_display()
    return headed


# 模块清理：程序退出时停止虚拟显示器
import atexit
atexit.register(stop_virtual_display)


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print(f"操作系统: {platform.system()}")
    print(f"显示器可用: {is_display_available()}")
    
    headed, mode = ensure_display()
    print(f"显示模式: {mode}")
    print(f"使用有头模式: {headed}")
    
    # 测试不同模式
    for test_mode in ["auto", "virtual", "headless", "real"]:
        print(f"\n--- 测试模式: {test_mode} ---")
        headed, actual = ensure_display(test_mode)
        print(f"  结果: headed={headed}, mode={actual}")
    
    stop_virtual_display()

