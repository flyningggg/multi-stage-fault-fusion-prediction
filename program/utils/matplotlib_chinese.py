# -*- coding: utf-8 -*-
"""
统一配置 matplotlib 中文字体，避免图表中出现乱码或方框。
支持 Windows、macOS、Linux 跨平台。
"""
import os
import sys


def _ensure_mpl_cache_dir() -> None:
    """
    确保 matplotlib 缓存目录可写，必须在 import matplotlib 之前调用。
    macOS 上 ~/.matplotlib 因 com.apple.provenance 扩展属性可能不可写，
    导致每次启动重建临时缓存、字体扫描极慢。
    此函数自动回退到 ~/.cache/matplotlib_fracture（始终可写），
    使字体缓存持久化，第二次及以后启动直接命中缓存。
    注意：main.py 已在更早的时机设置了 MPLCONFIGDIR，此处主要服务于独立脚本。
    """
    if "MPLCONFIGDIR" in os.environ:
        return
    fallback = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib_fracture")
    try:
        os.makedirs(fallback, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = fallback
    except OSError:
        pass


# macOS 上直接注册的字体文件路径（按优先级排列）
_MACOS_FONT_FILES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/simsun.ttc",
]


def _register_fonts_by_path(font_files: list) -> list:
    """
    直接用字体文件路径注册到 matplotlib，返回成功注册的字体名列表。
    此方法绕过字体缓存扫描，无论缓存目录是否可写都能立即生效。
    """
    import matplotlib.font_manager as fm

    registered = []
    for path in font_files:
        if not os.path.isfile(path):
            continue
        try:
            fm.fontManager.addfont(path)
            # addfont 后该文件对应的字体条目已加入 fontManager.ttflist
            # 取刚加入的最后一条记录的 name
            prop = fm.FontProperties(fname=path)
            name = prop.get_name()
            if name and name not in registered:
                registered.append(name)
        except Exception:
            pass
    return registered


def setup_matplotlib_chinese():
    """
    配置 matplotlib 正确显示中文及负号。
    应在任何绘图操作之前调用（如 main.py 入口、独立脚本开头）。
    """
    _ensure_mpl_cache_dir()

    import matplotlib
    import matplotlib.pyplot as plt

    if sys.platform == "darwin":
        # 直接注册字体文件，不依赖缓存扫描
        registered = _register_fonts_by_path(_MACOS_FONT_FILES)
        # 已注册的字体名放最前，其余备用名兜底
        fallback = ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
                    "Arial Unicode MS", "Microsoft YaHei", "SimHei"]
        fonts = registered + [f for f in fallback if f not in registered]
    elif sys.platform == "win32":
        fonts = ["Microsoft YaHei", "SimHei", "KaiTi", "Arial Unicode MS"]
    else:
        fonts = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                 "Noto Sans CJK SC", "Droid Sans Fallback",
                 "Microsoft YaHei", "SimHei"]

    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
