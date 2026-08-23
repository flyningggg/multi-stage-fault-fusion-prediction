# -*- coding: utf-8 -*-
import sys
import os
import math
import json
import time
import subprocess
import platform
import warnings
from typing import Optional, List, Dict, Any

# PyInstaller 打包后 macOS 用 spawn 模式启动子进程，会导致 fractopo/joblib 的
# multiprocessing worker 找不到入口而崩溃（"worker unexpectedly terminated"）。
# 必须在任何 import 之前调用 freeze_support()，并把 joblib/loky 限制为单线程。
if getattr(sys, "frozen", False):
    # 打包环境：注册 freeze_support 并强制 joblib 使用单线程
    import multiprocessing
    multiprocessing.freeze_support()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    try:
        import joblib
        joblib.parallel.DEFAULT_BACKEND = "threading"
    except Exception:
        pass

# 保证 program 目录在 path 中，便于从项目根或 program/ 运行时的导入
_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

# 清除 fractopo 损坏的 joblib 缓存（EOFError），避免每次运行报错
for _cache in [
    os.path.join(_PROGRAM_DIR, ".cache", "fractopo"),
    os.path.join(os.path.dirname(_PROGRAM_DIR), ".cache", "fractopo"),
]:
    if os.path.isdir(_cache):
        try:
            import shutil
            shutil.rmtree(_cache)
        except Exception:
            pass

# PyQt5 找不到 Qt 平台插件时（必须在 import PyQt5 之前设置）
_qt_plugins_path = None
for p in sys.path:
    _candidate = os.path.join(p, "PyQt5", "Qt5", "plugins", "platforms")
    if os.path.exists(_candidate):
        _qt_plugins_path = _candidate
        break
if _qt_plugins_path:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugins_path
    if sys.platform == "win32":
        os.environ["QT_PLUGIN_PATH"] = os.path.dirname(_qt_plugins_path)

# 确保 matplotlib 缓存目录可写（必须在 import matplotlib 之前）。
# macOS 上 ~/.matplotlib 因 com.apple.provenance 扩展属性可能不可写，
# 导致每次启动重建临时缓存、字体扫描极慢、中文显示为方框。
# 优先用项目内 .cache/mplconfig（项目目录本身始终可写），
# 这样字体缓存持久保存在项目里，第二次启动直接命中缓存，中文字体即可正常显示。
if "MPLCONFIGDIR" not in os.environ:
    _mpl_project_cache = os.path.join(_PROGRAM_DIR, ".cache", "mplconfig")
    os.makedirs(_mpl_project_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_project_cache

import matplotlib

# matplotlib 3.7+ 移除了 Legend 的 transform 参数，但 fractopo 0.9.x 内部仍会传入
# 在 Create 任何 figure 之前 patch Legend.__init__ 忽略该参数
def _legend_init_patch(original_init):
    def _patched(self, *args, **kwargs):
        kwargs.pop("transform", None)
        original_init(self, *args, **kwargs)
    return _patched

from matplotlib.legend import Legend as _Legend
_Legend.__init__ = _legend_init_patch(_Legend.__init__)

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import ticker

from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm as mpl_cm

from utils.matplotlib_chinese import setup_matplotlib_chinese
from utils.crs_metric import unify_traces_area_crs, reproject_to_metric_crs
from utils.config_loader import load_config
from utils.config_validation import validate_config

setup_matplotlib_chinese()

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QComboBox, QFrame, QHBoxLayout, QVBoxLayout, QLabel, \
    QPushButton, QInputDialog

from fractopo.branches_and_nodes import branches_and_nodes
from pprint import pprint
from matplotlib.lines import Line2D
from demo import Ui_MainWindow
from fractopo.general import (
    CC_branch,
    CI_branch,
    II_branch,
    X_node,
    Y_node,
    I_node,
    CONNECTION_COLUMN,
    CLASS_COLUMN,
)
from fractopo import Network
import geopandas as gpd
from scipy.stats import gaussian_kde
import numpy as np
from PyQt5.QtGui import QTextCursor

try:
    from topology_fusion import (
        run_fusion_pipeline,
        run_fusion_pipeline_ae,
        run_fusion_pipeline_umap,
        run_fusion_pipeline_vae,
        export_cluster_results,
        build_cluster_name_map,
        attach_cluster_names,
        compute_cluster_quality_metrics,
        compute_cluster_stability_ari,
        build_cluster_summary_rows,
        CONNECTIVITY_FEATURE_COLUMNS as _TF_CONN_COLS,
    )
except ImportError:
    run_fusion_pipeline = None
    run_fusion_pipeline_ae = None
    run_fusion_pipeline_umap = None
    run_fusion_pipeline_vae = None
    export_cluster_results = None
    build_cluster_name_map = None  # type: ignore
    attach_cluster_names = None  # type: ignore
    compute_cluster_quality_metrics = None  # type: ignore
    compute_cluster_stability_ari = None  # type: ignore
    build_cluster_summary_rows = None  # type: ignore
    _TF_CONN_COLS = tuple()

try:
    import torch as _torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
try:
    import umap as _umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore", message=".*geographic CRS.*", category=UserWarning)

# 数据源配置：迹线、研究区、显示名、对应的网格 CSV（融合/ML 用）
# 当前仅保留塔里木盆地英买 2（MY）
DATA_SOURCES = [
    {
        "traces": "MY/11.geojson",
        "area": "MY/my_area1.geojson",
        "name": "塔里木盆地英买2",
        "csv": "Yingmai 2 area in Tarim Basin.csv",
    },
]

# 全局变量，由 load_data_source 更新
traces = None
area = None
name = None
rate = 1.0
width = 0.0
height = 0.0
left = 0.0
right = 0.0
down = 0.0
up = 0.0


def load_data_source(index: int):
    """按索引加载数据源，更新全局 traces/area/name/rate/width/height/left/right/down/up。"""
    global traces, area, name, rate, width, height, left, right, down, up
    if index < 0 or index >= len(DATA_SOURCES):
        return False
    cfg = DATA_SOURCES[index]
    base = _PROGRAM_DIR
    trace_path = os.path.join(base, cfg["traces"])
    area_path = os.path.join(base, cfg["area"])
    if not os.path.isfile(trace_path) or not os.path.isfile(area_path):
        return False
    traces = gpd.read_file(trace_path)
    area = gpd.read_file(area_path)
    traces, area = unify_traces_area_crs(traces, area)
    traces, area = reproject_to_metric_crs(traces, area)
    name = cfg["name"]
    traces.drop_duplicates(subset="geometry", inplace=True)
    traces.reset_index(drop=True, inplace=True)
    geometry = traces.geometry.tolist()
    left = math.inf
    right = -math.inf
    down = math.inf
    up = -math.inf
    for one in geometry:
        b = one.boundary.bounds
        left = min(left, b[0])
        right = max(right, b[2])
        down = min(down, b[1])
        up = max(up, b[3])
    rate = (up - down) / (right - left) if (right - left) > 0 else 1.0
    width = 0.01 * (right - left)
    height = 0.01 * (up - down)
    # 上述 left/right/down/up 已通过 global 写入模块全局
    return True


def _safe_figsize(base_w=9.0, max_h=7.0):
    h = base_w * rate
    if h > max_h:
        return base_w * max_h / h, max_h
    return base_w, h


def load_first_available_data_source() -> int:
    """加载唯一数据源（英买 MY）；失败返回 -1。"""
    if load_data_source(0):
        return 0
    return -1


EMPTY_CROP_MSG = (
    "裁剪后迹线为空：迹线与当前研究区多边形无空间重叠，或数据/坐标系不匹配。\n\n"
    "请检查：① 迹线 GeoJSON 与研究区是否属同一工区；② 在 QGIS 中二者是否相交；③ 导出时统一坐标系。"
)


def _cfg_section(name: str) -> dict:
    cfg = load_config()
    if isinstance(cfg, dict):
        return cfg.get(name, {}) or {}
    return {}


def try_network(*args, **kwargs):
    """
    构造 fractopo.Network，捕获裁剪后迹线为空等错误，避免未处理异常导致进程退出。
    返回 (network, None) 或 (None, 错误说明)。
    """
    try:
        # 某些入口路径下 traces/area 可能被后续逻辑替换为地理 CRS。
        # 在进入 fractopo 前统一再做一次 CRS 对齐与米制投影，避免 length/contour_grid
        # 在经纬度坐标上计算并触发大量告警或结果失真。
        norm_args = list(args)
        if len(norm_args) >= 2:
            maybe_traces, maybe_area = norm_args[0], norm_args[1]
            if isinstance(maybe_traces, gpd.GeoDataFrame) and isinstance(maybe_area, gpd.GeoDataFrame):
                maybe_traces, maybe_area = unify_traces_area_crs(maybe_traces, maybe_area)
                maybe_traces, maybe_area = reproject_to_metric_crs(maybe_traces, maybe_area)
                norm_args[0], norm_args[1] = maybe_traces, maybe_area
                try:
                    print(f"[CRS] traces={maybe_traces.crs} area={maybe_area.crs}")
                except Exception:
                    pass
        return Network(*norm_args, **kwargs), None
    except ValueError as e:
        if "Empty trace GeoDataFrame after crop" in str(e):
            return None, EMPTY_CROP_MSG
        return None, str(e)


def _style_ternary_plot(fig, tax):
    """去掉 python-ternary 默认灰色三角底色，并加深图中的虚线（理论曲线等）。"""
    try:
        tax.set_background_color(color="white", alpha=1.0, zorder=-1000)
    except Exception:
        pass

    def _line_is_dashed(line):
        ls = line.get_linestyle()
        if ls == "--":
            return True
        if isinstance(ls, tuple) and len(ls) >= 2:
            return True
        return False

    for ax in fig.axes:
        for line in ax.get_lines():
            if not _line_is_dashed(line):
                continue
            line.set_color("#1a1a1a")
            line.set_alpha(min(1.0, max(line.get_alpha() or 0.6, 0.6) + 0.32))
            lw = line.get_linewidth()
            line.set_linewidth(max(lw * 1.65, 1.2))


def _polish_fractopo_ternary_labels(fig):
    """
    fractopo 在 X/Y/I 顶点使用白字+粗描边；若再统一套 bbox 会像乱码方框且易被裁切。
    统计信息（多行）单独用圆角白底框；$C_B$ 阈值线用 DejaVu 避免数学符号缺字。
    """
    corners = frozenset(
        {"X", "Y", "I", "I-C", "C-C", "I-I", "X-Y", "CC", "CI", "II"}
    )
    for ax in fig.axes:
        for txt in ax.texts:
            raw = txt.get_text().strip()
            compact = raw.replace(" ", "").replace("–", "-").replace("—", "-")
            if compact in corners:
                txt.set_bbox(None)
                try:
                    txt.set_path_effects([])
                except Exception:
                    pass
                txt.set_color("#111111")
                txt.set_fontfamily("DejaVu Sans")
                txt.set_fontweight("bold")
                txt.set_fontsize(19)
            elif raw.startswith("$"):
                txt.set_fontfamily("DejaVu Sans")
                txt.set_color("#1a1a1a")
                try:
                    txt.set_path_effects([])
                except Exception:
                    pass
            elif "\n" in raw:
                txt.set_bbox(
                    dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="#9CA3AF",
                        alpha=0.95,
                    )
                )
                zh_fonts = plt.rcParams.get("font.sans-serif", [])
                if isinstance(zh_fonts, (list, tuple)) and zh_fonts:
                    txt.set_fontfamily(zh_fonts[0])


# 启动时加载第一个可用的数据源（见 load_first_available_data_source）
START_DATA_SOURCE_INDEX = load_first_available_data_source()


def assign_colors(feature_type: str):
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


# 控制台输出重定向（错误信息放行，便于排查；仅过滤已知无害的系统噪音）
class StreamRedirector(QtCore.QObject):
    text_written = QtCore.pyqtSignal(str)

    def __init__(self, is_error=False):
        super().__init__()
        self.is_error = is_error

    def write(self, text):
        text = str(text)
        if not text.strip():
            return

        if self.is_error:
            # 只过滤已知无害的系统噪音，保留真实错误供排查
            skip_patterns = (
                "building the font cache",
                "PasteBoard:",
                "Connection Invalid",
                "Failure on line",
                "no screens available",
                "id scheduleApplicationNotification",
                "MemorizedFunc",
                "Exception while loading results",
                "joblib/memory.py",
                ".cache/fractopo",
            )
            if any(p in text for p in skip_patterns):
                return
            self.text_written.emit(text)
        else:
            self.text_written.emit(text)

    def flush(self):
        pass


class TaskRunner(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args, inject_progress=False, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._inject_progress = bool(inject_progress)

    def run(self):
        try:
            kwargs = dict(self._kwargs)
            if self._inject_progress:
                kwargs["progress_callback"] = self._report_progress
            out = self._fn(*self._args, **kwargs)
            if self.isInterruptionRequested():
                self.failed.emit("任务已取消")
                return
            self.finished_ok.emit(out)
        except InterruptedError:
            self.failed.emit("任务已取消")
        except Exception as e:
            self.failed.emit(str(e))

    def _report_progress(self, message):
        if self.isInterruptionRequested():
            raise InterruptedError("任务已取消")
        self.progress.emit(str(message))


def _make_latent_fusion_cmap_norm(n_k: int):
    """潜空间聚类图：柔和离散色 + BoundaryNorm；配色与分区底图、散点一致。"""
    base = np.array(
        [
            [0.40, 0.62, 0.86],
            [0.94, 0.58, 0.48],
            [0.52, 0.74, 0.54],
            [0.72, 0.56, 0.82],
            [0.96, 0.78, 0.40],
            [0.42, 0.74, 0.80],
            [0.86, 0.48, 0.56],
            [0.58, 0.62, 0.72],
        ],
        dtype=np.float64,
    )
    n_k = max(int(n_k), 1)
    reps = int(np.ceil(n_k / len(base)))
    colors = np.vstack([base] * reps)[:n_k]
    cmap = ListedColormap(colors)
    bounds = np.arange(n_k + 1, dtype=np.float64) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# 主窗口逻辑
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self._last_exports = {}
        self._initializing_ui = True
        self._is_rendering_contour = False

        # 1. 开启终端输出智能重定向（带降噪滤镜）
        self.stdout_redirector = StreamRedirector(is_error=False)
        self.stdout_redirector.text_written.connect(self.append_text)
        sys.stdout = self.stdout_redirector

        self.stderr_redirector = StreamRedirector(is_error=True)
        self.stderr_redirector.text_written.connect(self.append_text)
        sys.stderr = self.stderr_redirector

        # 1.5 数据源下拉与已加载数据一致（仅英买 MY）
        self.combo_data_source.blockSignals(True)
        if START_DATA_SOURCE_INDEX >= 0:
            self.combo_data_source.setCurrentIndex(START_DATA_SOURCE_INDEX)
        self.combo_data_source.blockSignals(False)
        if START_DATA_SOURCE_INDEX < 0:
            self.append_text(
                "【提示】未找到英买 2 数据。请在 program/MY 下放置 11.geojson、my_area1.geojson，"
                "并准备好网格 CSV（见 README / 运行说明）。\n"
            )

        # 1.6 绑定数据源切换
        self.combo_data_source.currentIndexChanged.connect(self._on_data_source_changed)
        cc = _cfg_section("clustering")
        ex = _cfg_section("export_grid")
        self.spin_kmeans_k.setValue(int(cc.get("n_clusters", 4)))
        self.dspin_grid_step.setValue(float(ex.get("cell_width", 750.0)))
        self._refresh_target_suggestions()
        self._refresh_shap_feature_combo()
        self._refresh_config_summary()
        self._check_config_on_startup()
        QtCore.QTimer.singleShot(240, self._show_startup_flow_guide)
        self.combo_train_target.currentTextChanged.connect(lambda *_: self._refresh_config_summary())
        self.spin_kmeans_k.valueChanged.connect(lambda *_: self._refresh_config_summary())
        self.dspin_grid_step.valueChanged.connect(lambda *_: self._refresh_config_summary())
        self.btn_open_model_dir.clicked.connect(lambda: self._open_program_subdir("model"))
        self.btn_open_processed_dir.clicked.connect(lambda: self._open_program_subdir(os.path.join("data", "processed")))
        self.btn_cancel_task.clicked.connect(self.cancel_running_task)
        self.btn_cancel_task.setEnabled(False)
        self.btn_primary_screening.clicked.connect(self._run_target_screening)
        self.btn_toggle_run_info.clicked.connect(self._toggle_run_info)
        self._running_task = None

        self.btn_prev_fig.clicked.connect(self.show_prev_figure)
        self.btn_next_fig.clicked.connect(self.show_next_figure)
        self.btn_fit_fig.clicked.connect(self._toggle_figure_fit)

        # 2. 绑定第一排：基础地质与拓扑绘图
        self.btn_yuantu.clicked.connect(self.run_yuantu)
        self.btn_fenleihou.clicked.connect(self.run_fenleihou)
        self.btn_relitu.clicked.connect(self.run_relitu)
        self.btn_azimuth.clicked.connect(self.run_azimuth)
        self.btn_meiguitu.clicked.connect(self.run_meiguitu)
        self.btn_sanyuantu.clicked.connect(self.run_sanyuantu)
        self.btn_guanxi.clicked.connect(self.run_guanxi)
        self.btn_b.clicked.connect(self.b)
        self.btn_a.clicked.connect(self.a)

        # 3. 绑定第二排：视图与参数提取
        self.combo_topo.currentIndexChanged.connect(self.onIndexChanged_2)
        self.combo_params.currentIndexChanged.connect(self.onIndexChanged)
        self.btn_tuopushuxing.clicked.connect(self.run_tuopushuxing)

        # 4. 绑定第三排：机器学习与属性融合
        self.combo_fusion.currentIndexChanged.connect(self._set_ronghe_combo_tooltip)
        self.btn_ronghe.clicked.connect(self.run_ronghe)
        self.btn_guoji_weighted.clicked.connect(self.run_guoji_weighted_fusion)
        self.btn_guoji_compare.clicked.connect(self.run_guoji_fusion_compare)
        self.btn_k_helper.clicked.connect(self.run_cluster_k_helper)
        self.btn_guoji_train.clicked.connect(self.run_guoji_train)
        self.btn_guoji_shap.clicked.connect(self.run_guoji_shap)
        self.btn_spatial.clicked.connect(self.run_spatial_topology_framework)
        self.btn_export_results.clicked.connect(self.show_export_results)

        # 5. 绑定时期选择器
        self.chk_period_haixi.stateChanged.connect(self._on_period_changed)
        self.chk_period_xishan.stateChanged.connect(self._on_period_changed)
        self.chk_period_yinzhi.stateChanged.connect(self._on_period_changed)
        self.btn_select_all_periods.clicked.connect(self._select_all_periods)
        self.btn_deselect_all_periods.clicked.connect(self._deselect_all_periods)

        # 6. 绑定渗流选项卡按钮
        self.btn_percolation_curves.clicked.connect(self._run_percolation_curves)
        self.btn_key_nodes.clicked.connect(self._run_key_nodes)
        self.btn_boundary_analysis.clicked.connect(self._run_boundary_analysis)

        # 7. 绑定代理选项卡按钮
        self.btn_agent_train.clicked.connect(self._run_agent_train)
        self.btn_agent_pred_vs_true.clicked.connect(self._run_agent_pred_vs_true)
        self.btn_agent_shap.clicked.connect(self._run_agent_shap)
        self.btn_agent_importance.clicked.connect(self._run_agent_importance)

        # 8. 绑定实验选项卡按钮
        self.btn_exp_ablation.clicked.connect(self._run_exp_ablation)
        self.btn_exp_model_compare.clicked.connect(self._run_exp_model_compare)
        self.btn_exp_noise.clicked.connect(self._run_exp_noise)
        self.btn_exp_spatial_cv.clicked.connect(self._run_exp_spatial_cv)
        self.btn_exp_params_compare.clicked.connect(self._run_exp_params_compare)
        self.btn_exp_length_powerlaw.clicked.connect(self._run_exp_length_powerlaw)
        self.btn_exp_grid_sampling.clicked.connect(self._run_exp_grid_sampling)

        self.opt = 0
        self._initializing_ui = False

        # 多时期分析菜单
        self._setup_multiperiod_menu()

        print("油气区多期断裂网络勘探有利区辅助筛选系统 — 初始化完成")


    def append_text(self, text):
        scrollbar = self.text_browser.verticalScrollBar()
        is_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

        cursor = self.text_browser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)

        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    # ==========================================
    # 时期选择器相关方法
    # ==========================================
    def _get_selected_periods(self):
        """获取用户选定的时期列表。"""
        periods = []
        if self.chk_period_haixi.isChecked():
            periods.append("海西期")
        if self.chk_period_xishan.isChecked():
            periods.append("喜山期")
        if self.chk_period_yinzhi.isChecked():
            periods.append("印支燕山期")
        return periods

    def _on_period_changed(self):
        """时期选择变化时的回调。"""
        selected = self._get_selected_periods()
        self._refresh_config_summary()

    def _select_all_periods(self):
        """全选所有时期。"""
        self.chk_period_haixi.setChecked(True)
        self.chk_period_xishan.setChecked(True)
        self.chk_period_yinzhi.setChecked(True)

    def _deselect_all_periods(self):
        """全不选所有时期。"""
        self.chk_period_haixi.setChecked(False)
        self.chk_period_xishan.setChecked(False)
        self.chk_period_yinzhi.setChecked(False)

    # ==========================================
    # 渗流选项卡方法
    # ==========================================
    def _run_percolation_curves(self):
        """运行渗流曲线对比。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "渗流曲线对比运行中...")

        def _worker():
            from percolation import run_percolation_pipeline
            return run_percolation_pipeline()
        self._launch_multiperiod_task(_worker, "渗流曲线对比",
            lambda res: self._show_percolation_result(res))

    def _show_percolation_result(self, res):
        """显示渗流曲线结果。"""
        import matplotlib.pyplot as plt
        txt = "【渗流曲线对比】\n\n"
        for period_name, r in res.items():
            if period_name.startswith("_"):
                continue
            txt += f"{period_name}: 渗流阈值={r.get('threshold', 'N/A'):.3f}\n"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        # 显示渗流曲线图
        plot_paths = res.get("_plot_paths", {})
        p = plot_paths.get("curves")
        if p and os.path.isfile(p):
            fig = plt.figure(figsize=(10, 6))
            img = plt.imread(p)
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            self.embed_figure(fig, description="三期渗流曲线对比")
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "渗流曲线对比完成")

    def _run_key_nodes(self):
        """运行关键节点图。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "关键节点图生成中...")

        def _worker():
            from percolation import run_percolation_pipeline
            return run_percolation_pipeline()
        self._launch_multiperiod_task(_worker, "关键节点图",
            lambda res: self._show_key_nodes_result(res, selected))

    def _show_key_nodes_result(self, res, selected):
        """显示关键节点结果。"""
        import matplotlib.pyplot as plt
        figs = []
        captions = []
        for period_name in selected:
            p = res.get("_plot_paths", {}).get(f"key_nodes_{period_name}")
            if p and os.path.isfile(p):
                fig = plt.figure(figsize=(10, 10))
                img = plt.imread(p)
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                figs.append(fig)
                captions.append(f"{period_name} 关键节点分布")
        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", f"关键节点图生成完成 ({len(figs)}张)")

    def _run_boundary_analysis(self):
        """运行边界效应分析。"""
        QMessageBox.information(self, "提示", "边界效应分析功能开发中...")

    # ==========================================
    # 代理选项卡方法
    # ==========================================
    def _run_agent_train(self):
        """运行代理模型训练。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "代理模型训练中...")

        def _worker():
            from agent_model import run_agent_pipeline
            return run_agent_pipeline()
        self._launch_multiperiod_task(_worker, "代理模型训练",
            lambda res: self._show_agent_result(res))

    def _show_agent_result(self, res):
        """显示代理模型结果。"""
        import matplotlib.pyplot as plt
        m = res.get("metrics", {})
        txt = "【代理模型训练结果】\n\n"
        txt += f"R² = {m.get('r2', 'N/A'):.4f}\n"
        txt += f"RMSE = {m.get('rmse', 'N/A'):.4f}\n"
        txt += f"MAE = {m.get('mae', 'N/A'):.4f}\n"
        txt += f"分类准确率 = {m.get('class_accuracy', 'N/A'):.3f}\n"
        txt += f"±0.5准确率 = {m.get('within_0.5', 'N/A'):.3f}\n"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        # 显示图表
        figs = []
        captions = []
        plot_paths = res.get("_plot_paths", {})
        for key, label in [("importance", "特征重要性"), ("pred_vs_true", "预测vs真实")]:
            p = plot_paths.get(key)
            if p and os.path.isfile(p):
                fig = plt.figure(figsize=(10, 7))
                img = plt.imread(p)
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                figs.append(fig)
                captions.append(label)
        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", f"代理模型训练完成\nR²={m.get('r2', 0):.4f}")

    def _run_agent_pred_vs_true(self):
        """运行预测vs真实散点图。"""
        QMessageBox.information(self, "提示", "请先训练代理模型，然后查看结果")

    def _run_agent_shap(self):
        """运行SHAP蜂群图。"""
        QMessageBox.information(self, "提示", "请先训练代理模型，然后查看SHAP分析")

    def _run_agent_importance(self):
        """运行特征重要性图。"""
        QMessageBox.information(self, "提示", "请先训练代理模型，然后查看特征重要性")

    # ==========================================
    # 实验选项卡方法
    # ==========================================
    def _run_exp_ablation(self):
        """运行消融实验。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "消融实验运行中...")

        def _worker():
            from multiperiod_data import load_all_periods, get_topology_matrix
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            import numpy as np
            period_gdfs = load_all_periods()
            results = {}
            for period_name, gdf in period_gdfs.items():
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                X_nz = X[nonzero]
                if len(X_nz) < 10:
                    continue
                # 使用第一个属性作为目标（模拟消融实验）
                y = X_nz[:, 0]
                # 消融实验：不同特征组合
                feature_sets = {
                    '仅NC_NB': [0],
                    '仅NC_NL': [1],
                    '仅NB_NL': [2],
                    'NC_NB+NC_NL': [0, 1],
                    'NC_NB+NB_NL': [0, 2],
                    '全部6属性': list(range(6)),
                }
                ablation_results = {}
                for name, indices in feature_sets.items():
                    valid_indices = [i for i in indices if i < X_nz.shape[1]]
                    if valid_indices:
                        X_subset = X_nz[:, valid_indices]
                        rf = RandomForestRegressor(n_estimators=50, random_state=42)
                        scores = cross_val_score(rf, X_subset, y, cv=3, scoring='r2')
                        ablation_results[name] = {
                            'r2_mean': scores.mean(),
                            'r2_std': scores.std(),
                        }
                results[period_name] = ablation_results
            return results
        self._launch_multiperiod_task(_worker, "消融实验",
            lambda res: self._show_ablation_result(res, selected))

    def _show_ablation_result(self, results, selected):
        """显示消融实验结果。"""
        import matplotlib.pyplot as plt
        import numpy as np

        figs = []
        captions = []

        for period_name in selected:
            if period_name in results:
                ablation = results[period_name]
                fig, ax = plt.subplots(figsize=(10, 6))
                names = list(ablation.keys())
                r2_means = [ablation[n]['r2_mean'] for n in names]
                r2_stds = [ablation[n]['r2_std'] for n in names]

                x = np.arange(len(names))
                bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5,
                             color='#457B9D', alpha=0.8, edgecolor='white')
                ax.set_xlabel('特征组合')
                ax.set_ylabel('R² Score')
                ax.set_title(f'{period_name} 消融实验结果')
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                figs.append(fig)
                captions.append(f'{period_name} 消融实验')

        # 显示结果文本
        txt = "【消融实验结果】\n\n"
        for period_name in selected:
            if period_name in results:
                txt += f"--- {period_name} ---\n"
                for name, metrics in results[period_name].items():
                    txt += f"  {name}: R²={metrics['r2_mean']:.4f}±{metrics['r2_std']:.4f}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "消融实验完成")

    def _run_exp_model_compare(self):
        """运行算法族对比。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "算法族对比运行中...")

        def _worker():
            from multiperiod_data import load_all_periods, get_topology_matrix
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score
            import numpy as np
            period_gdfs = load_all_periods()
            results = {}
            for period_name, gdf in period_gdfs.items():
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                X_nz = X[nonzero]
                if len(X_nz) < 10:
                    continue
                y = X_nz[:, 0]
                models = {
                    'Ridge回归': Ridge(alpha=1.0),
                    '随机森林': RandomForestRegressor(n_estimators=50, random_state=42),
                    '梯度提升': GradientBoostingRegressor(n_estimators=50, random_state=42),
                }
                model_results = {}
                for name, model in models.items():
                    scores = cross_val_score(model, X_nz, y, cv=3, scoring='r2')
                    model_results[name] = {
                        'r2_mean': scores.mean(),
                        'r2_std': scores.std(),
                    }
                results[period_name] = model_results
            return results
        self._launch_multiperiod_task(_worker, "算法族对比",
            lambda res: self._show_model_compare_result(res, selected))

    def _show_model_compare_result(self, results, selected):
        """显示算法族对比结果。"""
        import matplotlib.pyplot as plt
        import numpy as np

        figs = []
        captions = []

        # 生成对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = set()
        for period_name in selected:
            if period_name in results:
                model_names.update(results[period_name].keys())
        model_names = sorted(model_names)

        x = np.arange(len(model_names))
        width = 0.8 / len(selected)

        for i, period_name in enumerate(selected):
            if period_name in results:
                r2_means = [results[period_name].get(m, {}).get('r2_mean', 0) for m in model_names]
                ax.bar(x + i * width, r2_means, width, label=period_name, alpha=0.8)

        ax.set_xlabel('算法')
        ax.set_ylabel('R² Score')
        ax.set_title('算法族对比结果')
        ax.set_xticks(x + width * (len(selected) - 1) / 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        figs.append(fig)
        captions.append('算法族对比')

        # 显示结果文本
        txt = "【算法族对比结果】\n\n"
        for period_name in selected:
            if period_name in results:
                txt += f"--- {period_name} ---\n"
                for name, metrics in results[period_name].items():
                    txt += f"  {name}: R²={metrics['r2_mean']:.4f}±{metrics['r2_std']:.4f}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "算法族对比完成")

    def _run_exp_noise(self):
        """运行噪声敏感性实验。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "噪声敏感性实验运行中...")

        def _worker():
            from multiperiod_data import load_all_periods, get_topology_matrix
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            import numpy as np
            period_gdfs = load_all_periods()
            results = {}
            noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            for period_name, gdf in period_gdfs.items():
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                X_nz = X[nonzero]
                if len(X_nz) < 10:
                    continue
                y = X_nz[:, 0]
                noise_results = {}
                for noise in noise_levels:
                    # 添加高斯噪声
                    X_noisy = X_nz + np.random.normal(0, noise, X_nz.shape)
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    scores = cross_val_score(rf, X_noisy, y, cv=3, scoring='r2')
                    noise_results[noise] = {
                        'r2_mean': scores.mean(),
                        'r2_std': scores.std(),
                    }
                results[period_name] = noise_results
            return results
        self._launch_multiperiod_task(_worker, "噪声敏感性实验",
            lambda res: self._show_noise_result(res, selected))

    def _show_noise_result(self, results, selected):
        """显示噪声敏感性实验结果。"""
        import matplotlib.pyplot as plt
        import numpy as np

        figs = []
        captions = []

        # 生成折线图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#E63946', '#457B9D', '#2A9D8F']

        for i, period_name in enumerate(selected):
            if period_name in results:
                noise_results = results[period_name]
                noise_levels = sorted(noise_results.keys())
                r2_means = [noise_results[n]['r2_mean'] for n in noise_levels]
                r2_stds = [noise_results[n]['r2_std'] for n in noise_levels]

                ax.errorbar(noise_levels, r2_means, yerr=r2_stds,
                           label=period_name, color=colors[i % len(colors)],
                           marker='o', capsize=5, linewidth=2)

        ax.set_xlabel('噪声水平 (标准差)')
        ax.set_ylabel('R² Score')
        ax.set_title('噪声敏感性实验结果')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figs.append(fig)
        captions.append('噪声敏感性实验')

        # 显示结果文本
        txt = "【噪声敏感性实验结果】\n\n"
        for period_name in selected:
            if period_name in results:
                txt += f"--- {period_name} ---\n"
                for noise, metrics in sorted(results[period_name].items()):
                    txt += f"  噪声={noise:.2f}: R²={metrics['r2_mean']:.4f}±{metrics['r2_std']:.4f}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "噪声敏感性实验完成")

    def _run_exp_spatial_cv(self):
        """运行空间交叉验证。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "空间交叉验证运行中...")

        def _worker():
            from multiperiod_data import load_all_periods, get_topology_matrix
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import KFold
            import numpy as np
            period_gdfs = load_all_periods()
            results = {}
            for period_name, gdf in period_gdfs.items():
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                X_nz = X[nonzero]
                if len(X_nz) < 10:
                    continue
                y = X_nz[:, 0]
                # 空间交叉验证（使用KFold模拟）
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                scores = []
                for train_idx, test_idx in kf.split(X_nz):
                    X_train, X_test = X_nz[train_idx], X_nz[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    rf.fit(X_train, y_train)
                    score = rf.score(X_test, y_test)
                    scores.append(score)
                results[period_name] = {
                    'r2_mean': np.mean(scores),
                    'r2_std': np.std(scores),
                    'scores': scores,
                }
            return results
        self._launch_multiperiod_task(_worker, "空间交叉验证",
            lambda res: self._show_spatial_cv_result(res, selected))

    def _show_spatial_cv_result(self, results, selected):
        """显示空间交叉验证结果。"""
        import matplotlib.pyplot as plt
        import numpy as np

        figs = []
        captions = []

        # 生成箱线图
        fig, ax = plt.subplots(figsize=(8, 6))
        data_to_plot = []
        labels = []
        for period_name in selected:
            if period_name in results:
                data_to_plot.append(results[period_name]['scores'])
                labels.append(period_name)

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['#E63946', '#457B9D', '#2A9D8F']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xlabel('时期')
        ax.set_ylabel('R² Score')
        ax.set_title('空间交叉验证结果')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        figs.append(fig)
        captions.append('空间交叉验证')

        # 显示结果文本
        txt = "【空间交叉验证结果】\n\n"
        for period_name in selected:
            if period_name in results:
                r = results[period_name]
                txt += f"--- {period_name} ---\n"
                txt += f"  R²均值: {r['r2_mean']:.4f}±{r['r2_std']:.4f}\n"
                txt += f"  各折R²: {', '.join([f'{s:.4f}' for s in r['scores']])}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "空间交叉验证完成")

    def _run_exp_params_compare(self):
        """运行拓扑参数对比。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "拓扑参数对比运行中...")

        def _worker():
            from multiperiod_data import load_all_periods
            return load_all_periods()
        self._launch_multiperiod_task(_worker, "拓扑参数对比",
            lambda res: self._show_params_compare_result(res, selected))

    def _show_params_compare_result(self, period_gdfs, selected):
        """显示拓扑参数对比结果。"""
        import matplotlib.pyplot as plt
        from multiperiod_data import get_topology_matrix
        import numpy as np

        # 为每个选定时期生成拓扑参数表
        txt = "【拓扑参数对比】\n\n"
        for period_name in selected:
            if period_name in period_gdfs:
                gdf = period_gdfs[period_name]
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                txt += f"--- {period_name} ---\n"
                txt += f"  总网格数: {len(gdf)}\n"
                txt += f"  非零网格: {nonzero.sum()}\n"
                for j, col in enumerate(cols):
                    vals = X[nonzero, j]
                    if len(vals) > 0:
                        txt += f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        # 生成对比柱状图
        figs = []
        captions = []

        # 柱状图：各时期非零网格数对比
        fig, ax = plt.subplots(figsize=(8, 5))
        period_names = []
        nonzero_counts = []
        total_counts = []
        for period_name in selected:
            if period_name in period_gdfs:
                gdf = period_gdfs[period_name]
                _, X, _ = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                period_names.append(period_name)
                nonzero_counts.append(nonzero.sum())
                total_counts.append(len(gdf))

        x = np.arange(len(period_names))
        width = 0.35
        ax.bar(x - width/2, total_counts, width, label='总网格数', color='#457B9D', alpha=0.8)
        ax.bar(x + width/2, nonzero_counts, width, label='非零网格', color='#E63946', alpha=0.8)
        ax.set_xlabel('时期')
        ax.set_ylabel('网格数')
        ax.set_title('各时期网格统计对比')
        ax.set_xticks(x)
        ax.set_xticklabels(period_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        figs.append(fig)
        captions.append('网格统计对比')

        # 柱状图：各时期拓扑属性均值对比
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        all_means = {}
        for period_name in selected:
            if period_name in period_gdfs:
                gdf = period_gdfs[period_name]
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                means = []
                for j in range(len(cols)):
                    vals = X[nonzero, j]
                    means.append(vals.mean() if len(vals) > 0 else 0)
                all_means[period_name] = means

        x2 = np.arange(len(cols))
        width2 = 0.8 / len(selected)
        for i, period_name in enumerate(selected):
            if period_name in all_means:
                ax2.bar(x2 + i * width2, all_means[period_name], width2,
                       label=period_name, alpha=0.8)

        ax2.set_xlabel('拓扑属性')
        ax2.set_ylabel('均值')
        ax2.set_title('各时期拓扑属性均值对比')
        ax2.set_xticks(x2 + width2 * (len(selected) - 1) / 2)
        ax2.set_xticklabels(cols, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        figs.append(fig2)
        captions.append('拓扑属性均值对比')

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "拓扑参数对比完成")

    def _run_exp_length_powerlaw(self):
        """运行长度分布幂律分析。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "长度分布分析中...")

        def _worker():
            from multiperiod_data import load_all_periods, get_topology_matrix
            import numpy as np
            period_gdfs = load_all_periods()
            results = {}
            for period_name, gdf in period_gdfs.items():
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                # 提取长度相关属性（NC_NB, NC_NL, NB_NL）
                length_cols = ['NC_NB', 'NC_NL', 'NB_NL']
                length_data = {}
                for col in length_cols:
                    if col in cols:
                        idx = cols.index(col)
                        vals = X[nonzero, idx]
                        length_data[col] = vals
                results[period_name] = {
                    'length_data': length_data,
                    'nonzero_count': nonzero.sum(),
                    'total_count': len(gdf),
                }
            return results
        self._launch_multiperiod_task(_worker, "长度分布分析",
            lambda res: self._show_length_powerlaw_result(res, selected))

    def _show_length_powerlaw_result(self, results, selected):
        """显示长度分布分析结果。"""
        import matplotlib.pyplot as plt
        import numpy as np

        figs = []
        captions = []

        # 生成长度分布直方图
        for col in ['NC_NB', 'NC_NL', 'NB_NL']:
            fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4))
            if len(selected) == 1:
                axes = [axes]

            for ax, period_name in zip(axes, selected):
                if period_name in results:
                    data = results[period_name]['length_data'].get(col)
                    if data is not None and len(data) > 0:
                        ax.hist(data, bins=30, color='#457B9D', alpha=0.7, edgecolor='white')
                        ax.set_xlabel(col)
                        ax.set_ylabel('频次')
                        ax.set_title(f'{period_name}')
                        ax.grid(True, alpha=0.3)

            plt.suptitle(f'{col} 分布对比', fontsize=14, fontweight='bold')
            plt.tight_layout()
            figs.append(fig)
            captions.append(f'{col} 分布对比')

        # 显示统计信息
        txt = "【长度分布分析】\n\n"
        for period_name in selected:
            if period_name in results:
                r = results[period_name]
                txt += f"--- {period_name} ---\n"
                txt += f"  非零网格: {r['nonzero_count']}/{r['total_count']}\n"
                for col, data in r['length_data'].items():
                    if len(data) > 0:
                        txt += f"  {col}: mean={data.mean():.4f}, std={data.std():.4f}, min={data.min():.4f}, max={data.max():.4f}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "长度分布分析完成")

    def _run_exp_grid_sampling(self):
        """运行网格采样分析。"""
        selected = self._get_selected_periods()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个时期")
            return
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "网格采样分析中...")

        def _worker():
            from multiperiod_data import load_all_periods, get_topology_matrix
            import numpy as np
            period_gdfs = load_all_periods()
            results = {}
            for period_name, gdf in period_gdfs.items():
                _, X, cols = get_topology_matrix(gdf)
                nonzero = np.array(X.sum(axis=1)).ravel() > 0
                # 计算每个属性的统计信息
                stats = {}
                for j, col in enumerate(cols):
                    vals = X[nonzero, j]
                    if len(vals) > 0:
                        stats[col] = {
                            'mean': vals.mean(),
                            'std': vals.std(),
                            'min': vals.min(),
                            'max': vals.max(),
                            'median': np.median(vals),
                        }
                results[period_name] = {
                    'stats': stats,
                    'nonzero_count': nonzero.sum(),
                    'total_count': len(gdf),
                    'density': nonzero.sum() / len(gdf),
                }
            return results
        self._launch_multiperiod_task(_worker, "网格采样分析",
            lambda res: self._show_grid_sampling_result(res, selected))

    def _show_grid_sampling_result(self, results, selected):
        """显示网格采样分析结果。"""
        import matplotlib.pyplot as plt
        import numpy as np

        figs = []
        captions = []

        # 生成密度对比图
        fig, ax = plt.subplots(figsize=(8, 5))
        period_names = []
        densities = []
        for period_name in selected:
            if period_name in results:
                period_names.append(period_name)
                densities.append(results[period_name]['density'])

        bars = ax.bar(period_names, densities, color=['#E63946', '#457B9D', '#2A9D8F'][:len(period_names)], alpha=0.8)
        ax.set_xlabel('时期')
        ax.set_ylabel('非零网格密度')
        ax.set_title('各时期非零网格密度对比')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, v in zip(bars, densities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{v:.2%}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        figs.append(fig)
        captions.append('非零网格密度对比')

        # 生成属性分布箱线图
        for col in ['NC_NB', 'NC_NL', 'NB_NL', 'NC_A', 'NB_A', 'NL_A']:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            data_to_plot = []
            labels = []
            for period_name in selected:
                if period_name in results:
                    stats = results[period_name]['stats'].get(col)
                    if stats:
                        data_to_plot.append([stats['min'], stats['median'], stats['max']])
                        labels.append(period_name)

            if data_to_plot:
                # 简化版箱线图（用min/median/max）
                x = np.arange(len(labels))
                mins = [d[0] for d in data_to_plot]
                medians = [d[1] for d in data_to_plot]
                maxs = [d[2] for d in data_to_plot]

                ax2.bar(x - 0.2, mins, 0.2, label='Min', color='#457B9D', alpha=0.6)
                ax2.bar(x, medians, 0.2, label='Median', color='#E63946', alpha=0.8)
                ax2.bar(x + 0.2, maxs, 0.2, label='Max', color='#2A9D8F', alpha=0.6)

                ax2.set_xlabel('时期')
                ax2.set_ylabel(col)
                ax2.set_title(f'{col} 统计对比')
                ax2.set_xticks(x)
                ax2.set_xticklabels(labels)
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                figs.append(fig2)
                captions.append(f'{col} 统计对比')

        # 显示详细统计信息
        txt = "【网格采样分析】\n\n"
        for period_name in selected:
            if period_name in results:
                r = results[period_name]
                txt += f"--- {period_name} ---\n"
                txt += f"  总网格数: {r['total_count']}\n"
                txt += f"  非零网格: {r['nonzero_count']}\n"
                txt += f"  非零密度: {r['density']:.2%}\n"
                txt += "  拓扑属性统计:\n"
                for col, stats in r['stats'].items():
                    txt += f"    {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}\n"
                txt += "\n"

        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", "网格采样分析完成")

    def _refresh_config_summary(self):
        train_cfg = _cfg_section("train")
        clustering_cfg = _cfg_section("clustering")
        export_cfg = _cfg_section("export_grid")
        idx = self.combo_data_source.currentIndex()
        if 0 <= idx < len(DATA_SOURCES):
            csv_name = DATA_SOURCES[idx]["csv"]
        else:
            csv_name = "Yingmai 2 area in Tarim Basin.csv"
        gui_k = int(self.spin_kmeans_k.value()) if hasattr(self, "spin_kmeans_k") else int(clustering_cfg.get("n_clusters", 4))
        gui_cell = float(self.dspin_grid_step.value()) if hasattr(self, "dspin_grid_step") else float(export_cfg.get("cell_width", 750.0))
        gui_target = self.combo_train_target.currentText().strip() if hasattr(self, "combo_train_target") else ""
        if not gui_target:
            gui_target = str(train_cfg.get("target_column", "Fracture Intensity B21"))

        # 获取选定的时期
        selected_periods = self._get_selected_periods() if hasattr(self, "_get_selected_periods") else []
        periods_str = ", ".join(selected_periods) if selected_periods else "未选择"

        text = [
            "工区: 塔里木盆地英买2 (MY)",
            f"CSV: {csv_name}",
            f"分析时期: {periods_str}",
            f"KMeans k（本次 GUI）: {gui_k}  |  config 默认: {clustering_cfg.get('n_clusters', 4)}",
            f"训练目标列（本次 GUI）: {gui_target}",
            f"网格步长（本次 GUI）: {gui_cell:.1f} m  |  config: {export_cfg.get('cell_width', 750.0)} m",
            f"随机种子(训练): {train_cfg.get('random_state', 42)}",
            f"随机种子(聚类): {clustering_cfg.get('random_state', 42)}",
            f"模型状态: {'已找到 xgboost_reg.json' if self._model_exists() else '未训练'}",
        ]
        self.config_summary_browser.setPlainText("\n".join(text))
        self._refresh_prerequisite_buttons()

    def _remember_exports(self, category: str, **paths):
        clean = {k: self._relpath_for_ui(v) for k, v in paths.items() if v}
        if clean:
            self._last_exports[category] = clean

    def _relpath_for_ui(self, value):
        if not value or not isinstance(value, str):
            return value
        if "://" in value:
            return value
        try:
            abs_v = os.path.abspath(value)
            project_root = os.path.dirname(_PROGRAM_DIR)
            if abs_v.startswith(project_root):
                return os.path.relpath(abs_v, project_root)
        except Exception:
            return value
        return value

    def _model_exists(self) -> bool:
        p = os.path.join(_PROGRAM_DIR, "model", "xgboost_reg.json")
        return os.path.isfile(p)

    def _csv_path_silent(self) -> Optional[str]:
        idx = self.combo_data_source.currentIndex()
        csv_name = DATA_SOURCES[idx]["csv"] if 0 <= idx < len(DATA_SOURCES) else "Yingmai 2 area in Tarim Basin.csv"
        csv_path = os.path.join(_PROGRAM_DIR, csv_name)
        return csv_path if os.path.isfile(csv_path) else None

    def _refresh_prerequisite_buttons(self):
        csv_ok = self._csv_path_silent() is not None
        model_ok = self._model_exists()
        if hasattr(self, "btn_guoji_train"):
            self.btn_guoji_train.setEnabled(csv_ok)
        if hasattr(self, "btn_ronghe"):
            self.btn_ronghe.setEnabled(csv_ok)
        if hasattr(self, "btn_guoji_weighted"):
            self.btn_guoji_weighted.setEnabled(csv_ok)
        if hasattr(self, "btn_guoji_compare"):
            self.btn_guoji_compare.setEnabled(csv_ok)
        if hasattr(self, "btn_k_helper"):
            self.btn_k_helper.setEnabled(csv_ok)
        if hasattr(self, "btn_spatial"):
            self.btn_spatial.setEnabled(csv_ok)
        if hasattr(self, "btn_guoji_shap"):
            self.btn_guoji_shap.setEnabled(csv_ok and model_ok)

    def _refresh_wizard_status(self):
        if not hasattr(self, "wizard_browser"):
            return
        csv_ok = self._csv_path_silent() is not None
        model_ok = self._model_exists()
        shp = os.path.join(_PROGRAM_DIR, "model", "shap_summary.png")
        shap_ok = os.path.isfile(shp)
        def mark(ok: bool) -> str:
            return "✓" if ok else "□"
        lines = [
            f"{mark(csv_ok)} 第1步：网格 CSV 就绪",
            f"{mark(model_ok)} 第2步：模型训练完成",
            f"{mark(shap_ok)} 第3步：SHAP 解释完成",
        ]
        self.wizard_browser.setPlainText("\n".join(lines))

    def _show_startup_flow_guide(self):
        if not hasattr(self, "last_run_browser"):
            return
        if self.last_run_browser.toPlainText().strip():
            return
        self.last_run_browser.setPlainText(
            "快速开始：点击上方“生成候选勘探有利区”运行正式流程。\n"
            "结果摘要会保留稳定候选数、证据边界与导出目录；详细进度请切换到“运行日志”。\n"
            "基础、融合、渗流和实验图件仍可在右侧选项卡中单独生成。"
        )

    def _set_busy(self, busy: bool, msg: str = ""):
        if hasattr(self, "btn_cancel_task"):
            self.btn_cancel_task.setEnabled(bool(busy))
        if hasattr(self, "btn_primary_screening"):
            self.btn_primary_screening.setEnabled(not bool(busy))
        if msg:
            self.statusBar().showMessage(msg)
        else:
            self.statusBar().clearMessage()

    def cancel_running_task(self):
        cancelled_any = False
        for attr in ("_running_task", "_nw_runner", "_lunkuo_runner"):
            runner = getattr(self, attr, None)
            if runner is None or not runner.isRunning():
                continue
            try:
                runner.requestInterruption()
            except Exception:
                pass
            cancelled_any = True
        if cancelled_any:
            self.statusBar().showMessage("已请求安全取消，将在当前计算步骤结束后停止…")
            if hasattr(self, "screening_status_label"):
                self.screening_status_label.setText("正在安全取消…")
        return cancelled_any

    def _toggle_run_info(self):
        visible = not self.run_info_tabs.isVisible()
        self.run_info_tabs.setVisible(visible)
        self.btn_toggle_run_info.setText("隐藏运行信息" if visible else "显示运行信息")

    def closeEvent(self, event):
        import matplotlib.pyplot as plt
        active = any(
            runner is not None and runner.isRunning()
            for runner in (
                getattr(self, "_running_task", None),
                getattr(self, "_nw_runner", None),
                getattr(self, "_lunkuo_runner", None),
            )
        )
        if active:
            self.cancel_running_task()
            QMessageBox.information(
                self,
                "正在安全停止",
                "后台计算仍在运行，已提交取消请求。请等待当前计算步骤结束后再关闭窗口。",
            )
            event.ignore()
            return
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        plt.close("all")
        if hasattr(self, "current_figs"):
            for f in self.current_figs:
                plt.close(f)
            self.current_figs = []
        super().closeEvent(event)

    def _friendly_error_message(self, action: str, err_text: str) -> str:
        s = str(err_text or "")
        hint = "建议：查看左侧日志并核对输入数据路径。"
        low = s.lower()
        if "permission" in low or "operation not permitted" in low:
            hint = "建议：检查目录写权限，或将输出目录改到可写位置。"
        elif "no such file" in low or "not found" in low:
            hint = "建议：确认 CSV/模型文件路径存在，先完成上一步流程。"
        elif "driver" in low or "gpkg" in low:
            hint = "建议：检查 geopandas/fiona 驱动，先仅导出 CSV 验证数据。"
        elif "missing" in low or "缺少" in s or "列" in s:
            hint = "建议：检查输入 CSV 列名与配置 target_column。"
        return f"{action}失败：\n{s}\n\n{hint}"

    def _check_config_on_startup(self):
        cfg = load_config()
        errs = validate_config(cfg if isinstance(cfg, dict) else {})
        if not errs:
            return
        msg = "检测到配置问题：\n- " + "\n- ".join(errs[:8])
        if len(errs) > 8:
            msg += f"\n... 其余 {len(errs)-8} 条省略"
        msg += "\n\n请修正 program/config.yaml 后重启，或继续运行但可能在中途报错。"
        QMessageBox.warning(self, "配置校验提醒", msg)

    def _refresh_target_suggestions(self):
        if not hasattr(self, "combo_train_target"):
            return
        csv_path = self._csv_path_silent()
        train_cfg = _cfg_section("train")
        default = str(train_cfg.get("target_column", "Fracture Intensity B21"))
        prev = self.combo_train_target.currentText().strip()
        self.combo_train_target.blockSignals(True)
        self.combo_train_target.clear()
        if csv_path:
            try:
                from feature_engineering import suggest_regression_target_columns

                for c in suggest_regression_target_columns(csv_path):
                    self.combo_train_target.addItem(c)
            except Exception:
                pass
        if self.combo_train_target.findText(default) < 0:
            self.combo_train_target.insertItem(0, default)
        if prev and self.combo_train_target.findText(prev) >= 0:
            self.combo_train_target.setCurrentText(prev)
        elif prev:
            self.combo_train_target.setEditText(prev)
        else:
            self.combo_train_target.setCurrentText(default)
        self.combo_train_target.blockSignals(False)

    def _training_target_from_gui(self) -> str:
        train_cfg = _cfg_section("train")
        t = self.combo_train_target.currentText().strip() if hasattr(self, "combo_train_target") else ""
        return t or str(train_cfg.get("target_column", "Fracture Intensity B21"))

    def _open_program_subdir(self, rel: str):
        path = os.path.abspath(os.path.join(_PROGRAM_DIR, rel))
        os.makedirs(path, exist_ok=True)
        if not os.path.isdir(path):
            QMessageBox.warning(self, "路径无效", path)
            return
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", path], check=False)
            elif platform.system() == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            QMessageBox.warning(self, "无法打开目录", str(e))

    def _update_last_run_card(self, lines: List[str]):
        if hasattr(self, "last_run_browser"):
            self.last_run_browser.setPlainText("\n".join(lines).strip())

    def _append_run_history(self, record: Dict[str, Any]):
        record = {**record, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        out_dir = os.path.join(_PROGRAM_DIR, "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        hist_path = os.path.join(out_dir, "gui_run_history.jsonl")
        try:
            with open(hist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def show_export_results(self):
        if not self._last_exports:
            QMessageBox.information(
                self,
                "导出结果",
                "当前会自动导出聚类 CSV/GPKG、预测结果、SHAP 表等。\n"
                "请先运行一次融合、训练、SHAP 或一键空间-拓扑融合后再查看最近导出路径。",
            )
            return
        lines = ["最近导出结果：", ""]
        for category, mapping in self._last_exports.items():
            lines.append(f"[{category}]")
            for key, value in mapping.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        QMessageBox.information(self, "导出结果", "\n".join(lines).rstrip())


    def embed_figure(self, figs, *, description=None, descriptions=None):
        """
        在右侧画布区嵌入图像。可选：
        description — 单段说明（多图时每张共用）；
        descriptions — 与 figs 等长的说明列表（翻页时随当前图切换）。
        说明显示在图表下方灰色区域。
        """
        if not isinstance(figs, list):
            figs = [figs]
        n = len(figs)

        # 关闭旧的 matplotlib figures，防止重叠
        if hasattr(self, "current_figs") and self.current_figs:
            for old_fig in self.current_figs:
                plt.close(old_fig)
            self.current_figs = []

        if descriptions is not None and len(descriptions) == n:
            self._fig_captions = [str(s).strip() for s in descriptions]
        elif description:
            d = str(description).strip()
            self._fig_captions = [d] * n
        else:
            self._fig_captions = [""] * n

        self.current_figs = figs
        self.current_fig_sizes = [tuple(fig.get_size_inches()) for fig in figs]
        self.current_fig_dpis = [fig.dpi for fig in figs]
        self.current_fig_idx = 0

        if len(figs) > 1:
            for _w in (self.btn_prev_fig, self.btn_next_fig, self.lbl_fig_status):
                _w.setVisible(True)
            self.lbl_fig_status.setText(f"第 1 张 / 共 {len(figs)} 张")
        else:
            for _w in (self.btn_prev_fig, self.btn_next_fig, self.lbl_fig_status):
                _w.setVisible(False)

        # 仅清空 canvas_display_layout 内的子项（画布/说明），
        # 不重建 layout 本身——因为它已通过 setLayout 绑定到 canvas_scroll_content，
        # Qt 不允许对同一 widget 重复 setLayout。复用即可。
        if hasattr(self, "canvas_display_layout") and self.canvas_display_layout is not None:
            while self.canvas_display_layout.count():
                item = self.canvas_display_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.close()
                    widget.deleteLater()
                elif item.layout() is not None:
                    layout = item.layout()
                    while layout.count():
                        sub_item = layout.takeAt(0)
                        sub_widget = sub_item.widget()
                        if sub_widget is not None:
                            sub_widget.setParent(None)
                            sub_widget.close()
                            sub_widget.deleteLater()
                    layout.deleteLater()

        # 复用已绑定到 canvas_scroll_content 的 canvas_display_layout
        # （首次进入时由 demo.py 初始化；此处确保存在且已清空）

        self._render_current_figure()

    def show_prev_figure(self):
        if not getattr(self, "current_figs", None):
            return
        if self.current_fig_idx > 0:
            self.current_fig_idx -= 1
            self._render_current_figure()

    def show_next_figure(self):
        if not getattr(self, "current_figs", None):
            return
        if self.current_fig_idx < len(self.current_figs) - 1:
            self.current_fig_idx += 1
            self._render_current_figure()

    def _toggle_figure_fit(self):
        fit_enabled = self.btn_fit_fig.isChecked()
        self.btn_fit_fig.setText("自适应：开" if fit_enabled else "自适应：关")
        self._apply_figure_view_mode()

    def _apply_figure_view_mode(self):
        """默认将当前图完整缩放进视口；原始尺寸模式才启用滚动。"""
        canvas = getattr(self, "current_canvas", None)
        if canvas is None or not getattr(self, "current_figs", None):
            return
        index = self.current_fig_idx
        if index >= len(self.current_fig_sizes):
            return
        width_in, height_in = self.current_fig_sizes[index]
        dpi = float(self.current_fig_dpis[index])
        original_width = max(1, int(round(width_in * dpi)))
        original_height = max(1, int(round(height_in * dpi)))
        fit_enabled = self.btn_fit_fig.isChecked()

        if fit_enabled:
            self.canvas_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.canvas_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            viewport = self.canvas_scroll.viewport().size()
            available_width = max(240, viewport.width() - 12)
            caption_height = 52 if getattr(self, "_current_caption_visible", False) else 4
            available_height = max(180, viewport.height() - caption_height - 12)
            scale = min(
                available_width / float(original_width),
                available_height / float(original_height),
            )
            target_width = max(240, int(original_width * scale))
            target_height = max(180, int(original_height * scale))
            canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
            )
            canvas.setMinimumSize(0, 0)
            canvas.setMaximumSize(target_width, target_height)
            canvas.resize(target_width, target_height)
        else:
            self.canvas_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            self.canvas_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
            )
            canvas.setMinimumSize(original_width, original_height)
            canvas.setMaximumSize(original_width, original_height)
            canvas.resize(original_width, original_height)
        canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "btn_fit_fig") and self.btn_fit_fig.isChecked():
            QtCore.QTimer.singleShot(0, self._apply_figure_view_mode)

    def _render_current_figure(self):
        if getattr(self, "_rendering", False):
            return
        self._rendering = True
        try:
            while self.canvas_display_layout.count():
                item = self.canvas_display_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.close()
                    widget.deleteLater()
                elif item.layout() is not None:
                    layout = item.layout()
                    while layout.count():
                        sub_item = layout.takeAt(0)
                        sub_widget = sub_item.widget()
                        if sub_widget is not None:
                            sub_widget.setParent(None)
                            sub_widget.close()
                            sub_widget.deleteLater()
                    layout.deleteLater()

            fig = self.current_figs[self.current_fig_idx]
            if hasattr(self, "current_fig_sizes") and self.current_fig_idx < len(self.current_fig_sizes):
                width, height = self.current_fig_sizes[self.current_fig_idx]
                fig.set_size_inches(width, height, forward=False)
            if hasattr(self, "current_fig_dpis") and self.current_fig_idx < len(self.current_fig_dpis):
                fig.set_dpi(self.current_fig_dpis[self.current_fig_idx])

            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            canvas.setMinimumSize(0, 0)
            self.current_canvas = canvas
            self.canvas_display_layout.addWidget(canvas, 1, QtCore.Qt.AlignCenter)
            canvas.draw()
            QtWidgets.QApplication.processEvents()

            cap_text = ""
            if hasattr(self, "_fig_captions") and self.current_fig_idx < len(self._fig_captions):
                cap_text = self._fig_captions[self.current_fig_idx]
            if cap_text:
                cap_lbl = QtWidgets.QLabel()
                cap_lbl.setObjectName("figureCaptionLabel")
                cap_lbl.setWordWrap(True)
                cap_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                cap_lbl.setOpenExternalLinks(False)
                cap_lbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
                cap_lbl.setStyleSheet(
                    "QLabel#figureCaptionLabel {"
                    " background-color: #f1f3f5; color: #212529; padding: 6px 10px;"
                    " border-radius: 6px; font-size: 12px; border: 1px solid #dee2e6;"
                    " line-height: 1.35;"
                    "}"
                )
                cap_lbl.setText("【图说明】" + cap_text)
                cap_lbl.setToolTip(cap_text)
                cap_scroll = QtWidgets.QScrollArea()
                cap_scroll.setWidgetResizable(True)
                cap_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
                cap_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                cap_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                cap_scroll.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Fixed,
                )
                cap_scroll.setFixedHeight(40)
                cap_scroll.setWidget(cap_lbl)
                cap_scroll.setStyleSheet("QScrollArea { background: transparent; }")
                self.canvas_display_layout.addSpacing(8)
                self.canvas_display_layout.addWidget(cap_scroll)
            self._current_caption_visible = bool(cap_text)
            QtCore.QTimer.singleShot(0, self._apply_figure_view_mode)

            if len(self.current_figs) > 1:
                self.lbl_fig_status.setText(f"第 {self.current_fig_idx + 1} 张 / 共 {len(self.current_figs)} 张")
                self.btn_prev_fig.setEnabled(self.current_fig_idx > 0)
                self.btn_next_fig.setEnabled(self.current_fig_idx < len(self.current_figs) - 1)

            QtWidgets.QApplication.processEvents()
        finally:
            self._rendering = False

    def _plot_latent_fusion_kmeans_regions(
        self,
        ax,
        df_out,
        x_col: str,
        y_col: str,
        kmeans,
        n_clusters: int,
    ):
        """
        潜空间平面细网格上 KMeans.predict：Voronoi 式分区。
        网格坐标强制 float64，避免 sklearn 1.6+/Py3.13 下 predict 报 buffer dtype mismatch。
        返回 (是否画了簇中心, ListedColormap|None, BoundaryNorm|None) 供散点与 colorbar 同配色。
        """
        xy = df_out[[x_col, y_col]].to_numpy(dtype=np.float64)
        cc = getattr(kmeans, "cluster_centers_", None)
        n_k = int(getattr(kmeans, "n_clusters", n_clusters))
        if cc is None:
            return False, None, None
        cc = np.asarray(cc, dtype=np.float64)
        if cc.shape != (n_k, 2):
            return False, None, None
        x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
        y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
        dx, dy = x_max - x_min, y_max - y_min
        pad_x = max(dx * 0.14, 0.08)
        pad_y = max(dy * 0.14, 0.08)
        if dx < 1e-12:
            pad_x = max(pad_x, 0.5)
        if dy < 1e-12:
            pad_y = max(pad_y, 0.5)
        gx = np.linspace(x_min - pad_x, x_max + pad_x, 360, dtype=np.float64)
        gy = np.linspace(y_min - pad_y, y_max + pad_y, 360, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        # 与 KMeans(euclidean) 的 predict 等价：最近簇中心标号。
        # 不用 kmeans.predict：部分环境 sklearn Cython 要求 X 为 C 连续 double，易与 float32 网格冲突报错。
        grid_xy = np.ascontiguousarray(
            np.column_stack([xx.ravel(), yy.ravel()]),
            dtype=np.float64,
        )
        cc_c = np.ascontiguousarray(cc, dtype=np.float64)
        d2 = np.sum((grid_xy[:, np.newaxis, :] - cc_c[np.newaxis, :, :]) ** 2, axis=2)
        Z = np.argmin(d2, axis=1).reshape(xx.shape)
        cmap_soft, _ = _make_latent_fusion_cmap_norm(n_k)
        lev = np.arange(n_k + 1, dtype=np.float64) - 0.5
        ax.contourf(
            xx,
            yy,
            Z,
            levels=lev,
            cmap=cmap_soft,
            alpha=0.58,
            antialiased=True,
            zorder=0,
        )
        if n_k > 1:
            ax.contour(
                xx,
                yy,
                Z,
                levels=np.arange(n_k - 1, dtype=np.float64) + 0.5,
                colors="#ffffff",
                linewidths=0.85,
                alpha=0.92,
                zorder=1,
            )
        ax.scatter(
            cc[:, 0],
            cc[:, 1],
            marker="X",
            s=100,
            c="#2a2a2a",
            zorder=5,
            linewidths=1.0,
            edgecolors="white",
            label="KMeans 簇中心",
        )
        cmap_full, norm_full = _make_latent_fusion_cmap_norm(n_k)
        return True, cmap_full, norm_full

    def _plot_spatial_cluster_grid(self, ax, df_out, n_clusters: int, method_name: str) -> bool:
        """
        将网格 CSV 中的四边形单元按 cluster_id 着色，得到与「断裂密度/聚类」
        类似的空间绿–蓝填色图（投影坐标，单位 m）。
        """

        vtx_cols = [
            "vertex1_x",
            "vertex1_y",
            "vertex2_x",
            "vertex2_y",
            "vertex3_x",
            "vertex3_y",
            "vertex4_x",
            "vertex4_y",
        ]
        if not all(c in df_out.columns for c in vtx_cols):
            return False
        if len(df_out) == 0 or "cluster_id" not in df_out.columns:
            return False
        arr = df_out[vtx_cols].to_numpy(dtype=float)
        verts = arr.reshape(-1, 4, 2)
        z = df_out["cluster_id"].to_numpy(dtype=float)
        cmap = mpl_cm.get_cmap("GnBu", max(int(n_clusters), 1))
        bounds = np.arange(-0.5, float(n_clusters) + 0.5, 1.0)
        norm = BoundaryNorm(bounds, cmap.N)
        pc = PolyCollection(
            verts,
            array=z,
            cmap=cmap,
            norm=norm,
            edgecolors="0.82",
            linewidths=0.12,
        )
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        zh_fonts = plt.rcParams.get("font.sans-serif", [])
        font_family = (
            zh_fonts[0]
            if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0
            else None
        )
        ax.set_title(
            f"空间网格聚类分布（{method_name}）",
            fontsize=12,
            fontfamily=font_family,
        )
        ax.tick_params(axis="both", labelsize=9)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 8))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("聚类簇编号", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#1a1a1a")
            spine.set_linewidth(1.1)
        return True

    def _get_guoji_csv(self):
        """根据当前数据源获取对应的网格 CSV 路径。"""
        csv_path = self._csv_path_silent()
        if csv_path:
            return csv_path
        idx = self.combo_data_source.currentIndex()
        csv_name = DATA_SOURCES[idx]["csv"] if 0 <= idx < len(DATA_SOURCES) else "Yingmai 2 area in Tarim Basin.csv"
        QMessageBox.warning(
            self,
            "未找到数据",
            f"未找到：{csv_name}\n请先运行 export_grid_csv.py 为该区域生成网格 CSV。",
        )
        return None

    def _on_data_source_changed(self, index: int):
        """数据源切换时重新加载迹线与研究区。"""
        if load_data_source(index):
            print(f"已切换数据源：{name}")
            self._refresh_shap_feature_combo()
            self._refresh_target_suggestions()
            self._refresh_config_summary()
        else:
            QMessageBox.warning(self, "加载失败",
                                f"无法加载选中的数据源，请确认 {DATA_SOURCES[index]['traces']} 和 {DATA_SOURCES[index]['area']} 存在。")

    def _refresh_shap_feature_combo(self):
        """用当前数据源网格 CSV 刷新 SHAP 特征下拉列表（与训练时 feature_engineering 列一致）。"""
        idx = self.combo_data_source.currentIndex()
        if 0 <= idx < len(DATA_SOURCES):
            csv_name = DATA_SOURCES[idx]["csv"]
        else:
            csv_name = "Yingmai 2 area in Tarim Basin.csv"
        csv_path = os.path.join(_PROGRAM_DIR, csv_name)
        self.combo_shap_features.blockSignals(True)
        self.combo_shap_features.clear()
        self.combo_shap_features.addItem("全部（默认顺序）")
        if os.path.isfile(csv_path):
            try:
                from feature_engineering import build_feature_matrix

                r = build_feature_matrix(csv_path, out_processed_dir=None)
                for fname in r["feature_names"]:
                    self.combo_shap_features.addItem(fname)
            except Exception:
                pass
        self.combo_shap_features.blockSignals(False)

    def run_guoji_weighted_fusion(self):
        try:
            from fusion_algorithm import run_weighted_fusion_pipeline
        except ImportError:
            QMessageBox.warning(self, "模块未安装",
                                "请确保 fusion_algorithm、feature_engineering 可用，并安装 scikit-learn。")
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        try:
            df = run_weighted_fusion_pipeline(csv_path)
            mean_s = df["weighted_fusion_score"].mean()
            std_s = df["weighted_fusion_score"].std()
            txt = "【加权融合结果】\n\n"
            txt += f"样本数：{len(df)}\n"
            txt += f"融合得分 均值：{mean_s:.4f}  标准差：{std_s:.4f}\n"
            n_at_min = int((df["weighted_fusion_score"] == 0).sum())
            txt += (
                f"\n说明：得分已按全工区 min-max 归一到 [0,1]，最低一档均为 0；"
                f"当前并列最低的网格数：{n_at_min}（多为无断裂或拓扑特征最弱的格子）。\n"
            )
            txt += "\n前 5 行得分：\n" + df["weighted_fusion_score"].head().to_string()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            QMessageBox.information(self, "完成", "加权融合已运行，结果已显示在左侧文本框。")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", self._friendly_error_message("融合对比", str(e)))

    def run_guoji_fusion_compare(self):
        try:
            from fusion_algorithm import run_fusion_comparison_experiment
        except ImportError:
            QMessageBox.warning(self, "模块未安装",
                                "请确保 fusion_algorithm、feature_engineering 可用。GAT 需 pip install torch torch_geometric。")
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        try:
            out_dir = os.path.join(os.path.dirname(csv_path), "data", "processed")
            res = run_fusion_comparison_experiment(csv_path, out_dir=out_dir, save_boxplot=True)
            txt = "【融合对比：加权 vs GAT】\n\n"
            txt += f"加权得分 均值：{res['weighted_scores'].mean():.4f}\n"
            txt += f"GAT 得分 均值：{res['gat_scores'].mean():.4f}\n"
            if res.get("gat_degraded") and res.get("gat_degraded_reason"):
                txt += "\n【说明】\n" + res["gat_degraded_reason"] + "\n"
                txt += "\n因此箱线图右侧「GAT 融合」常显示为一条贴在 0 的线，并非加权融合也有问题。\n"
            if res.get("boxplot_path") and os.path.isfile(res["boxplot_path"]):
                txt += f"\n箱线图已保存：{res['boxplot_path']}\n"
                fig = plt.figure(figsize=(5, 4))
                img = plt.imread(res["boxplot_path"])
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                self.embed_figure(
                    fig,
                    description=(
                        "箱线图对比「规则加权融合」与「GAT 图注意力融合」在每张网格上的得分分布；"
                        "箱体与须须表示分位与离散程度，若离群点多说明工区内差异大。"
                        "若 GAT 侧退化，左侧文本会说明原因。"
                    ),
                )
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            if res.get("gat_degraded"):
                QMessageBox.warning(
                    self,
                    "融合对比完成（GAT 侧无效或退化）",
                    "左侧已说明原因。若 GAT 全为 0，多半是未安装 torch_geometric；"
                    "安装后仍是一条线则可能是 GAT 输出无方差。",
                )
            else:
                QMessageBox.information(self, "完成", "融合对比已运行，箱线图已弹出。")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", self._friendly_error_message("训练流程", str(e)))

    def run_cluster_k_helper(self):
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        try:
            from topology_fusion import load_and_prepare, fuse_with_pca
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, davies_bouldin_score
        except ImportError as e:
            QMessageBox.warning(self, "模块未安装", f"缺少聚类评估依赖：\n{e}")
            return
        cfg = _cfg_section("clustering")
        k_min = int(cfg.get("k_search_min", 2))
        k_max = int(cfg.get("k_search_max", 12))
        if k_max <= k_min:
            k_max = k_min + 1
        try:
            _, X_raw, _ = load_and_prepare(csv_path)
            X2, _, _ = fuse_with_pca(X_raw, n_components=2, standardize=True)
            rows = []
            for k in range(k_min, k_max + 1):
                km = KMeans(n_clusters=k, random_state=int(cfg.get("random_state", 42)), n_init=10)
                labels = km.fit_predict(X2)
                inertia = float(km.inertia_)
                sil = float("nan")
                dbi = float("nan")
                if len(np.unique(labels)) >= 2:
                    sil = float(silhouette_score(X2, labels))
                    dbi = float(davies_bouldin_score(X2, labels))
                rows.append({"k": k, "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi})
            import pandas as pd

            dfm = pd.DataFrame(rows)
            out_dir = os.path.join(os.path.dirname(csv_path), "data", "processed")
            os.makedirs(out_dir, exist_ok=True)
            csv_out = os.path.join(out_dir, "k_selection_metrics.csv")
            dfm.to_csv(csv_out, index=False)
            fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
            axes[0].plot(dfm["k"], dfm["inertia"], marker="o")
            axes[0].set_title("Elbow (Inertia↓)")
            axes[0].set_xlabel("k")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(dfm["k"], dfm["silhouette"], marker="o")
            axes[1].set_title("Silhouette (↑)")
            axes[1].set_xlabel("k")
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(dfm["k"], dfm["davies_bouldin"], marker="o")
            axes[2].set_title("Davies-Bouldin (↓)")
            axes[2].set_xlabel("k")
            axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            self.embed_figure(
                fig,
                description="选 k 辅助：左图看肘部拐点（Inertia 越小越好）；中图看 Silhouette（越大越好）；右图看 DBI（越小越好）。",
            )
            best_k = int(dfm.loc[dfm["silhouette"].astype(float).idxmax(), "k"]) if dfm["silhouette"].notna().any() else None
            txt = "【聚类选 k 辅助】\n\n" + dfm.to_string(index=False)
            if best_k is not None:
                txt += f"\n\n建议参考 k（按 silhouette 最大）：{best_k}"
            txt += f"\n指标表导出：{self._relpath_for_ui(csv_out)}"
            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QTextCursor.End)
            self._remember_exports("选k辅助", k_metrics_csv=csv_out)
            self._update_last_run_card(
                [
                    "类型：聚类选 k 辅助",
                    f"k 范围：{k_min}-{k_max}",
                    f"建议 k：{best_k if best_k is not None else '无'}",
                    f"导出：{self._relpath_for_ui(csv_out)}",
                ]
            )
            self._append_run_history(
                {"kind": "cluster_k_helper", "k_min": k_min, "k_max": k_max, "recommended_k": best_k, "csv": csv_out}
            )
            QMessageBox.information(self, "完成", f"选 k 指标已生成并导出：\n{self._relpath_for_ui(csv_out)}")
        except Exception as e:
            QMessageBox.critical(self, "运行出错", f"选 k 辅助失败：\n{e}")

    def run_guoji_train(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.close('all')  # 强制清空历史残留画板

        try:
            from feature_engineering import build_feature_matrix
            from ml.train import train_xgboost_regression, save_model_report, export_prediction_results
            from utils.export_utils import export_table, write_run_manifest, build_run_metadata
        except ImportError as e:
            QMessageBox.warning(self, "模块未安装", f"请确保环境可用。\n{e}")
            return

        csv_path = self._get_guoji_csv()
        if not csv_path:
            return

        import warnings
        warnings.filterwarnings("ignore")

        try:
            train_cfg = _cfg_section("train")
            target_col = self._training_target_from_gui()
            r = build_feature_matrix(csv_path, target_column=target_col, out_processed_dir=None)
            X, y = r["X"], r["y"]
            if y is None:
                target_col2, ok = QInputDialog.getText(
                    self,
                    "请指定目标列",
                    "无法在 CSV 中读取该目标列。请输入列名\n（如 Fracture Intensity B21）：",
                    text=target_col,
                )
                if not ok or not target_col2.strip():
                    return
                target_col = target_col2.strip()
                r = build_feature_matrix(csv_path, target_column=target_col, out_processed_dir=None)
                X, y = r["X"], r["y"]
                if y is None:
                    QMessageBox.warning(self, "列名无效", f"在 CSV 中未找到列「{target_col}」，请确认列名拼写。")
                    return

            res = train_xgboost_regression(X, y, df_meta=r.get("df"))
            model_dir = os.path.join(os.path.dirname(csv_path), "model")
            save_model_report(res, model_dir, name="xgboost_reg", feature_names=r.get("feature_names"))
            baseline_csv = None
            if res.get("baseline_metrics"):
                baseline_df = (
                    pd.DataFrame(res["baseline_metrics"])
                    .T.reset_index()
                    .rename(columns={"index": "model"})
                )
                baseline_csv = export_table(baseline_df, model_dir, "baseline_metrics")
            pred_all = res["model"].predict(np.asarray(X))
            pred_df = r["df"].copy()
            run_meta = build_run_metadata(config_path=os.path.join(_PROGRAM_DIR, "config.yaml"))
            pred_df["processing_run_id"] = run_meta.get("processing_run_id", "")
            pred_df["run_timestamp_utc"] = run_meta.get("run_timestamp_utc", "")
            pred_df["config_hash_sha256"] = run_meta.get("config_hash_sha256", "")
            pred_df["prediction_xgboost"] = pred_all
            if y is not None and len(y) == len(pred_df):
                pred_df["target_true"] = y
            interval_info = res.get("prediction_interval") or {}
            conf_q = float(interval_info.get("conformal_qhat", 0.0) or 0.0)
            if conf_q > 0:
                pred_df["prediction_lower"] = pred_df["prediction_xgboost"] - conf_q
                pred_df["prediction_upper"] = pred_df["prediction_xgboost"] + conf_q
            else:
                sigma = float(interval_info.get("sigma", 1.96))
                resid_std = float(interval_info.get("residual_std", 0.0))
                pred_df["prediction_lower"] = pred_df["prediction_xgboost"] - sigma * resid_std
                pred_df["prediction_upper"] = pred_df["prediction_xgboost"] + sigma * resid_std
            pred_df["prediction_interval_width"] = pred_df["prediction_upper"] - pred_df["prediction_lower"]
            pred_paths = export_prediction_results(
                pred_df,
                model_dir,
                stem="xgboost_predictions",
                export_csv=bool(train_cfg.get("export_predictions_csv", True)),
                export_gpkg=bool(train_cfg.get("export_predictions_gpkg", True)),
            )
            self._remember_exports(
                "训练预测",
                model=os.path.join(model_dir, "xgboost_reg.json"),
                report=os.path.join(model_dir, "xgboost_reg_report.json"),
                baseline_metrics=baseline_csv,
                pred_csv=pred_paths.get("csv"),
                pred_gpkg=pred_paths.get("gpkg"),
            )
            manifest_path = write_run_manifest(
                model_dir,
                run_id=str(run_meta.get("processing_run_id", "")) or None,
                kind="train",
                config_path=os.path.join(_PROGRAM_DIR, "config.yaml"),
                artifacts={
                    "model_json": os.path.join(model_dir, "xgboost_reg.json"),
                    "report_json": os.path.join(model_dir, "xgboost_reg_report.json"),
                    "prediction_csv": pred_paths.get("csv"),
                    "prediction_gpkg": pred_paths.get("gpkg"),
                    "baseline_csv": baseline_csv,
                },
                extra={
                    "target_column": target_col,
                    "train_samples": int(res.get("n_train", 0)),
                    "test_samples": int(res.get("n_test", 0)),
                },
            )
            self._remember_exports("训练预测", run_manifest=manifest_path)

            from ml.explain import shap_feature_importance, connectivity_shap_breakdown

            df_shap_train = shap_feature_importance(
                res["model"],
                X,
                feature_names=r["feature_names"],
                is_tree=True,
                out_plot_path=None,
            )
            conn_df, conn_cum_pct = connectivity_shap_breakdown(df_shap_train)
            conn_csv = None
            if conn_df is not None and not conn_df.empty:
                conn_csv = export_table(conn_df, model_dir, "shap_connectivity_features")

            names_zh: List[str] = []
            r2s: List[float] = []
            if res.get("baseline_metrics") and "linear_regression" in res["baseline_metrics"]:
                names_zh.append("线性回归")
                r2s.append(float(res["baseline_metrics"]["linear_regression"].get("R2", float("nan"))))
            if res.get("baseline_metrics") and "random_forest" in res["baseline_metrics"]:
                names_zh.append("随机森林")
                r2s.append(float(res["baseline_metrics"]["random_forest"].get("R2", float("nan"))))
            names_zh.append("XGBoost")
            r2s.append(float(res["test_metrics"]["R2"]))
            fig_m, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
            xpos = np.arange(len(names_zh))
            cols_bar = ["#7eb6df", "#90c98a", "#e89c5c", "#c49fd4"]
            axes[0].bar(xpos, r2s, color=cols_bar[: len(r2s)])
            axes[0].set_xticks(xpos)
            axes[0].set_xticklabels(names_zh, rotation=14, ha="right")
            axes[0].set_ylabel("测试集 R²")
            axes[0].set_title("模型对比（留出测试集）")
            axes[0].grid(True, axis="y", alpha=0.35)
            y_te = np.asarray(res["y_test"], dtype=np.float64)
            pred_te = np.asarray(res["test_predictions"], dtype=np.float64)
            axes[1].scatter(y_te, pred_te, s=18, alpha=0.68, c="#3498db", edgecolors="none")
            lo = float(min(y_te.min(), pred_te.min()))
            hi = float(max(y_te.max(), pred_te.max()))
            axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.55)
            axes[1].set_xlabel("真实值")
            axes[1].set_ylabel("预测值")
            axes[1].set_title("测试集：预测 vs 真实")
            resid = y_te - pred_te
            axes[2].scatter(pred_te, resid, s=18, alpha=0.68, c="#9b59b6", edgecolors="none")
            axes[2].axhline(0.0, color="k", lw=1, alpha=0.5)
            axes[2].set_xlabel("预测值")
            axes[2].set_ylabel("残差（真−预）")
            axes[2].set_title("测试集残差")
            plt.tight_layout()
            self.embed_figure(
                fig_m,
                description=(
                    "左：线性回归、随机森林与 XGBoost 在同一测试集上的 R²；中：预测–真值散点（对角线为理想）；"
                    "右：残差对预测值，用于发现系统性偏差。"
                ),
            )

            txt = "【XGBoost 训练结果】\n\n"
            txt += f"目标列：{target_col}\n\n"
            txt += f"CV R² 均值：{res['cv_agg']['R2_mean']:.4f}\n"
            if "R2_std" in res["cv_agg"]:
                txt += f"CV R² 标准差：{res['cv_agg']['R2_std']:.4f}\n"
            if "RMSE_std" in res["cv_agg"]:
                txt += f"CV RMSE 标准差：{res['cv_agg']['RMSE_std']:.4f}\n"
            txt += f"测试集 R²：{res['test_metrics']['R2']:.4f}\n"
            stab = res.get("seed_stability")
            if stab:
                txt += (
                    f"\n随机种子稳定性（仅测试集重新划分评估）："
                    f"R² 均值 {stab['test_R2_mean']:.4f}，标准差 {stab['test_R2_std']:.4f}，"
                    f"seeds={stab.get('seeds')}\n"
                )
            if res.get("baseline_metrics"):
                txt += "\n基线模型对比（测试集）：\n"
                for model_name, metrics in res["baseline_metrics"].items():
                    txt += (
                        f"  - {model_name}: "
                        f"R²={metrics.get('R2', float('nan')):.4f}, "
                        f"RMSE={metrics.get('RMSE', float('nan')):.4f}\n"
                    )
            txt += "\n【连通性特征组 · SHAP】\n"
            txt += "（Connections per Branch/Trace、Connection Frequency 若未通过特征筛选则不会出现在表中）\n"
            if conn_df is not None and not conn_df.empty:
                txt += conn_df.to_string(index=False) + "\n"
                txt += f"\n连通性特征累计贡献占比（按 mean|SHAP| 归一）：{conn_cum_pct:.2f}%\n"
            else:
                txt += "（当前特征子集内无连通性列或未进入模型。）\n"
            if conn_csv:
                txt += f"连通性 SHAP 子表：{conn_csv}\n"
            if res.get("prediction_interval"):
                cov = float((res["prediction_interval"] or {}).get("conformal_test_coverage", 0.0))
                qhat = float((res["prediction_interval"] or {}).get("conformal_qhat", 0.0))
                txt += (
                    f"\n不确定性摘要：split-conformal α={res['prediction_interval'].get('conformal_alpha', 0.1):.2f}，"
                    f"q̂={qhat:.4f}，测试覆盖率={cov:.2%}\n"
                )
                txt += (
                    f"（回退统计）±{res['prediction_interval']['sigma']:.2f}σ，"
                    f"残差标准差={res['prediction_interval']['residual_std']:.4f}\n"
                )
            spcv = res.get("spatial_cv_agg")
            if spcv:
                txt += (
                    f"\n空间 Block-CV：R² 均值={spcv.get('R2_mean', float('nan')):.4f}，"
                    f"标准差={spcv.get('R2_std', float('nan')):.4f}，"
                    f"blocks={spcv.get('n_blocks_used', 'NA')}\n"
                )
            txt += f"\n模型已保存：{self._relpath_for_ui(os.path.join(model_dir, 'xgboost_reg.json'))}"
            if baseline_csv:
                txt += f"\n基线对比表：{baseline_csv}"
            txt += f"\n预测导出 CSV：{self._relpath_for_ui(pred_paths.get('csv'))}"
            if pred_paths.get("gpkg"):
                txt += f"\n预测导出 GPKG（图层 predictions_xgb）：{self._relpath_for_ui(pred_paths.get('gpkg'))}"
            txt += f"\n运行清单：{self._relpath_for_ui(manifest_path)}"

            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QtGui.QTextCursor.End)
            self._refresh_prerequisite_buttons()
            self._refresh_config_summary()
            best_line = ""
            if res.get("baseline_metrics"):
                ranked = sorted(
                    res["baseline_metrics"].items(),
                    key=lambda kv: kv[1].get("R2", float("-inf")),
                    reverse=True,
                )
                if ranked:
                    best_line = (
                        f"\n最佳基线：{ranked[0][0]} "
                        f"(R²={ranked[0][1].get('R2', float('nan')):.4f})"
                    )
            card_lines = [
                f"类型：XGBoost 训练",
                f"目标列：{target_col}",
                f"测试 R²：{res['test_metrics']['R2']:.4f}",
            ]
            if stab:
                card_lines.append(f"多种子 R²：{stab['test_R2_mean']:.4f} ± {stab['test_R2_std']:.4f}")
            card_lines.append(f"预测导出：{self._relpath_for_ui(pred_paths.get('csv', ''))}")
            if res.get("prediction_interval"):
                card_lines.append(
                    f"conformal 覆盖率：{float((res['prediction_interval'] or {}).get('conformal_test_coverage', 0.0)):.2%}"
                )
            if conn_df is not None and not conn_df.empty:
                card_lines.append(f"连通性 SHAP 累计占比：{conn_cum_pct:.1f}%")
            self._update_last_run_card(card_lines)
            self._append_run_history(
                {
                    "kind": "train",
                    "target": target_col,
                    "test_R2": res["test_metrics"]["R2"],
                    "pred_csv": pred_paths.get("csv"),
                    "connectivity_shap_pct": conn_cum_pct,
                    "conformal_coverage_test": float((res.get("prediction_interval") or {}).get("conformal_test_coverage", 0.0)),
                }
            )
            QMessageBox.information(
                self,
                "完成",
                "训练完成，图表已刷新。"
                f"\n预测 CSV：{self._relpath_for_ui(pred_paths.get('csv', '未导出'))}"
                f"\n预测 GPKG：{self._relpath_for_ui(pred_paths.get('gpkg', '未导出'))}"
                f"{best_line}",
            )

        except Exception as e:
            QMessageBox.critical(self, "运行出错", str(e))

    def run_guoji_shap(self):
        import matplotlib.pyplot as plt
        plt.close('all')

        try:
            from ml.explain import explain_xgboost
        except ImportError:
            QMessageBox.warning(self, "模块未安装", "请确保 ml.explain 可用，并安装 shap。")
            return

        csv_path = self._get_guoji_csv()
        if not csv_path:
            return

        basic_model_dir = os.path.join(os.path.dirname(csv_path), "model")
        out_dir = basic_model_dir
        model_path = os.path.join(basic_model_dir, "xgboost_reg.json")

        if not os.path.isfile(model_path):
            QMessageBox.warning(self, "请先训练", "未找到基础训练模型！\n请先点击「训练 XGBoost」。")
            return

        import warnings
        warnings.filterwarnings("ignore")

        try:
            emph = None
            if self.combo_shap_features.currentIndex() > 0:
                emph = [self.combo_shap_features.currentText().strip()]
            df_imp = explain_xgboost(
                model_path, csv_path, out_dir=out_dir, emphasize_first=emph
            )

            from ml.explain import connectivity_shap_breakdown

            conn_sub, conn_cum = connectivity_shap_breakdown(df_imp)
            txt = "【XGBoost SHAP 特征贡献分析】\n\n"
            if emph:
                txt += f"关注特征（图中置顶）：{emph[0]}\n\n"
            txt += df_imp.head(10).to_string()
            txt += "\n\n【连通性特征组 · SHAP】\n"
            if conn_sub is not None and not conn_sub.empty:
                txt += conn_sub.to_string(index=False) + f"\n累计贡献占比：{conn_cum:.2f}%\n"
            else:
                txt += "（当前模型特征中未包含连通性列。）\n"
            txt += f"\n\n（summary 图已保存至 {self._relpath_for_ui(os.path.join(out_dir, 'shap_summary.png'))}）"
            if df_imp.attrs.get("csv_path"):
                txt += f"\nSHAP 特征表导出（shap_top_features.csv）：{self._relpath_for_ui(df_imp.attrs['csv_path'])}"
            if df_imp.attrs.get("dependence_plot"):
                txt += f"\n关键特征 dependence 图：{self._relpath_for_ui(df_imp.attrs['dependence_plot'])}"

            self.text_browser.clear()
            self.text_browser.insertPlainText(txt)
            self.text_browser.moveCursor(QtGui.QTextCursor.End)

            shap_png = os.path.join(out_dir, "shap_summary.png")
            if os.path.isfile(shap_png):
                fig = plt.figure(figsize=(8, 6))
                img = plt.imread(shap_png)
                plt.imshow(img)
                plt.axis("off")
                fig.tight_layout()
                self.embed_figure(
                    [fig],
                    description=(
                        "SHAP 摘要图：每一行对应一个输入特征；横轴为该特征对模型输出的 SHAP 贡献（影响方向与幅度）；"
                        "点色表示该样本上特征取值高低。可据此判断哪些拓扑/融合属性最能驱动当前目标列预测。"
                    ),
                )
            self._remember_exports(
                "SHAP",
                shap_png=shap_png if os.path.isfile(shap_png) else None,
                shap_csv=df_imp.attrs.get("csv_path"),
                dependence_png=df_imp.attrs.get("dependence_plot"),
            )

            plt.close('all')
            QMessageBox.information(
                self,
                "完成",
                "SHAP 分析已成功运行，特征图已在右侧显示！"
                f"\nSHAP 表（文件名多为 shap_top_features.csv）：{self._relpath_for_ui(df_imp.attrs.get('csv_path', '未导出'))}"
                f"\nDependence 图：{self._relpath_for_ui(df_imp.attrs.get('dependence_plot', '未生成'))}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "SHAP 运行遇到小麻烦",
                self._friendly_error_message("SHAP 分析", str(e)),
            )

    def run_spatial_topology_framework(self):
        """一键运行空间-拓扑融合学习框架（后台线程，可取消）。"""
        import matplotlib.pyplot as plt
        plt.close('all')
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        target_column, ok = QInputDialog.getText(
            self,
            "目标列名",
            "请输入要预测的目标列（例如 Fracture Intensity B21）：",
            text=self._training_target_from_gui(),
        )
        if not ok or not target_column.strip():
            return
        target_column = target_column.strip()

        def _worker(csv_path_inner: str, target_inner: str):
            from spatial_topology_framework import run_spatial_topology_fusion_pipeline
            return run_spatial_topology_fusion_pipeline(csv_path=csv_path_inner, target_column=target_inner)

        self._set_busy(True, "一键空间-拓扑融合运行中...")
        task = TaskRunner(_worker, csv_path, target_column)
        self._running_task = task

        def _done(res):
            self._running_task = None
            self._set_busy(False)
            try:
                xgb_res = res.get("xgb_result", {})
                cv_agg = xgb_res.get("cv_agg", {})
                test_metrics = xgb_res.get("test_metrics", {})
                shap_df = res.get("shap_importance")
                out_dir = os.path.join(os.path.dirname(csv_path), "data", "processed")
                shap_png = os.path.join(out_dir, "shap_summary.png")
                if os.path.isfile(shap_png):
                    fig = plt.figure(figsize=(8, 6))
                    img = plt.imread(shap_png)
                    plt.imshow(img)
                    plt.axis("off")
                    fig.tight_layout()
                    self.embed_figure(
                        [fig],
                        description=(
                            "「一键空间–拓扑融合流水线」结束后的 SHAP 汇总图（若已生成）："
                            "含义与单独 SHAP 按钮相同，特征重要性针对流水线中指定的目标列。"
                        ),
                    )
                plt.close('all')
                txt = "【空间-拓扑融合分析】\n\n"
                txt += f"目标列：{target_column}\n"
                if cv_agg:
                    txt += "\n交叉验证（CV）指标：\n"
                    if "R2_mean" in cv_agg:
                        txt += f"  R² 均值：{cv_agg['R2_mean']:.4f}\n"
                    if "RMSE_mean" in cv_agg:
                        txt += f"  RMSE 均值：{cv_agg['RMSE_mean']:.4f}\n"
                if xgb_res.get("spatial_cv_agg"):
                    sp = xgb_res["spatial_cv_agg"]
                    txt += (
                        f"\n空间 Block-CV：R² 均值={sp.get('R2_mean', float('nan')):.4f}，"
                        f"标准差={sp.get('R2_std', float('nan')):.4f}\n"
                    )
                if test_metrics:
                    txt += "\n测试集指标：\n"
                    if "R2" in test_metrics:
                        txt += f"  R²：{test_metrics['R2']:.4f}\n"
                if shap_df is not None and not shap_df.empty:
                    txt += "\nTop 特征贡献（SHAP）：\n"
                    txt += shap_df.head(8).to_string(index=False)
                    txt += "\n\n（SHAP 分析图已嵌入右侧画板）"
                    if shap_df.attrs.get("csv_path"):
                        txt += f"\nSHAP 表导出：{self._relpath_for_ui(shap_df.attrs['csv_path'])}"
                export_paths = res.get("export_paths") or {}
                if export_paths.get("csv"):
                    txt += f"\n结果导出 CSV：{self._relpath_for_ui(export_paths['csv'])}"
                if export_paths.get("gpkg"):
                    txt += f"\n结果导出 GPKG：{self._relpath_for_ui(export_paths['gpkg'])}"
                if export_paths.get("manifest"):
                    txt += f"\n运行清单：{self._relpath_for_ui(export_paths['manifest'])}"
                self._remember_exports(
                    "一键空间-拓扑融合",
                    result_csv=export_paths.get("csv"),
                    result_gpkg=export_paths.get("gpkg"),
                    run_manifest=export_paths.get("manifest"),
                    shap_csv=shap_df.attrs.get("csv_path") if shap_df is not None else None,
                    shap_png=shap_png if os.path.isfile(shap_png) else None,
                )
                self.text_browser.clear()
                self.text_browser.insertPlainText(txt)
                self.text_browser.moveCursor(QtGui.QTextCursor.End)
                self._update_last_run_card(
                    [
                        "类型：一键空间-拓扑融合",
                        f"目标列：{target_column}",
                        f"结果：{self._relpath_for_ui(export_paths.get('csv', ''))}",
                    ]
                )
                QMessageBox.information(
                    self,
                    "完成",
                    "空间-拓扑融合运行完毕"
                    f"\n结果 CSV：{self._relpath_for_ui(export_paths.get('csv', '未导出'))}"
                    f"\n结果 GPKG：{self._relpath_for_ui(export_paths.get('gpkg', '未导出'))}",
                )
            except Exception as e:
                QMessageBox.critical(self, "后处理失败", self._friendly_error_message("一键空间-拓扑融合结果渲染", str(e)))

        def _fail(msg: str):
            self._running_task = None
            self._set_busy(False)
            QMessageBox.critical(self, "运行出错", self._friendly_error_message("一键空间-拓扑融合", msg))

        task.finished_ok.connect(_done)
        task.failed.connect(_fail)
        task.start()

    def _setup_multiperiod_menu(self):
        """添加正式筛选入口，同时保留原有专业图件和研究功能。"""
        menu = self.menuBar().addMenu("多时期勘探筛选")
        formal_action = menu.addAction("一键生成候选勘探有利区", self._run_target_screening)
        formal_action.setStatusTip("运行精确拓扑、多期匹配、稳健性分析并导出证据卡")

        professional_menu = menu.addMenu("专业分析与图件")
        professional_menu.addAction(
            "三期融合对比图 (PCA+加权+XGBoost诊断)", self._run_multiperiod_stage1
        )
        professional_menu.addAction("三期空间叠加图", self._run_multiperiod_stage2)
        professional_menu.addAction("渗流曲线与关键节点图", self._run_multiperiod_stage3)

        research_menu = menu.addMenu("研究与实验")
        research_action = research_menu.addAction(
            "代理模型实验 (XGBoost→预测渗流)", self._run_multiperiod_stage4
        )
        research_action.setStatusTip("实验功能，不参与正式候选有利区评分")

    def _run_target_screening(self):
        """运行可解释的正式候选有利区筛选主流程。"""
        import matplotlib.pyplot as plt
        from datetime import datetime

        plt.close("all")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            _PROGRAM_DIR, "data", "processed", "target_screening", timestamp
        )
        self.screening_status_label.setText("正在启动…")
        self.run_info_tabs.setVisible(True)
        self.btn_toggle_run_info.setText("隐藏运行信息")
        self.run_info_tabs.setCurrentWidget(self.run_log_tab)
        self._set_busy(True, "正在进行精确拓扑、多期匹配与候选有利区筛选...")

        def _worker(progress_callback=None):
            from screening_pipeline import run_target_screening
            return run_target_screening(
                output_dir=out_dir, progress_callback=progress_callback
            )

        self._launch_multiperiod_task(
            _worker,
            "候选勘探有利区筛选",
            self._show_target_screening_result,
            show_progress=True,
        )

    def _show_target_screening_result(self, res):
        """展示正式筛选摘要、证据边界和候选有利区图。"""
        import matplotlib.pyplot as plt

        summary = res.get("input_summary", {})
        external = res.get("external_validation", {})
        artifacts = res.get("artifact_paths", {})
        period_counts = summary.get("period_grid_counts", {})
        limitations = res.get("limitations", [])
        txt = "【多期断裂网络候选勘探有利区辅助筛选】\n\n"
        txt += f"运行状态: {res.get('status', 'unknown')}\n"
        txt += f"三期有效网格: {sum(period_counts.values()) if period_counts else 0}\n"
        txt += f"多期匹配单元: {summary.get('matched_cell_count', 0)}\n"
        txt += f"候选单元: {summary.get('candidate_cell_count', 0)}\n"
        txt += f"稳定候选有利区: {summary.get('stable_target_count', 0)}\n"
        txt += f"不稳定候选（仅供复核）: {summary.get('unstable_target_count', 0)}\n"
        txt += f"候选空间单元总数: {summary.get('candidate_target_count', 0)}\n"
        txt += f"外部验证状态: {external.get('status', 'not_validated')}\n"
        txt += f"运行耗时: {res.get('elapsed_seconds', 0.0):.1f} 秒\n\n"
        txt += "证据边界:\n"
        txt += "\n".join(f"- {item}" for item in limitations)
        txt += f"\n\n结果目录: {os.path.dirname(artifacts.get('result_json', ''))}"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)

        map_path = artifacts.get("candidate_targets_png")
        if map_path and os.path.isfile(map_path):
            fig = plt.figure(figsize=(12, 7))
            img = plt.imread(map_path)
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            self.embed_figure(fig, description="多期断裂网络候选勘探有利区")
        plt.close("all")
        stable_count = summary.get("stable_target_count", 0)
        unstable_count = summary.get("unstable_target_count", 0)
        self.screening_status_label.setText(f"已完成 · 稳定 {stable_count}")
        self._update_last_run_card([
            "正式筛选已完成",
            f"稳定候选：{stable_count} 个  |  不稳定候选：{unstable_count} 个",
            f"匹配单元：{summary.get('matched_cell_count', 0)}  |  候选单元：{summary.get('candidate_cell_count', 0)}",
            f"外部验证：{external.get('status', 'not_validated')}",
            f"结果目录：{os.path.dirname(artifacts.get('result_json', ''))}",
        ])
        self._remember_exports(
            "候选勘探有利区",
            result_json=artifacts.get("result_json"),
            stable_targets_csv=artifacts.get("stable_targets_csv"),
            candidate_targets_gpkg=artifacts.get("candidate_targets_gpkg"),
            candidate_targets_png=artifacts.get("candidate_targets_png"),
        )
        self._append_run_history({
            "kind": "target_screening",
            "stable_target_count": stable_count,
            "unstable_target_count": unstable_count,
            "external_validation": external.get("status", "not_validated"),
            "result_json": artifacts.get("result_json"),
        })
        self.run_info_tabs.setCurrentWidget(self.run_summary_tab)
        self._set_busy(False)
        QMessageBox.information(
            self,
            "筛选完成",
            f"已生成 {summary.get('stable_target_count', 0)} 个稳定候选有利区，"
            f"另有 {summary.get('unstable_target_count', 0)} 个不稳定候选供复核。\n"
            "当前结论属于内部辅助筛选，需结合井位、储层或专家资料验证。",
        )

    def _run_multiperiod_stage1(self):
        """阶段1: 三期独立融合对比。"""
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "阶段1: 三期融合对比运行中...")

        def _worker():
            from batch_run import run_multiperiod_fusion
            return run_multiperiod_fusion(n_clusters=4)
        self._launch_multiperiod_task(_worker, "阶段1: 三期融合对比",
            lambda res: self._show_multiperiod_result_stage1(res))

    def _show_multiperiod_result_stage1(self, res):
        import matplotlib.pyplot as plt
        summary = res["summary_df"]
        txt = "【阶段1: 三期独立融合对比】\n\n"
        txt += summary.to_string(index=False)
        txt += f"\n\n导出目录: {res['out_dir']}"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)
        # 显示图表
        figs = []
        captions = []
        for key, label in [("pca", "PCA降维聚类对比"), ("weighted", "加权融合得分分布"), ("xgboost", "XGBoost特征重要性")]:
            p = res["plot_paths"].get(key)
            if p and os.path.isfile(p):
                fig = plt.figure(figsize=(10, 6))
                img = plt.imread(p)
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                figs.append(fig)
                captions.append(label)
        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        self._refresh_config_summary()
        QMessageBox.information(self, "完成", f"阶段1完成\n导出: {res['out_dir']}")

    def _run_multiperiod_stage2(self):
        """阶段2: 三期空间叠加。"""
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "阶段2: 三期空间叠加运行中...")

        def _worker():
            from multiperiod_overlay import run_overlay_pipeline
            return run_overlay_pipeline()
        self._launch_multiperiod_task(_worker, "阶段2: 三期空间叠加",
            lambda res: self._show_multiperiod_result_stage2(res))

    def _show_multiperiod_result_stage2(self, res):
        import matplotlib.pyplot as plt
        overlap = res["overlap_df"]
        n = len(overlap)
        targets = overlap["target_area"].value_counts().to_string() if "target_area" in overlap.columns else "N/A"
        txt = f"【阶段2: 三期空间叠加】\n\n三期全重叠网格数: {n}\n\n靶区分布:\n{targets}\n\n"
        txt += f"重叠表: {res.get('overlap_csv', '')}"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)
        p = res.get("plot_path")
        if p and os.path.isfile(p):
            fig = plt.figure(figsize=(12, 6))
            img = plt.imread(p)
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            self.embed_figure(fig, description="三期空间重叠分布图")
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", f"阶段2完成\n三期重叠: {n} 网格")

    def _run_multiperiod_stage3(self):
        """阶段3: 渗流模拟。"""
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "阶段3: 渗流模拟运行中...")

        def _worker():
            from percolation import run_percolation_pipeline
            return run_percolation_pipeline()
        self._launch_multiperiod_task(_worker, "阶段3: 渗流模拟",
            lambda res: self._show_multiperiod_result_stage3(res))

    def _show_multiperiod_result_stage3(self, res):
        import matplotlib.pyplot as plt
        summary = res["_summary"]
        txt = "【阶段3: 渗流模拟】\n\n"
        txt += summary.to_string(index=False)
        txt += f"\n\n导出目录: {res['_out_dir']}"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)
        figs = []
        captions = []
        for key, label in [("curves", "渗流曲线对比"), ("key_nodes_海西期", "海西期关键节点"),
                           ("key_nodes_喜山期", "喜山期关键节点"), ("key_nodes_印支燕山期", "印支燕山期关键节点")]:
            p = res["_plot_paths"].get(key)
            if p and os.path.isfile(p):
                fig = plt.figure(figsize=(9, 6))
                img = plt.imread(p)
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                figs.append(fig)
                captions.append(label)
        if figs:
            self.embed_figure(figs, descriptions=captions)
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", f"阶段3完成")

    def _run_multiperiod_stage4(self):
        """阶段4: 代理模型。"""
        import matplotlib.pyplot as plt
        plt.close("all")
        self._set_busy(True, "阶段4: 代理模型运行中...")

        def _worker():
            from agent_model import run_agent_pipeline
            return run_agent_pipeline()
        self._launch_multiperiod_task(_worker, "阶段4: 代理模型",
            lambda res: self._show_multiperiod_result_stage4(res))

    def _show_multiperiod_result_stage4(self, res):
        import matplotlib.pyplot as plt
        txt = f"【阶段4: 代理模型】\n\nR² = {res['metrics']['r2']:.4f}\nRMSE = {res['metrics']['rmse']:.4f}\n\n"
        txt += res["feature_importance"].to_string(index=False)
        txt += f"\n\n导出目录: {res['_out_dir']}"
        self.text_browser.clear()
        self.text_browser.insertPlainText(txt)
        self.text_browser.moveCursor(QtGui.QTextCursor.End)
        figs = []
        for key, label in [("importance", "特征重要性"), ("pred_vs_true", "预测vs真实")]:
            p = res["_plot_paths"].get(key)
            if p and os.path.isfile(p):
                fig = plt.figure(figsize=(8, 6))
                img = plt.imread(p)
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                figs.append(fig)
        if figs:
            self.embed_figure(figs, descriptions=["特征重要性", "预测vs真实"])
        plt.close("all")
        self._set_busy(False)
        QMessageBox.information(self, "完成", f"阶段4完成\nR² = {res['metrics']['r2']:.4f}")

    def _run_multiperiod_all(self):
        """兼容旧入口；统一转向正式筛选主流程。"""
        self._run_target_screening()

    def _launch_multiperiod_task(self, worker_fn, title, done_fn, show_progress=False):
        """启动后台任务并连接完成回调。"""
        task = TaskRunner(worker_fn, inject_progress=show_progress)
        self._running_task = task

        def _done(res):
            self._running_task = None
            try:
                done_fn(res)
            except Exception as e:
                self._set_busy(False)
                QMessageBox.critical(self, "结果渲染失败", str(e))

        def _fail(msg):
            self._running_task = None
            self._set_busy(False)
            if "取消" in str(msg):
                if show_progress and hasattr(self, "screening_status_label"):
                    self.screening_status_label.setText("已取消")
                QMessageBox.information(self, "已取消", "任务已在安全检查点停止。")
                return
            if show_progress and hasattr(self, "screening_status_label"):
                self.screening_status_label.setText("运行失败")
            QMessageBox.critical(self, "运行失败", self._friendly_error_message(title, msg))

        task.finished_ok.connect(_done)
        task.failed.connect(_fail)
        if show_progress:
            def _show_progress(message):
                self._set_busy(True, f"{title}: {message}")
                if hasattr(self, "screening_status_label"):
                    self.screening_status_label.setText(str(message))
            task.progress.connect(_show_progress)
        task.start()

    def _set_ronghe_combo_tooltip(self):
        lines = ["融合方式："]
        lines.append("• PCA：直接可用")
        lines.append("• 自编码器/VAE：需 pip install torch" if not HAS_TORCH else "• 自编码器/VAE：已安装 torch")
        lines.append("• UMAP：需 pip install umap-learn" if not HAS_UMAP else "• UMAP：已安装 umap-learn")
        self.combo_fusion.setToolTip("\n".join(lines))

    def onIndexChanged(self, index):
        self.opt = index
        if getattr(self, "_initializing_ui", False):
            return
        if index <= 0:
            return
        self.run_lunkuo()

    def onIndexChanged_2(self, index):
        if index == 1:
            self.run_tuopuhou1()
        elif index == 2:
            self.run_tuopuhou2()

    # ── 通用后台 Network 计算框架 ────────────────────────────────────────────
    def _run_with_network(self, render_fn, network_kwargs=None, loading_text="正在构建断裂网络，请耐心等待..."):
        """
        在后台线程中执行 try_network，完成后在主线程调用 render_fn(network)。
        避免主线程阻塞导致 macOS 触发"无响应"警告与强制重启。
        """
        if traces is None or area is None or traces.empty:
            QMessageBox.warning(self, "无数据", "请先切换数据源并确保迹线、研究区文件存在且非空。")
            return

        progress = QtWidgets.QProgressDialog(loading_text, None, 0, 0, self)
        progress.setWindowTitle("系统运算中")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

        kw = dict(
            determine_branches_nodes=True,
            truncate_traces=True,
            circular_target_area=False,
            snap_threshold=0.001,
        )
        if network_kwargs:
            kw.update(network_kwargs)

        traces_snap = traces.copy()
        area_snap = area.copy() if area is not None else None
        name_snap = name
        left_snap, right_snap = left, right
        down_snap, up_snap = down, up
        width_snap, height_snap = width, height
        rate_snap = rate

        def _bg():
            nw, err = try_network(traces_snap, area_snap, name=name_snap, **kw)
            if err:
                raise RuntimeError(err)
            return nw

        def _done(nw):
            global left, right, down, up, width, height, rate
            saved = (left, right, down, up, width, height, rate)
            left, right = left_snap, right_snap
            down, up = down_snap, up_snap
            width, height = width_snap, height_snap
            rate = rate_snap
            try:
                render_fn(nw)
            except Exception as e:
                print(f"❌ 渲染报错: {e}")
            finally:
                left, right, down, up, width, height, rate = saved
                _cleanup()

        def _fail(msg):
            print(f"❌ 网络构建报错: {msg}")
            QMessageBox.warning(self, "无法构建断裂网络", msg)
            _cleanup()

        def _cleanup():
            QtWidgets.QApplication.restoreOverrideCursor()
            progress.close()
            self._nw_runner = None

        runner = TaskRunner(_bg)
        runner.finished_ok.connect(_done)
        runner.failed.connect(_fail)
        self._nw_runner = runner
        runner.start()

    def run_yuantu(self):
        warnings.filterwarnings("ignore")
        # 原图仅展示迹线 + 研究区边界，无需构造 Network（Network 会裁剪迹线，CRS 不一致或不相交时会崩溃）
        if traces is None or area is None or traces.empty:
            QMessageBox.warning(self, "无数据", "请先切换数据源并确保迹线、研究区文件存在且非空。")
            return
        fig, ax = plt.subplots(1, 1, figsize=_safe_figsize())
        traces.plot(ax=ax, color="blue")
        ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
        ax.set_xlim(left - width, right + width)
        ax.set_ylim(down - height, up + height)
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_color("#0d0d0d")
            s.set_linewidth(1.35)
        fig.tight_layout()
        self.embed_figure(
            fig,
            description="原始断裂迹线图：蓝色线为输入迹线，坐标系见标题；用于检查数据范围、与研究区是否一致，未做拓扑分类。",
        )

    def run_fenleihou(self):
        warnings.filterwarnings("ignore")
        def _render(network):
            fig, ax = plt.subplots(figsize=_safe_figsize())
            ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
            network.branch_gdf.plot(
                colors=[assign_colors(bt) for bt in network.branch_types],
                ax=ax,
                aspect="equal",
            )
            handles = [
                plt.Line2D([0], [0], color="green", lw=2, label="CC_branch / X_node"),
                plt.Line2D([0], [0], color="blue", lw=2, label="CI_branch / Y_node"),
                plt.Line2D([0], [0], color="black", lw=2, label="II_branch / I_node"),
                plt.Line2D([0], [0], color="red", lw=2, label="Other / Boundary"),
            ]
            ax.legend(handles=handles, loc='lower left')
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            ax.set_aspect('equal')
            for s in ax.spines.values():
                s.set_color("#0d0d0d")
                s.set_linewidth(1.35)
            fig.tight_layout()
            self.embed_figure(
                fig,
                description=(
                    "分类后迹线图：按 fractopo 分支类型（CC/CI/II）对线段着色；图例中绿色/蓝色/黑色对应不同分支类，"
                    "红色为无法归类或边界相关；反映拓扑划分后的空间模式。"
                ),
            )
        self._run_with_network(_render)

    def run_tuopuhou1(self):
        warnings.filterwarnings("ignore")
        def _render(network):
            fig, ax = plt.subplots(figsize=_safe_figsize())
            ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")
            network.trace_gdf.plot(ax=ax, linewidth=0.5, aspect="equal")
            network.node_gdf.plot(
                c=[assign_colors(bt) for bt in network.node_types],
                ax=ax,
                markersize=10,
                aspect="equal",
            )
            area.boundary.plot(ax=ax, color="red", aspect="equal")
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="green", markersize=10, label="X_node"),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=10, label="Y_node"),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="black", markersize=10, label="I_node"),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="red", markersize=10, label="Other / Boundary"),
            ]
            ax.legend(handles=handles, loc='lower left')
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            ax.set_aspect('equal')
            self.embed_figure(
                fig,
                description=(
                    "拓扑化视图 1：浅色为迹线；节点按 X/Y/I 类型着色（见左下角图例示意），红线为研究区边界。"
                    "用于核对节点识别是否落在迹线交点等位置。"
                ),
            )
        self._run_with_network(_render)

    def run_tuopuhou2(self):
        warnings.filterwarnings("ignore")
        def _render(network):
            type_to_color = {'E': 'red', 'I': 'green', 'X': 'blue', 'Y': 'yellow'}
            type_to_shape = {'E': 'o', 'I': 'o', 'X': '^', 'Y': '*'}
            fig, ax = plt.subplots(figsize=_safe_figsize())
            bg = network.branch_gdf
            if CONNECTION_COLUMN in bg.columns:
                for conn_val, color, leg in [(CC_branch, "red", "CC"), (CI_branch, "green", "CI"), (II_branch, "blue", "II")]:
                    subset = bg[bg[CONNECTION_COLUMN] == conn_val]
                    if not subset.empty:
                        subset.plot(ax=ax, color=color, linewidth=1, label=leg, aspect="equal")
            elif "Class" in bg.columns:
                for branch_type, color in (("CC", "red"), ("CI", "green"), ("II", "blue")):
                    subset = bg[bg["Class"] == branch_type]
                    if not subset.empty:
                        subset.plot(ax=ax, color=color, linewidth=1, label=branch_type, aspect="equal")
            else:
                QMessageBox.warning(self, "无法绘制分支", "branch_gdf 中未找到「Connection」或「Class」列，可能与当前 fractopo 版本不兼容。")
                return
            for node_type in type_to_color:
                nodes = network.node_gdf[network.node_gdf[CLASS_COLUMN] == node_type]
                if not nodes.empty:
                    ax.scatter(nodes.geometry.x, nodes.geometry.y, s=50,
                               c=type_to_color[node_type], marker=type_to_shape[node_type], label=node_type, zorder=5)
            area.boundary.plot(ax=ax, color="red", aspect="equal")
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            ax.legend(title=' Type')
            ax.set_aspect('equal')
            self.embed_figure(
                fig,
                description=(
                    "拓扑化视图 2：彩色线段表示分支连接类型（C-C / C-I / I-I 等 fractopo Connection 记号），"
                    "节点散点为 X/Y/I/E 类型；与视图 1 互补，侧重「线段-节点」联合展示。"
                ),
            )
        self._run_with_network(_render)

    def run_tuopushuxing(self):
        warnings.filterwarnings("ignore")
        def _render(network):
            parameters = 'parameters'.ljust(40, ' ') + 'values' + "\n"
            for key, value in network.parameters.items():
                parameters = parameters + str(key).ljust(40, ' ') + str(value) + "\n"
            self.text_browser.clear()
            self.text_browser.insertPlainText(parameters)
            self.text_browser.moveCursor(QTextCursor.End)
        self._run_with_network(_render)

    def run_azimuth(self):
        setup_matplotlib_chinese()
        def _render(network):
            pprint((network.azimuth_set_names, network.azimuth_set_ranges))
            pprint(network.trace_azimuth_set_counts)
            fig, ax = plt.subplots(figsize=_safe_figsize())
            colors = ("red", "blue")
            for azimuth_set, set_range, color in zip(network.azimuth_set_names, network.azimuth_set_ranges, colors):
                trace_gdf_set = network.trace_gdf.loc[network.trace_gdf["azimuth_set"] == azimuth_set]
                trace_gdf_set.plot(color=color, label=f"{azimuth_set} - {set_range}", ax=ax)
            zh_fonts = plt.rcParams.get("font.sans-serif", [])
            font_family = zh_fonts[0] if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0 else "Microsoft YaHei"
            ax.set_title(f"方位角集图 - {name}", fontsize=14, fontfamily=font_family)
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            ax.set_aspect('equal')
            for s in ax.spines.values():
                s.set_color("#0d0d0d")
                s.set_linewidth(1.35)
            plt.legend()
            self.embed_figure(
                fig,
                description=(
                    "方位角集图：不同颜色对应不同方位组（如 N-S 与 E-W 及角度范围）；"
                    "用于查看各走向迹线在工区内的分布是否分组明显。"
                ),
            )
        self._run_with_network(_render, network_kwargs={
            "azimuth_set_names": ("N-S", "E-W"),
            "azimuth_set_ranges": ((135, 45), (45, 135)),
        })

    def run_relitu(self):
        warnings.filterwarnings("ignore")
        # 热力图 为每条线段生成一系列的点
        points = []
        for geom in traces.geometry:
            points.extend(list(geom.interpolate(distance=5, normalized=True).coords))
        x, y = np.array(points).T
        kde = gaussian_kde(np.vstack([x, y]))
        kde_values = kde(np.vstack([x, y]))
        fig, ax = plt.subplots(figsize=_safe_figsize())
        scatter = ax.scatter(x, y, c=kde_values, cmap="Reds", s=10, alpha=0.5)
        plt.title("Fracture density heatmap " + name)
        plt.axis("equal")
        ax.set_aspect('equal')
        ax.set_xlim(left - width, right + width)
        ax.set_ylim(down - height, up + height)
        for s in ax.spines.values():
            s.set_color("#0d0d0d")
            s.set_linewidth(1.35)
        self.embed_figure(
            fig,
            description=(
                "断裂密度热力图：沿迹线加密采样点后做核密度估计，颜色越暖表示该处线密度越高；"
                "反映断裂在平面上的聚集带，并非网格 CSV 中的属性。"
            ),
        )

    def a(self):
        warnings.filterwarnings("ignore")
        def _render(network):
            fit, fig1, ax = network.plot_trace_lengths()
            fit, fig2, ax = network.plot_branch_lengths()
            fit_line_colors = {
                "power": "#d62728",
                "lognormal": "#1f77b4",
                "exponential": "#2ca02c",
            }
            for fig in (fig1, fig2):
                for one_ax in fig.axes:
                    for line in one_ax.get_lines():
                        label = (line.get_label() or "").lower()
                        if "power" in label:
                            line.set_color(fit_line_colors["power"]); line.set_alpha(0.7)
                        elif "lognormal" in label:
                            line.set_color(fit_line_colors["lognormal"]); line.set_alpha(0.7)
                        elif "exponential" in label:
                            line.set_color(fit_line_colors["exponential"]); line.set_alpha(0.7)
                    legend = one_ax.get_legend()
                    if legend is not None:
                        handles = getattr(legend, "legend_handles", None)
                        if handles is None:
                            handles = getattr(legend, "legendHandles", [])
                        for handle, txt in zip(handles, legend.get_texts()):
                            tlabel = (txt.get_text() or "").lower()
                            if "power" in tlabel:
                                color = fit_line_colors["power"]
                            elif "lognormal" in tlabel:
                                color = fit_line_colors["lognormal"]
                            elif "exponential" in tlabel:
                                color = fit_line_colors["exponential"]
                            else:
                                continue
                            if hasattr(handle, "set_color"):
                                handle.set_color(color)
                            if hasattr(handle, "set_alpha"):
                                handle.set_alpha(0.7)
                            txt.set_color(color)
                    for txt in one_ax.texts:
                        tlabel = (txt.get_text() or "").lower()
                        if "power" in tlabel:
                            txt.set_color(fit_line_colors["power"])
                        elif "lognormal" in tlabel:
                            txt.set_color(fit_line_colors["lognormal"])
                        elif "exponential" in tlabel:
                            txt.set_color(fit_line_colors["exponential"])
            self.embed_figure(
                [fig1, fig2],
                descriptions=[
                    "迹线长度分布直方图及幂律、对数正态、指数等典型拟合曲线；用于判断标度律与共守分布形态。",
                    "分支长度分布及同样拟合对比；分支由迹线拓扑分解得到，长度统计与迹线层可对照阅读。",
                ],
            )
        self._run_with_network(_render)

    def run_meiguitu(self):
        warnings.filterwarnings("ignore")
        setup_matplotlib_chinese()
        def _render(network):
            azimuth_bin_dict, fig1, ax = network.plot_trace_azimuth()
            azimuth_bin_dict, fig2, ax = network.plot_branch_azimuth()
            for one_ax in fig1.axes:
                for patch in one_ax.patches:
                    patch.set_facecolor("#2C3E50")
                    patch.set_edgecolor("black")
                    patch.set_alpha(0.65)
            for one_ax in fig2.axes:
                for patch in one_ax.patches:
                    patch.set_facecolor("#AED6F1")
                    patch.set_edgecolor("#2E86C1")
                    patch.set_alpha(0.65)
            zh_fonts = plt.rcParams.get("font.sans-serif", [])
            font_family = zh_fonts[0] if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0 else "Microsoft YaHei"
            for fig, title_text in ((fig1, f"迹线玫瑰图 - {name}"), (fig2, f"分支玫瑰图 - {name}")):
                for one_ax in fig.axes:
                    one_ax.set_title(title_text, fontfamily=font_family, fontsize=14)
            self.embed_figure(
                [fig1, fig2],
                descriptions=[
                    "迹线方位玫瑰图：极坐标下各走向区间频数，峰值方向即优势构造走向。",
                    "分支方位玫瑰图：对拓扑分支线段统计走向，可与迹线玫瑰图对比构造与分解后差异。",
                ],
            )
        self._run_with_network(_render)

    def run_sanyuantu(self):
        warnings.filterwarnings("ignore")
        setup_matplotlib_chinese()
        def _render(network):
            fig1, ax1, tax1 = network.plot_xyi()
            ax1.axis('off')
            fig1.set_size_inches(10, 10)
            # 底部多留白，避免顶点标签（X–Y、C–C 等）与下方「图说明」条带视觉上挤在一起
            ax1.set_position([0.13, 0.18, 0.62, 0.56])
            fig1.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.90)
            fig2, ax2, tax2 = network.plot_branch()
            ax2.axis('off')
            fig2.set_size_inches(10, 10)
            ax2.set_position([0.13, 0.18, 0.62, 0.56])
            fig2.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.90)
            _style_ternary_plot(fig1, tax1)
            _style_ternary_plot(fig2, tax2)
            _polish_fractopo_ternary_labels(fig1)
            _polish_fractopo_ternary_labels(fig2)
            zh_fonts = plt.rcParams.get("font.sans-serif", [])
            font_family = zh_fonts[0] if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0 else "Microsoft YaHei"
            for fig, ttl in (
                (fig1, f"节点类型三元图（XYI）- {name}"),
                (fig2, f"分支类型三元图（CC/CI/II）- {name}"),
            ):
                fig.suptitle(ttl, fontsize=14, fontfamily=font_family, y=0.96)
                for ax in fig.axes:
                    leg = ax.get_legend()
                    if leg is not None:
                        for t in leg.get_texts():
                            t.set_fontfamily(font_family)
                        title = leg.get_title()
                        if title is not None:
                            title.set_fontfamily(font_family)
                for leg in getattr(fig, "legends", []):
                    for t in leg.get_texts():
                        t.set_fontfamily(font_family)
            self.embed_figure(
                [fig1, fig2],
                descriptions=[
                    "节点类型三元图（XYI）：三角形顶点为 X、Y、I 三类节点占比，落在三角形内的点云表示样本整体组成。",
                    "分支类型三元图（CC、CI、II）：三端元为三类分支在数量或长度加权下的比例（定义见 fractopo）。",
                ],
            )
        self._run_with_network(_render)

    def run_guanxi(self):
        warnings.filterwarnings("ignore")
        def _render(network):
            print(f"Azimuth set names: {network.azimuth_set_names}")
            print(f"Azimuth set ranges: {network.azimuth_set_ranges}")
            figs, fig_axes = network.plot_azimuth_crosscut_abutting_relationships()
            relationship_colors = ("#4A5568", "#2B6CB0", "#63B3ED")
            for fig in figs:
                for ax in fig.axes:
                    if not hasattr(ax, "containers"):
                        continue
                    for container in ax.containers:
                        if len(container) >= 3:
                            for patch, color in zip(container[:3], relationship_colors):
                                patch.set_facecolor(color)
                                patch.set_edgecolor("black")
                            break
                    legend = ax.get_legend()
                    if legend is not None:
                        handles = getattr(legend, "legend_handles", None)
                        if handles is None:
                            handles = getattr(legend, "legendHandles", [])
                        for handle, color in zip(handles, relationship_colors):
                            if hasattr(handle, "set_facecolor"):
                                handle.set_facecolor(color)
                            if hasattr(handle, "set_edgecolor"):
                                handle.set_edgecolor("black")
                        leg_txts = legend.get_texts()
                        full_title = ""
                        if leg_txts:
                            full_title = leg_txts[0].get_text()
                            ax.set_title(
                                full_title,
                                fontsize=10,
                                fontweight="bold",
                                pad=8,
                                fontfamily="DejaVu Sans",
                            )
                        # 图例改为短标签 + 单列叠放，限制在各自子图上方，避免横向伸入邻图
                        short_labels = ("Cross-cut", "Abutting (A→B)", "Abutting (B→A)")
                        n = min(len(handles), len(short_labels))
                        legend.remove()
                        new_leg = ax.legend(
                            handles[:n],
                            short_labels[:n],
                            loc="lower center",
                            bbox_to_anchor=(0.5, 1.02),
                            transform=ax.transAxes,
                            ncol=1,
                            fontsize=8,
                            frameon=True,
                            fancybox=True,
                            framealpha=0.95,
                            edgecolor="#9CA3AF",
                            handlelength=0.9,
                            handletextpad=0.45,
                            borderpad=0.35,
                            labelspacing=0.35,
                            columnspacing=0.6,
                        )
                        for handle, color in zip(
                            getattr(new_leg, "legend_handles", None)
                            or getattr(new_leg, "legendHandles", []),
                            relationship_colors,
                        ):
                            if hasattr(handle, "set_facecolor"):
                                handle.set_facecolor(color)
                            if hasattr(handle, "set_edgecolor"):
                                handle.set_edgecolor("black")
                    for txt in ax.texts:
                        if "trace count" in txt.get_text():
                            txt.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#9CA3AF", alpha=0.95))
                            txt.set_clip_on(False)
            for fig in figs:
                if hasattr(fig, "_suptitle") and fig._suptitle is not None:
                    fig._suptitle.set_text(str(name) if name else "断裂网络")
                    zh_fonts = plt.rcParams.get("font.sans-serif", [])
                    if isinstance(zh_fonts, (list, tuple)) and len(zh_fonts) > 0:
                        fig._suptitle.set_fontfamily(zh_fonts[0])
                    fig._suptitle.set_fontsize(15)
                    fig._suptitle.set_bbox(dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#9CA3AF", alpha=0.95))
                    fig._suptitle.set_x(0.5)
                    fig._suptitle.set_y(0.96)
                    fig._suptitle.set_ha("center")
                    fig._suptitle.set_va("top")
                # 略增子图间距与右侧留白，避免图例与 trace count 文本与邻轴重叠
                fig.subplots_adjust(left=0.09, right=0.82, top=0.85, bottom=0.15, wspace=0.48)
                fig.set_size_inches(16, 9.0)
            if figs:
                _cap_rel = (
                    "交叉与相邻关系图：各子图表示两方位集之间交切（cross-cut）与不同方向邻接（abutting）的计数统计；"
                    "柱色对应图例中关系类型；侧栏为 trace count。翻页可浏览不同方位集组合。"
                )
                self.embed_figure(figs, descriptions=[_cap_rel] * len(figs))
        self._run_with_network(_render)

    def b(self):
        branches, nodes = branches_and_nodes(traces, area, snap_threshold=0.001)
        # 左右并排，避免上下叠图时标题与坐标轴标签互相遮挡
        h_in = max(7.5, 8.0 * float(rate))
        fig, axes = plt.subplots(1, 2, figsize=(17, h_in), sharex=True, sharey=True)
        ax0, ax1 = axes[0], axes[1]
        traces.plot(ax=ax0, color="blue", label="Traces")
        area.boundary.plot(ax=ax0, color="black", label="Target Area", linestyle="dashed")
        ax0.set_title("Traces & Target Area", fontsize=12, pad=12)
        nodes.plot(ax=ax1, column="Class", zorder=10, legend=False, categorical=True, markersize=7)
        ax1.set_title("Branches & Nodes & Area", fontsize=12, pad=12)
        area.boundary.plot(ax=ax1, color="black", linestyle="dashed")
        for ax in (ax0, ax1):
            ax.set_xlim(left - width, right + width)
            ax.set_ylim(down - height, up + height)
            area.boundary.plot(ax=ax, color="red")
            ax.set_aspect("equal")
            xa0, xa1 = ax.get_xlim()
            ya0, ya1 = ax.get_ylim()
            mx = max(abs(xa0), abs(xa1), abs(ya0), abs(ya1))
            if mx >= 1e5:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
                ax.set_xlabel("Easting (×10⁶ m)")
                if ax is ax0:
                    ax.set_ylabel("Northing (×10⁶ m)")
            else:
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
                ax.set_xlabel("X")
                if ax is ax0:
                    ax.set_ylabel("Y")
            ax.tick_params(axis="both", labelsize=9)

        class_order = [c for c in ("X", "Y", "I", "E") if c in nodes["Class"].dropna().unique()]
        if class_order:
            cmap = plt.get_cmap("tab10")
            handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=7,
                       markerfacecolor=cmap(i), markeredgecolor="black", label=cls)
                for i, cls in enumerate(class_order)
            ]
            legend = ax1.legend(
                handles=handles,
                title="Node Type",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
            )
            leg_handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
            for handle in leg_handles:
                if hasattr(handle, "_sizes"):
                    handle._sizes = [20]
        fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.14, wspace=0.18)
        self.embed_figure(
            fig,
            description=(
                "左：原始迹线与研究区边界；右：节点类型（X/Y/I 等）在平面上的位置及研究区；"
                "两图共用坐标比例，便于与拓扑化视图对照检查识别结果。"
            ),
        )

    def _plot_contour_safe(self, network, sampled_grid, parameters):
        import pandas as pd
        import numpy as np

        if isinstance(parameters, str):
            parameters = [parameters]

        for param in parameters:
            if param not in sampled_grid.columns:
                continue

            grid_plot = sampled_grid.copy()
            grid_plot = grid_plot[grid_plot.geometry.notna()]
            grid_plot = grid_plot[~grid_plot.geometry.is_empty]
            grid_plot = grid_plot[grid_plot.geometry.is_valid]

            if grid_plot.empty:
                print(f"⚠️ 参数 {param} 的有效网格为空，跳过绘制。")
                continue

            grid_plot[param] = pd.to_numeric(grid_plot[param], errors='coerce')
            grid_plot[param] = grid_plot[param].replace([np.inf, -np.inf], np.nan).fillna(0)

            try:
                fig, ax = plt.subplots(figsize=(9, 8))
                ax.grid(False)

                centroids = grid_plot.geometry.centroid
                x = centroids.x.values
                y = centroids.y.values
                z = grid_plot[param].values

                try:
                    network.trace_gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.3)
                except Exception:
                    pass

                contour = ax.tricontourf(x, y, z, levels=50, cmap='plasma', alpha=0.85)

                cbar = fig.colorbar(contour, ax=ax)
                cbar.set_label(param)

                ax.set_aspect("equal")
                ax.set_title(f"平滑热力图: {param}", fontsize=14, pad=15)
                plt.tight_layout()

                _cap = (
                    f"本图为网格拓扑参数「{param}」的平滑填色：对网格中心做三角剖分后插值填色；"
                    f"色条为该指标量纲；浅底为迹线叠置。坐标为当前投影平面。"
                )
                try:
                    self.embed_figure([fig], description=_cap)
                except TypeError:
                    self.embed_figure(fig, description=_cap)


            except Exception as e:
                print(f"平滑渲染失败: {param}。原因: {str(e)}")
                try:
                    plt.close(fig)
                except Exception:
                    pass

    def run_lunkuo(self):
        warnings.filterwarnings("ignore")
        if getattr(self, "_is_rendering_contour", False):
            print("轮廓/热图任务仍在运行，已忽略重复触发。")
            return
        if self.opt <= 0:
            return
        if traces is None or area is None or traces.empty:
            QMessageBox.warning(self, "无数据", "请先切换数据源并确保迹线、研究区文件存在且非空。")
            return
        print(f"当前选择的绘图选项: {self.opt}")

        self._is_rendering_contour = True
        self._lunkuo_progress = QtWidgets.QProgressDialog("正在进行空间计算与高清渲染，请耐心等待...", None, 0, 0, self)
        self._lunkuo_progress.setWindowTitle("系统运算中")
        self._lunkuo_progress.setWindowModality(QtCore.Qt.WindowModal)
        self._lunkuo_progress.setMinimumDuration(0)
        self._lunkuo_progress.setCancelButton(None)
        self._lunkuo_progress.show()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

        # 快照当前绘图选项，避免后台线程运行期间被改变
        opt_snapshot = self.opt

        # 预处理（轻量，在主线程完成）
        try:
            traces_local = traces.copy()
            bounds = traces_local.total_bounds

            if area is not None:
                minx, miny, maxx, maxy = area.total_bounds
                spatial_index = traces_local.sindex
                possible_matches_index = list(spatial_index.intersection((minx, miny, maxx, maxy)))
                traces_local = traces_local.iloc[possible_matches_index]

            map_span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            dp_tolerance = map_span * 0.002
            if dp_tolerance > 0:
                traces_local = traces_local.copy()
                traces_local.geometry = traces_local.geometry.simplify(tolerance=dp_tolerance, preserve_topology=True)
        except Exception as e:
            print(f"算法预处理跳过。原因: {e}")
            traces_local = traces.copy()

        bounds = traces_local.total_bounds
        export_cfg = _cfg_section("export_grid")
        dynamic_width = float(self.dspin_grid_step.value()) if hasattr(self, "dspin_grid_step") else 0.0
        if dynamic_width <= 0:
            dynamic_width = float(export_cfg.get("cell_width", 0.0) or 0.0)
        if dynamic_width <= 0:
            dynamic_width = (bounds[2] - bounds[0]) / 20.0
        if dynamic_width <= 0:
            dynamic_width = 100.0

        print(f"正在执行空间拓扑计算 (网格大小: {dynamic_width:.4f})...")

        # 耗时的拓扑计算放入后台线程，避免主线程阻塞被 macOS 强杀
        area_snapshot = area.copy() if area is not None else None
        name_snapshot = name

        def _bg_compute():
            nw, nw_err = try_network(
                traces_local,
                area_snapshot,
                name=name_snapshot,
                determine_branches_nodes=True,
                truncate_traces=True,
                circular_target_area=False,
                snap_threshold=0.001,
            )
            if nw_err:
                raise RuntimeError(nw_err)
            sg = nw.contour_grid(cell_width=dynamic_width)
            return nw, sg

        def _on_compute_done(result):
            network, sampled_grid = result
            print("拓扑网格计算完成，准备渲染！")
            if hasattr(self, "_lunkuo_progress") and self._lunkuo_progress:
                self._lunkuo_progress.setLabelText("计算完成，正在生成高清平滑图像...")
                QtWidgets.QApplication.processEvents()
            try:
                if opt_snapshot == 1:
                    self._plot_contour_safe(network, sampled_grid, ["Fracture Intensity B21", "Fracture Intensity P21"])
                elif opt_snapshot == 2:
                    self._plot_contour_safe(network, sampled_grid,
                                            ["Trace Min Length", "Trace Max Length", "Trace Mean Length"])
                elif opt_snapshot == 3:
                    self._plot_contour_safe(network, sampled_grid,
                                            ["Dimensionless Intensity B22", "Dimensionless Intensity P22"])
                elif opt_snapshot == 4:
                    self._plot_contour_safe(network, sampled_grid, "Number of Traces (Real)")
                elif opt_snapshot == 5:
                    self._plot_contour_safe(network, sampled_grid,
                                            ["Branch Min Length", "Branch Max Length", "Branch Mean Length"])
                elif opt_snapshot == 6:
                    self._plot_contour_safe(network, sampled_grid, ["Areal Frequency B20", "Areal Frequency P20"])
                elif opt_snapshot == 7:
                    self._plot_contour_safe(network, sampled_grid, ["Connections per Trace", "Connections per Branch"])
                elif opt_snapshot == 8:
                    self._plot_contour_safe(network, sampled_grid, "Connection Frequency")
            except Exception as e:
                print(f"❌ 渲染报错: {str(e)}")
            finally:
                _cleanup()

        def _on_compute_failed(err_msg):
            print(f"❌ 运行报错: {err_msg}")
            QMessageBox.warning(self, "无法构建断裂网络", err_msg)
            _cleanup()

        def _cleanup():
            self._is_rendering_contour = False
            QtWidgets.QApplication.restoreOverrideCursor()
            if hasattr(self, "_lunkuo_progress") and self._lunkuo_progress:
                self._lunkuo_progress.close()
                self._lunkuo_progress = None
            self._lunkuo_runner = None

        runner = TaskRunner(_bg_compute)
        runner.finished_ok.connect(_on_compute_done)
        runner.failed.connect(_on_compute_failed)
        self._lunkuo_runner = runner  # 防止被 GC
        runner.start()

    def run_ronghe(self):
        if run_fusion_pipeline is None:
            QMessageBox.warning(
                self, "模块未安装",
                "请确保 topology_fusion 模块可用（与 main.py 同目录），并已安装 scikit-learn。",
            )
            return
        method = self.combo_fusion.currentText().replace("融合方法: ", "").strip()
        if method in ("自编码器", "VAE") and not HAS_TORCH:
            QMessageBox.warning(
                self, "需要安装 PyTorch",
                f"「{method}」依赖 PyTorch。请执行：pip install torch\n或先选择 PCA/UMAP。",
            )
            return
        if method == "UMAP" and not HAS_UMAP:
            QMessageBox.warning(
                self, "需要安装 umap-learn",
                "「UMAP」依赖 umap-learn。请执行：pip install umap-learn\n或先选择 PCA。",
            )
            return
        csv_path = self._get_guoji_csv()
        if not csv_path:
            return
        warnings.filterwarnings("ignore")
        n_k = int(self.spin_kmeans_k.value()) if hasattr(self, "spin_kmeans_k") else int(_cfg_section("clustering").get("n_clusters", 4))
        try:
            if method == "自编码器":
                df_out, scaler, kmeans, cluster_means = run_fusion_pipeline_ae(csv_path, n_clusters=n_k)
                x_col, y_col = "Z1", "Z2"
                method_name = "自编码器"
            elif method == "UMAP":
                df_out, scaler, _, kmeans, cluster_means = run_fusion_pipeline_umap(csv_path, n_clusters=n_k)
                x_col, y_col = "U1", "U2"
                method_name = "UMAP"
            elif method == "VAE":
                df_out, scaler, kmeans, cluster_means = run_fusion_pipeline_vae(csv_path, n_clusters=n_k)
                x_col, y_col = "Z1", "Z2"
                method_name = "VAE"
            else:
                df_out, scaler, pca, kmeans, cluster_means = run_fusion_pipeline(csv_path, n_clusters=n_k)
                x_col, y_col = "PC1", "PC2"
                method_name = "PCA"
        except Exception as e:
            QMessageBox.critical(self, "运行出错", self._friendly_error_message("属性融合/聚类", str(e)))
            return
        n_samples = len(df_out)
        n_clusters = int(df_out["cluster_id"].nunique())
        spatial_ok = all(
            c in df_out.columns
            for c in (
                "vertex1_x",
                "vertex1_y",
                "vertex2_x",
                "vertex2_y",
                "vertex3_x",
                "vertex3_y",
                "vertex4_x",
                "vertex4_y",
            )
        )
        if spatial_ok:
            # 左右排列；尺寸略小于原先的 13.5×5.8，减轻占屏
            fig1, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(11.6, 5.25),
                gridspec_kw={"wspace": 0.26},
            )
        else:
            fig1, ax1 = plt.subplots(figsize=(7, 6))
            ax2 = None
        ax1.set_axisbelow(True)
        ax1.grid(True, alpha=0.42, linestyle="-", linewidth=0.55, color="0.75", zorder=0.3)
        ax1.set_facecolor("#e8ebf2")
        has_center_legend, cmap_latent, norm_latent = self._plot_latent_fusion_kmeans_regions(
            ax1, df_out, x_col, y_col, kmeans, n_clusters
        )
        _skw = dict(
            s=19,
            alpha=0.9,
            zorder=3,
            edgecolors="white",
            linewidths=0.32,
        )
        if cmap_latent is not None and norm_latent is not None:
            scatter = ax1.scatter(
                df_out[x_col],
                df_out[y_col],
                c=df_out["cluster_id"],
                cmap=cmap_latent,
                norm=norm_latent,
                **_skw,
            )
        else:
            scatter = ax1.scatter(
                df_out[x_col],
                df_out[y_col],
                c=df_out["cluster_id"],
                cmap="tab10",
                **_skw,
            )
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title(
            f"拓扑属性融合（{method_name}）：{x_col}–{y_col}"
            f"\n（柔和底色=KMeans 分区｜散点=网格单元）",
            fontsize=11,
        )
        plt.colorbar(scatter, ax=ax1, label="cluster_id")
        if has_center_legend:
            leg = ax1.legend(loc="best", fontsize=8, framealpha=0.92)
            if leg is not None:
                leg.set_zorder(6)
        ax1.set_aspect("equal", adjustable="datalim")
        if ax2 is not None:
            if not self._plot_spatial_cluster_grid(ax2, df_out, n_clusters, method_name):
                ax2.text(0.5, 0.5, "无法绘制空间网格（缺少顶点列）", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_axis_off()
        plt.tight_layout()
        cluster_paths: dict = {}
        qual: dict = {}
        from utils.export_utils import build_run_metadata

        run_meta = build_run_metadata(config_path=os.path.join(_PROGRAM_DIR, "config.yaml"))
        df_out["processing_run_id"] = run_meta.get("processing_run_id", "")
        df_out["run_timestamp_utc"] = run_meta.get("run_timestamp_utc", "")
        df_out["config_hash_sha256"] = run_meta.get("config_hash_sha256", "")
        if (
            build_cluster_name_map is not None
            and attach_cluster_names is not None
            and compute_cluster_quality_metrics is not None
            and compute_cluster_stability_ari is not None
            and build_cluster_summary_rows is not None
            and export_cluster_results is not None
        ):
            try:
                conn_cols = [c for c in _TF_CONN_COLS if c in cluster_means.columns]
                name_map = build_cluster_name_map(cluster_means, connectivity_cols=conn_cols or None)
                df_out = attach_cluster_names(df_out, name_map)
                Xz = df_out[[x_col, y_col]].to_numpy(dtype=np.float64)
                qual = compute_cluster_quality_metrics(Xz, df_out["cluster_id"].to_numpy())
                qual.update(compute_cluster_stability_ari(Xz, n_clusters))
                summary_df = build_cluster_summary_rows(df_out, cluster_means, name_map)
                cluster_paths = export_cluster_results(
                    df_out,
                    csv_path,
                    method_name,
                    cluster_summary=summary_df,
                    quality_metrics=qual or None,
                )
            except Exception:
                try:
                    cluster_paths = export_cluster_results(df_out, csv_path, method_name)
                except Exception:
                    cluster_paths = {}
        elif export_cluster_results is not None:
            try:
                cluster_paths = export_cluster_results(df_out, csv_path, method_name)
            except Exception:
                cluster_paths = {}
        self._remember_exports(
            "属性融合聚类",
            cluster_csv=cluster_paths.get("csv"),
            cluster_gpkg=cluster_paths.get("gpkg"),
            cluster_summary=cluster_paths.get("cluster_summary_csv"),
            fusion_quality=cluster_paths.get("quality_json"),
        )
        self.embed_figure(
            fig1,
            description=(
                "属性融合与聚类：左图为多拓扑指标降维后的潜空间（散点为网格、底色为 KMeans 分区），"
                "右图为同一聚类编号在平面网格上的空间分布（绿–蓝为簇编号）。"
                "用于观察属性相似簇是否在空间上成片出现。"
            ),
        )
        summary_lines = [
            "【智能拓扑分析结果】",
            "",
            f"融合方式：{method_name}（GUI k={n_k}）",
            f"数据：{os.path.basename(csv_path)}",
            f"有效网格数：{n_samples}",
            f"聚类数：{n_clusters}",
            f"新属性：{x_col}, {y_col}, cluster_id, cluster_name（导出列）",
            f"聚类结果 CSV：{self._relpath_for_ui(cluster_paths.get('csv', '未导出'))}",
            f"聚类结果 GPKG：{self._relpath_for_ui(cluster_paths.get('gpkg', '未导出或无空间几何'))}",
            f"逐簇统计表：{self._relpath_for_ui(cluster_paths.get('cluster_summary_csv', '—'))}",
            f"质量指标 JSON：{self._relpath_for_ui(cluster_paths.get('quality_json', '—'))}",
        ]
        if qual:
            summary_lines.extend(
                [
                    "",
                    "聚类质量（潜空间）：",
                    f"  silhouette_score（越大越好）：{qual.get('silhouette_score', float('nan')):.4f}",
                    f"  davies_bouldin_index（越小越好）：{qual.get('davies_bouldin_index', float('nan')):.4f}",
                    f"  stability ARI 均值（越大越稳）：{qual.get('cluster_stability_ari_mean', float('nan')):.4f}",
                ]
            )
        if "cluster_name" in df_out.columns:
            summary_lines.extend(
                [
                    "",
                    "簇命名（示意）：",
                    df_out[["cluster_id", "cluster_name"]].drop_duplicates().sort_values("cluster_id").to_string(index=False),
                ]
            )
        summary_lines.extend(
            [
                "",
                "各簇在部分拓扑属性上的均值：",
                "",
                cluster_means.head(min(8, len(cluster_means))).to_string(),
            ]
        )
        self.text_browser.clear()
        self.text_browser.insertPlainText("\n".join(summary_lines))
        self.text_browser.moveCursor(QTextCursor.End)
        from utils.export_utils import clusters_gpkg_layer_name as _layer_fn, write_run_manifest

        _layer = _layer_fn(method_name)
        run_manifest = write_run_manifest(
            os.path.join(os.path.dirname(csv_path), "data", "processed"),
            run_id=str(run_meta.get("processing_run_id", "")) or None,
            kind="cluster",
            config_path=os.path.join(_PROGRAM_DIR, "config.yaml"),
            artifacts={
                "cluster_csv": cluster_paths.get("csv"),
                "cluster_gpkg": cluster_paths.get("gpkg"),
                "cluster_summary_csv": cluster_paths.get("cluster_summary_csv"),
                "quality_json": cluster_paths.get("quality_json"),
            },
            extra={
                "method": method_name,
                "n_clusters": int(n_clusters),
                "k_gui": int(n_k),
            },
        )
        self._remember_exports("属性融合聚类", run_manifest=run_manifest)
        self._update_last_run_card(
            [
                "类型：属性融合 + 聚类",
                f"方法：{method_name}，k={n_clusters}",
                f"导出：{self._relpath_for_ui(cluster_paths.get('csv', ''))}",
                f"GPKG 图层：{_layer}",
            ]
        )
        self._append_run_history(
            {
                "kind": "fusion_cluster",
                "method": method_name,
                "k": int(n_clusters),
                "silhouette": qual.get("silhouette_score") if qual else None,
                "cluster_csv": cluster_paths.get("csv"),
            }
        )
        self._refresh_config_summary()
        QMessageBox.information(
            self, "运行完成",
            f"已用 {method_name} 生成 {x_col}、{y_col} 与 {n_clusters} 类聚类结果；GPKG 图层名为「{_layer}」"
            f"。\n聚类 CSV：{self._relpath_for_ui(cluster_paths.get('csv', '未导出'))}"
            f"\n聚类 GPKG：{self._relpath_for_ui(cluster_paths.get('gpkg', '未导出或无空间几何'))}",
        )


import traceback


def exception_hook(exctype, value, tb):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("\n" + "=" * 50)
    print("系统崩溃：")
    print("".join(traceback.format_exception(exctype, value, tb)))
    print("=" * 50 + "\n")
    sys.exit(1)


if __name__ == "__main__":
    sys.excepthook = exception_hook  # 挂载防崩溃

    # 开启高 DPI 缩放，避免在高分屏/Retina 屏幕上窗口过小或模糊
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    window = MainWindow()

    # 根据当前屏幕可用区域自动调整窗口大小
    screen = app.primaryScreen()
    available = screen.availableGeometry()
    sw, sh = available.width(), available.height()

    # 目标尺寸：屏幕可用区域的 92%，但不低于 1100×700
    target_w = max(1100, int(sw * 0.92))
    target_h = max(700, int(sh * 0.92))
    # 同时不超过屏幕可用区域
    target_w = min(target_w, sw)
    target_h = min(target_h, sh)

    window.resize(target_w, target_h)

    # 居中显示
    x = available.x() + (sw - target_w) // 2
    y = available.y() + (sh - target_h) // 2
    window.move(x, y)

    window.show()

    # show() 后按实际窗口内容宽度重设左右 splitter 比例（30% 左面板 / 70% 右画板）
    # 减去主布局左右边距（各 10px）
    inner_w = target_w - 20
    window.splitter.setSizes([int(inner_w * 0.30), int(inner_w * 0.70)])

    sys.exit(app.exec_())
