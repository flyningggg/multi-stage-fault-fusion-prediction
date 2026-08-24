# -*- coding: utf-8 -*-
"""开发期GUI可见性与自适应画布冒烟截图。"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
import main


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = None
    try:
        window = main.MainWindow()
        window.resize(1600, 980)
        window.show()
        app.processEvents()

        window._show_project_evidence()
        app.processEvents()
        window._apply_figure_view_mode()
        app.processEvents()

        smart_size = window.current_canvas.size()
        viewport_size = window.canvas_scroll.viewport().size()
        window.combo_figure_view_mode.setCurrentIndex(1)  # 完整显示
        app.processEvents()
        fit_size = window.current_canvas.size()
        window.combo_figure_view_mode.setCurrentIndex(0)  # 清晰适配
        app.processEvents()
        smart_size = window.current_canvas.size()
        checks = {
            "primary_visible": window.btn_primary_screening.isVisible(),
            "evidence_button_visible": window.btn_evidence_overview.isVisible(),
            "evidence_summary_truthful": all(
                text in window.last_run_browser.toPlainText()
                for text in (
                    "稳定候选 18 个", "P2 高置信 5 个", "KB11 方法演示几何",
                    "原质心门槛未完全通过", "证据链一致性：15/15 项通过",
                    "真实同位物理验证：待数据",
                )
            ),
            "evidence_figures_loaded": len(window.current_figs) == 4,
            "run_info_visible": window.run_info_tabs.isVisible(),
            "smart_mode_default": window.combo_figure_view_mode.currentData() == "smart",
            "smart_readability_gain": (
                smart_size.width() * smart_size.height()
                >= fit_size.width() * fit_size.height() * 1.5
            ),
            "smart_no_horizontal_scroll": smart_size.width() <= viewport_size.width(),
            "smart_vertical_overflow_bounded": smart_size.height() <= int(viewport_size.height() * 1.40),
            "smart_canvas_size": [smart_size.width(), smart_size.height()],
            "strict_fit_canvas_size": [fit_size.width(), fit_size.height()],
            "viewport_size": [viewport_size.width(), viewport_size.height()],
        }
        window.combo_figure_view_mode.setCurrentIndex(3)  # 原始尺寸
        app.processEvents()
        original_size = window.current_canvas.size()
        checks["original_mode_exceeds_viewport"] = (
            original_size.width() > viewport_size.width()
            or original_size.height() > viewport_size.height()
        )
        checks["original_canvas_size"] = [original_size.width(), original_size.height()]
        window.combo_figure_view_mode.setCurrentIndex(0)
        app.processEvents()
        output = ROOT / "artifacts" / "experiment" / "target-screening-mvp-v2" / "gui_v2_3_preview.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        saved = window.grab().save(str(output))

        window.btn_focus_fig.setChecked(True)
        app.processEvents()
        window._apply_figure_view_mode()
        app.processEvents()
        focus_size = window.current_canvas.size()
        checks["focus_hides_secondary_chrome"] = all([
            not window.left_panel.isVisible(),
            not window.tab_widget.isVisible(),
            not window.run_info_tabs.isVisible(),
        ])
        checks["focus_enlarges_canvas"] = (
            focus_size.width() * focus_size.height()
            > smart_size.width() * smart_size.height() * 1.5
        )
        checks["focus_canvas_size"] = [focus_size.width(), focus_size.height()]
        focus_output = output.with_name("gui_v2_3_focus_preview.png")
        focus_saved = window.grab().save(str(focus_output))
        window.btn_focus_fig.setChecked(False)
        app.processEvents()
        sys.__stdout__.write(f"GUI_PREVIEW_SAVED={saved} {output}\n")
        sys.__stdout__.write(f"GUI_FOCUS_PREVIEW_SAVED={focus_saved} {focus_output}\n")
        sys.__stdout__.write(f"GUI_CHECKS={checks}\n")
        sys.__stdout__.flush()
        return 0 if saved and all([
            checks["primary_visible"], checks["evidence_button_visible"],
            checks["evidence_summary_truthful"], checks["evidence_figures_loaded"],
            checks["run_info_visible"],
            checks["smart_mode_default"], checks["smart_readability_gain"],
            checks["smart_no_horizontal_scroll"], checks["smart_vertical_overflow_bounded"],
            checks["original_mode_exceeds_viewport"],
            checks["focus_hides_secondary_chrome"], checks["focus_enlarges_canvas"],
        ]) and focus_saved else 1
    except Exception:
        sys.__stderr__.write(traceback.format_exc())
        sys.__stderr__.flush()
        return 2
    finally:
        if window is not None:
            window.close()
        app.quit()


if __name__ == "__main__":
    raise SystemExit(run())
