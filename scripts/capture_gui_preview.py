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

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.plot([0, 1, 2, 3], [0, 1, 0.4, 1.4], color="#356f70", linewidth=3)
        ax.scatter([0, 1, 2, 3], [0, 1, 0.4, 1.4], color="#a95757", s=70)
        ax.set_title("图件自适应显示冒烟验证")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        window.embed_figure(fig, description="大尺寸图默认适应右侧可见区域，无需滚动查看全图。")
        app.processEvents()
        window._apply_figure_view_mode()
        app.processEvents()

        canvas_size = window.current_canvas.size()
        viewport_size = window.canvas_scroll.viewport().size()
        checks = {
            "primary_visible": window.btn_primary_screening.isVisible(),
            "run_info_visible": window.run_info_tabs.isVisible(),
            "fit_checked": window.btn_fit_fig.isChecked(),
            "canvas_within_viewport": (
                canvas_size.width() <= viewport_size.width()
                and canvas_size.height() <= viewport_size.height()
            ),
            "canvas_size": [canvas_size.width(), canvas_size.height()],
            "viewport_size": [viewport_size.width(), viewport_size.height()],
        }
        window.btn_fit_fig.setChecked(False)
        window._toggle_figure_fit()
        app.processEvents()
        original_size = window.current_canvas.size()
        checks["original_mode_exceeds_viewport"] = (
            original_size.width() > viewport_size.width()
            or original_size.height() > viewport_size.height()
        )
        checks["original_canvas_size"] = [original_size.width(), original_size.height()]
        window.btn_fit_fig.setChecked(True)
        window._toggle_figure_fit()
        app.processEvents()
        output = ROOT / "artifacts" / "experiment" / "target-screening-mvp-v1" / "gui_v2_1_preview.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        saved = window.grab().save(str(output))
        sys.__stdout__.write(f"GUI_PREVIEW_SAVED={saved} {output}\n")
        sys.__stdout__.write(f"GUI_CHECKS={checks}\n")
        sys.__stdout__.flush()
        return 0 if saved and all([
            checks["primary_visible"], checks["run_info_visible"],
            checks["fit_checked"], checks["canvas_within_viewport"],
            checks["original_mode_exceeds_viewport"],
        ]) else 1
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
