from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_primary_screening_entry_is_visible_in_main_layout():
    source = (ROOT / "program" / "demo.py").read_text(encoding="utf-8")
    assert 'self.btn_primary_screening = QtWidgets.QPushButton("生成候选勘探有利区  →")' in source
    assert 'self.main_vbox.addWidget(self.screening_banner)' in source
    assert 'self.screening_status_label = QtWidgets.QLabel("尚未运行")' in source


def test_run_summary_and_log_share_secondary_collapsible_surface():
    source = (ROOT / "program" / "demo.py").read_text(encoding="utf-8")
    assert 'self.run_info_tabs.addTab(self.run_summary_tab, "结果摘要")' in source
    assert 'self.run_info_tabs.addTab(self.run_log_tab, "运行日志")' in source
    assert "self.btn_toggle_run_info" in source


def test_startup_guide_is_non_modal_and_matches_formal_workflow():
    source = (ROOT / "program" / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _show_startup_flow_guide(self):")
    end = source.index("    def _set_busy", start)
    guide = source[start:end]
    assert "生成候选勘探有利区" in guide
    assert "QMessageBox" not in guide


def test_task_cancellation_never_force_terminates_qthread():
    source = (ROOT / "program" / "main.py").read_text(encoding="utf-8")
    start = source.index("    def cancel_running_task(self):")
    end = source.index("    def _toggle_run_info", start)
    cancellation = source[start:end]
    assert "requestInterruption" in cancellation
    assert ".terminate()" not in cancellation


def test_figure_viewer_defaults_to_fit_mode_without_forced_canvas_height():
    demo_source = (ROOT / "program" / "demo.py").read_text(encoding="utf-8")
    main_source = (ROOT / "program" / "main.py").read_text(encoding="utf-8")
    assert "self.btn_fit_fig.setChecked(True)" in demo_source
    assert "def _apply_figure_view_mode(self):" in main_source
    assert "ScrollBarAlwaysOff" in main_source
    assert "canvas.setMinimumHeight(int(fig.get_size_inches()[1] * fig.dpi))" not in main_source
