from pathlib import Path


MAIN_SOURCE = Path(__file__).resolve().parents[1] / "program" / "main.py"


def _source() -> str:
    return MAIN_SOURCE.read_text(encoding="utf-8")


def test_formal_screening_is_the_primary_multiperiod_entry():
    source = _source()
    assert 'menu.addAction("一键生成候选勘探有利区", self._run_target_screening)' in source
    assert 'menu.addMenu("专业分析与图件")' in source
    assert 'menu.addMenu("研究与实验")' in source


def test_formal_gui_path_does_not_call_agent_model():
    source = _source()
    start = source.index("    def _run_target_screening(self):")
    end = source.index("    def _show_target_screening_result(self, res):")
    formal_path = source[start:end]
    assert "screening_pipeline" in formal_path
    assert "agent_model" not in formal_path
    assert "show_progress=True" in formal_path


def test_legacy_all_entry_redirects_to_formal_pipeline():
    source = _source()
    start = source.index("    def _run_multiperiod_all(self):")
    end = source.index("    def _launch_multiperiod_task", start)
    compatibility_path = source[start:end]
    assert "self._run_target_screening()" in compatibility_path
    assert "for name, fn in stages" not in compatibility_path


def test_existing_plotting_functions_remain_available():
    source = _source()
    required_methods = [
        "run_yuantu",
        "run_fenleihou",
        "run_relitu",
        "run_azimuth",
        "run_meiguitu",
        "run_sanyuantu",
        "run_guanxi",
        "run_tuopushuxing",
        "run_lunkuo",
    ]
    for method in required_methods:
        assert f"    def {method}(self" in source
