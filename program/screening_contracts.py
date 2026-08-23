# -*- coding: utf-8 -*-
"""候选靶区正式主流程的稳定数据合同。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CandidateTargetRecord:
    target_id: str
    target_level: str
    centroid_x: float
    centroid_y: float
    area: float
    total_score: float
    network_criticality: float
    removal_impact: float
    period_persistence: float
    parameter_stability: float
    supporting_periods: List[str]
    cell_count: int
    evidence_status: str
    recommendation_reason: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningRunResult:
    run_id: str
    status: str
    input_summary: Dict[str, Any]
    period_results: Dict[str, Any]
    candidate_targets: List[Dict[str, Any]]
    stability_summary: Dict[str, Any]
    external_validation: Dict[str, Any]
    artifact_paths: Dict[str, str]
    limitations: List[str]
    warnings: List[str] = field(default_factory=list)
    failure: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
