from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContractClause:
    clause_id: str
    title: str
    content: str
    clause_type: str = "unknown"


@dataclass
class LegalRisk:
    clause_id: str
    clause_text: str
    risk_level: str
    issue: str
    legal_basis: list[dict] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class ContractAnalysisResult:
    summary: str
    parties: list[str]
    assets: list[str]
    obligations: list[str]
    risks: list[LegalRisk]
    sources: list[dict]
    timings: dict