"""Zero-dependency structural compatibility adapters for external optimizer interfaces."""

from __future__ import annotations

from .dspy import GEPA, HillclimbGEPA, HillclimbMIPRO, MIPROv2
from .gepa_ai import AdapterProtocol, EvaluationBatchProtocol, GepaResult, optimize

__all__ = [
    "AdapterProtocol",
    "EvaluationBatchProtocol",
    "GEPA",
    "GepaResult",
    "HillclimbGEPA",
    "HillclimbMIPRO",
    "MIPROv2",
    "optimize",
]
