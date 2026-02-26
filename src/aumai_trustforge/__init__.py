"""AumAI TrustForge — generic trust scoring for AI agents.

Public API::

    from aumai_trustforge import (
        BehaviorEvidence,
        CapabilityEvidence,
        DimensionScore,
        ProvenanceEvidence,
        SecurityEvidence,
        TrustDimension,
        TrustScore,
        TrustScorer,
        TrustWeights,
    )
"""

from aumai_trustforge.core import TrustScorer
from aumai_trustforge.models import (
    BehaviorEvidence,
    CapabilityEvidence,
    DimensionScore,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustDimension,
    TrustScore,
    TrustWeights,
)

__version__ = "0.1.0"

__all__ = [
    # models
    "BehaviorEvidence",
    "CapabilityEvidence",
    "DimensionScore",
    "ProvenanceEvidence",
    "SecurityEvidence",
    "TrustDimension",
    "TrustScore",
    "TrustWeights",
    # core
    "TrustScorer",
]
