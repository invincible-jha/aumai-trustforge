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
        # async
        AsyncTrustScorer,
        # store
        TrustRecord,
        TrustStore,
        TrustStoreConfig,
        TrustStoreMetrics,
        # llm
        LLMTrustAssessment,
        LLMTrustAssessor,
        build_mock_assessor,
        # integration
        TrustForgeIntegration,
        setup_trustforge,
    )
"""

from aumai_trustforge.async_core import AsyncTrustScorer
from aumai_trustforge.core import TrustScorer
from aumai_trustforge.integration import TrustForgeIntegration, setup_trustforge
from aumai_trustforge.llm_assessor import (
    LLMTrustAssessment,
    LLMTrustAssessor,
    build_mock_assessor,
)
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
from aumai_trustforge.store import (
    TrustRecord,
    TrustStore,
    TrustStoreConfig,
    TrustStoreMetrics,
)

__version__ = "0.1.0"

__all__ = [
    # package metadata
    "__version__",
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
    # async
    "AsyncTrustScorer",
    # store
    "TrustRecord",
    "TrustStore",
    "TrustStoreConfig",
    "TrustStoreMetrics",
    # llm assessor
    "LLMTrustAssessment",
    "LLMTrustAssessor",
    "build_mock_assessor",
    # integration
    "TrustForgeIntegration",
    "setup_trustforge",
]
