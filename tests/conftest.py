"""Shared pytest fixtures for aumai-trustforge tests."""

from __future__ import annotations

import datetime

import pytest

from aumai_trustforge.core import TrustScorer
from aumai_trustforge.models import (
    BehaviorEvidence,
    CapabilityEvidence,
    DimensionScore,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustWeights,
)

# ---------------------------------------------------------------------------
# Weight fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def equal_weights() -> TrustWeights:
    """Default equal weights (0.25 each)."""
    return TrustWeights()


@pytest.fixture()
def custom_weights() -> TrustWeights:
    """Non-equal but valid weights summing to 1.0."""
    return TrustWeights(
        provenance=0.30,
        behavior=0.30,
        capability=0.20,
        security=0.20,
    )


@pytest.fixture()
def scorer(equal_weights: TrustWeights) -> TrustScorer:
    """TrustScorer with default equal weights."""
    return TrustScorer(equal_weights)


@pytest.fixture()
def custom_scorer(custom_weights: TrustWeights) -> TrustScorer:
    """TrustScorer with custom asymmetric weights."""
    return TrustScorer(custom_weights)


# ---------------------------------------------------------------------------
# Perfect evidence fixtures — every signal set to its highest possible value
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_provenance() -> ProvenanceEvidence:
    """All four provenance signals present and verified."""
    return ProvenanceEvidence(
        model_card_present=True,
        license_verified=True,
        author_verified=True,
        source_url="https://example.com/model-card",
    )


@pytest.fixture()
def perfect_behavior() -> BehaviorEvidence:
    """Zero error rate, zero latency, 100% uptime, ample samples."""
    return BehaviorEvidence(
        error_rate=0.0,
        avg_latency_ms=0.0,
        uptime_pct=100.0,
        sample_count=1000,
    )


@pytest.fixture()
def perfect_capability() -> CapabilityEvidence:
    """All claimed capabilities verified via a structured method."""
    return CapabilityEvidence(
        claimed_capabilities=["summarize", "translate", "classify"],
        verified_capabilities=["summarize", "translate", "classify"],
        verification_method="benchmark_suite",
    )


@pytest.fixture()
def perfect_security() -> SecurityEvidence:
    """Sandbox compliant, no vulnerabilities, audited today."""
    return SecurityEvidence(
        sandbox_compliant=True,
        vulnerability_count=0,
        last_audit_date=datetime.date.today(),
    )


# ---------------------------------------------------------------------------
# Worst-case evidence fixtures — every signal at its lowest value
# ---------------------------------------------------------------------------


@pytest.fixture()
def worst_provenance() -> ProvenanceEvidence:
    """No signals present."""
    return ProvenanceEvidence(
        model_card_present=False,
        license_verified=False,
        author_verified=False,
        source_url=None,
    )


@pytest.fixture()
def worst_behavior() -> BehaviorEvidence:
    """100% error rate, maximum latency, zero uptime, no samples."""
    return BehaviorEvidence(
        error_rate=1.0,
        avg_latency_ms=10_000.0,
        uptime_pct=0.0,
        sample_count=0,
    )


@pytest.fixture()
def worst_capability() -> CapabilityEvidence:
    """No claims, no verified capabilities, no verification method."""
    return CapabilityEvidence(
        claimed_capabilities=[],
        verified_capabilities=[],
        verification_method="",
    )


@pytest.fixture()
def worst_security() -> SecurityEvidence:
    """Not sandbox compliant, many vulnerabilities, no audit."""
    return SecurityEvidence(
        sandbox_compliant=False,
        vulnerability_count=100,
        last_audit_date=None,
    )


# ---------------------------------------------------------------------------
# Pre-built DimensionScore fixtures (bypass evidence scoring)
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_dim_scores(
    scorer: TrustScorer,
    perfect_provenance: ProvenanceEvidence,
    perfect_behavior: BehaviorEvidence,
    perfect_capability: CapabilityEvidence,
    perfect_security: SecurityEvidence,
) -> dict[str, DimensionScore]:
    """All four dimension scores at maximum for a fully-ideal agent."""
    return {
        "provenance": scorer.score_provenance(perfect_provenance),
        "behavior": scorer.score_behavior(perfect_behavior),
        "capability": scorer.score_capability(perfect_capability),
        "security": scorer.score_security(perfect_security),
    }


@pytest.fixture()
def worst_dim_scores(
    scorer: TrustScorer,
    worst_provenance: ProvenanceEvidence,
    worst_behavior: BehaviorEvidence,
    worst_capability: CapabilityEvidence,
    worst_security: SecurityEvidence,
) -> dict[str, DimensionScore]:
    """All four dimension scores at minimum for a fully-degraded agent."""
    return {
        "provenance": scorer.score_provenance(worst_provenance),
        "behavior": scorer.score_behavior(worst_behavior),
        "capability": scorer.score_capability(worst_capability),
        "security": scorer.score_security(worst_security),
    }
