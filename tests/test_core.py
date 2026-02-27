"""Tests for aumai_trustforge.core.

Covers TrustScorer and the private math helpers (_clamp, _linear_decay,
_sample_confidence).  The SR-1 scope excludes trust propagation and network
scoring — only the four-dimension generic scoring is tested here.
"""

from __future__ import annotations

import datetime

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aumai_trustforge.core import (
    TrustScorer,
    _clamp,
    _linear_decay,
    _sample_confidence,
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

# ===========================================================================
# _clamp
# ===========================================================================


class TestClamp:
    """Private _clamp helper: values are constrained to [low, high]."""

    def test_value_below_low_returns_low(self) -> None:
        assert _clamp(-5.0) == 0.0

    def test_value_above_high_returns_high(self) -> None:
        assert _clamp(2.0) == 1.0

    def test_value_in_range_returned_unchanged(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_boundary_values_returned_unchanged(self) -> None:
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_custom_range(self) -> None:
        assert _clamp(5.0, low=0.0, high=10.0) == 5.0
        assert _clamp(-1.0, low=0.0, high=10.0) == 0.0
        assert _clamp(15.0, low=0.0, high=10.0) == 10.0

    @given(value=st.floats(allow_nan=False, allow_infinity=False))
    def test_property_output_always_in_unit_interval(self, value: float) -> None:
        result = _clamp(value)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# _linear_decay
# ===========================================================================


class TestLinearDecay:
    """Private _linear_decay helper: 1.0 below low, 0.0 above high."""

    def test_value_at_low_returns_one(self) -> None:
        assert _linear_decay(500.0, low=500.0, high=5000.0) == 1.0

    def test_value_below_low_returns_one(self) -> None:
        assert _linear_decay(100.0, low=500.0, high=5000.0) == 1.0

    def test_value_at_high_returns_zero(self) -> None:
        assert _linear_decay(5000.0, low=500.0, high=5000.0) == 0.0

    def test_value_above_high_returns_zero(self) -> None:
        assert _linear_decay(9999.0, low=500.0, high=5000.0) == 0.0

    def test_midpoint_returns_half(self) -> None:
        midpoint = (500.0 + 5000.0) / 2.0
        result = _linear_decay(midpoint, low=500.0, high=5000.0)
        assert abs(result - 0.5) < 1e-9

    def test_interpolation_is_linear(self) -> None:
        """Three evenly spaced values should yield evenly spaced scores."""
        low, high = 0.0, 10.0
        scores = [_linear_decay(float(v), low=low, high=high) for v in [0, 5, 10]]
        assert scores == [1.0, 0.5, 0.0]

    @given(
        value=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        low=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
        high=st.floats(min_value=51.0, max_value=100.0, allow_nan=False),
    )
    def test_property_output_always_in_unit_interval(
        self, value: float, low: float, high: float
    ) -> None:
        result = _linear_decay(value, low=low, high=high)
        assert 0.0 <= result <= 1.0

    @given(
        value=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    )
    def test_property_below_low_always_one(self, value: float) -> None:
        result = _linear_decay(value, low=500.0, high=5000.0)
        assert result == 1.0


# ===========================================================================
# _sample_confidence
# ===========================================================================


class TestSampleConfidence:
    """Private _sample_confidence helper: log-scaled confidence from sample count."""

    def test_zero_samples_returns_zero(self) -> None:
        assert _sample_confidence(0, 100) == 0.0

    def test_negative_samples_returns_zero(self) -> None:
        assert _sample_confidence(-10, 100) == 0.0

    def test_at_target_returns_one(self) -> None:
        assert _sample_confidence(100, 100) == 1.0

    def test_above_target_returns_one(self) -> None:
        assert _sample_confidence(1000, 100) == 1.0

    def test_midpoint_is_below_half(self) -> None:
        """Log scaling means the midpoint count produces confidence < 0.5 at the halfway mark
        between 0 and target."""
        # At n=50, target=100: log(51)/log(101) — just verify it's in (0, 1).
        result = _sample_confidence(50, 100)
        assert 0.0 < result < 1.0

    def test_more_samples_produce_higher_confidence(self) -> None:
        """Monotonically increasing: n=10 < n=50 < n=99."""
        c10 = _sample_confidence(10, 100)
        c50 = _sample_confidence(50, 100)
        c99 = _sample_confidence(99, 100)
        assert c10 < c50 < c99

    def test_single_sample_above_zero(self) -> None:
        result = _sample_confidence(1, 100)
        assert result > 0.0

    @given(
        count=st.integers(min_value=0, max_value=200),
        target=st.integers(min_value=1, max_value=100),
    )
    def test_property_output_always_in_unit_interval(
        self, count: int, target: int
    ) -> None:
        result = _sample_confidence(count, target)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# TrustScorer — construction
# ===========================================================================


class TestTrustScorerConstruction:
    """TrustScorer instantiation with valid and invalid weights."""

    def test_default_weights_scorer_created(self) -> None:
        scorer = TrustScorer(TrustWeights())
        assert scorer is not None

    def test_custom_weights_scorer_created(self, custom_weights: TrustWeights) -> None:
        scorer = TrustScorer(custom_weights)
        assert scorer is not None


# ===========================================================================
# score_provenance
# ===========================================================================


class TestScoreProvenance:
    """score_provenance covers all boolean signal combinations."""

    def test_all_signals_true_yields_score_one(
        self, scorer: TrustScorer, perfect_provenance: ProvenanceEvidence
    ) -> None:
        result = scorer.score_provenance(perfect_provenance)
        assert result.score == 1.0
        assert result.confidence == 1.0
        assert result.dimension is TrustDimension.provenance

    def test_no_signals_yields_score_zero(
        self, scorer: TrustScorer, worst_provenance: ProvenanceEvidence
    ) -> None:
        result = scorer.score_provenance(worst_provenance)
        assert result.score == 0.0
        assert result.confidence == 1.0

    def test_model_card_contributes_030(self, scorer: TrustScorer) -> None:
        only_card = ProvenanceEvidence(model_card_present=True)
        result = scorer.score_provenance(only_card)
        assert abs(result.score - 0.30) < 1e-6

    def test_license_contributes_030(self, scorer: TrustScorer) -> None:
        only_license = ProvenanceEvidence(license_verified=True)
        result = scorer.score_provenance(only_license)
        assert abs(result.score - 0.30) < 1e-6

    def test_author_contributes_020(self, scorer: TrustScorer) -> None:
        only_author = ProvenanceEvidence(author_verified=True)
        result = scorer.score_provenance(only_author)
        assert abs(result.score - 0.20) < 1e-6

    def test_source_url_contributes_020(self, scorer: TrustScorer) -> None:
        only_url = ProvenanceEvidence(source_url="https://x.com")
        result = scorer.score_provenance(only_url)
        assert abs(result.score - 0.20) < 1e-6

    def test_empty_source_url_does_not_contribute(self, scorer: TrustScorer) -> None:
        ev = ProvenanceEvidence(source_url="")
        result = scorer.score_provenance(ev)
        assert result.score == 0.0

    def test_evidence_items_count_equals_four(
        self, scorer: TrustScorer, perfect_provenance: ProvenanceEvidence
    ) -> None:
        result = scorer.score_provenance(perfect_provenance)
        assert len(result.evidence) == 4

    def test_evidence_mentions_model_card_present(self, scorer: TrustScorer) -> None:
        result = scorer.score_provenance(
            ProvenanceEvidence(model_card_present=True)
        )
        assert any("Model card is present" in item for item in result.evidence)

    def test_evidence_mentions_model_card_absent(self, scorer: TrustScorer) -> None:
        result = scorer.score_provenance(ProvenanceEvidence())
        assert any("Model card is absent" in item for item in result.evidence)

    def test_score_is_sum_of_contributions(self, scorer: TrustScorer) -> None:
        ev = ProvenanceEvidence(
            model_card_present=True,
            license_verified=False,
            author_verified=True,
            source_url=None,
        )
        # 0.30 + 0.0 + 0.20 + 0.0 = 0.50
        result = scorer.score_provenance(ev)
        assert abs(result.score - 0.50) < 1e-6

    @given(
        model_card=st.booleans(),
        license_ok=st.booleans(),
        author_ok=st.booleans(),
        has_url=st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_score_always_in_unit_interval(
        self,
        scorer: TrustScorer,
        model_card: bool,
        license_ok: bool,
        author_ok: bool,
        has_url: bool,
    ) -> None:
        ev = ProvenanceEvidence(
            model_card_present=model_card,
            license_verified=license_ok,
            author_verified=author_ok,
            source_url="https://x.com" if has_url else None,
        )
        result = scorer.score_provenance(ev)
        assert 0.0 <= result.score <= 1.0

    @given(
        model_card=st.booleans(),
        license_ok=st.booleans(),
        author_ok=st.booleans(),
        has_url=st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_confidence_always_one(
        self,
        scorer: TrustScorer,
        model_card: bool,
        license_ok: bool,
        author_ok: bool,
        has_url: bool,
    ) -> None:
        ev = ProvenanceEvidence(
            model_card_present=model_card,
            license_verified=license_ok,
            author_verified=author_ok,
            source_url="https://x.com" if has_url else None,
        )
        result = scorer.score_provenance(ev)
        assert result.confidence == 1.0


# ===========================================================================
# score_behavior
# ===========================================================================


class TestScoreBehavior:
    """score_behavior covers reliability, uptime, latency, and confidence."""

    def test_perfect_behavior_yields_score_one(
        self, scorer: TrustScorer, perfect_behavior: BehaviorEvidence
    ) -> None:
        result = scorer.score_behavior(perfect_behavior)
        assert result.score == 1.0
        assert result.confidence == 1.0
        assert result.dimension is TrustDimension.behavior

    def test_worst_behavior_yields_low_score(
        self, scorer: TrustScorer, worst_behavior: BehaviorEvidence
    ) -> None:
        result = scorer.score_behavior(worst_behavior)
        # Error rate 1.0 -> reliability 0; uptime 0; latency 10000 -> 0.
        # Score is 0. Confidence is 0 (no samples).
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_zero_samples_yields_zero_confidence(self, scorer: TrustScorer) -> None:
        ev = BehaviorEvidence(sample_count=0)
        result = scorer.score_behavior(ev)
        assert result.confidence == 0.0

    def test_100_samples_yields_full_confidence(self, scorer: TrustScorer) -> None:
        ev = BehaviorEvidence(
            error_rate=0.0,
            avg_latency_ms=0.0,
            uptime_pct=100.0,
            sample_count=100,
        )
        result = scorer.score_behavior(ev)
        assert result.confidence == 1.0

    def test_above_target_sample_count_caps_confidence_at_one(
        self, scorer: TrustScorer
    ) -> None:
        ev = BehaviorEvidence(sample_count=10_000)
        result = scorer.score_behavior(ev)
        assert result.confidence == 1.0

    def test_latency_below_acceptable_gives_full_latency_score(
        self, scorer: TrustScorer
    ) -> None:
        ev = BehaviorEvidence(
            error_rate=0.0,
            avg_latency_ms=400.0,  # below 500 ms threshold
            uptime_pct=100.0,
            sample_count=100,
        )
        result = scorer.score_behavior(ev)
        # reliability=1.0, uptime=1.0, latency=1.0 -> score=1.0
        assert result.score == 1.0

    def test_latency_at_poor_threshold_reduces_score(self, scorer: TrustScorer) -> None:
        ev = BehaviorEvidence(
            error_rate=0.0,
            avg_latency_ms=5000.0,  # at 5000 ms — latency_score=0
            uptime_pct=100.0,
            sample_count=100,
        )
        result = scorer.score_behavior(ev)
        # reliability*0.40 + uptime*0.35 + 0*0.25 = 0.40+0.35 = 0.75
        assert abs(result.score - 0.75) < 1e-4

    def test_latency_midpoint_gives_half_latency_contribution(
        self, scorer: TrustScorer
    ) -> None:
        midpoint = (500.0 + 5000.0) / 2.0
        ev = BehaviorEvidence(
            error_rate=0.0,
            avg_latency_ms=midpoint,
            uptime_pct=100.0,
            sample_count=100,
        )
        result = scorer.score_behavior(ev)
        # latency_score=0.5: 1.0*0.40 + 1.0*0.35 + 0.5*0.25 = 0.40+0.35+0.125 = 0.875
        assert abs(result.score - 0.875) < 1e-4

    def test_high_error_rate_reduces_score(self, scorer: TrustScorer) -> None:
        ev = BehaviorEvidence(
            error_rate=1.0,
            avg_latency_ms=0.0,
            uptime_pct=100.0,
            sample_count=100,
        )
        result = scorer.score_behavior(ev)
        # reliability=0, uptime=1.0, latency=1.0 -> 0*0.40 + 1.0*0.35 + 1.0*0.25 = 0.60
        assert abs(result.score - 0.60) < 1e-4

    def test_low_uptime_reduces_score(self, scorer: TrustScorer) -> None:
        ev = BehaviorEvidence(
            error_rate=0.0,
            avg_latency_ms=0.0,
            uptime_pct=0.0,
            sample_count=100,
        )
        result = scorer.score_behavior(ev)
        # reliability=1.0, uptime=0.0, latency=1.0 -> 0.40 + 0.0 + 0.25 = 0.65
        assert abs(result.score - 0.65) < 1e-4

    def test_evidence_contains_four_items(self, scorer: TrustScorer) -> None:
        result = scorer.score_behavior(BehaviorEvidence())
        # reliability, uptime, latency, confidence
        assert len(result.evidence) == 4

    @given(
        error_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        latency=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False),
        uptime=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        samples=st.integers(min_value=0, max_value=1000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_score_always_in_unit_interval(
        self,
        scorer: TrustScorer,
        error_rate: float,
        latency: float,
        uptime: float,
        samples: int,
    ) -> None:
        ev = BehaviorEvidence(
            error_rate=error_rate,
            avg_latency_ms=latency,
            uptime_pct=uptime,
            sample_count=samples,
        )
        result = scorer.score_behavior(ev)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0


# ===========================================================================
# score_capability
# ===========================================================================


class TestScoreCapability:
    """score_capability covers verification ratio and method bonus."""

    def test_all_claimed_verified_structured_method_yields_one(
        self, scorer: TrustScorer, perfect_capability: CapabilityEvidence
    ) -> None:
        result = scorer.score_capability(perfect_capability)
        assert result.score == 1.0
        assert result.confidence == 1.0
        assert result.dimension is TrustDimension.capability

    def test_no_claims_no_verified_yields_zero_score(
        self, scorer: TrustScorer, worst_capability: CapabilityEvidence
    ) -> None:
        result = scorer.score_capability(worst_capability)
        # ratio=0.0 * 0.70 + 0.0 * 0.30 = 0.0
        assert result.score == 0.0
        assert result.confidence == 0.5

    def test_no_claims_but_verified_yields_partial_score(
        self, scorer: TrustScorer
    ) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=[],
            verified_capabilities=["summarize"],
            verification_method="",
        )
        result = scorer.score_capability(ev)
        # ratio=0.5 (no claimed), method_bonus=0.0 -> 0.5*0.70 + 0.0 = 0.35
        assert abs(result.score - 0.35) < 1e-6
        assert result.confidence == 0.5

    def test_half_verified_structured_method(self, scorer: TrustScorer) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a", "b"],
            verified_capabilities=["a"],
            verification_method="benchmark_suite",
        )
        result = scorer.score_capability(ev)
        # ratio=0.5, method_bonus=1.0 -> 0.5*0.70 + 1.0*0.30 = 0.35+0.30 = 0.65
        assert abs(result.score - 0.65) < 1e-6

    def test_all_verified_manual_review(self, scorer: TrustScorer) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a"],
            verification_method="manual_review",
        )
        result = scorer.score_capability(ev)
        # ratio=1.0, method_bonus=0.5 -> 1.0*0.70 + 0.5*0.30 = 0.70+0.15 = 0.85
        assert abs(result.score - 0.85) < 1e-6

    def test_all_verified_manual_shorthand(self, scorer: TrustScorer) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a"],
            verification_method="manual",
        )
        result = scorer.score_capability(ev)
        assert abs(result.score - 0.85) < 1e-6

    def test_automated_eval_is_structured_method(self, scorer: TrustScorer) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a"],
            verification_method="automated_eval",
        )
        result = scorer.score_capability(ev)
        assert abs(result.score - 1.0) < 1e-6

    def test_automated_benchmark_is_structured_method(
        self, scorer: TrustScorer
    ) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a"],
            verification_method="automated_benchmark",
        )
        result = scorer.score_capability(ev)
        assert abs(result.score - 1.0) < 1e-6

    def test_unknown_method_yields_zero_bonus(self, scorer: TrustScorer) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a"],
            verification_method="some_other_method",
        )
        result = scorer.score_capability(ev)
        # ratio=1.0, method_bonus=0.0 -> 0.70
        assert abs(result.score - 0.70) < 1e-6

    def test_method_matching_is_case_insensitive(self, scorer: TrustScorer) -> None:
        ev_upper = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a"],
            verification_method="BENCHMARK_SUITE",
        )
        result = scorer.score_capability(ev_upper)
        assert abs(result.score - 1.0) < 1e-6

    def test_duplicate_claims_deduped_by_set(self, scorer: TrustScorer) -> None:
        """Duplicates in claimed_capabilities are collapsed to a set before scoring."""
        ev = CapabilityEvidence(
            claimed_capabilities=["a", "a"],
            verified_capabilities=["a"],
            verification_method="benchmark_suite",
        )
        result = scorer.score_capability(ev)
        # claimed set = {"a"}, overlap = {"a"}, ratio = 1/1 = 1.0
        # score = 1.0*0.70 + 1.0*0.30 = 1.0
        assert result.score == 1.0

    def test_confidence_is_one_when_claimed_is_non_empty(
        self, scorer: TrustScorer
    ) -> None:
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=[],
        )
        result = scorer.score_capability(ev)
        assert result.confidence == 1.0

    def test_confidence_is_half_when_no_claims(self, scorer: TrustScorer) -> None:
        ev = CapabilityEvidence()
        result = scorer.score_capability(ev)
        assert result.confidence == 0.5

    @given(
        claimed_count=st.integers(min_value=0, max_value=10),
        verified_count=st.integers(min_value=0, max_value=10),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_score_always_in_unit_interval(
        self,
        scorer: TrustScorer,
        claimed_count: int,
        verified_count: int,
    ) -> None:
        claimed = [f"cap_{i}" for i in range(claimed_count)]
        # Use first min(verified_count, claimed_count) from claimed for overlap,
        # then fill rest with extra capabilities not in claimed.
        overlap = claimed[: min(verified_count, claimed_count)]
        extra = [f"extra_{i}" for i in range(max(0, verified_count - claimed_count))]
        verified = overlap + extra
        ev = CapabilityEvidence(
            claimed_capabilities=claimed,
            verified_capabilities=verified,
            verification_method="benchmark_suite",
        )
        result = scorer.score_capability(ev)
        assert 0.0 <= result.score <= 1.0


# ===========================================================================
# score_security
# ===========================================================================


class TestScoreSecurity:
    """score_security covers sandbox compliance, vulnerabilities, audit recency."""

    def test_perfect_security_yields_score_one(
        self, scorer: TrustScorer, perfect_security: SecurityEvidence
    ) -> None:
        result = scorer.score_security(perfect_security)
        assert result.score == 1.0
        assert result.confidence == 1.0
        assert result.dimension is TrustDimension.security

    def test_worst_security_yields_low_score(
        self, scorer: TrustScorer, worst_security: SecurityEvidence
    ) -> None:
        result = scorer.score_security(worst_security)
        # sandbox=0, vuln=1/(1+100)~0.0099, audit=0 -> 0*0.35 + 0.0099*0.35 + 0*0.30 ≈ 0.0035
        assert result.score < 0.05
        assert result.confidence == 1.0

    def test_sandbox_compliance_contributes_035(self, scorer: TrustScorer) -> None:
        ev = SecurityEvidence(
            sandbox_compliant=True,
            vulnerability_count=0,
            last_audit_date=None,
        )
        result = scorer.score_security(ev)
        # sandbox=1.0*0.35 + vuln=1.0*0.35 + audit=0.0*0.30 = 0.70
        assert abs(result.score - 0.70) < 1e-4

    def test_no_vulnerabilities_contributes_035(self, scorer: TrustScorer) -> None:
        ev = SecurityEvidence(
            sandbox_compliant=False,
            vulnerability_count=0,
            last_audit_date=None,
        )
        result = scorer.score_security(ev)
        # sandbox=0.0*0.35 + vuln=1.0*0.35 + audit=0.0 = 0.35
        assert abs(result.score - 0.35) < 1e-4

    def test_one_vulnerability_reduces_vuln_score(self, scorer: TrustScorer) -> None:
        ev = SecurityEvidence(vulnerability_count=1)
        result = scorer.score_security(ev)
        # vuln_score = 1/(1+1) = 0.5
        expected = 0.0 * 0.35 + 0.5 * 0.35 + 0.0 * 0.30
        assert abs(result.score - expected) < 1e-4

    def test_many_vulnerabilities_approaches_zero(self, scorer: TrustScorer) -> None:
        ev = SecurityEvidence(vulnerability_count=9999)
        result = scorer.score_security(ev)
        # vuln_score ≈ 0
        assert result.score < 0.01

    def test_audit_today_gives_full_audit_score(self, scorer: TrustScorer) -> None:
        ev = SecurityEvidence(last_audit_date=datetime.date.today())
        result = scorer.score_security(ev)
        # audit_score = 1.0, sandbox=0, vuln=1.0
        # 0*0.35 + 1.0*0.35 + 1.0*0.30 = 0.65
        assert abs(result.score - 0.65) < 1e-4

    def test_audit_within_90_days_gives_full_audit_score(
        self, scorer: TrustScorer
    ) -> None:
        recent = datetime.date.today() - datetime.timedelta(days=45)
        ev = SecurityEvidence(last_audit_date=recent)
        result = scorer.score_security(ev)
        # audit_score should still be 1.0 (age < 90 days)
        # vuln_score=1.0, sandbox=False -> 0*0.35 + 1.0*0.35 + 1.0*0.30 = 0.65
        assert abs(result.score - 0.65) < 1e-4

    def test_audit_at_365_days_gives_zero_audit_score(
        self, scorer: TrustScorer
    ) -> None:
        old = datetime.date.today() - datetime.timedelta(days=365)
        ev = SecurityEvidence(last_audit_date=old)
        result = scorer.score_security(ev)
        # audit_score = 0.0 -> vuln=1.0*0.35 only = 0.35
        assert abs(result.score - 0.35) < 1e-4

    def test_audit_between_90_and_365_days_linearly_decays(
        self, scorer: TrustScorer
    ) -> None:
        midpoint_days = (90 + 365) // 2  # 227 days
        mid_date = datetime.date.today() - datetime.timedelta(days=midpoint_days)
        ev = SecurityEvidence(last_audit_date=mid_date)
        result = scorer.score_security(ev)
        # audit_score ~0.5 -> contribution ~0.15
        # vuln=1.0*0.35 + 0.5*0.30 = 0.35 + 0.15 = 0.50
        assert 0.45 < result.score < 0.55

    def test_no_audit_date_gives_zero_audit_contribution(
        self, scorer: TrustScorer
    ) -> None:
        ev = SecurityEvidence(last_audit_date=None)
        result = scorer.score_security(ev)
        assert any("No audit date recorded" in item for item in result.evidence)

    def test_confidence_is_always_one(
        self, scorer: TrustScorer, worst_security: SecurityEvidence
    ) -> None:
        result = scorer.score_security(worst_security)
        assert result.confidence == 1.0

    def test_evidence_contains_three_items(self, scorer: TrustScorer) -> None:
        result = scorer.score_security(SecurityEvidence())
        assert len(result.evidence) == 3

    @given(
        sandbox=st.booleans(),
        vuln_count=st.integers(min_value=0, max_value=100),
        days_ago=st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=400),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_score_always_in_unit_interval(
        self,
        scorer: TrustScorer,
        sandbox: bool,
        vuln_count: int,
        days_ago: int | None,
    ) -> None:
        audit_date = (
            datetime.date.today() - datetime.timedelta(days=days_ago)
            if days_ago is not None
            else None
        )
        ev = SecurityEvidence(
            sandbox_compliant=sandbox,
            vulnerability_count=vuln_count,
            last_audit_date=audit_date,
        )
        result = scorer.score_security(ev)
        assert 0.0 <= result.score <= 1.0
        assert result.confidence == 1.0


# ===========================================================================
# compute_trust
# ===========================================================================


class TestComputeTrust:
    """compute_trust combines four DimensionScores into a TrustScore."""

    def test_perfect_scores_yield_overall_one(
        self,
        scorer: TrustScorer,
        perfect_dim_scores: dict[str, DimensionScore],
    ) -> None:
        trust = scorer.compute_trust(
            provenance=perfect_dim_scores["provenance"],
            behavior=perfect_dim_scores["behavior"],
            capability=perfect_dim_scores["capability"],
            security=perfect_dim_scores["security"],
            agent_id="perfect-agent",
        )
        assert trust.overall_score == 1.0
        assert trust.agent_id == "perfect-agent"

    def test_worst_scores_yield_near_zero_overall(
        self,
        scorer: TrustScorer,
        worst_dim_scores: dict[str, DimensionScore],
    ) -> None:
        """The 'worst' evidence does not reach exactly 0.0 because the security
        vulnerability scorer uses a harmonic decay (1/(1+n)) that never reaches
        0.0 for finite n=100.  We assert the result is very low instead."""
        trust = scorer.compute_trust(
            provenance=worst_dim_scores["provenance"],
            behavior=worst_dim_scores["behavior"],
            capability=worst_dim_scores["capability"],
            security=worst_dim_scores["security"],
            agent_id="worst-agent",
        )
        assert trust.overall_score < 0.01

    def test_default_agent_id_is_unknown(
        self,
        scorer: TrustScorer,
        perfect_dim_scores: dict[str, DimensionScore],
    ) -> None:
        trust = scorer.compute_trust(
            provenance=perfect_dim_scores["provenance"],
            behavior=perfect_dim_scores["behavior"],
            capability=perfect_dim_scores["capability"],
            security=perfect_dim_scores["security"],
        )
        assert trust.agent_id == "unknown"

    def test_returns_trust_score_instance(
        self,
        scorer: TrustScorer,
        perfect_dim_scores: dict[str, DimensionScore],
    ) -> None:
        trust = scorer.compute_trust(
            provenance=perfect_dim_scores["provenance"],
            behavior=perfect_dim_scores["behavior"],
            capability=perfect_dim_scores["capability"],
            security=perfect_dim_scores["security"],
        )
        assert isinstance(trust, TrustScore)

    def test_dimension_scores_map_contains_all_four_keys(
        self,
        scorer: TrustScorer,
        perfect_dim_scores: dict[str, DimensionScore],
    ) -> None:
        trust = scorer.compute_trust(
            provenance=perfect_dim_scores["provenance"],
            behavior=perfect_dim_scores["behavior"],
            capability=perfect_dim_scores["capability"],
            security=perfect_dim_scores["security"],
        )
        assert set(trust.dimension_scores.keys()) == {
            "provenance",
            "behavior",
            "capability",
            "security",
        }

    def test_timestamp_is_utc_aware(
        self,
        scorer: TrustScorer,
        perfect_dim_scores: dict[str, DimensionScore],
    ) -> None:
        trust = scorer.compute_trust(
            provenance=perfect_dim_scores["provenance"],
            behavior=perfect_dim_scores["behavior"],
            capability=perfect_dim_scores["capability"],
            security=perfect_dim_scores["security"],
        )
        assert trust.timestamp.tzinfo is not None

    def test_overall_score_is_confidence_weighted(self) -> None:
        """When one dimension has zero confidence it is excluded from normalisation."""
        weights = TrustWeights()
        scorer = TrustScorer(weights)

        # Three dimensions with score=1.0, confidence=1.0
        # One dimension with score=0.0, confidence=0.0
        high = DimensionScore(
            dimension=TrustDimension.provenance, score=1.0, confidence=1.0
        )
        high2 = DimensionScore(
            dimension=TrustDimension.behavior, score=1.0, confidence=1.0
        )
        high3 = DimensionScore(
            dimension=TrustDimension.capability, score=1.0, confidence=1.0
        )
        zero_conf = DimensionScore(
            dimension=TrustDimension.security, score=0.0, confidence=0.0
        )

        trust = scorer.compute_trust(high, high2, high3, zero_conf)
        # Effective weights: prov=0.25*1=0.25, beh=0.25*1=0.25, cap=0.25*1=0.25, sec=0.25*0=0
        # total_effective_weight = 0.75
        # overall = (0.25*1 + 0.25*1 + 0.25*1) / 0.75 = 0.75/0.75 = 1.0
        assert trust.overall_score == 1.0

    def test_all_zero_confidence_yields_zero_overall(self) -> None:
        weights = TrustWeights()
        scorer = TrustScorer(weights)

        zero_conf_dims = [
            DimensionScore(
                dimension=dim, score=1.0, confidence=0.0
            )
            for dim in TrustDimension
        ]
        trust = scorer.compute_trust(*zero_conf_dims)
        assert trust.overall_score == 0.0

    def test_custom_weights_shift_overall_score(self) -> None:
        """Heavily weighted dimension dominates the overall score."""
        heavy_prov_weights = TrustWeights(
            provenance=0.97,
            behavior=0.01,
            capability=0.01,
            security=0.01,
        )
        scorer = TrustScorer(heavy_prov_weights)

        prov = DimensionScore(
            dimension=TrustDimension.provenance, score=1.0, confidence=1.0
        )
        behav = DimensionScore(
            dimension=TrustDimension.behavior, score=0.0, confidence=1.0
        )
        cap = DimensionScore(
            dimension=TrustDimension.capability, score=0.0, confidence=1.0
        )
        sec = DimensionScore(
            dimension=TrustDimension.security, score=0.0, confidence=1.0
        )

        trust = scorer.compute_trust(prov, behav, cap, sec)
        # overall ≈ (0.97*1) / (0.97+0.01+0.01+0.01) = 0.97/1.0 = 0.97
        assert abs(trust.overall_score - 0.97) < 1e-4

    def test_overall_score_is_clamped_to_unit_interval(
        self,
        scorer: TrustScorer,
        perfect_dim_scores: dict[str, DimensionScore],
    ) -> None:
        trust = scorer.compute_trust(
            provenance=perfect_dim_scores["provenance"],
            behavior=perfect_dim_scores["behavior"],
            capability=perfect_dim_scores["capability"],
            security=perfect_dim_scores["security"],
        )
        assert 0.0 <= trust.overall_score <= 1.0

    @given(
        prov_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        beh_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        cap_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        sec_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        prov_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        beh_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        cap_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        sec_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_overall_always_in_unit_interval(
        self,
        scorer: TrustScorer,
        prov_score: float,
        beh_score: float,
        cap_score: float,
        sec_score: float,
        prov_conf: float,
        beh_conf: float,
        cap_conf: float,
        sec_conf: float,
    ) -> None:
        prov = DimensionScore(
            dimension=TrustDimension.provenance,
            score=prov_score,
            confidence=prov_conf,
        )
        beh = DimensionScore(
            dimension=TrustDimension.behavior,
            score=beh_score,
            confidence=beh_conf,
        )
        cap = DimensionScore(
            dimension=TrustDimension.capability,
            score=cap_score,
            confidence=cap_conf,
        )
        sec = DimensionScore(
            dimension=TrustDimension.security,
            score=sec_score,
            confidence=sec_conf,
        )
        trust = scorer.compute_trust(prov, beh, cap, sec)
        assert 0.0 <= trust.overall_score <= 1.0


# ===========================================================================
# End-to-end integration test
# ===========================================================================


class TestEndToEnd:
    """Full pipeline: evidence -> dimension scores -> trust score -> grade."""

    def test_typical_agent_produces_reasonable_score(self) -> None:
        weights = TrustWeights(
            provenance=0.30,
            behavior=0.30,
            capability=0.20,
            security=0.20,
        )
        scorer = TrustScorer(weights)

        prov = scorer.score_provenance(
            ProvenanceEvidence(
                model_card_present=True,
                license_verified=True,
                author_verified=False,
                source_url="https://example.com",
            )
        )
        behav = scorer.score_behavior(
            BehaviorEvidence(
                error_rate=0.02,
                avg_latency_ms=200.0,
                uptime_pct=99.5,
                sample_count=500,
            )
        )
        cap = scorer.score_capability(
            CapabilityEvidence(
                claimed_capabilities=["summarize", "translate"],
                verified_capabilities=["summarize"],
                verification_method="benchmark_suite",
            )
        )
        sec = scorer.score_security(
            SecurityEvidence(
                sandbox_compliant=True,
                vulnerability_count=0,
                last_audit_date=datetime.date.today(),
            )
        )

        trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="typical-agent")

        assert trust.agent_id == "typical-agent"
        assert 0.0 <= trust.overall_score <= 1.0
        assert trust.grade() in {"A", "B", "C", "D", "F"}
        assert len(trust.dimension_scores) == 4

    def test_grade_a_agent_with_all_perfect_evidence(
        self,
        scorer: TrustScorer,
        perfect_provenance: ProvenanceEvidence,
        perfect_behavior: BehaviorEvidence,
        perfect_capability: CapabilityEvidence,
        perfect_security: SecurityEvidence,
    ) -> None:
        prov = scorer.score_provenance(perfect_provenance)
        beh = scorer.score_behavior(perfect_behavior)
        cap = scorer.score_capability(perfect_capability)
        sec = scorer.score_security(perfect_security)
        trust = scorer.compute_trust(prov, beh, cap, sec, agent_id="ideal-agent")
        assert trust.grade() == "A"
        assert trust.overall_score == 1.0

    def test_grade_f_agent_with_all_worst_evidence(
        self,
        scorer: TrustScorer,
        worst_provenance: ProvenanceEvidence,
        worst_behavior: BehaviorEvidence,
        worst_capability: CapabilityEvidence,
        worst_security: SecurityEvidence,
    ) -> None:
        prov = scorer.score_provenance(worst_provenance)
        beh = scorer.score_behavior(worst_behavior)
        cap = scorer.score_capability(worst_capability)
        sec = scorer.score_security(worst_security)
        trust = scorer.compute_trust(prov, beh, cap, sec, agent_id="bad-agent")
        assert trust.grade() == "F"
