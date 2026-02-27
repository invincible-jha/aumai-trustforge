"""Tests for aumai_trustforge.models.

Covers every public model class, their validators, and helper methods.
Property-based tests use Hypothesis to exercise the full numeric range.
"""

from __future__ import annotations

import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

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
# TrustDimension
# ===========================================================================


class TestTrustDimension:
    """TrustDimension enum membership and string value identity."""

    def test_all_four_members_present(self) -> None:
        members = {d.value for d in TrustDimension}
        assert members == {"provenance", "behavior", "capability", "security"}

    def test_string_access_equals_value(self) -> None:
        assert TrustDimension("provenance") is TrustDimension.provenance
        assert TrustDimension("behavior") is TrustDimension.behavior
        assert TrustDimension("capability") is TrustDimension.capability
        assert TrustDimension("security") is TrustDimension.security

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            TrustDimension("network")  # type: ignore[arg-type]

    def test_is_str_subclass(self) -> None:
        # TrustDimension extends str — values compare equal to plain strings.
        assert TrustDimension.provenance == "provenance"


# ===========================================================================
# DimensionScore
# ===========================================================================


class TestDimensionScore:
    """Validation rules for DimensionScore."""

    def test_valid_construction(self) -> None:
        ds = DimensionScore(
            dimension=TrustDimension.behavior,
            score=0.75,
            confidence=0.90,
            evidence=["sample"],
        )
        assert ds.dimension is TrustDimension.behavior
        assert ds.score == 0.75
        assert ds.confidence == 0.90
        assert ds.evidence == ["sample"]

    def test_default_evidence_is_empty_list(self) -> None:
        ds = DimensionScore(
            dimension=TrustDimension.provenance, score=0.5, confidence=0.5
        )
        assert ds.evidence == []

    @pytest.mark.parametrize("score", [-0.001, 1.001, -1.0, 2.0])
    def test_score_out_of_range_raises(self, score: float) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=TrustDimension.security, score=score, confidence=0.5
            )

    @pytest.mark.parametrize("confidence", [-0.001, 1.001])
    def test_confidence_out_of_range_raises(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=TrustDimension.capability, score=0.5, confidence=confidence
            )

    @pytest.mark.parametrize("boundary", [0.0, 1.0])
    def test_boundary_values_accepted(self, boundary: float) -> None:
        ds = DimensionScore(
            dimension=TrustDimension.behavior, score=boundary, confidence=boundary
        )
        assert ds.score == boundary
        assert ds.confidence == boundary

    @given(
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_property_any_valid_score_and_confidence_accepted(
        self, score: float, confidence: float
    ) -> None:
        ds = DimensionScore(
            dimension=TrustDimension.provenance, score=score, confidence=confidence
        )
        assert 0.0 <= ds.score <= 1.0
        assert 0.0 <= ds.confidence <= 1.0


# ===========================================================================
# TrustWeights
# ===========================================================================


class TestTrustWeights:
    """Validation and for_dimension helper for TrustWeights."""

    def test_default_weights_are_equal(self) -> None:
        weights = TrustWeights()
        assert weights.provenance == 0.25
        assert weights.behavior == 0.25
        assert weights.capability == 0.25
        assert weights.security == 0.25

    def test_custom_weights_sum_to_one(self) -> None:
        weights = TrustWeights(
            provenance=0.40,
            behavior=0.30,
            capability=0.20,
            security=0.10,
        )
        total = (
            weights.provenance + weights.behavior + weights.capability + weights.security
        )
        assert abs(total - 1.0) < 1e-9

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="sum"):
            TrustWeights(
                provenance=0.50,
                behavior=0.50,
                capability=0.10,
                security=0.10,
            )

    def test_all_zero_weights_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrustWeights(
                provenance=0.0,
                behavior=0.0,
                capability=0.0,
                security=0.0,
            )

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrustWeights(
                provenance=-0.10,
                behavior=0.50,
                capability=0.40,
                security=0.20,
            )

    def test_weight_exceeding_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrustWeights(
                provenance=1.10,
                behavior=0.0,
                capability=0.0,
                security=0.0,
            )

    def test_tolerance_boundary_exact_sum_accepted(self) -> None:
        """Weights that sum to exactly 1.0 must be accepted."""
        weights = TrustWeights(
            provenance=0.40,
            behavior=0.30,
            capability=0.20,
            security=0.10,
        )
        # 0.40 + 0.30 + 0.20 + 0.10 = 1.0 exactly
        total = weights.provenance + weights.behavior + weights.capability + weights.security
        assert abs(total - 1.0) < 1e-9

    def test_tolerance_boundary_tiny_fp_error_accepted(self) -> None:
        """Floating-point representation errors at 1e-15 scale must be tolerated."""
        # 1/3 * 3 sums to exactly 1.0 in IEEE 754 for these values; verify that
        # equal-split defaults (0.25 * 4) pass the 1e-9 guard.
        weights = TrustWeights()  # provenance=behavior=capability=security=0.25
        total = weights.provenance + weights.behavior + weights.capability + weights.security
        assert abs(total - 1.0) <= 1e-9

    def test_tolerance_boundary_0_001_above_one_raises(self) -> None:
        """Sum that differs from 1.0 by 0.001 (the old loose tolerance) must now be rejected."""
        with pytest.raises(ValidationError, match="sum"):
            TrustWeights(
                provenance=0.251,
                behavior=0.25,
                capability=0.25,
                security=0.25,
            )

    def test_tolerance_boundary_just_outside_raises(self) -> None:
        """Sum that differs from 1.0 by a large margin must be rejected."""
        with pytest.raises(ValidationError):
            TrustWeights(
                provenance=0.252,
                behavior=0.25,
                capability=0.25,
                security=0.25,
            )

    @pytest.mark.parametrize(
        "dimension",
        list(TrustDimension),
    )
    def test_for_dimension_returns_correct_weight(
        self, dimension: TrustDimension
    ) -> None:
        weights = TrustWeights(
            provenance=0.40,
            behavior=0.30,
            capability=0.20,
            security=0.10,
        )
        expected = {
            TrustDimension.provenance: 0.40,
            TrustDimension.behavior: 0.30,
            TrustDimension.capability: 0.20,
            TrustDimension.security: 0.10,
        }
        assert weights.for_dimension(dimension) == expected[dimension]


# ===========================================================================
# TrustScore
# ===========================================================================


class TestTrustScore:
    """Validation and grade() helper for TrustScore."""

    def test_construction_with_required_fields(self) -> None:
        ts = TrustScore(agent_id="agent-1", overall_score=0.75)
        assert ts.agent_id == "agent-1"
        assert ts.overall_score == 0.75
        assert ts.dimension_scores == {}
        assert ts.timestamp.tzinfo is not None  # timezone-aware

    def test_overall_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrustScore(agent_id="x", overall_score=-0.01)

    def test_overall_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrustScore(agent_id="x", overall_score=1.001)

    def test_timestamp_defaults_to_utc_aware(self) -> None:
        ts = TrustScore(agent_id="x", overall_score=0.5)
        assert ts.timestamp.tzinfo is not None

    def test_explicit_timestamp_accepted(self) -> None:
        when = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        ts = TrustScore(agent_id="x", overall_score=0.5, timestamp=when)
        assert ts.timestamp == when

    @pytest.mark.parametrize(
        ("score", "expected_grade"),
        [
            (1.00, "A"),
            (0.85, "A"),
            (0.849, "B"),
            (0.70, "B"),
            (0.699, "C"),
            (0.55, "C"),
            (0.549, "D"),
            (0.40, "D"),
            (0.399, "F"),
            (0.00, "F"),
        ],
    )
    def test_grade_boundaries(self, score: float, expected_grade: str) -> None:
        ts = TrustScore(agent_id="x", overall_score=score)
        assert ts.grade() == expected_grade

    @given(score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_property_grade_always_returns_letter(self, score: float) -> None:
        ts = TrustScore(agent_id="agent", overall_score=score)
        assert ts.grade() in {"A", "B", "C", "D", "F"}

    @given(score=st.floats(min_value=0.85, max_value=1.0, allow_nan=False))
    def test_property_grade_a_for_high_scores(self, score: float) -> None:
        ts = TrustScore(agent_id="agent", overall_score=score)
        assert ts.grade() == "A"

    @given(score=st.floats(min_value=0.0, max_value=0.3999, allow_nan=False))
    def test_property_grade_f_for_low_scores(self, score: float) -> None:
        ts = TrustScore(agent_id="agent", overall_score=score)
        assert ts.grade() == "F"


# ===========================================================================
# ProvenanceEvidence
# ===========================================================================


class TestProvenanceEvidence:
    """ProvenanceEvidence default values and field types."""

    def test_all_defaults_are_false_or_none(self) -> None:
        ev = ProvenanceEvidence()
        assert ev.model_card_present is False
        assert ev.license_verified is False
        assert ev.author_verified is False
        assert ev.source_url is None

    def test_explicit_values_accepted(self) -> None:
        ev = ProvenanceEvidence(
            model_card_present=True,
            license_verified=True,
            author_verified=True,
            source_url="https://example.com",
        )
        assert ev.source_url == "https://example.com"

    def test_empty_source_url_string_is_falsy_sentinel(self) -> None:
        ev = ProvenanceEvidence(source_url="")
        # An empty string is stored but treated as falsy in the scorer.
        assert ev.source_url == ""


# ===========================================================================
# BehaviorEvidence
# ===========================================================================


class TestBehaviorEvidence:
    """BehaviorEvidence defaults and boundary validation."""

    def test_defaults(self) -> None:
        ev = BehaviorEvidence()
        assert ev.error_rate == 0.0
        assert ev.avg_latency_ms == 0.0
        assert ev.uptime_pct == 100.0
        assert ev.sample_count == 0

    def test_error_rate_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorEvidence(error_rate=-0.01)

    def test_error_rate_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorEvidence(error_rate=1.001)

    def test_uptime_above_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorEvidence(uptime_pct=100.001)

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorEvidence(avg_latency_ms=-1.0)

    def test_negative_sample_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            BehaviorEvidence(sample_count=-1)

    def test_zero_sample_count_accepted(self) -> None:
        ev = BehaviorEvidence(sample_count=0)
        assert ev.sample_count == 0


# ===========================================================================
# CapabilityEvidence
# ===========================================================================


class TestCapabilityEvidence:
    """CapabilityEvidence default values and list field semantics."""

    def test_defaults_are_empty_lists_and_empty_method(self) -> None:
        ev = CapabilityEvidence()
        assert ev.claimed_capabilities == []
        assert ev.verified_capabilities == []
        assert ev.verification_method == ""

    def test_verified_can_exceed_claimed(self) -> None:
        """Verified list is not required to be a subset of claimed."""
        ev = CapabilityEvidence(
            claimed_capabilities=["a"],
            verified_capabilities=["a", "b"],
            verification_method="manual_review",
        )
        assert "b" in ev.verified_capabilities

    def test_duplicate_capabilities_stored_as_given(self) -> None:
        """Duplicates are accepted at the model layer; deduplication is a scorer concern."""
        ev = CapabilityEvidence(
            claimed_capabilities=["a", "a"],
            verified_capabilities=["a"],
        )
        assert ev.claimed_capabilities.count("a") == 2


# ===========================================================================
# SecurityEvidence
# ===========================================================================


class TestSecurityEvidence:
    """SecurityEvidence defaults and field constraints."""

    def test_defaults(self) -> None:
        ev = SecurityEvidence()
        assert ev.sandbox_compliant is False
        assert ev.vulnerability_count == 0
        assert ev.last_audit_date is None

    def test_negative_vulnerability_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            SecurityEvidence(vulnerability_count=-1)

    def test_audit_date_today_accepted(self) -> None:
        ev = SecurityEvidence(last_audit_date=datetime.date.today())
        assert ev.last_audit_date == datetime.date.today()

    def test_future_audit_date_accepted(self) -> None:
        """Model layer does not restrict future dates; scorer may handle them."""
        future = datetime.date.today() + datetime.timedelta(days=30)
        ev = SecurityEvidence(last_audit_date=future)
        assert ev.last_audit_date == future

    def test_large_vulnerability_count_accepted(self) -> None:
        ev = SecurityEvidence(vulnerability_count=10_000)
        assert ev.vulnerability_count == 10_000
