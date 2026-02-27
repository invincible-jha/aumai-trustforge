"""Pydantic models for aumai-trustforge."""

from __future__ import annotations

import datetime
import enum

from pydantic import BaseModel, Field, field_validator, model_validator


class TrustDimension(str, enum.Enum):
    """The four independent trust dimensions scored by TrustScorer."""

    provenance = "provenance"
    behavior = "behavior"
    capability = "capability"
    security = "security"


class DimensionScore(BaseModel):
    """A scored result for one trust dimension."""

    dimension: TrustDimension
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score in [0, 1]")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level of the score in [0, 1]"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Human-readable evidence items supporting the score",
    )


class TrustWeights(BaseModel):
    """Per-dimension weights used to compute the overall trust score.

    All four weights must be non-negative and must sum to exactly 1.0.

    Example::

        weights = TrustWeights(
            provenance=0.30,
            behavior=0.30,
            capability=0.20,
            security=0.20,
        )
    """

    provenance: float = Field(default=0.25, ge=0.0, le=1.0)
    behavior: float = Field(default=0.25, ge=0.0, le=1.0)
    capability: float = Field(default=0.25, ge=0.0, le=1.0)
    security: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> TrustWeights:
        """Reject weight configurations that don't sum to 1.0 (±1e-9 tolerance).

        The tolerance of 1e-9 accommodates floating-point representation errors
        (e.g. 0.1 + 0.2 != 0.3 in IEEE 754) while still rejecting weight sets
        like (0.2503, 0.2503, 0.2503, 0.2503) that would allow the overall score
        to exceed 1.0 before clamping when all dimension scores are 1.0.
        """
        total = self.provenance + self.behavior + self.capability + self.security
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"TrustWeights must sum to 1.0, but sum is {total:.10f}. "
                "Adjust weights so provenance + behavior + capability + security = 1.0."
            )
        return self

    def for_dimension(self, dimension: TrustDimension) -> float:
        """Return the weight for *dimension*."""
        mapping: dict[TrustDimension, float] = {
            TrustDimension.provenance: self.provenance,
            TrustDimension.behavior: self.behavior,
            TrustDimension.capability: self.capability,
            TrustDimension.security: self.security,
        }
        return mapping[dimension]


class TrustScore(BaseModel):
    """Aggregated trust score for one agent at one point in time."""

    agent_id: str = Field(..., description="Unique identifier of the scored agent")
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted overall score in [0, 1]"
    )
    dimension_scores: dict[str, DimensionScore] = Field(
        default_factory=dict,
        description="Dimension name -> DimensionScore mapping",
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    def grade(self) -> str:
        """Return a letter grade.

        A (>=0.85), B (>=0.70), C (>=0.55), D (>=0.40), F (<0.40).
        """
        if self.overall_score >= 0.85:
            return "A"
        if self.overall_score >= 0.70:
            return "B"
        if self.overall_score >= 0.55:
            return "C"
        if self.overall_score >= 0.40:
            return "D"
        return "F"

    def __repr__(self) -> str:
        """Return a compact, human-readable representation.

        Example::

            TrustScore(agent_id='agent-42', overall_score=0.8750, grade='A')
        """
        return (
            f"TrustScore(agent_id={self.agent_id!r}, "
            f"overall_score={self.overall_score:.4f}, "
            f"grade={self.grade()!r})"
        )


# ---------------------------------------------------------------------------
# Evidence models — one per dimension
# ---------------------------------------------------------------------------


class ProvenanceEvidence(BaseModel):
    """Evidence for the provenance dimension.

    Captures whether the agent has a published model card, an approved
    license, a verified author, and a public source URL.
    """

    model_card_present: bool = Field(default=False)
    license_verified: bool = Field(default=False)
    author_verified: bool = Field(default=False)
    source_url: str | None = Field(
        default=None, description="URL to agent source/model card"
    )


class BehaviorEvidence(BaseModel):
    """Evidence for the behavior dimension.

    Derived from operational metrics over a representative sample.
    """

    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of requests that resulted in an error",
    )
    avg_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average response latency in milliseconds",
    )
    uptime_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Availability percentage over the measurement window",
    )
    sample_count: int = Field(
        default=0,
        ge=0,
        description="Number of samples the metrics are based on (affects confidence)",
    )


class CapabilityEvidence(BaseModel):
    """Evidence for the capability dimension.

    Compares what the agent claims to be able to do against what has been
    independently verified.
    """

    claimed_capabilities: list[str] = Field(default_factory=list)
    verified_capabilities: list[str] = Field(default_factory=list)
    verification_method: str = Field(
        default="",
        description=(
            "How capabilities were verified, e.g. 'benchmark_suite', 'manual_review'"
        ),
    )

    @field_validator("verified_capabilities")
    @classmethod
    def verified_must_be_subset_of_claimed(cls, verified: list[str]) -> list[str]:
        """Warn-only: verified may include items not in claimed (cross-validation)."""
        return verified


class SecurityEvidence(BaseModel):
    """Evidence for the security dimension."""

    sandbox_compliant: bool = Field(
        default=False,
        description="Agent has been validated against a sandbox capability declaration",
    )
    vulnerability_count: int = Field(
        default=0,
        ge=0,
        description="Number of known unmitigated vulnerabilities",
    )
    last_audit_date: datetime.date | None = Field(
        default=None,
        description="Date of the most recent security audit",
    )


__all__ = [
    "BehaviorEvidence",
    "CapabilityEvidence",
    "DimensionScore",
    "ProvenanceEvidence",
    "SecurityEvidence",
    "TrustDimension",
    "TrustScore",
    "TrustWeights",
]
