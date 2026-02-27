"""Trust scoring logic for aumai-trustforge."""

from __future__ import annotations

import datetime
import math

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

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Behavior: latency above this threshold (ms) starts reducing the score.
_LATENCY_ACCEPTABLE_MS: float = 500.0
_LATENCY_POOR_MS: float = 5000.0

# Behavior: minimum sample count for full confidence.
_MIN_SAMPLES_FOR_FULL_CONFIDENCE: int = 100

# Security: audit age (days) beyond which the score decays.
_AUDIT_ACCEPTABLE_DAYS: int = 90
_AUDIT_MAX_DAYS: int = 365


class TrustScorer:
    """Compute trust scores for AI agents across four dimensions.

    Each scoring method accepts a typed evidence object and returns a
    :class:`~aumai_trustforge.models.DimensionScore` with a real numeric score,
    a calibrated confidence value, and human-readable evidence items explaining
    the score.

    :meth:`compute_trust` combines the four dimension scores using the supplied
    :class:`~aumai_trustforge.models.TrustWeights`.

    Example::

        weights = TrustWeights(provenance=0.30, behavior=0.30,
                               capability=0.20, security=0.20)
        scorer = TrustScorer(weights)

        prov = scorer.score_provenance(ProvenanceEvidence(
            model_card_present=True, license_verified=True,
            author_verified=False, source_url="https://example.com/card"))

        behav = scorer.score_behavior(BehaviorEvidence(
            error_rate=0.02, avg_latency_ms=200, uptime_pct=99.5, sample_count=500))

        cap = scorer.score_capability(CapabilityEvidence(
            claimed_capabilities=["summarize", "translate"],
            verified_capabilities=["summarize"],
            verification_method="benchmark_suite"))

        sec = scorer.score_security(SecurityEvidence(
            sandbox_compliant=True, vulnerability_count=0,
            last_audit_date=datetime.date.today()))

        trust = scorer.compute_trust(prov, behav, cap, sec)
        print(trust.overall_score, trust.grade())
    """

    def __init__(self, weights: TrustWeights) -> None:
        self._weights = weights

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def score_provenance(self, evidence: ProvenanceEvidence) -> DimensionScore:
        """Score the provenance dimension.

        Scoring formula (weights sum to 1.0):
        - model_card_present : 0.30
        - license_verified   : 0.30
        - author_verified    : 0.20
        - source_url present : 0.20

        Confidence is 1.0 because all signals are boolean (no sampling
        uncertainty).
        """
        evidence_items: list[str] = []

        model_card_contribution = 0.30 if evidence.model_card_present else 0.0
        if evidence.model_card_present:
            evidence_items.append("Model card is present (+0.30)")
        else:
            evidence_items.append("Model card is absent (-0.30)")

        license_contribution = 0.30 if evidence.license_verified else 0.0
        if evidence.license_verified:
            evidence_items.append("License has been verified (+0.30)")
        else:
            evidence_items.append("License not verified (-0.30)")

        author_contribution = 0.20 if evidence.author_verified else 0.0
        if evidence.author_verified:
            evidence_items.append("Author identity verified (+0.20)")
        else:
            evidence_items.append("Author identity unverified (-0.20)")

        source_url_contribution = 0.20 if evidence.source_url else 0.0
        if evidence.source_url:
            evidence_items.append(f"Source URL present: {evidence.source_url} (+0.20)")
        else:
            evidence_items.append("No source URL provided (-0.20)")

        raw_score = (
            model_card_contribution
            + license_contribution
            + author_contribution
            + source_url_contribution
        )
        score = _clamp(raw_score)

        return DimensionScore(
            dimension=TrustDimension.provenance,
            score=round(score, 4),
            confidence=1.0,
            evidence=evidence_items,
        )

    def score_behavior(self, evidence: BehaviorEvidence) -> DimensionScore:
        """Score the behavior dimension from operational metrics.

        Sub-scores (each in [0, 1]):

        - **reliability** (weight 0.40): ``1 - error_rate``
        - **uptime** (weight 0.35): ``uptime_pct / 100``
        - **latency** (weight 0.25): decays linearly from 1.0 at
          ``<= 500 ms`` to 0.0 at ``>= 5000 ms``

        Confidence scales with sample count — full confidence at >= 100 samples.
        """
        evidence_items: list[str] = []

        # Reliability sub-score.
        reliability = _clamp(1.0 - evidence.error_rate)
        evidence_items.append(
            f"Error rate {evidence.error_rate:.1%} "
            f"-> reliability={reliability:.3f} (weight 0.40)"
        )

        # Uptime sub-score.
        uptime_score = _clamp(evidence.uptime_pct / 100.0)
        evidence_items.append(
            f"Uptime {evidence.uptime_pct:.2f}% "
            f"-> uptime_score={uptime_score:.3f} (weight 0.35)"
        )

        # Latency sub-score: linear decay from good (500 ms) to poor (5000 ms).
        latency_score = _linear_decay(
            value=evidence.avg_latency_ms,
            low=_LATENCY_ACCEPTABLE_MS,
            high=_LATENCY_POOR_MS,
        )
        evidence_items.append(
            f"Avg latency {evidence.avg_latency_ms:.0f} ms "
            f"-> latency_score={latency_score:.3f} (weight 0.25)"
        )

        raw_score = (
            reliability * 0.40
            + uptime_score * 0.35
            + latency_score * 0.25
        )
        score = _clamp(raw_score)

        # Confidence: log-scaled with sample count, capped at 1.0.
        confidence = _sample_confidence(
            evidence.sample_count, _MIN_SAMPLES_FOR_FULL_CONFIDENCE
        )
        evidence_items.append(
            f"Sample count {evidence.sample_count} -> confidence={confidence:.3f}"
        )

        return DimensionScore(
            dimension=TrustDimension.behavior,
            score=round(score, 4),
            confidence=round(confidence, 4),
            evidence=evidence_items,
        )

    def score_capability(self, evidence: CapabilityEvidence) -> DimensionScore:
        """Score the capability dimension.

        Scoring formula:

        - **verification ratio** (weight 0.70): ``len(verified) / len(claimed)``
          when claimed > 0; 0.0 when claimed == 0 and verified == 0.
        - **verification method bonus** (weight 0.30): 1.0 for structured
          methods (benchmark_suite, automated_eval), 0.5 for manual_review,
          0.0 for unknown/empty.

        Confidence is 1.0 when there are claimed capabilities to compare
        against, 0.5 otherwise (no claim context).
        """
        evidence_items: list[str] = []

        claimed = set(evidence.claimed_capabilities)
        verified = set(evidence.verified_capabilities)

        if claimed:
            overlap = claimed & verified
            verification_ratio = len(overlap) / len(claimed)
            evidence_items.append(
                f"{len(overlap)}/{len(claimed)} claimed capabilities verified "
                f"-> ratio={verification_ratio:.3f} (weight 0.70)"
            )
            confidence = 1.0
        else:
            verification_ratio = 0.0 if not verified else 0.5
            evidence_items.append(
                "No claimed capabilities declared "
                f"-> ratio={verification_ratio:.3f} (weight 0.70)"
            )
            confidence = 0.5

        # Verification method bonus.
        method = evidence.verification_method.lower().strip()
        structured_methods = {
            "benchmark_suite",
            "automated_eval",
            "automated_benchmark",
        }
        partial_methods = {"manual_review", "manual"}
        if method in structured_methods:
            method_bonus = 1.0
            evidence_items.append(
                f"Structured verification method '{evidence.verification_method}' "
                f"-> bonus=1.0 (weight 0.30)"
            )
        elif method in partial_methods:
            method_bonus = 0.5
            evidence_items.append(
                f"Manual review method '{evidence.verification_method}' "
                f"-> bonus=0.5 (weight 0.30)"
            )
        else:
            method_bonus = 0.0
            evidence_items.append(
                f"Unknown/missing verification method '{evidence.verification_method}' "
                f"-> bonus=0.0 (weight 0.30)"
            )

        raw_score = verification_ratio * 0.70 + method_bonus * 0.30
        score = _clamp(raw_score)

        return DimensionScore(
            dimension=TrustDimension.capability,
            score=round(score, 4),
            confidence=round(confidence, 4),
            evidence=evidence_items,
        )

    def score_security(self, evidence: SecurityEvidence) -> DimensionScore:
        """Score the security dimension.

        Sub-scores:

        - **sandbox_compliant** (weight 0.35): 1.0 if True, 0.0 otherwise.
        - **vulnerability penalty** (weight 0.35): ``1 / (1 + vulnerability_count)``
          — decays toward 0 as vulnerability count grows.
        - **audit recency** (weight 0.30): 1.0 within 90 days, linear decay
          to 0.0 at 365 days; 0.0 if no audit date recorded.

        Confidence is 1.0 because all inputs are directly measured facts.
        """
        evidence_items: list[str] = []

        # Sandbox compliance.
        sandbox_score = 1.0 if evidence.sandbox_compliant else 0.0
        evidence_items.append(
            f"Sandbox compliant: {evidence.sandbox_compliant} "
            f"-> score={sandbox_score:.1f} (weight 0.35)"
        )

        # Vulnerability score: harmonic decay.
        vuln_score = 1.0 / (1.0 + evidence.vulnerability_count)
        evidence_items.append(
            f"{evidence.vulnerability_count} vulnerability/vulnerabilities "
            f"-> vuln_score={vuln_score:.3f} (weight 0.35)"
        )

        # Audit recency.
        if evidence.last_audit_date is not None:
            today = datetime.date.today()
            age_days = (today - evidence.last_audit_date).days
            audit_score = _linear_decay(
                value=float(age_days),
                low=float(_AUDIT_ACCEPTABLE_DAYS),
                high=float(_AUDIT_MAX_DAYS),
            )
            evidence_items.append(
                f"Last audit {age_days} days ago "
                f"-> audit_score={audit_score:.3f} (weight 0.30)"
            )
        else:
            audit_score = 0.0
            evidence_items.append(
                "No audit date recorded -> audit_score=0.0 (weight 0.30)"
            )

        raw_score = sandbox_score * 0.35 + vuln_score * 0.35 + audit_score * 0.30
        score = _clamp(raw_score)

        return DimensionScore(
            dimension=TrustDimension.security,
            score=round(score, 4),
            confidence=1.0,
            evidence=evidence_items,
        )

    # ------------------------------------------------------------------
    # Overall trust computation
    # ------------------------------------------------------------------

    def compute_trust(
        self,
        provenance: DimensionScore,
        behavior: DimensionScore,
        capability: DimensionScore,
        security: DimensionScore,
        agent_id: str = "unknown",
    ) -> TrustScore:
        """Combine four dimension scores into a single :class:`TrustScore`.

        The overall score is the confidence-weighted sum of dimension scores,
        then re-weighted by the configured
        :class:`~aumai_trustforge.models.TrustWeights`.

        Each dimension's contribution = ``weight * dimension_score * confidence``,
        normalized by the total effective weight (sum of ``weight * confidence``
        across all dimensions) so the output remains in [0, 1].

        Args:
            provenance: Scored provenance dimension.
            behavior: Scored behavior dimension.
            capability: Scored capability dimension.
            security: Scored security dimension.
            agent_id: Identifier for the agent being scored.

        Returns:
            A :class:`TrustScore` with overall score, per-dimension breakdown,
            and timestamp.
        """
        dimensions = [provenance, behavior, capability, security]

        weighted_score_sum = 0.0
        total_effective_weight = 0.0

        for dim_score in dimensions:
            w = self._weights.for_dimension(dim_score.dimension)
            effective_weight = w * dim_score.confidence
            weighted_score_sum += effective_weight * dim_score.score
            total_effective_weight += effective_weight

        if total_effective_weight == 0.0:
            overall = 0.0
        else:
            overall = weighted_score_sum / total_effective_weight

        overall = _clamp(overall)

        dimension_scores_map = {dim.dimension.value: dim for dim in dimensions}

        return TrustScore(
            agent_id=agent_id,
            overall_score=round(overall, 4),
            dimension_scores=dimension_scores_map,
            timestamp=datetime.datetime.now(datetime.UTC),
        )


# ---------------------------------------------------------------------------
# Private math helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* to [*low*, *high*]."""
    return max(low, min(high, value))


def _linear_decay(value: float, low: float, high: float) -> float:
    """Return a score that is 1.0 below *low* and 0.0 above *high*.

    Linearly interpolates in between.  The decay is inverted — higher values
    produce lower scores (used for latency and vulnerability age where larger
    is worse).
    """
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return 1.0 - (value - low) / (high - low)


def _sample_confidence(sample_count: int, target: int) -> float:
    """Return a confidence value that increases with sample count.

    Uses a logarithmic scale so confidence climbs quickly for the first few
    samples and plateaus toward 1.0 near *target*.

    Returns:
        Value in [0.0, 1.0].
    """
    if sample_count <= 0:
        return 0.0
    if sample_count >= target:
        return 1.0
    # log-scale: confidence = log(1 + sample_count) / log(1 + target)
    return _clamp(math.log(1.0 + sample_count) / math.log(1.0 + target))


__all__ = ["TrustScorer"]
