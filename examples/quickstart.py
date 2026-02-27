"""Quickstart examples for aumai-trustforge.

This script demonstrates the four core use cases of aumai-trustforge:
scoring each dimension individually, computing an overall trust score,
working with custom weights, and interpreting the results.

Run directly to verify your installation:

    python examples/quickstart.py

All examples use fictional agent data and require no external services.
"""

from __future__ import annotations

import datetime
import json

from aumai_trustforge import (
    BehaviorEvidence,
    CapabilityEvidence,
    DimensionScore,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustScorer,
    TrustWeights,
)


# ---------------------------------------------------------------------------
# Demo 1: Score a well-trusted agent
# ---------------------------------------------------------------------------


def demo_well_trusted_agent() -> None:
    """Score a production-ready agent with strong evidence across all dimensions.

    This represents the ideal case: fully documented, low error rate, all
    capabilities verified, recently audited, and sandboxed.
    """
    print("\n" + "=" * 60)
    print("Demo 1: Well-trusted agent (expected grade: A)")
    print("=" * 60)

    # Configure weights — compliance-focused deployment
    weights = TrustWeights(
        provenance=0.30,
        behavior=0.30,
        capability=0.20,
        security=0.20,
    )
    scorer = TrustScorer(weights)

    # Provenance: full documentation, verified author, public source
    prov = scorer.score_provenance(ProvenanceEvidence(
        model_card_present=True,
        license_verified=True,
        author_verified=True,
        source_url="https://huggingface.co/my-org/summarizer-v2",
    ))
    print(f"\nProvenance score : {prov.score:.4f}  "
          f"(confidence={prov.confidence:.2f})")
    for item in prov.evidence:
        print(f"  * {item}")

    # Behavior: excellent metrics from 10,000 real requests
    behav = scorer.score_behavior(BehaviorEvidence(
        error_rate=0.005,       # 0.5% error rate
        avg_latency_ms=280.0,   # well under the 500 ms acceptable threshold
        uptime_pct=99.95,       # very high uptime
        sample_count=10_000,    # large sample — full confidence
    ))
    print(f"\nBehavior score   : {behav.score:.4f}  "
          f"(confidence={behav.confidence:.2f})")

    # Capability: all claimed capabilities verified by a benchmark suite
    cap = scorer.score_capability(CapabilityEvidence(
        claimed_capabilities=["summarize", "translate", "classify"],
        verified_capabilities=["summarize", "translate", "classify"],
        verification_method="benchmark_suite",
    ))
    print(f"\nCapability score : {cap.score:.4f}  "
          f"(confidence={cap.confidence:.2f})")

    # Security: sandboxed, no known vulnerabilities, audited recently
    sec = scorer.score_security(SecurityEvidence(
        sandbox_compliant=True,
        vulnerability_count=0,
        last_audit_date=datetime.date.today() - datetime.timedelta(days=15),
    ))
    print(f"\nSecurity score   : {sec.score:.4f}  "
          f"(confidence={sec.confidence:.2f})")

    # Combine all four dimensions into a single TrustScore
    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="summarizer-v2")

    print(f"\nOverall score    : {trust.overall_score:.4f}")
    print(f"Grade            : {trust.grade()}")
    print(f"Repr             : {trust!r}")


# ---------------------------------------------------------------------------
# Demo 2: Score a partially-trusted agent
# ---------------------------------------------------------------------------


def demo_partially_trusted_agent() -> None:
    """Score an agent with mixed evidence — good behavior, weak provenance.

    This represents a common real-world case: an internal agent with great
    operational metrics but incomplete documentation and no formal audit.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Partially trusted agent (expected grade: B or C)")
    print("=" * 60)

    scorer = TrustScorer(TrustWeights())  # equal weights: 0.25 each

    # Provenance: no model card, no verified author, no source URL
    prov = scorer.score_provenance(ProvenanceEvidence(
        model_card_present=False,
        license_verified=True,   # license was checked
        author_verified=False,
        source_url=None,
    ))

    # Behavior: good metrics but from a small sample (low confidence)
    behav = scorer.score_behavior(BehaviorEvidence(
        error_rate=0.03,
        avg_latency_ms=600.0,    # slightly above the 500 ms acceptable threshold
        uptime_pct=98.0,
        sample_count=25,         # only 25 samples — reduced confidence
    ))
    print(f"\nBehavior confidence at 25 samples: {behav.confidence:.3f}")

    # Capability: only manual review, 1 of 3 capabilities unverified
    cap = scorer.score_capability(CapabilityEvidence(
        claimed_capabilities=["summarize", "translate", "classify"],
        verified_capabilities=["summarize", "translate"],
        verification_method="manual_review",   # partial bonus of 0.5
    ))

    # Security: no formal audit, 2 known vulnerabilities
    sec = scorer.score_security(SecurityEvidence(
        sandbox_compliant=True,
        vulnerability_count=2,
        last_audit_date=None,    # never audited — audit score = 0.0
    ))

    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="internal-classifier")

    print(f"\nProvenance score : {prov.score:.4f}")
    print(f"Behavior score   : {behav.score:.4f}  (conf={behav.confidence:.3f})")
    print(f"Capability score : {cap.score:.4f}")
    print(f"Security score   : {sec.score:.4f}")
    print(f"\nOverall score    : {trust.overall_score:.4f}")
    print(f"Grade            : {trust.grade()}")


# ---------------------------------------------------------------------------
# Demo 3: Compare two weight configurations
# ---------------------------------------------------------------------------


def demo_weight_comparison() -> None:
    """Show how different weight configurations change the overall score.

    The same agent evidence yields different overall scores depending on
    whether you prioritize compliance (provenance/security) vs. operations
    (behavior).
    """
    print("\n" + "=" * 60)
    print("Demo 3: Weight configuration comparison")
    print("=" * 60)

    # Evidence with strong behavior/security but weak provenance/capability
    prov_ev = ProvenanceEvidence(
        model_card_present=False,
        license_verified=True,
        author_verified=False,
        source_url=None,
    )
    behav_ev = BehaviorEvidence(
        error_rate=0.01,
        avg_latency_ms=300.0,
        uptime_pct=99.9,
        sample_count=5000,
    )
    cap_ev = CapabilityEvidence(
        claimed_capabilities=["summarize"],
        verified_capabilities=[],         # nothing verified yet
        verification_method="",
    )
    sec_ev = SecurityEvidence(
        sandbox_compliant=True,
        vulnerability_count=0,
        last_audit_date=datetime.date.today() - datetime.timedelta(days=30),
    )

    weight_configs: dict[str, TrustWeights] = {
        "compliance (prov+sec heavy)": TrustWeights(
            provenance=0.40, behavior=0.20, capability=0.20, security=0.20
        ),
        "operations  (behavior heavy)": TrustWeights(
            provenance=0.15, behavior=0.55, capability=0.15, security=0.15
        ),
        "balanced    (equal weights) ": TrustWeights(),
    }

    print()
    for config_name, weights in weight_configs.items():
        scorer = TrustScorer(weights)
        prov = scorer.score_provenance(prov_ev)
        behav = scorer.score_behavior(behav_ev)
        cap = scorer.score_capability(cap_ev)
        sec = scorer.score_security(sec_ev)
        trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="test-agent")
        print(f"  [{config_name}]  "
              f"score={trust.overall_score:.4f}  grade={trust.grade()}")


# ---------------------------------------------------------------------------
# Demo 4: Serialize and deserialize a TrustScore
# ---------------------------------------------------------------------------


def demo_serialization() -> None:
    """Show how to persist a TrustScore to JSON and reload it.

    TrustScore is a Pydantic v2 model, so serialization is built-in.
    This is useful for archiving scores, comparing over time, or sending
    scores to a monitoring dashboard.
    """
    print("\n" + "=" * 60)
    print("Demo 4: JSON serialization / deserialization")
    print("=" * 60)

    scorer = TrustScorer(TrustWeights())
    prov = scorer.score_provenance(ProvenanceEvidence(
        model_card_present=True, license_verified=True,
        author_verified=True, source_url="https://example.com/agent",
    ))
    behav = scorer.score_behavior(BehaviorEvidence(
        error_rate=0.01, avg_latency_ms=200, uptime_pct=99.9, sample_count=500,
    ))
    cap = scorer.score_capability(CapabilityEvidence(
        claimed_capabilities=["summarize"],
        verified_capabilities=["summarize"],
        verification_method="automated_eval",
    ))
    sec = scorer.score_security(SecurityEvidence(
        sandbox_compliant=True, vulnerability_count=0,
        last_audit_date=datetime.date(2025, 12, 1),
    ))
    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="serialization-demo")

    # Serialize to a JSON string
    trust_dict = trust.model_dump(mode="json")
    trust_json = json.dumps(trust_dict, default=str, indent=2)

    print(f"\nSerialized (first 300 chars):\n{trust_json[:300]}...")

    # Deserialize back from JSON
    from aumai_trustforge import TrustScore
    reloaded = TrustScore.model_validate(json.loads(trust_json))

    print(f"\nReloaded score : {reloaded.overall_score:.4f}")
    print(f"Grade matches  : {reloaded.grade() == trust.grade()}")
    print(f"ID matches     : {reloaded.agent_id == trust.agent_id}")


# ---------------------------------------------------------------------------
# Demo 5: Understand confidence scaling
# ---------------------------------------------------------------------------


def demo_confidence_scaling() -> None:
    """Show how sample_count affects behavior confidence and final score.

    With very few samples, the behavior dimension's influence on the overall
    score is automatically reduced — even if the behavior metrics themselves
    look excellent.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Confidence scaling (behavior dimension)")
    print("=" * 60)

    scorer = TrustScorer(TrustWeights())

    # Same excellent metrics, different sample counts
    sample_counts = [1, 5, 10, 25, 50, 100, 500]
    excellent_behavior = dict(error_rate=0.001, avg_latency_ms=100.0, uptime_pct=99.99)

    print(f"\n{'Samples':>8}  {'Confidence':>12}  {'Behavior score':>15}")
    print("-" * 40)

    for count in sample_counts:
        ev = BehaviorEvidence(**excellent_behavior, sample_count=count)
        score: DimensionScore = scorer.score_behavior(ev)
        print(f"{count:>8}  {score.confidence:>12.4f}  {score.score:>15.4f}")

    print("\nNote: confidence affects how much the behavior dimension")
    print("contributes to the overall trust score.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quickstart demos in sequence."""
    print("aumai-trustforge quickstart examples")
    print("=" * 60)

    demo_well_trusted_agent()
    demo_partially_trusted_agent()
    demo_weight_comparison()
    demo_serialization()
    demo_confidence_scaling()

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("See docs/getting-started.md for step-by-step tutorials.")
    print("See docs/api-reference.md for full API documentation.")


if __name__ == "__main__":
    main()
