# Getting Started with aumai-trustforge

This guide walks you from a fresh environment to a working trust score in under
10 minutes. It then covers the most common usage patterns and how to troubleshoot
problems when they arise.

---

## Prerequisites

- Python 3.11 or later.
- `pip` (bundled with Python).
- Optional: a virtual environment tool (`venv`, `pyenv`, `conda`).

No external services, API keys, or model downloads are required.

---

## Installation

### From PyPI (recommended)

```bash
pip install aumai-trustforge
```

Verify:

```bash
aumai-trustforge --version
python -c "import aumai_trustforge; print(aumai_trustforge.__version__)"
```

### From source

```bash
git clone https://github.com/aumai/aumai-trustforge
cd aumai-trustforge
pip install -e .
```

### Development mode (with test dependencies)

```bash
pip install -e ".[dev]"
make test   # should show all tests passing
```

---

## Your First Trust Score

This step-by-step tutorial scores a hypothetical summarization agent.

### Step 1: Understand what evidence you need

`aumai-trustforge` requires you to supply evidence for up to four dimensions.
Each dimension has a Pydantic model with sensible defaults — you only need to
fill in what you know.

```
ProvenanceEvidence — Is the agent documented and its origin verifiable?
BehaviorEvidence   — How has the agent behaved in production?
CapabilityEvidence — Do its claimed skills match what was verified?
SecurityEvidence   — Is it sandboxed, patched, and audited?
```

### Step 2: Create evidence files (CLI path)

```bash
mkdir my-agent-evidence

# Provenance: we have a model card and verified license/author
cat > my-agent-evidence/provenance.json << 'EOF'
{
  "model_card_present": true,
  "license_verified": true,
  "author_verified": true,
  "source_url": "https://huggingface.co/my-org/summarizer-v2"
}
EOF

# Behavior: metrics from last 30 days of production traffic
cat > my-agent-evidence/behavior.json << 'EOF'
{
  "error_rate": 0.02,
  "avg_latency_ms": 450.0,
  "uptime_pct": 99.5,
  "sample_count": 12000
}
EOF

# Capability: 3 claimed, 2 verified, using a benchmark suite
cat > my-agent-evidence/capability.json << 'EOF'
{
  "claimed_capabilities": ["summarize", "translate", "sentiment-analysis"],
  "verified_capabilities": ["summarize", "translate"],
  "verification_method": "benchmark_suite"
}
EOF

# Security: no vulnerabilities, audited last month
cat > my-agent-evidence/security.json << 'EOF'
{
  "sandbox_compliant": true,
  "vulnerability_count": 0,
  "last_audit_date": "2025-12-01"
}
EOF
```

### Step 3: Run the scorer

```bash
aumai-trustforge score --agent-dir my-agent-evidence
```

You will see output like:

```
══════════════════════════════════════════════════════════════
  AumAI TrustForge — Agent Trust Report
══════════════════════════════════════════════════════════════
  Agent ID   : my-agent-evidence
  Timestamp  : 2026-02-27T10:00:00+00:00
  Overall    : 0.8542  (Grade: A)
──────────────────────────────────────────────────────────────
  Dimension Scores:

  PROVENANCE    0.8000  conf=1.00  [################----]
              * Model card is present (+0.30)
              * License has been verified (+0.30)
              * Author identity verified (+0.20)
              * Source URL present: https://huggingface.co/my-org/summarizer-v2 (+0.20)

  BEHAVIOR      0.9080  conf=1.00  [##################--]
              * Error rate 2.0% -> reliability=0.980 (weight 0.40)
              * Uptime 99.50% -> uptime_score=0.995 (weight 0.35)
              * Avg latency 450 ms -> latency_score=0.971 (weight 0.25)
              * Sample count 12000 -> confidence=1.000
  ...
══════════════════════════════════════════════════════════════
```

### Step 4: Save and reuse the score

```bash
# Save as JSON
aumai-trustforge score --agent-dir my-agent-evidence --output json > trust.json

# Re-render later without re-running evidence collection
aumai-trustforge report --input trust.json
```

### Step 5: Use custom weights

Create a `weights.json` file:

```json
{
  "provenance": 0.35,
  "behavior": 0.35,
  "capability": 0.15,
  "security": 0.15
}
```

```bash
aumai-trustforge score --agent-dir my-agent-evidence --weights weights.json
```

---

## Common Patterns

### Pattern 1: Scoring from Python code

The most common use case — integrate trust scoring into your agent's deployment
pipeline or monitoring system.

```python
import datetime
from aumai_trustforge import (
    TrustScorer, TrustWeights,
    ProvenanceEvidence, BehaviorEvidence,
    CapabilityEvidence, SecurityEvidence,
)

def score_agent(
    agent_id: str,
    metrics: dict,
    audit_date: datetime.date,
) -> float:
    """Return the overall trust score for an agent given live metrics."""
    scorer = TrustScorer(TrustWeights(
        provenance=0.30, behavior=0.30,
        capability=0.20, security=0.20,
    ))

    prov = scorer.score_provenance(ProvenanceEvidence(
        model_card_present=metrics.get("has_model_card", False),
        license_verified=metrics.get("license_ok", False),
        author_verified=metrics.get("author_verified", False),
        source_url=metrics.get("source_url"),
    ))

    behav = scorer.score_behavior(BehaviorEvidence(
        error_rate=metrics["error_rate"],
        avg_latency_ms=metrics["avg_latency_ms"],
        uptime_pct=metrics["uptime_pct"],
        sample_count=metrics["sample_count"],
    ))

    cap = scorer.score_capability(CapabilityEvidence(
        claimed_capabilities=metrics.get("claimed_caps", []),
        verified_capabilities=metrics.get("verified_caps", []),
        verification_method=metrics.get("verification_method", ""),
    ))

    sec = scorer.score_security(SecurityEvidence(
        sandbox_compliant=metrics.get("sandbox_ok", False),
        vulnerability_count=metrics.get("vuln_count", 0),
        last_audit_date=audit_date,
    ))

    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id=agent_id)
    return trust.overall_score
```

### Pattern 2: Gating deployment on trust score

```python
from aumai_trustforge import TrustScorer, TrustWeights

MIN_TRUST_FOR_PRODUCTION = 0.75

def can_promote_to_production(agent_id: str, evidence: dict) -> bool:
    scorer = TrustScorer(TrustWeights())
    # ... build evidence objects from `evidence` dict ...
    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id=agent_id)

    if trust.overall_score < MIN_TRUST_FOR_PRODUCTION:
        print(f"Agent {agent_id} blocked: score {trust.overall_score:.4f} "
              f"< threshold {MIN_TRUST_FOR_PRODUCTION}")
        return False
    return True
```

### Pattern 3: Tracking trust over time

```python
import json
import datetime
from pathlib import Path
from aumai_trustforge import TrustScorer, TrustWeights

def score_and_archive(agent_id: str, evidence_dir: str, archive_dir: str) -> None:
    """Score an agent and archive the result with a datestamp."""
    # ... build scorer and evidence ...
    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id=agent_id)

    today = datetime.date.today().isoformat()
    archive_path = Path(archive_dir) / f"{agent_id}_{today}.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        json.dumps(trust.model_dump(mode="json"), default=str, indent=2)
    )
    print(f"Archived: {archive_path}")
```

### Pattern 4: Comparing two scoring configurations

```python
from aumai_trustforge import TrustScorer, TrustWeights

# Score with two different weight configurations
configs = {
    "compliance": TrustWeights(provenance=0.40, behavior=0.20,
                               capability=0.20, security=0.20),
    "operations": TrustWeights(provenance=0.15, behavior=0.50,
                               capability=0.15, security=0.20),
}

for name, weights in configs.items():
    scorer = TrustScorer(weights)
    trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="my-agent")
    print(f"[{name}] score={trust.overall_score:.4f}  grade={trust.grade()}")
```

### Pattern 5: Inspecting confidence values

```python
from aumai_trustforge import TrustScorer, TrustWeights, BehaviorEvidence

scorer = TrustScorer(TrustWeights())

# Low sample count — confidence is reduced
low_sample = scorer.score_behavior(BehaviorEvidence(
    error_rate=0.01, avg_latency_ms=200, uptime_pct=99.9, sample_count=5
))
print(f"confidence with 5 samples: {low_sample.confidence:.3f}")   # ~0.483

# High sample count — full confidence
high_sample = scorer.score_behavior(BehaviorEvidence(
    error_rate=0.01, avg_latency_ms=200, uptime_pct=99.9, sample_count=500
))
print(f"confidence with 500 samples: {high_sample.confidence:.3f}")  # 1.000
```

---

## Troubleshooting FAQ

**Q: I get `ValueError: TrustWeights must sum to 1.0`.**

Your weights do not sum exactly to 1.0. Check for typos:

```python
# Wrong
TrustWeights(provenance=0.30, behavior=0.30, capability=0.20, security=0.21)

# Correct — sums to exactly 1.0
TrustWeights(provenance=0.30, behavior=0.30, capability=0.20, security=0.20)
```

The validator allows a tolerance of 1e-9 for floating-point rounding, but
values that differ by more than that will be rejected.

---

**Q: My behavior score is lower than expected even though my metrics are good.**

Check `sample_count`. The behavior confidence is logarithmically scaled against
a target of 100 samples. With only 5 samples the confidence is ~0.48, meaning
the behavior dimension has roughly half its configured weight in the final score.

---

**Q: The CLI says "missing files are treated as empty evidence." What does that mean?**

If `provenance.json` is absent from the `--agent-dir`, a `ProvenanceEvidence`
with all defaults is used: `model_card_present=False`, `license_verified=False`,
`author_verified=False`, `source_url=None`. This yields a provenance score of
0.0. Create the file (even with some `false` values) to get the correct score.

---

**Q: Can I pass evidence programmatically without JSON files?**

Yes. The JSON files are only used by the CLI. When using the Python API, you
construct `ProvenanceEvidence`, `BehaviorEvidence`, `CapabilityEvidence`, and
`SecurityEvidence` directly in code. The CLI is a convenience wrapper around
the same Python API.

---

**Q: My `last_audit_date` is in the past — will the score decay over time?**

Yes, intentionally. The security dimension uses a linear decay from 1.0 (audit
within 90 days) to 0.0 (audit older than 365 days). If you run the same
evidence files 6 months later, the security score will be lower. This reflects
the real-world reality that security posture degrades without regular audits.

---

**Q: How do I handle an agent with no security audit on record?**

Set `last_audit_date=None` (or omit the field in JSON). The audit recency
sub-score will be 0.0, which reduces the overall security dimension score but
does not cause an error. This is the correct signal: an unaudited agent is
less trustworthy from a security standpoint.

---

**Q: Can I extend the scoring with custom dimensions?**

Not in this library — it is scope-reduced to the four standard dimensions.
To add custom dimensions, subclass `TrustScorer` and override `compute_trust`,
or build a wrapper that combines the standard `TrustScore` with your own
scoring logic before presenting a unified result.
