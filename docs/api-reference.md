# API Reference — aumai-trustforge

All public symbols are importable from the top-level package:

```python
from aumai_trustforge import (
    TrustScorer, TrustWeights,
    TrustScore, TrustDimension, DimensionScore,
    ProvenanceEvidence, BehaviorEvidence,
    CapabilityEvidence, SecurityEvidence,
)
```

---

## `aumai_trustforge.core`

### `TrustScorer`

The central scoring engine. Accepts a `TrustWeights` configuration and exposes
one scoring method per dimension plus a final aggregation method.

```python
class TrustScorer:
    def __init__(self, weights: TrustWeights) -> None: ...
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `weights` | `TrustWeights` | Per-dimension weights used in `compute_trust`. Must sum to 1.0. |

**Example:**
```python
from aumai_trustforge import TrustScorer, TrustWeights

scorer = TrustScorer(TrustWeights(
    provenance=0.30, behavior=0.30,
    capability=0.20, security=0.20,
))
```

---

#### `TrustScorer.score_provenance`

```python
def score_provenance(self, evidence: ProvenanceEvidence) -> DimensionScore
```

Score the provenance dimension from four boolean signals.

**Scoring formula:**

| Signal | Weight |
|---|---|
| `model_card_present` | 0.30 |
| `license_verified` | 0.30 |
| `author_verified` | 0.20 |
| `source_url is not None` | 0.20 |

Confidence is always 1.0 — boolean inputs carry no sampling uncertainty.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `evidence` | `ProvenanceEvidence` | Provenance signals for the agent. |

**Returns:** `DimensionScore` with `dimension=TrustDimension.provenance`,
`score` in [0, 1], `confidence=1.0`, and human-readable `evidence` items.

**Example:**
```python
from aumai_trustforge import ProvenanceEvidence

score = scorer.score_provenance(ProvenanceEvidence(
    model_card_present=True,
    license_verified=True,
    author_verified=False,
    source_url="https://huggingface.co/my-org/my-agent",
))
# score.score == 0.80 (card=0.30 + license=0.30 + url=0.20)
# score.confidence == 1.0
```

---

#### `TrustScorer.score_behavior`

```python
def score_behavior(self, evidence: BehaviorEvidence) -> DimensionScore
```

Score the behavior dimension from operational metrics.

**Scoring formula (sub-scores):**

| Sub-score | Formula | Weight |
|---|---|---|
| reliability | `clamp(1.0 - error_rate)` | 0.40 |
| uptime | `clamp(uptime_pct / 100.0)` | 0.35 |
| latency | `linear_decay(avg_latency_ms, low=500, high=5000)` | 0.25 |

**Confidence formula:**

```
confidence = log(1 + sample_count) / log(1 + 100)
```

Plateaus at 1.0 when `sample_count >= 100`. Returns 0.0 when `sample_count == 0`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `evidence` | `BehaviorEvidence` | Operational metric snapshot for the agent. |

**Returns:** `DimensionScore` with `dimension=TrustDimension.behavior`,
`score` in [0, 1], `confidence` in [0, 1], and human-readable `evidence` items.

**Example:**
```python
from aumai_trustforge import BehaviorEvidence

score = scorer.score_behavior(BehaviorEvidence(
    error_rate=0.02,
    avg_latency_ms=400.0,
    uptime_pct=99.5,
    sample_count=1000,
))
# score.score ≈ 0.9480
# score.confidence == 1.0  (1000 >= 100)
```

---

#### `TrustScorer.score_capability`

```python
def score_capability(self, evidence: CapabilityEvidence) -> DimensionScore
```

Score the capability dimension by comparing claimed and verified capabilities,
plus the quality of the verification method.

**Scoring formula:**

| Component | Formula | Weight |
|---|---|---|
| verification ratio | `len(claimed ∩ verified) / len(claimed)` | 0.70 |
| method bonus | 1.0 for structured, 0.5 for manual, 0.0 for unknown | 0.30 |

**Confidence:**

- 1.0 when `claimed_capabilities` is non-empty (full comparison possible).
- 0.5 when `claimed_capabilities` is empty (no basis for comparison).

**Structured verification methods** (full bonus = 1.0):
`benchmark_suite`, `automated_eval`, `automated_benchmark`

**Manual review methods** (partial bonus = 0.5):
`manual_review`, `manual`

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `evidence` | `CapabilityEvidence` | Claimed and verified capabilities with method. |

**Returns:** `DimensionScore` with `dimension=TrustDimension.capability`.

**Example:**
```python
from aumai_trustforge import CapabilityEvidence

score = scorer.score_capability(CapabilityEvidence(
    claimed_capabilities=["summarize", "translate", "classify"],
    verified_capabilities=["summarize", "translate"],
    verification_method="benchmark_suite",
))
# ratio = 2/3 ≈ 0.667, bonus = 1.0
# score.score = 0.667*0.70 + 1.0*0.30 = 0.7669
# score.confidence = 1.0
```

---

#### `TrustScorer.score_security`

```python
def score_security(self, evidence: SecurityEvidence) -> DimensionScore
```

Score the security dimension from sandbox compliance, vulnerability count,
and security audit recency.

**Scoring formula:**

| Sub-score | Formula | Weight |
|---|---|---|
| sandbox | `1.0 if sandbox_compliant else 0.0` | 0.35 |
| vulnerabilities | `1 / (1 + vulnerability_count)` | 0.35 |
| audit recency | `linear_decay(age_days, low=90, high=365)` | 0.30 |

Confidence is always 1.0 — all inputs are directly measured facts.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `evidence` | `SecurityEvidence` | Security posture evidence for the agent. |

**Returns:** `DimensionScore` with `dimension=TrustDimension.security`,
`confidence=1.0`.

**Example:**
```python
import datetime
from aumai_trustforge import SecurityEvidence

score = scorer.score_security(SecurityEvidence(
    sandbox_compliant=True,
    vulnerability_count=2,
    last_audit_date=datetime.date(2025, 11, 1),
))
# sandbox_score = 1.0
# vuln_score    = 1 / (1 + 2) = 0.333
# audit_score   = depends on days since 2025-11-01
```

---

#### `TrustScorer.compute_trust`

```python
def compute_trust(
    self,
    provenance: DimensionScore,
    behavior: DimensionScore,
    capability: DimensionScore,
    security: DimensionScore,
    agent_id: str = "unknown",
) -> TrustScore
```

Combine four dimension scores into a single overall `TrustScore` using
confidence-weighted aggregation.

**Aggregation formula:**

```
effective_weight(dim) = configured_weight(dim) * confidence(dim)
overall = sum(effective_weight * score) / sum(effective_weight)
```

This normalization keeps the result in [0, 1] regardless of confidence values
and automatically reduces the influence of low-confidence dimensions.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `provenance` | `DimensionScore` | Scored provenance dimension. |
| `behavior` | `DimensionScore` | Scored behavior dimension. |
| `capability` | `DimensionScore` | Scored capability dimension. |
| `security` | `DimensionScore` | Scored security dimension. |
| `agent_id` | `str` | Identifier for the agent being scored. Default: `"unknown"`. |

**Returns:** `TrustScore` with `overall_score` in [0, 1], `grade()`, a
`dimension_scores` mapping, and a UTC `timestamp`.

**Example:**
```python
trust = scorer.compute_trust(prov, behav, cap, sec, agent_id="prod-agent-v3")
print(trust.overall_score)              # e.g. 0.8542
print(trust.grade())                    # "A"
print(trust.dimension_scores["behavior"].score)  # e.g. 0.9480
```

---

## `aumai_trustforge.models`

### `TrustDimension`

```python
class TrustDimension(str, enum.Enum):
    provenance = "provenance"
    behavior   = "behavior"
    capability = "capability"
    security   = "security"
```

Enumeration of the four trust dimensions. Used as the `dimension` field on
`DimensionScore`.

---

### `DimensionScore`

```python
class DimensionScore(BaseModel):
    dimension: TrustDimension
    score:     float  # ge=0.0, le=1.0
    confidence: float  # ge=0.0, le=1.0
    evidence:  list[str]  # default_factory=list
```

Scored result for one trust dimension.

**Fields**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `dimension` | `TrustDimension` | — | Which dimension this score represents. |
| `score` | `float` | [0.0, 1.0] | Normalized score. 1.0 is the best possible. |
| `confidence` | `float` | [0.0, 1.0] | How much to trust this score. Affects aggregation weight. |
| `evidence` | `list[str]` | — | Human-readable explanation items. |

**Example:**
```python
from aumai_trustforge import TrustDimension, DimensionScore

ds = DimensionScore(
    dimension=TrustDimension.behavior,
    score=0.9480,
    confidence=1.0,
    evidence=["Error rate 2.0% -> reliability=0.980 (weight 0.40)"],
)
```

---

### `TrustWeights`

```python
class TrustWeights(BaseModel):
    provenance: float  # default=0.25, ge=0.0, le=1.0
    behavior:   float  # default=0.25
    capability: float  # default=0.25
    security:   float  # default=0.25
```

Per-dimension weights for computing the overall trust score. All four values
must sum to exactly 1.0 (validated with 1e-9 tolerance).

**Fields**

| Field | Type | Default | Constraints |
|---|---|---|---|
| `provenance` | `float` | 0.25 | [0.0, 1.0] |
| `behavior` | `float` | 0.25 | [0.0, 1.0] |
| `capability` | `float` | 0.25 | [0.0, 1.0] |
| `security` | `float` | 0.25 | [0.0, 1.0] |

**Raises:** `ValidationError` if `provenance + behavior + capability + security != 1.0 ± 1e-9`.

**Methods**

##### `for_dimension(dimension: TrustDimension) -> float`

Return the weight for the given dimension.

```python
weights = TrustWeights(provenance=0.30, behavior=0.30, capability=0.20, security=0.20)
weights.for_dimension(TrustDimension.behavior)  # 0.30
```

---

### `TrustScore`

```python
class TrustScore(BaseModel):
    agent_id:         str
    overall_score:    float           # ge=0.0, le=1.0
    dimension_scores: dict[str, DimensionScore]  # default_factory=dict
    timestamp:        datetime.datetime
```

Aggregated trust score for one agent at one point in time. Produced by
`TrustScorer.compute_trust`.

**Fields**

| Field | Type | Description |
|---|---|---|
| `agent_id` | `str` | Unique identifier of the scored agent. |
| `overall_score` | `float` | Weighted overall score in [0, 1]. |
| `dimension_scores` | `dict[str, DimensionScore]` | Keyed by dimension name (`"provenance"`, `"behavior"`, etc.). |
| `timestamp` | `datetime.datetime` | UTC timestamp of when the score was computed. |

**Methods**

##### `grade() -> str`

Return a letter grade based on `overall_score`:

| Grade | Score range |
|---|---|
| A | >= 0.85 |
| B | >= 0.70 |
| C | >= 0.55 |
| D | >= 0.40 |
| F | < 0.40 |

```python
trust.grade()  # "A", "B", "C", "D", or "F"
```

##### `__repr__() -> str`

Returns a compact representation:

```python
repr(trust)
# "TrustScore(agent_id='my-agent', overall_score=0.8542, grade='A')"
```

---

### `ProvenanceEvidence`

```python
class ProvenanceEvidence(BaseModel):
    model_card_present: bool        # default=False
    license_verified:   bool        # default=False
    author_verified:    bool        # default=False
    source_url:         str | None  # default=None
```

Evidence for the provenance dimension. All fields default to the most
conservative (lowest-scoring) values so that missing information is penalized.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `model_card_present` | `bool` | `False` | True if the agent has a published model card. |
| `license_verified` | `bool` | `False` | True if the license has been reviewed and approved. |
| `author_verified` | `bool` | `False` | True if the author's identity has been confirmed. |
| `source_url` | `str \| None` | `None` | URL to the agent source or model card. Non-None earns 0.20. |

---

### `BehaviorEvidence`

```python
class BehaviorEvidence(BaseModel):
    error_rate:      float  # default=0.0, ge=0.0, le=1.0
    avg_latency_ms:  float  # default=0.0, ge=0.0
    uptime_pct:      float  # default=100.0, ge=0.0, le=100.0
    sample_count:    int    # default=0, ge=0
```

Evidence for the behavior dimension derived from operational metrics.

**Fields**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `error_rate` | `float` | 0.0 | [0.0, 1.0] | Fraction of requests that resulted in an error (0.01 = 1%). |
| `avg_latency_ms` | `float` | 0.0 | >= 0.0 | Average response latency in milliseconds. |
| `uptime_pct` | `float` | 100.0 | [0.0, 100.0] | Availability percentage over the measurement window. |
| `sample_count` | `int` | 0 | >= 0 | Number of samples used to derive the metrics. Affects confidence. |

---

### `CapabilityEvidence`

```python
class CapabilityEvidence(BaseModel):
    claimed_capabilities:  list[str]  # default_factory=list
    verified_capabilities: list[str]  # default_factory=list
    verification_method:   str        # default=""
```

Evidence for the capability dimension. Compares self-reported capabilities
against independently verified ones.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `claimed_capabilities` | `list[str]` | `[]` | List of capabilities the agent declares it can perform. |
| `verified_capabilities` | `list[str]` | `[]` | Subset that has been independently verified. |
| `verification_method` | `str` | `""` | How verification was performed. Affects method bonus. |

**Verification method values and their bonuses:**

| Value | Bonus |
|---|---|
| `"benchmark_suite"` | 1.0 |
| `"automated_eval"` | 1.0 |
| `"automated_benchmark"` | 1.0 |
| `"manual_review"` | 0.5 |
| `"manual"` | 0.5 |
| Any other / empty | 0.0 |

---

### `SecurityEvidence`

```python
class SecurityEvidence(BaseModel):
    sandbox_compliant:    bool             # default=False
    vulnerability_count:  int              # default=0, ge=0
    last_audit_date:      datetime.date | None  # default=None
```

Evidence for the security dimension.

**Fields**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `sandbox_compliant` | `bool` | `False` | — | True if agent has been validated against a sandbox capability declaration. |
| `vulnerability_count` | `int` | `0` | >= 0 | Number of known, unmitigated vulnerabilities. |
| `last_audit_date` | `datetime.date \| None` | `None` | — | Date of the most recent security audit. `None` yields an audit score of 0.0. |

---

## Private Math Helpers (in `core.py`)

These functions are not part of the public API but are documented here for
contributors who need to understand or extend the scoring logic.

### `_clamp(value, low=0.0, high=1.0) -> float`

Clamp `value` to the range `[low, high]`. Used throughout to prevent scores
from leaving [0, 1] due to floating-point edge cases.

### `_linear_decay(value, low, high) -> float`

Returns 1.0 when `value <= low`, 0.0 when `value >= high`, and interpolates
linearly in between. Used for latency scoring and audit recency scoring where
larger values are worse.

### `_sample_confidence(sample_count, target) -> float`

Logarithmic confidence scale:

```
confidence = log(1 + sample_count) / log(1 + target)
```

Clamped to [0.0, 1.0]. Returns 0.0 for `sample_count <= 0` and 1.0 for
`sample_count >= target`.
