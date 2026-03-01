"""LLM-powered trust assessor using aumai-llm-core foundation library.

Provides LLMTrustAssessor, which uses a language model to perform semantic
analysis of agent behavior and context for trust assessment beyond what static
scoring can capture.  Falls back to static heuristics when the LLM is
unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aumai_llm_core import (
    CompletionRequest,
    LLMClient,
    Message,
    MockProvider,
    ModelConfig,
)
from pydantic import BaseModel, Field

from aumai_trustforge.models import (
    BehaviorEvidence,
    CapabilityEvidence,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustDimension,
    TrustWeights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

_CONFIDENCE_LEVELS = ("very_low", "low", "medium", "high", "very_high")
_RISK_LEVELS = ("none", "low", "medium", "high", "critical")


class LLMTrustAssessment(BaseModel):
    """Structured output returned by the LLM-powered trust assessment.

    Attributes:
        trust_level: Overall assessment — one of ``"none"``, ``"low"``,
            ``"medium"``, ``"high"``, or ``"critical"`` risk.
        confidence: Confidence in the assessment — one of ``"very_low"``,
            ``"low"``, ``"medium"``, ``"high"``, or ``"very_high"``.
        risk_factors: List of identified risk factors that reduce trustworthiness.
        recommendations: Ordered list of actionable steps to improve trust.
        assessed_dimensions: Which trust dimensions were evaluated in this
            assessment (e.g. ``["provenance", "behavior"]``).
        llm_powered: ``True`` when the result came from an LLM call, ``False``
            when it came from the static heuristic fallback.
        summary: Short natural-language summary of the trust assessment.
    """

    trust_level: str = Field(
        default="medium",
        description="Overall risk level: none | low | medium | high | critical",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence: very_low | low | medium | high | very_high",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risk factors reducing agent trustworthiness.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Ordered list of actionable recommendations.",
    )
    assessed_dimensions: list[str] = Field(
        default_factory=list,
        description="Trust dimensions evaluated in this assessment.",
    )
    llm_powered: bool = Field(
        default=True,
        description="True when the result was produced by an LLM call.",
    )
    summary: str = Field(
        default="",
        description="Short natural-language summary of the trust assessment.",
    )

    def risk_score(self) -> float:
        """Convert the trust level string to a normalized risk score in [0, 1].

        Returns:
            A float where 0.0 = no risk and 1.0 = critical risk.
        """
        mapping = {
            "none": 0.0,
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75,
            "critical": 1.0,
        }
        return mapping.get(self.trust_level, 0.5)

    def suggested_trust_score(self) -> float:
        """Return a suggested overall trust score (inverse of risk score).

        Returns:
            A float in [0, 1] where 1.0 = fully trusted.
        """
        return 1.0 - self.risk_score()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a trust assessment specialist for AI agents and autonomous systems.

Given a JSON object containing evidence about an AI agent — including provenance
data (model card, license, author, source URL), behavioral metrics (error rate,
latency, uptime, sample count), capability evidence (claimed vs verified
capabilities), and security posture (sandbox compliance, vulnerabilities, audit
date) — assess the agent's overall trustworthiness.

Consider the following trust dimensions:
  - provenance: Is the agent's origin verifiable and documented?
  - behavior: Does the agent perform reliably with low error rates and high uptime?
  - capability: Are the agent's capabilities backed by independent verification?
  - security: Does the agent meet security requirements (sandbox, audits, no vulns)?

Respond ONLY with a valid JSON object matching this exact schema — no markdown,
no prose outside the JSON:
{
  "trust_level": "<none|low|medium|high|critical>",
  "confidence": "<very_low|low|medium|high|very_high>",
  "risk_factors": ["<string>", ...],
  "recommendations": ["<string>", ...],
  "assessed_dimensions": ["<dimension>", ...],
  "summary": "<short natural-language summary>"
}

Where trust_level represents the RISK level (high risk = low trust).
If the agent appears fully trustworthy, set trust_level to "none".
"""


class LLMTrustAssessor:
    """LLM-powered assessor that performs semantic trust analysis.

    Sends agent evidence to an LLM for holistic trust assessment and returns
    a structured :class:`LLMTrustAssessment`.  Automatically falls back to
    static heuristics when the LLM call fails or is not configured.

    Args:
        client: An :class:`~aumai_llm_core.core.LLMClient` instance.  When
            ``None`` the assessor operates in **fallback-only mode** (static
            heuristics only).

    Example (production)::

        config = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-6")
        client = LLMClient(config)
        assessor = LLMTrustAssessor(client=client)
        assessment = await assessor.assess(
            agent_id="agent-1",
            provenance=ProvenanceEvidence(model_card_present=True),
            behavior=BehaviorEvidence(error_rate=0.02, sample_count=500),
        )

    Example (testing with MockProvider)::

        assessor = build_mock_assessor()
        assessment = await assessor.assess(agent_id="agent-1")
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def assess(
        self,
        agent_id: str,
        provenance: ProvenanceEvidence | None = None,
        behavior: BehaviorEvidence | None = None,
        capability: CapabilityEvidence | None = None,
        security: SecurityEvidence | None = None,
        weights: TrustWeights | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> LLMTrustAssessment:
        """Assess an agent's trustworthiness using LLM-powered analysis.

        The method first attempts an LLM call.  If the client is not
        configured, or if the LLM call fails, it falls back to static
        heuristic analysis automatically.

        Args:
            agent_id: Unique identifier of the agent being assessed.
            provenance: Provenance evidence (optional).
            behavior: Behavior evidence (optional).
            capability: Capability evidence (optional).
            security: Security evidence (optional).
            weights: Trust weights — included in context for the LLM.
            extra_context: Any additional key/value pairs to include in the
                LLM prompt context.

        Returns:
            A :class:`LLMTrustAssessment` with risk level, confidence,
            risk factors, and recommendations.
        """
        if self._client is None:
            logger.debug(
                "LLMTrustAssessor: no LLM client configured, using heuristic fallback."
            )
            return self._heuristic_fallback(
                agent_id, provenance, behavior, capability, security
            )

        try:
            return await self._llm_assess(
                agent_id, provenance, behavior, capability, security, weights,
                extra_context
            )
        except Exception as exc:
            logger.warning(
                "LLMTrustAssessor: LLM call failed (%s), falling back to heuristics.",
                exc,
            )
            return self._heuristic_fallback(
                agent_id, provenance, behavior, capability, security
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _llm_assess(
        self,
        agent_id: str,
        provenance: ProvenanceEvidence | None,
        behavior: BehaviorEvidence | None,
        capability: CapabilityEvidence | None,
        security: SecurityEvidence | None,
        weights: TrustWeights | None,
        extra_context: dict[str, Any] | None,
    ) -> LLMTrustAssessment:
        """Perform the actual LLM call and parse the structured response.

        Args:
            agent_id: Agent identifier.
            provenance: Provenance evidence.
            behavior: Behavior evidence.
            capability: Capability evidence.
            security: Security evidence.
            weights: Trust weights.
            extra_context: Additional context for the prompt.

        Returns:
            Parsed :class:`LLMTrustAssessment`.
        """
        evidence_payload: dict[str, Any] = {"agent_id": agent_id}

        if provenance is not None:
            evidence_payload["provenance"] = provenance.model_dump()
        if behavior is not None:
            evidence_payload["behavior"] = behavior.model_dump()
        if capability is not None:
            evidence_payload["capability"] = capability.model_dump()
        if security is not None:
            evidence_payload["security"] = security.model_dump()
        if weights is not None:
            evidence_payload["weights"] = weights.model_dump()
        if extra_context:
            evidence_payload["extra_context"] = extra_context

        evidence_json = json.dumps(evidence_payload, indent=2, default=str)
        user_message = (
            f"Assess the trustworthiness of the following AI agent:\n\n"
            f"```json\n{evidence_json}\n```"
        )

        request = CompletionRequest(
            messages=[
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=user_message),
            ],
            temperature=0.0,
        )

        assert self._client is not None
        response = await self._client.complete(request)
        return self._parse_llm_response(response.content)

    def _parse_llm_response(self, raw_content: str) -> LLMTrustAssessment:
        """Parse the LLM's JSON response into a :class:`LLMTrustAssessment`.

        Strips markdown code fences if present before JSON parsing.  If
        parsing fails, returns a conservative ``"medium"`` risk assessment.

        Args:
            raw_content: Raw text content from the LLM response.

        Returns:
            A :class:`LLMTrustAssessment`.
        """
        content = raw_content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data: dict[str, Any] = json.loads(content)
            trust_level = str(data.get("trust_level", "medium"))
            if trust_level not in _RISK_LEVELS:
                trust_level = "medium"
            confidence = str(data.get("confidence", "medium"))
            if confidence not in _CONFIDENCE_LEVELS:
                confidence = "medium"
            return LLMTrustAssessment(
                trust_level=trust_level,
                confidence=confidence,
                risk_factors=list(data.get("risk_factors", [])),
                recommendations=list(data.get("recommendations", [])),
                assessed_dimensions=list(data.get("assessed_dimensions", [])),
                summary=str(data.get("summary", "")),
                llm_powered=True,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "LLMTrustAssessor: could not parse LLM JSON response: %s", exc
            )
            return LLMTrustAssessment(
                trust_level="medium",
                confidence="low",
                risk_factors=["LLM response could not be parsed — treating as uncertain."],
                recommendations=["Manually review this agent's trust evidence."],
                assessed_dimensions=[],
                summary=(
                    "LLM response unparseable — defaulting to medium risk. "
                    f"Parse error: {exc}"
                ),
                llm_powered=True,
            )

    def _heuristic_fallback(
        self,
        agent_id: str,
        provenance: ProvenanceEvidence | None,
        behavior: BehaviorEvidence | None,
        capability: CapabilityEvidence | None,
        security: SecurityEvidence | None,
    ) -> LLMTrustAssessment:
        """Run static heuristic trust assessment from available evidence.

        Args:
            agent_id: Agent identifier.
            provenance: Provenance evidence.
            behavior: Behavior evidence.
            capability: Capability evidence.
            security: Security evidence.

        Returns:
            A :class:`LLMTrustAssessment` marked with ``llm_powered=False``.
        """
        risk_factors: list[str] = []
        recommendations: list[str] = []
        assessed_dimensions: list[str] = []
        risk_score: float = 0.0
        dimension_count = 0

        if provenance is not None:
            assessed_dimensions.append(TrustDimension.provenance.value)
            dimension_count += 1
            prov_risk = 0.0
            if not provenance.model_card_present:
                prov_risk += 0.25
                risk_factors.append("No model card present")
                recommendations.append("Publish a model card documenting the agent.")
            if not provenance.license_verified:
                prov_risk += 0.25
                risk_factors.append("License not verified")
                recommendations.append("Verify and publish the agent's license.")
            if not provenance.author_verified:
                prov_risk += 0.15
                risk_factors.append("Author identity unverified")
            if not provenance.source_url:
                prov_risk += 0.10
                risk_factors.append("No source URL provided")
            risk_score += min(prov_risk, 1.0)

        if behavior is not None:
            assessed_dimensions.append(TrustDimension.behavior.value)
            dimension_count += 1
            behav_risk = 0.0
            if behavior.error_rate > 0.10:
                behav_risk += 0.40
                risk_factors.append(f"High error rate: {behavior.error_rate:.1%}")
                recommendations.append("Investigate and reduce error rate below 5%.")
            elif behavior.error_rate > 0.05:
                behav_risk += 0.20
                risk_factors.append(f"Elevated error rate: {behavior.error_rate:.1%}")
            if behavior.uptime_pct < 95.0:
                behav_risk += 0.30
                risk_factors.append(f"Low uptime: {behavior.uptime_pct:.1f}%")
                recommendations.append("Improve availability to above 99%.")
            if behavior.sample_count < 10:
                behav_risk += 0.20
                risk_factors.append(
                    f"Very low sample count ({behavior.sample_count}) — "
                    "behavioral metrics unreliable"
                )
            risk_score += min(behav_risk, 1.0)

        if capability is not None:
            assessed_dimensions.append(TrustDimension.capability.value)
            dimension_count += 1
            cap_risk = 0.0
            if not capability.claimed_capabilities:
                cap_risk += 0.30
                risk_factors.append("No capabilities claimed")
            elif not capability.verified_capabilities:
                cap_risk += 0.50
                risk_factors.append("No capabilities independently verified")
                recommendations.append("Run independent capability verification.")
            else:
                claimed = set(capability.claimed_capabilities)
                verified = set(capability.verified_capabilities)
                overlap = claimed & verified
                if len(claimed) > 0:
                    unverified_ratio = 1.0 - len(overlap) / len(claimed)
                    if unverified_ratio > 0.5:
                        cap_risk += 0.40
                        risk_factors.append(
                            f"More than half of claimed capabilities unverified "
                            f"({len(claimed) - len(overlap)}/{len(claimed)})"
                        )
            risk_score += min(cap_risk, 1.0)

        if security is not None:
            assessed_dimensions.append(TrustDimension.security.value)
            dimension_count += 1
            sec_risk = 0.0
            if not security.sandbox_compliant:
                sec_risk += 0.35
                risk_factors.append("Agent is not sandbox compliant")
                recommendations.append("Validate agent against sandbox requirements.")
            if security.vulnerability_count > 0:
                sec_risk += min(0.40 * security.vulnerability_count, 0.60)
                risk_factors.append(
                    f"{security.vulnerability_count} known vulnerability/vulnerabilities"
                )
                recommendations.append("Remediate known vulnerabilities.")
            if security.last_audit_date is None:
                sec_risk += 0.25
                risk_factors.append("No security audit recorded")
                recommendations.append("Conduct a security audit.")
            risk_score += min(sec_risk, 1.0)

        if dimension_count == 0:
            # No evidence provided — maximum uncertainty.
            return LLMTrustAssessment(
                trust_level="medium",
                confidence="very_low",
                risk_factors=["No evidence provided for any trust dimension"],
                recommendations=["Provide evidence for all four trust dimensions."],
                assessed_dimensions=[],
                summary=f"Agent '{agent_id}': no evidence available — medium risk assumed.",
                llm_powered=False,
            )

        avg_risk = risk_score / dimension_count
        if avg_risk < 0.10:
            trust_level = "none"
            confidence = "high"
        elif avg_risk < 0.30:
            trust_level = "low"
            confidence = "medium"
        elif avg_risk < 0.55:
            trust_level = "medium"
            confidence = "medium"
        elif avg_risk < 0.75:
            trust_level = "high"
            confidence = "medium"
        else:
            trust_level = "critical"
            confidence = "high"

        summary = (
            f"Agent '{agent_id}': heuristic assessment across "
            f"{len(assessed_dimensions)} dimension(s) yields "
            f"risk_level='{trust_level}' (avg_risk={avg_risk:.2f}). "
            "LLM analysis was unavailable."
        )

        return LLMTrustAssessment(
            trust_level=trust_level,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendations=recommendations,
            assessed_dimensions=assessed_dimensions,
            summary=summary,
            llm_powered=False,
        )


def build_mock_assessor(responses: list[str] | None = None) -> LLMTrustAssessor:
    """Create an :class:`LLMTrustAssessor` backed by a :class:`~aumai_llm_core.MockProvider`.

    This is the canonical way to build a fully testable LLM assessor without
    making real API calls.

    Args:
        responses: Canned JSON response strings to return in round-robin order.
            Defaults to a single ``"none"``-risk response.

    Returns:
        A configured :class:`LLMTrustAssessor` using the mock provider.

    Example::

        assessor = build_mock_assessor([
            '{"trust_level":"low","confidence":"high","risk_factors":[],'
            '"recommendations":[],"assessed_dimensions":["provenance"],'
            '"summary":"Looks good."}'
        ])
        result = await assessor.assess(agent_id="agent-1")
        assert result.trust_level == "low"
    """
    default_response = json.dumps(
        {
            "trust_level": "none",
            "confidence": "high",
            "risk_factors": [],
            "recommendations": [],
            "assessed_dimensions": [
                "provenance",
                "behavior",
                "capability",
                "security",
            ],
            "summary": "Agent appears fully trustworthy.",
        }
    )
    effective_responses = responses if responses is not None else [default_response]

    mock_provider = MockProvider(responses=effective_responses)
    config = ModelConfig(provider="mock", model_id="mock-trustforge-assessor")
    client = LLMClient(config)
    client._provider = mock_provider  # type: ignore[attr-defined]
    return LLMTrustAssessor(client=client)


__all__ = [
    "LLMTrustAssessment",
    "LLMTrustAssessor",
    "build_mock_assessor",
]
