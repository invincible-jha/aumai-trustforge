"""Async API for aumai-trustforge using aumai-async-core foundation library.

Provides AsyncTrustScorer — a lifecycle-managed async service that wraps
the synchronous TrustScorer with event emission, concurrency control, and
health checks.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aumai_async_core import AsyncEventEmitter, AsyncService, AsyncServiceConfig

from aumai_trustforge.core import TrustScorer
from aumai_trustforge.models import (
    BehaviorEvidence,
    CapabilityEvidence,
    DimensionScore,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustScore,
    TrustWeights,
)


class AsyncTrustScorer(AsyncService):
    """Lifecycle-managed async service for agent trust scoring.

    Wraps the synchronous :class:`~aumai_trustforge.core.TrustScorer` with
    async-first ergonomics, event emission on trust score computation, and the
    full :class:`~aumai_async_core.core.AsyncService` lifecycle (start/stop,
    health checks, concurrency limits).

    Events emitted:
        - ``trust.scored``: fired after a full trust score is computed.
          Payload keys: ``agent_id``, ``overall_score``, ``grade``.
        - ``trust.updated``: fired when a dimension score is individually
          computed.  Payload keys: ``agent_id``, ``dimension``, ``score``,
          ``confidence``.
        - ``trust.violation``: fired when the overall score falls below the
          configured violation threshold.  Payload keys: ``agent_id``,
          ``overall_score``, ``threshold``.

    Example::

        config = AsyncServiceConfig(name="trustforge")
        service = AsyncTrustScorer(config)
        await service.start()

        weights = TrustWeights()
        score = await service.score_agent(
            agent_id="agent-1",
            weights=weights,
            provenance=ProvenanceEvidence(model_card_present=True, license_verified=True),
            behavior=BehaviorEvidence(error_rate=0.01, uptime_pct=99.9, sample_count=500),
            capability=CapabilityEvidence(
                claimed_capabilities=["summarize"],
                verified_capabilities=["summarize"],
                verification_method="benchmark_suite",
            ),
            security=SecurityEvidence(sandbox_compliant=True, vulnerability_count=0),
        )
        print(score.grade())

        @service.emitter.on_event("trust.violation")
        async def on_violation(agent_id: str, overall_score: float, **kw: Any) -> None:
            print(f"Trust violation for {agent_id}: score={overall_score}")

        await service.stop()
    """

    #: Score below this threshold triggers a ``trust.violation`` event.
    DEFAULT_VIOLATION_THRESHOLD: float = 0.40

    def __init__(
        self,
        config: AsyncServiceConfig | None = None,
        *,
        violation_threshold: float = DEFAULT_VIOLATION_THRESHOLD,
        run_in_executor: bool = True,
    ) -> None:
        """Initialise the async trust scorer service.

        Args:
            config: Service configuration.  Defaults to a sensible config
                with ``name="trustforge"``.
            violation_threshold: Overall scores below this value trigger a
                ``trust.violation`` event.  Default is ``0.40`` (grade ``F``).
            run_in_executor: When ``True`` (the default), CPU-bound scoring
                runs in the default thread executor to avoid blocking the
                event loop.  Set to ``False`` in tests to keep execution
                synchronous.
        """
        effective_config = config or AsyncServiceConfig(
            name="trustforge",
            health_check_interval_seconds=0.0,
        )
        super().__init__(effective_config)
        self._emitter: AsyncEventEmitter = AsyncEventEmitter()
        self._violation_threshold: float = violation_threshold
        self._run_in_executor = run_in_executor

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def emitter(self) -> AsyncEventEmitter:
        """The :class:`~aumai_async_core.events.AsyncEventEmitter` for this service.

        Register handlers here to receive ``trust.scored``, ``trust.updated``,
        and ``trust.violation`` events.
        """
        return self._emitter

    @property
    def violation_threshold(self) -> float:
        """Overall score threshold below which a ``trust.violation`` event fires."""
        return self._violation_threshold

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """No-op start hook — scorer is stateless and ready immediately."""

    async def on_stop(self) -> None:
        """Remove all event listeners on service shutdown."""
        self._emitter.remove_all_listeners()

    async def health_check(self) -> bool:
        """Return ``True`` when the underlying scorer is operational.

        Performs a trivial probe: score all-zero evidence and verify the
        result is a valid :class:`~aumai_trustforge.models.TrustScore`.
        """
        try:
            weights = TrustWeights()
            scorer = TrustScorer(weights)
            prov = scorer.score_provenance(ProvenanceEvidence())
            behav = scorer.score_behavior(BehaviorEvidence())
            cap = scorer.score_capability(CapabilityEvidence())
            sec = scorer.score_security(SecurityEvidence())
            result = scorer.compute_trust(prov, behav, cap, sec, agent_id="healthcheck")
            return isinstance(result, TrustScore)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def score_agent(
        self,
        agent_id: str,
        weights: TrustWeights,
        provenance: ProvenanceEvidence,
        behavior: BehaviorEvidence,
        capability: CapabilityEvidence,
        security: SecurityEvidence,
    ) -> TrustScore:
        """Score an agent across all four trust dimensions asynchronously.

        The CPU-bound scoring work is dispatched to a thread executor so the
        event loop remains unblocked.  Events are emitted for each dimension
        score (``trust.updated``) and the final result (``trust.scored``).
        If the overall score is below the violation threshold, a
        ``trust.violation`` event is also emitted.

        Args:
            agent_id: Unique identifier of the agent being scored.
            weights: Per-dimension weights for the overall score computation.
            provenance: Evidence for the provenance dimension.
            behavior: Evidence for the behavior dimension.
            capability: Evidence for the capability dimension.
            security: Evidence for the security dimension.

        Returns:
            A :class:`~aumai_trustforge.models.TrustScore` with overall score,
            per-dimension breakdown, and timestamp.
        """
        await self.increment_request_count()

        try:
            if self._run_in_executor:
                loop = asyncio.get_running_loop()
                trust_score: TrustScore = await loop.run_in_executor(
                    None,
                    self._compute_sync,
                    agent_id,
                    weights,
                    provenance,
                    behavior,
                    capability,
                    security,
                )
            else:
                trust_score = self._compute_sync(
                    agent_id, weights, provenance, behavior, capability, security
                )
        except Exception:
            await self.increment_error_count()
            raise

        # Emit per-dimension update events concurrently.
        await asyncio.gather(
            *[
                self._emit_dimension_updated(agent_id, dim_score)
                for dim_score in trust_score.dimension_scores.values()
            ]
        )

        # Emit final scored event.
        await self._emitter.emit(
            "trust.scored",
            agent_id=agent_id,
            overall_score=trust_score.overall_score,
            grade=trust_score.grade(),
        )

        # Emit violation event if score is below threshold.
        if trust_score.overall_score < self._violation_threshold:
            await self._emitter.emit(
                "trust.violation",
                agent_id=agent_id,
                overall_score=trust_score.overall_score,
                threshold=self._violation_threshold,
            )

        return trust_score

    async def score_provenance_async(
        self, agent_id: str, weights: TrustWeights, evidence: ProvenanceEvidence
    ) -> DimensionScore:
        """Score the provenance dimension asynchronously.

        Args:
            agent_id: Agent identifier used in the emitted event.
            weights: Trust weights (used to build the scorer).
            evidence: Provenance evidence to score.

        Returns:
            A :class:`~aumai_trustforge.models.DimensionScore` for provenance.
        """
        scorer = TrustScorer(weights)
        if self._run_in_executor:
            loop = asyncio.get_running_loop()
            dim_score: DimensionScore = await loop.run_in_executor(
                None, scorer.score_provenance, evidence
            )
        else:
            dim_score = scorer.score_provenance(evidence)
        await self._emit_dimension_updated(agent_id, dim_score)
        return dim_score

    async def score_behavior_async(
        self, agent_id: str, weights: TrustWeights, evidence: BehaviorEvidence
    ) -> DimensionScore:
        """Score the behavior dimension asynchronously.

        Args:
            agent_id: Agent identifier used in the emitted event.
            weights: Trust weights (used to build the scorer).
            evidence: Behavior evidence to score.

        Returns:
            A :class:`~aumai_trustforge.models.DimensionScore` for behavior.
        """
        scorer = TrustScorer(weights)
        if self._run_in_executor:
            loop = asyncio.get_running_loop()
            dim_score: DimensionScore = await loop.run_in_executor(
                None, scorer.score_behavior, evidence
            )
        else:
            dim_score = scorer.score_behavior(evidence)
        await self._emit_dimension_updated(agent_id, dim_score)
        return dim_score

    async def score_capability_async(
        self, agent_id: str, weights: TrustWeights, evidence: CapabilityEvidence
    ) -> DimensionScore:
        """Score the capability dimension asynchronously.

        Args:
            agent_id: Agent identifier used in the emitted event.
            weights: Trust weights (used to build the scorer).
            evidence: Capability evidence to score.

        Returns:
            A :class:`~aumai_trustforge.models.DimensionScore` for capability.
        """
        scorer = TrustScorer(weights)
        if self._run_in_executor:
            loop = asyncio.get_running_loop()
            dim_score: DimensionScore = await loop.run_in_executor(
                None, scorer.score_capability, evidence
            )
        else:
            dim_score = scorer.score_capability(evidence)
        await self._emit_dimension_updated(agent_id, dim_score)
        return dim_score

    async def score_security_async(
        self, agent_id: str, weights: TrustWeights, evidence: SecurityEvidence
    ) -> DimensionScore:
        """Score the security dimension asynchronously.

        Args:
            agent_id: Agent identifier used in the emitted event.
            weights: Trust weights (used to build the scorer).
            evidence: Security evidence to score.

        Returns:
            A :class:`~aumai_trustforge.models.DimensionScore` for security.
        """
        scorer = TrustScorer(weights)
        if self._run_in_executor:
            loop = asyncio.get_running_loop()
            dim_score: DimensionScore = await loop.run_in_executor(
                None, scorer.score_security, evidence
            )
        else:
            dim_score = scorer.score_security(evidence)
        await self._emit_dimension_updated(agent_id, dim_score)
        return dim_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sync(
        self,
        agent_id: str,
        weights: TrustWeights,
        provenance: ProvenanceEvidence,
        behavior: BehaviorEvidence,
        capability: CapabilityEvidence,
        security: SecurityEvidence,
    ) -> TrustScore:
        """Run the full synchronous scoring pipeline.

        Args:
            agent_id: Agent identifier.
            weights: Per-dimension weights.
            provenance: Provenance evidence.
            behavior: Behavior evidence.
            capability: Capability evidence.
            security: Security evidence.

        Returns:
            A :class:`~aumai_trustforge.models.TrustScore`.
        """
        scorer = TrustScorer(weights)
        prov_score = scorer.score_provenance(provenance)
        behav_score = scorer.score_behavior(behavior)
        cap_score = scorer.score_capability(capability)
        sec_score = scorer.score_security(security)
        return scorer.compute_trust(
            prov_score, behav_score, cap_score, sec_score, agent_id=agent_id
        )

    async def _emit_dimension_updated(
        self, agent_id: str, dim_score: DimensionScore
    ) -> None:
        """Emit a ``trust.updated`` event for a single dimension score.

        Args:
            agent_id: Agent identifier.
            dim_score: The dimension score to report.
        """
        await self._emitter.emit(
            "trust.updated",
            agent_id=agent_id,
            dimension=dim_score.dimension.value,
            score=dim_score.score,
            confidence=dim_score.confidence,
        )


__all__ = ["AsyncTrustScorer"]
