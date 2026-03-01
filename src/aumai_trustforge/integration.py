"""AumOS integration module for aumai-trustforge.

Registers trustforge as a named service in the AumOS discovery layer,
publishes trust domain events (trust.scored, trust.updated, trust.violation),
and subscribes to capability events from the broader agent ecosystem.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aumai_integration import AumOS, Event, EventBus, ServiceInfo

from aumai_trustforge.core import TrustScorer
from aumai_trustforge.models import (
    BehaviorEvidence,
    CapabilityEvidence,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustScore,
    TrustWeights,
)

logger = logging.getLogger(__name__)

# Service metadata constants.
_SERVICE_NAME = "trustforge"
_SERVICE_VERSION = "0.1.0"
_SERVICE_DESCRIPTION = (
    "AumAI TrustForge — generic trust scoring for AI agents across 4 dimensions."
)
_SERVICE_CAPABILITIES = [
    "trust-scoring",
    "provenance-scoring",
    "behavior-scoring",
    "capability-scoring",
    "security-scoring",
]


class TrustForgeIntegration:
    """AumOS integration facade for the trustforge service.

    Handles service registration, event subscriptions, and event publishing.
    One instance per application is expected; obtain via :meth:`from_aumos`.

    Publishes:
        - ``trust.scored``: emitted after a complete trust score is computed.
          Payload: ``agent_id``, ``overall_score``, ``grade``.
        - ``trust.updated``: emitted for each dimension score update.
          Payload: ``agent_id``, ``dimension``, ``score``, ``confidence``.
        - ``trust.violation``: emitted when overall score is below threshold.
          Payload: ``agent_id``, ``overall_score``, ``threshold``.

    Subscribes:
        - ``agent.capability_updated``: receives capability change events
          from the agent ecosystem to refresh the capability cache.

    Attributes:
        SERVICE_NAME: Constant string ``"trustforge"`` used as the service key.
        VIOLATION_THRESHOLD: Default score below which a violation event fires.

    Example::

        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        await integration.register()

        score = await integration.score_and_publish(
            agent_id="my-agent",
            weights=TrustWeights(),
            provenance=ProvenanceEvidence(model_card_present=True),
            behavior=BehaviorEvidence(error_rate=0.01, sample_count=100),
            capability=CapabilityEvidence(
                claimed_capabilities=["summarize"],
                verified_capabilities=["summarize"],
            ),
            security=SecurityEvidence(sandbox_compliant=True),
        )
    """

    SERVICE_NAME: str = _SERVICE_NAME
    VIOLATION_THRESHOLD: float = 0.40

    def __init__(self, aumos: AumOS) -> None:
        """Initialise the integration against an AumOS hub.

        Args:
            aumos: The AumOS hub to register with and subscribe events on.
        """
        self._aumos = aumos
        self._subscription_id: str | None = None
        self._registered: bool = False
        # Capability cache: agent_id -> list of known capabilities.
        self._capability_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_aumos(cls, aumos: AumOS) -> "TrustForgeIntegration":
        """Create a :class:`TrustForgeIntegration` bound to *aumos*.

        Args:
            aumos: The AumOS hub instance.

        Returns:
            A new :class:`TrustForgeIntegration`.
        """
        return cls(aumos)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """Register trustforge with AumOS and subscribe to capability events.

        Idempotent — calling this method more than once is safe.

        Steps:
            1. Register the service descriptor with the discovery layer.
            2. Subscribe to ``agent.capability_updated`` events.
        """
        if self._registered:
            logger.debug("TrustForgeIntegration: already registered, skipping.")
            return

        service_info = ServiceInfo(
            name=_SERVICE_NAME,
            version=_SERVICE_VERSION,
            description=_SERVICE_DESCRIPTION,
            capabilities=list(_SERVICE_CAPABILITIES),
            endpoints={},
            metadata={
                "trust_dimensions": [
                    "provenance",
                    "behavior",
                    "capability",
                    "security",
                ],
                "violation_threshold": self.VIOLATION_THRESHOLD,
            },
            status="healthy",
        )
        self._aumos.register(service_info)
        logger.info(
            "TrustForgeIntegration: registered service '%s' v%s with capabilities %s",
            _SERVICE_NAME,
            _SERVICE_VERSION,
            _SERVICE_CAPABILITIES,
        )

        # Subscribe to agent.capability_updated events.
        self._subscription_id = self._aumos.events.subscribe(
            pattern="agent.capability_updated",
            handler=self._handle_capability_updated,
            subscriber=_SERVICE_NAME,
        )
        logger.info(
            "TrustForgeIntegration: subscribed to 'agent.capability_updated' events "
            "(subscription_id=%s)",
            self._subscription_id,
        )

        self._registered = True

    async def unregister(self) -> None:
        """Unsubscribe from events and mark the service as not registered.

        Does not remove the service from the discovery layer (that is managed
        by the AumOS hub lifecycle).
        """
        if self._subscription_id is not None:
            self._aumos.events.unsubscribe(self._subscription_id)
            self._subscription_id = None
        self._registered = False
        logger.info("TrustForgeIntegration: unregistered.")

    # ------------------------------------------------------------------
    # Scoring and publishing
    # ------------------------------------------------------------------

    async def score_and_publish(
        self,
        agent_id: str,
        weights: TrustWeights,
        provenance: ProvenanceEvidence,
        behavior: BehaviorEvidence,
        capability: CapabilityEvidence,
        security: SecurityEvidence,
        source: str = _SERVICE_NAME,
    ) -> TrustScore:
        """Score an agent and publish trust domain events.

        Publishes:
        - One ``trust.updated`` event per dimension.
        - One ``trust.scored`` event with the final result.
        - One ``trust.violation`` event if ``overall_score < VIOLATION_THRESHOLD``.

        Args:
            agent_id: Unique identifier of the agent being scored.
            weights: Per-dimension weights for the overall score computation.
            provenance: Provenance evidence.
            behavior: Behavior evidence.
            capability: Capability evidence.
            security: Security evidence.
            source: The event source name (defaults to ``"trustforge"``).

        Returns:
            The computed :class:`~aumai_trustforge.models.TrustScore`.
        """
        scorer = TrustScorer(weights)
        prov_score = scorer.score_provenance(provenance)
        behav_score = scorer.score_behavior(behavior)
        cap_score = scorer.score_capability(capability)
        sec_score = scorer.score_security(security)
        trust_score = scorer.compute_trust(
            prov_score, behav_score, cap_score, sec_score, agent_id=agent_id
        )

        # Publish per-dimension update events.
        for dim_score in trust_score.dimension_scores.values():
            await self._aumos.events.publish_simple(
                "trust.updated",
                source=source,
                agent_id=agent_id,
                dimension=dim_score.dimension.value,
                score=dim_score.score,
                confidence=dim_score.confidence,
            )
            logger.debug(
                "TrustForgeIntegration: trust.updated for '%s' dimension='%s' "
                "score=%.4f confidence=%.4f",
                agent_id,
                dim_score.dimension.value,
                dim_score.score,
                dim_score.confidence,
            )

        # Publish the final scored event.
        await self._aumos.events.publish_simple(
            "trust.scored",
            source=source,
            agent_id=agent_id,
            overall_score=trust_score.overall_score,
            grade=trust_score.grade(),
        )
        logger.info(
            "TrustForgeIntegration: trust.scored for '%s' overall=%.4f grade='%s'",
            agent_id,
            trust_score.overall_score,
            trust_score.grade(),
        )

        # Publish violation event when threshold exceeded.
        if trust_score.overall_score < self.VIOLATION_THRESHOLD:
            await self._aumos.events.publish_simple(
                "trust.violation",
                source=source,
                agent_id=agent_id,
                overall_score=trust_score.overall_score,
                threshold=self.VIOLATION_THRESHOLD,
            )
            logger.warning(
                "TrustForgeIntegration: trust.violation for '%s' "
                "score=%.4f < threshold=%.2f",
                agent_id,
                trust_score.overall_score,
                self.VIOLATION_THRESHOLD,
            )

        return trust_score

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_capability_updated(self, event: Event) -> None:
        """Update the capability cache when an agent's capabilities change.

        Expected event payload keys:
            - ``agent_id`` (str): The agent whose capabilities changed.
            - ``capabilities`` (list[str]): Updated capability list.

        Missing or invalid payloads are logged as warnings and skipped.

        Args:
            event: The ``agent.capability_updated`` event received from the bus.
        """
        agent_id = str(event.data.get("agent_id", "unknown"))
        capabilities = event.data.get("capabilities")

        if not isinstance(capabilities, list):
            logger.warning(
                "TrustForgeIntegration: received 'agent.capability_updated' event "
                "for '%s' without a valid 'capabilities' list — skipping cache update.",
                agent_id,
            )
            return

        self._capability_cache[agent_id] = [str(c) for c in capabilities]
        logger.info(
            "TrustForgeIntegration: updated capability cache for '%s': %s",
            agent_id,
            self._capability_cache[agent_id],
        )

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def get_cached_capabilities(self, agent_id: str) -> list[str]:
        """Return the cached capability list for *agent_id*.

        Args:
            agent_id: The agent identifier to look up.

        Returns:
            List of capability strings, or an empty list if not cached.
        """
        return list(self._capability_cache.get(agent_id, []))

    def clear_capability_cache(self) -> None:
        """Clear the entire capability cache."""
        self._capability_cache.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_registered(self) -> bool:
        """``True`` when the service has been registered with AumOS."""
        return self._registered

    @property
    def aumos(self) -> AumOS:
        """The AumOS hub this integration is bound to."""
        return self._aumos

    @property
    def capability_cache(self) -> dict[str, list[str]]:
        """Read-only snapshot of the capability cache."""
        return dict(self._capability_cache)


async def setup_trustforge(aumos: AumOS) -> TrustForgeIntegration:
    """Convenience function: create and register a :class:`TrustForgeIntegration`.

    Args:
        aumos: The AumOS hub to register with.

    Returns:
        The registered :class:`TrustForgeIntegration` instance.

    Example::

        hub = AumOS()
        integration = await setup_trustforge(hub)
    """
    integration = TrustForgeIntegration.from_aumos(aumos)
    await integration.register()
    return integration


__all__ = [
    "TrustForgeIntegration",
    "setup_trustforge",
]
