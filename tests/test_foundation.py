"""Comprehensive tests for the four aumai-trustforge foundation modules.

Covers:
  - async_core.py  — AsyncTrustScorer lifecycle, events, scoring
  - store.py       — TrustStore CRUD, history, metrics, model_validator round-trip
  - llm_assessor.py — LLMTrustAssessor with MockProvider + heuristic fallback
  - integration.py  — TrustForgeIntegration registration, events, capability cache

All event handlers are ``async def`` functions; all EventBus publish calls are
awaited, per the aumai-integration contract.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aumai_async_core import AsyncServiceConfig
from aumai_integration import AumOS, Event

from aumai_trustforge.async_core import AsyncTrustScorer
from aumai_trustforge.integration import TrustForgeIntegration, setup_trustforge
from aumai_trustforge.llm_assessor import (
    LLMTrustAssessment,
    LLMTrustAssessor,
    build_mock_assessor,
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
from aumai_trustforge.store import (
    TrustRecord,
    TrustStore,
    TrustStoreConfig,
    TrustStoreMetrics,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture
def default_weights() -> TrustWeights:
    return TrustWeights()


@pytest.fixture
def high_weights() -> TrustWeights:
    return TrustWeights(provenance=0.40, behavior=0.30, capability=0.20, security=0.10)


@pytest.fixture
def good_provenance() -> ProvenanceEvidence:
    return ProvenanceEvidence(
        model_card_present=True,
        license_verified=True,
        author_verified=True,
        source_url="https://example.com/agent-card",
    )


@pytest.fixture
def poor_provenance() -> ProvenanceEvidence:
    return ProvenanceEvidence()


@pytest.fixture
def good_behavior() -> BehaviorEvidence:
    return BehaviorEvidence(
        error_rate=0.01,
        avg_latency_ms=200.0,
        uptime_pct=99.9,
        sample_count=500,
    )


@pytest.fixture
def poor_behavior() -> BehaviorEvidence:
    return BehaviorEvidence(
        error_rate=0.50,
        avg_latency_ms=8000.0,
        uptime_pct=60.0,
        sample_count=5,
    )


@pytest.fixture
def good_capability() -> CapabilityEvidence:
    return CapabilityEvidence(
        claimed_capabilities=["summarize", "translate", "classify"],
        verified_capabilities=["summarize", "translate", "classify"],
        verification_method="benchmark_suite",
    )


@pytest.fixture
def poor_capability() -> CapabilityEvidence:
    return CapabilityEvidence()


@pytest.fixture
def good_security() -> SecurityEvidence:
    import datetime
    return SecurityEvidence(
        sandbox_compliant=True,
        vulnerability_count=0,
        last_audit_date=datetime.date.today(),
    )


@pytest.fixture
def poor_security() -> SecurityEvidence:
    return SecurityEvidence(
        sandbox_compliant=False,
        vulnerability_count=5,
        last_audit_date=None,
    )


def _make_scorer(name: str = "test-trustforge") -> AsyncTrustScorer:
    """Return an AsyncTrustScorer in non-executor mode for synchronous testing."""
    config = AsyncServiceConfig(
        name=name,
        health_check_interval_seconds=0.0,
    )
    return AsyncTrustScorer(config, run_in_executor=False)


# ===========================================================================
# 1. async_core.py — AsyncTrustScorer
# ===========================================================================


class TestAsyncTrustScorerLifecycle:
    async def test_default_config_has_name_trustforge(self) -> None:
        service = AsyncTrustScorer()
        assert service.config.name == "trustforge"

    async def test_custom_config_respected(self) -> None:
        config = AsyncServiceConfig(name="custom-tf", health_check_interval_seconds=0.0)
        service = AsyncTrustScorer(config, run_in_executor=False)
        assert service.config.name == "custom-tf"

    async def test_service_starts_and_stops(self) -> None:
        service = _make_scorer()
        await service.start()
        assert service.status.state == "running"
        await service.stop()
        assert service.status.state == "stopped"

    async def test_start_sets_state_running(self) -> None:
        service = _make_scorer()
        assert service.status.state == "created"
        await service.start()
        assert service.status.state == "running"
        await service.stop()

    async def test_stop_removes_all_listeners(self) -> None:
        service = _make_scorer()
        await service.start()

        async def noop(**kw: Any) -> None:
            pass

        service.emitter.on("trust.scored", noop)
        assert service.emitter.listener_count("trust.scored") == 1
        await service.stop()
        assert service.emitter.listener_count("trust.scored") == 0

    async def test_health_check_returns_true(self) -> None:
        service = _make_scorer()
        await service.start()
        assert await service.health_check() is True
        await service.stop()

    async def test_emitter_property_is_async_event_emitter(self) -> None:
        from aumai_async_core import AsyncEventEmitter
        service = _make_scorer()
        assert isinstance(service.emitter, AsyncEventEmitter)

    async def test_violation_threshold_default(self) -> None:
        service = _make_scorer()
        assert service.violation_threshold == pytest.approx(0.40)

    async def test_violation_threshold_custom(self) -> None:
        service = AsyncTrustScorer(violation_threshold=0.60, run_in_executor=False)
        assert service.violation_threshold == pytest.approx(0.60)


class TestAsyncTrustScorerScoring:
    async def test_score_agent_returns_trust_score(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        result = await service.score_agent(
            agent_id="agent-good",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert isinstance(result, TrustScore)
        assert result.agent_id == "agent-good"
        await service.stop()

    async def test_score_agent_overall_score_in_range(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        result = await service.score_agent(
            agent_id="agent-range",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert 0.0 <= result.overall_score <= 1.0
        await service.stop()

    async def test_score_agent_good_evidence_high_score(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        result = await service.score_agent(
            agent_id="agent-high",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert result.overall_score > 0.5
        await service.stop()

    async def test_score_agent_poor_evidence_low_score(
        self,
        default_weights: TrustWeights,
        poor_provenance: ProvenanceEvidence,
        poor_behavior: BehaviorEvidence,
        poor_capability: CapabilityEvidence,
        poor_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        result = await service.score_agent(
            agent_id="agent-low",
            weights=default_weights,
            provenance=poor_provenance,
            behavior=poor_behavior,
            capability=poor_capability,
            security=poor_security,
        )
        assert result.overall_score < 0.5
        await service.stop()

    async def test_score_agent_emits_trust_scored_event(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        captured: list[dict[str, Any]] = []

        async def on_scored(**kw: Any) -> None:
            captured.append(kw)

        service.emitter.on("trust.scored", on_scored)
        await service.score_agent(
            agent_id="agent-ev",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert len(captured) == 1
        assert captured[0]["agent_id"] == "agent-ev"
        assert "overall_score" in captured[0]
        assert "grade" in captured[0]
        await service.stop()

    async def test_score_agent_emits_trust_updated_per_dimension(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        updates: list[dict[str, Any]] = []

        async def on_updated(**kw: Any) -> None:
            updates.append(kw)

        service.emitter.on("trust.updated", on_updated)
        await service.score_agent(
            agent_id="agent-upd",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        # 4 dimensions = 4 trust.updated events.
        assert len(updates) == 4
        dimensions = {u["dimension"] for u in updates}
        assert dimensions == {"provenance", "behavior", "capability", "security"}
        await service.stop()

    async def test_score_agent_no_violation_event_for_high_score(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        violations: list[dict[str, Any]] = []

        async def on_violation(**kw: Any) -> None:
            violations.append(kw)

        service.emitter.on("trust.violation", on_violation)
        result = await service.score_agent(
            agent_id="agent-ok",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        if result.overall_score >= 0.40:
            assert violations == []
        await service.stop()

    async def test_score_agent_violation_event_for_low_score(
        self,
        default_weights: TrustWeights,
        poor_provenance: ProvenanceEvidence,
        poor_behavior: BehaviorEvidence,
        poor_capability: CapabilityEvidence,
        poor_security: SecurityEvidence,
    ) -> None:
        service = AsyncTrustScorer(
            AsyncServiceConfig(name="tf-viol", health_check_interval_seconds=0.0),
            violation_threshold=0.99,
            run_in_executor=False,
        )
        await service.start()
        violations: list[dict[str, Any]] = []

        async def on_violation(**kw: Any) -> None:
            violations.append(kw)

        service.emitter.on("trust.violation", on_violation)
        result = await service.score_agent(
            agent_id="agent-viol",
            weights=default_weights,
            provenance=poor_provenance,
            behavior=poor_behavior,
            capability=poor_capability,
            security=poor_security,
        )
        # With threshold=0.99, anything below triggers a violation.
        if result.overall_score < 0.99:
            assert len(violations) == 1
            assert violations[0]["agent_id"] == "agent-viol"
            assert "threshold" in violations[0]
        await service.stop()

    async def test_score_agent_increments_request_count(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        await service.score_agent(
            agent_id="a",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert service.status.request_count == 1
        await service.score_agent(
            agent_id="a",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert service.status.request_count == 2
        await service.stop()

    async def test_score_provenance_async_returns_dimension_score(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        dim_score = await service.score_provenance_async(
            "agent-prov", default_weights, good_provenance
        )
        assert isinstance(dim_score, DimensionScore)
        assert dim_score.dimension == TrustDimension.provenance
        await service.stop()

    async def test_score_behavior_async_returns_dimension_score(
        self,
        default_weights: TrustWeights,
        good_behavior: BehaviorEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        dim_score = await service.score_behavior_async(
            "agent-behav", default_weights, good_behavior
        )
        assert isinstance(dim_score, DimensionScore)
        assert dim_score.dimension == TrustDimension.behavior
        await service.stop()

    async def test_score_capability_async_returns_dimension_score(
        self,
        default_weights: TrustWeights,
        good_capability: CapabilityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        dim_score = await service.score_capability_async(
            "agent-cap", default_weights, good_capability
        )
        assert isinstance(dim_score, DimensionScore)
        assert dim_score.dimension == TrustDimension.capability
        await service.stop()

    async def test_score_security_async_returns_dimension_score(
        self,
        default_weights: TrustWeights,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        dim_score = await service.score_security_async(
            "agent-sec", default_weights, good_security
        )
        assert isinstance(dim_score, DimensionScore)
        assert dim_score.dimension == TrustDimension.security
        await service.stop()

    async def test_trust_updated_event_has_required_keys(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        service = _make_scorer()
        await service.start()
        events: list[dict[str, Any]] = []

        async def on_updated(**kw: Any) -> None:
            events.append(kw)

        service.emitter.on("trust.updated", on_updated)
        await service.score_agent(
            agent_id="agent-keys",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        for event in events:
            assert "agent_id" in event
            assert "dimension" in event
            assert "score" in event
            assert "confidence" in event
        await service.stop()


# ===========================================================================
# 2. store.py — TrustStore
# ===========================================================================


def _make_trust_score(agent_id: str = "agent-1", overall: float = 0.75) -> TrustScore:
    """Return a TrustScore for testing."""
    from aumai_trustforge.core import TrustScorer
    weights = TrustWeights()
    scorer = TrustScorer(weights)
    prov = scorer.score_provenance(ProvenanceEvidence(model_card_present=True))
    behav = scorer.score_behavior(BehaviorEvidence(error_rate=0.05, sample_count=100))
    cap = scorer.score_capability(CapabilityEvidence(
        claimed_capabilities=["summarize"],
        verified_capabilities=["summarize"],
        verification_method="benchmark_suite",
    ))
    sec = scorer.score_security(SecurityEvidence(sandbox_compliant=True))
    return scorer.compute_trust(prov, behav, cap, sec, agent_id=agent_id)


class TestTrustRecord:
    def test_trust_record_default_id_is_uuid(self) -> None:
        record = TrustRecord(agent_id="agent-1")
        assert len(record.id) == 36
        assert record.id.count("-") == 4

    def test_trust_record_model_validator_coerces_dict_score_json(self) -> None:
        payload_dict = {"agent_id": "x", "overall_score": 0.8}
        record = TrustRecord(
            agent_id="x",
            score_json=payload_dict,  # type: ignore[arg-type]
        )
        assert isinstance(record.score_json, str)
        assert json.loads(record.score_json) == payload_dict

    def test_trust_record_score_json_string_unchanged(self) -> None:
        payload = '{"agent_id": "x"}'
        record = TrustRecord(agent_id="x", score_json=payload)
        assert record.score_json == payload

    def test_trust_record_grade_field_stored(self) -> None:
        record = TrustRecord(agent_id="x", grade="A")
        assert record.grade == "A"


class TestTrustStoreLifecycle:
    async def test_memory_factory_creates_instance(self) -> None:
        store = TrustStore.memory()
        assert isinstance(store, TrustStore)

    async def test_initialize_succeeds(self) -> None:
        store = TrustStore.memory()
        await store.initialize()
        await store.close()

    async def test_context_manager_initializes_and_closes(self) -> None:
        async with TrustStore.memory() as store:
            assert store._repo is not None

    async def test_operations_before_initialize_raise(self) -> None:
        store = TrustStore.memory()
        with pytest.raises(RuntimeError, match="not been initialised"):
            await store.get_all()

    async def test_close_can_be_called_before_initialize(self) -> None:
        store = TrustStore.memory()
        # Should not raise.
        await store.close()


class TestTrustStoreCRUD:
    async def test_save_score_returns_trust_record(self) -> None:
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("agent-save")
            record = await store.save_score(trust_score)
            assert isinstance(record, TrustRecord)
            assert record.agent_id == "agent-save"

    async def test_save_score_persists_overall_score(self) -> None:
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("agent-score")
            record = await store.save_score(trust_score)
            assert record.overall_score == pytest.approx(trust_score.overall_score)

    async def test_save_score_assigns_grade(self) -> None:
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("agent-grade")
            record = await store.save_score(trust_score)
            assert record.grade in ("A", "B", "C", "D", "F")

    async def test_save_score_serializes_json(self) -> None:
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("agent-json")
            record = await store.save_score(trust_score)
            assert isinstance(record.score_json, str)
            parsed = json.loads(record.score_json)
            assert parsed["agent_id"] == "agent-json"

    async def test_get_by_id_returns_saved_record(self) -> None:
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("agent-get")
            record = await store.save_score(trust_score)
            fetched = await store.get_by_id(record.id)
            assert fetched is not None
            assert fetched.id == record.id

    async def test_get_by_id_returns_none_for_missing(self) -> None:
        async with TrustStore.memory() as store:
            fetched = await store.get_by_id("nonexistent-uuid")
            assert fetched is None

    async def test_get_all_returns_all_saved_records(self) -> None:
        async with TrustStore.memory() as store:
            for i in range(3):
                await store.save_score(_make_trust_score(f"agent-{i}"))
            all_records = await store.get_all()
            assert len(all_records) == 3

    async def test_get_agent_history_filters_by_agent_id(self) -> None:
        async with TrustStore.memory() as store:
            await store.save_score(_make_trust_score("agent-alpha"))
            await store.save_score(_make_trust_score("agent-alpha"))
            await store.save_score(_make_trust_score("agent-beta"))
            history = await store.get_agent_history("agent-alpha")
            assert len(history) == 2
            assert all(r.agent_id == "agent-alpha" for r in history)

    async def test_get_agent_history_is_sorted_newest_first(self) -> None:
        async with TrustStore.memory() as store:
            for _ in range(3):
                await store.save_score(_make_trust_score("agent-sorted"))
            history = await store.get_agent_history("agent-sorted")
            timestamps = [r.timestamp for r in history]
            assert timestamps == sorted(timestamps, reverse=True)

    async def test_get_agent_history_respects_limit(self) -> None:
        async with TrustStore.memory() as store:
            for _ in range(10):
                await store.save_score(_make_trust_score("agent-limit"))
            history = await store.get_agent_history("agent-limit", limit=3)
            assert len(history) == 3

    async def test_get_agent_history_empty_for_unknown_agent(self) -> None:
        async with TrustStore.memory() as store:
            history = await store.get_agent_history("nobody")
            assert history == []

    async def test_get_by_grade_filters_correctly(self) -> None:
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("grade-test")
            record = await store.save_score(trust_score)
            by_grade = await store.get_by_grade(record.grade)
            assert any(r.id == record.id for r in by_grade)

    async def test_score_json_round_trips_cleanly(self) -> None:
        """Verify the model_validator handles JSON auto-parse correctly."""
        async with TrustStore.memory() as store:
            trust_score = _make_trust_score("agent-rt")
            record = await store.save_score(trust_score)
            # Re-fetch from store (triggers potential JSON auto-parse).
            fetched = await store.get_by_id(record.id)
            assert fetched is not None
            assert isinstance(fetched.score_json, str)
            parsed = json.loads(fetched.score_json)
            assert parsed["agent_id"] == "agent-rt"


class TestTrustStoreViolations:
    async def test_get_violations_filters_below_threshold(self) -> None:
        async with TrustStore.memory() as store:
            # Save a low-score record by directly inserting TrustRecord.
            low_record = TrustRecord(
                agent_id="agent-low",
                overall_score=0.10,
                grade="F",
            )
            await store._repo.save(low_record)  # type: ignore[union-attr]
            high_record = TrustRecord(
                agent_id="agent-high",
                overall_score=0.90,
                grade="A",
            )
            await store._repo.save(high_record)  # type: ignore[union-attr]
            violations = await store.get_violations(threshold=0.40)
            assert all(v.overall_score < 0.40 for v in violations)
            assert any(v.agent_id == "agent-low" for v in violations)


class TestTrustStoreMetrics:
    async def test_metrics_empty_store(self) -> None:
        async with TrustStore.memory() as store:
            metrics = await store.get_metrics()
            assert isinstance(metrics, TrustStoreMetrics)
            assert metrics.total == 0
            assert metrics.avg_overall_score is None
            assert metrics.violation_rate is None

    async def test_metrics_total_count(self) -> None:
        async with TrustStore.memory() as store:
            for i in range(4):
                await store.save_score(_make_trust_score(f"agent-{i}"))
            metrics = await store.get_metrics()
            assert metrics.total == 4

    async def test_metrics_avg_overall_score_is_float(self) -> None:
        async with TrustStore.memory() as store:
            for i in range(3):
                await store.save_score(_make_trust_score(f"agent-{i}"))
            metrics = await store.get_metrics()
            assert metrics.avg_overall_score is not None
            assert 0.0 <= metrics.avg_overall_score <= 1.0

    async def test_metrics_score_distribution_has_grade_keys(self) -> None:
        async with TrustStore.memory() as store:
            for i in range(2):
                await store.save_score(_make_trust_score(f"agent-{i}"))
            metrics = await store.get_metrics()
            assert isinstance(metrics.score_distribution, dict)
            total_in_dist = sum(metrics.score_distribution.values())
            assert total_in_dist == metrics.total

    async def test_metrics_agents_scored_distinct_agents(self) -> None:
        async with TrustStore.memory() as store:
            await store.save_score(_make_trust_score("alpha"))
            await store.save_score(_make_trust_score("alpha"))
            await store.save_score(_make_trust_score("beta"))
            metrics = await store.get_metrics()
            assert metrics.agents_scored == 2

    async def test_metrics_violation_rate_in_range(self) -> None:
        async with TrustStore.memory() as store:
            for i in range(3):
                await store.save_score(_make_trust_score(f"agent-{i}"))
            metrics = await store.get_metrics()
            assert metrics.violation_rate is not None
            assert 0.0 <= metrics.violation_rate <= 1.0

    async def test_trust_store_config_model(self) -> None:
        config = TrustStoreConfig(database_url="sqlite:///test.db", backend="sqlite")
        assert config.backend == "sqlite"
        assert "test.db" in config.database_url


# ===========================================================================
# 3. llm_assessor.py — LLMTrustAssessor
# ===========================================================================


class TestLLMTrustAssessmentModel:
    def test_defaults_are_medium_risk(self) -> None:
        assessment = LLMTrustAssessment()
        assert assessment.trust_level == "medium"

    def test_risk_score_none_is_zero(self) -> None:
        assessment = LLMTrustAssessment(trust_level="none")
        assert assessment.risk_score() == pytest.approx(0.0)

    def test_risk_score_critical_is_one(self) -> None:
        assessment = LLMTrustAssessment(trust_level="critical")
        assert assessment.risk_score() == pytest.approx(1.0)

    def test_risk_score_medium_is_half(self) -> None:
        assessment = LLMTrustAssessment(trust_level="medium")
        assert assessment.risk_score() == pytest.approx(0.50)

    def test_suggested_trust_score_inverse_of_risk(self) -> None:
        assessment = LLMTrustAssessment(trust_level="none")
        assert assessment.suggested_trust_score() == pytest.approx(1.0)

    def test_suggested_trust_score_critical_is_zero(self) -> None:
        assessment = LLMTrustAssessment(trust_level="critical")
        assert assessment.suggested_trust_score() == pytest.approx(0.0)

    def test_llm_powered_default_true(self) -> None:
        assessment = LLMTrustAssessment()
        assert assessment.llm_powered is True


class TestLLMTrustAssessorNoClient:
    async def test_assess_no_client_returns_heuristic(self) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="agent-noc")
        assert isinstance(result, LLMTrustAssessment)
        assert result.llm_powered is False

    async def test_assess_no_evidence_returns_medium_uncertainty(self) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="agent-noev")
        assert result.trust_level == "medium"
        assert result.confidence == "very_low"

    async def test_assess_good_evidence_low_risk(
        self,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(
            agent_id="agent-good",
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert result.trust_level in ("none", "low", "medium")

    async def test_assess_poor_provenance_adds_risk_factors(
        self, poor_provenance: ProvenanceEvidence
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="a", provenance=poor_provenance)
        assert len(result.risk_factors) > 0

    async def test_assess_high_error_rate_adds_risk(
        self, poor_behavior: BehaviorEvidence
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="a", behavior=poor_behavior)
        assert any("error rate" in rf.lower() for rf in result.risk_factors)

    async def test_assess_no_audit_adds_risk_factor(
        self, poor_security: SecurityEvidence
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="a", security=poor_security)
        assert any("audit" in rf.lower() or "sandbox" in rf.lower()
                   for rf in result.risk_factors)

    async def test_assess_missing_capabilities_adds_risk(
        self, poor_capability: CapabilityEvidence
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="a", capability=poor_capability)
        # No capabilities = risk factor.
        assert len(result.risk_factors) > 0

    async def test_assess_good_evidence_includes_dimensions(
        self,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(
            agent_id="a",
            provenance=good_provenance,
            behavior=good_behavior,
        )
        assert "provenance" in result.assessed_dimensions
        assert "behavior" in result.assessed_dimensions

    async def test_assess_heuristic_summary_is_non_empty(
        self, good_provenance: ProvenanceEvidence
    ) -> None:
        assessor = LLMTrustAssessor(client=None)
        result = await assessor.assess(agent_id="a", provenance=good_provenance)
        assert len(result.summary) > 0


class TestLLMTrustAssessorWithMockProvider:
    async def test_mock_assessor_returns_llm_powered_true(self) -> None:
        assessor = build_mock_assessor()
        result = await assessor.assess(agent_id="agent-mock")
        assert result.llm_powered is True

    async def test_mock_assessor_default_returns_none_risk(self) -> None:
        assessor = build_mock_assessor()
        result = await assessor.assess(agent_id="agent-mock")
        assert result.trust_level == "none"

    async def test_mock_assessor_custom_response_high_risk(self) -> None:
        response = json.dumps({
            "trust_level": "high",
            "confidence": "high",
            "risk_factors": ["Missing model card", "High error rate"],
            "recommendations": ["Add model card", "Reduce error rate"],
            "assessed_dimensions": ["provenance", "behavior"],
            "summary": "High risk agent.",
        })
        assessor = build_mock_assessor(responses=[response])
        result = await assessor.assess(agent_id="agent-risky")
        assert result.trust_level == "high"
        assert result.confidence == "high"
        assert len(result.risk_factors) == 2

    async def test_mock_assessor_recommendations_parsed(self) -> None:
        response = json.dumps({
            "trust_level": "low",
            "confidence": "medium",
            "risk_factors": ["Factor A"],
            "recommendations": ["Do X", "Do Y"],
            "assessed_dimensions": ["security"],
            "summary": "Low risk.",
        })
        assessor = build_mock_assessor(responses=[response])
        result = await assessor.assess(agent_id="a")
        assert result.recommendations == ["Do X", "Do Y"]

    async def test_mock_assessor_assessed_dimensions_parsed(self) -> None:
        response = json.dumps({
            "trust_level": "none",
            "confidence": "very_high",
            "risk_factors": [],
            "recommendations": [],
            "assessed_dimensions": ["provenance", "behavior", "capability", "security"],
            "summary": "All good.",
        })
        assessor = build_mock_assessor(responses=[response])
        result = await assessor.assess(agent_id="a")
        assert len(result.assessed_dimensions) == 4

    async def test_mock_assessor_invalid_json_falls_back_gracefully(self) -> None:
        assessor = build_mock_assessor(responses=["NOT VALID JSON {{{"])
        result = await assessor.assess(agent_id="a")
        # Should return a fallback response without crashing.
        assert isinstance(result, LLMTrustAssessment)

    async def test_build_mock_assessor_none_responses_uses_default(self) -> None:
        assessor = build_mock_assessor(responses=None)
        result = await assessor.assess(agent_id="a")
        assert result.trust_level == "none"

    async def test_assessor_with_extra_context(
        self, good_provenance: ProvenanceEvidence
    ) -> None:
        assessor = build_mock_assessor()
        result = await assessor.assess(
            agent_id="a",
            provenance=good_provenance,
            extra_context={"deployment": "production", "region": "us-east-1"},
        )
        assert isinstance(result, LLMTrustAssessment)

    async def test_assessor_with_weights(
        self,
        good_provenance: ProvenanceEvidence,
        default_weights: TrustWeights,
    ) -> None:
        assessor = build_mock_assessor()
        result = await assessor.assess(
            agent_id="a",
            provenance=good_provenance,
            weights=default_weights,
        )
        assert isinstance(result, LLMTrustAssessment)


# ===========================================================================
# 4. integration.py — TrustForgeIntegration
# ===========================================================================


class TestTrustForgeIntegrationRegistration:
    async def test_from_aumos_factory_creates_instance(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        assert isinstance(integration, TrustForgeIntegration)

    async def test_not_registered_before_register(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        assert integration.is_registered is False

    async def test_register_marks_as_registered(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        await integration.register()
        assert integration.is_registered is True

    async def test_register_is_idempotent(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        await integration.register()
        await integration.register()  # second call should not raise
        assert integration.is_registered is True

    async def test_register_publishes_service_info(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        await integration.register()
        # Service should be discoverable in the AumOS registry.
        service = hub.get_service("trustforge")
        assert service is not None
        assert service.name == "trustforge"

    async def test_unregister_sets_not_registered(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        await integration.register()
        await integration.unregister()
        assert integration.is_registered is False

    async def test_aumos_property_returns_hub(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        assert integration.aumos is hub

    async def test_setup_trustforge_convenience_function(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        assert isinstance(integration, TrustForgeIntegration)
        assert integration.is_registered is True


class TestTrustForgeIntegrationScoreAndPublish:
    async def test_score_and_publish_returns_trust_score(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        result = await integration.score_and_publish(
            agent_id="agent-int",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert isinstance(result, TrustScore)
        assert result.agent_id == "agent-int"

    async def test_score_and_publish_emits_trust_scored_event(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        captured: list[Event] = []

        async def on_scored(event: Event) -> None:
            captured.append(event)

        hub.events.subscribe("trust.scored", on_scored, subscriber="test")
        await integration.score_and_publish(
            agent_id="agent-ev",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        assert len(captured) == 1
        assert captured[0].data["agent_id"] == "agent-ev"
        assert "overall_score" in captured[0].data
        assert "grade" in captured[0].data

    async def test_score_and_publish_emits_trust_updated_per_dimension(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        updates: list[Event] = []

        async def on_updated(event: Event) -> None:
            updates.append(event)

        hub.events.subscribe("trust.updated", on_updated, subscriber="test")
        await integration.score_and_publish(
            agent_id="agent-upd",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        # 4 dimensions = 4 trust.updated events.
        assert len(updates) == 4
        dimensions_seen = {e.data["dimension"] for e in updates}
        assert dimensions_seen == {"provenance", "behavior", "capability", "security"}

    async def test_score_and_publish_violation_event_low_score(
        self,
        default_weights: TrustWeights,
        poor_provenance: ProvenanceEvidence,
        poor_behavior: BehaviorEvidence,
        poor_capability: CapabilityEvidence,
        poor_security: SecurityEvidence,
    ) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration(hub)
        # Override threshold to near-1 to guarantee a violation.
        integration.VIOLATION_THRESHOLD = 0.99
        await integration.register()

        violations: list[Event] = []

        async def on_violation(event: Event) -> None:
            violations.append(event)

        hub.events.subscribe("trust.violation", on_violation, subscriber="test")
        result = await integration.score_and_publish(
            agent_id="agent-viol",
            weights=default_weights,
            provenance=poor_provenance,
            behavior=poor_behavior,
            capability=poor_capability,
            security=poor_security,
        )
        if result.overall_score < 0.99:
            assert len(violations) == 1
            assert violations[0].data["agent_id"] == "agent-viol"

    async def test_trust_updated_event_has_required_payload_keys(
        self,
        default_weights: TrustWeights,
        good_provenance: ProvenanceEvidence,
        good_behavior: BehaviorEvidence,
        good_capability: CapabilityEvidence,
        good_security: SecurityEvidence,
    ) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        events: list[Event] = []

        async def on_updated(event: Event) -> None:
            events.append(event)

        hub.events.subscribe("trust.updated", on_updated, subscriber="test")
        await integration.score_and_publish(
            agent_id="agent-keys",
            weights=default_weights,
            provenance=good_provenance,
            behavior=good_behavior,
            capability=good_capability,
            security=good_security,
        )
        for event in events:
            assert "agent_id" in event.data
            assert "dimension" in event.data
            assert "score" in event.data
            assert "confidence" in event.data


class TestTrustForgeIntegrationCapabilityCache:
    async def test_capability_cache_empty_initially(self) -> None:
        hub = AumOS()
        integration = TrustForgeIntegration.from_aumos(hub)
        assert integration.get_cached_capabilities("any-agent") == []

    async def test_handle_capability_updated_caches_capabilities(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-cap",
            capabilities=["summarize", "translate"],
        )
        cached = integration.get_cached_capabilities("agent-cap")
        assert cached == ["summarize", "translate"]

    async def test_handle_capability_updated_overwrites_cache(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-cap",
            capabilities=["summarize"],
        )
        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-cap",
            capabilities=["summarize", "translate", "classify"],
        )
        cached = integration.get_cached_capabilities("agent-cap")
        assert len(cached) == 3

    async def test_invalid_capabilities_payload_skips_cache_update(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-bad",
            capabilities="not-a-list",  # invalid — should be list
        )
        # Cache should remain empty for this agent.
        cached = integration.get_cached_capabilities("agent-bad")
        assert cached == []

    async def test_clear_capability_cache_empties_cache(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-x",
            capabilities=["skill1"],
        )
        integration.clear_capability_cache()
        assert integration.get_cached_capabilities("agent-x") == []

    async def test_capability_cache_property_returns_snapshot(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-snap",
            capabilities=["a", "b"],
        )
        snapshot = integration.capability_cache
        assert "agent-snap" in snapshot
        assert snapshot["agent-snap"] == ["a", "b"]

    async def test_capability_cache_distinct_agents(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="alpha",
            capabilities=["skill-alpha"],
        )
        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="beta",
            capabilities=["skill-beta"],
        )
        assert integration.get_cached_capabilities("alpha") == ["skill-alpha"]
        assert integration.get_cached_capabilities("beta") == ["skill-beta"]

    async def test_unregister_then_capability_event_not_received(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        await integration.unregister()

        await hub.events.publish_simple(
            "agent.capability_updated",
            source="test",
            agent_id="agent-unreg",
            capabilities=["skill"],
        )
        # After unregister the subscription is removed; cache should stay empty.
        cached = integration.get_cached_capabilities("agent-unreg")
        assert cached == []


class TestTrustForgeIntegrationServiceMetadata:
    async def test_service_name_constant(self) -> None:
        assert TrustForgeIntegration.SERVICE_NAME == "trustforge"

    async def test_registered_service_has_capabilities(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        service = hub.get_service("trustforge")
        assert service is not None
        assert "trust-scoring" in service.capabilities

    async def test_registered_service_status_is_healthy(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        service = hub.get_service("trustforge")
        assert service is not None
        assert service.status == "healthy"

    async def test_registered_service_version(self) -> None:
        hub = AumOS()
        integration = await setup_trustforge(hub)
        service = hub.get_service("trustforge")
        assert service is not None
        assert service.version == "0.1.0"
