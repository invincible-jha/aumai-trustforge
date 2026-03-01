"""Persistence layer for aumai-trustforge using aumai-store foundation library.

Provides TrustStore — a repository-backed persistence service for trust score
records and history — and TrustRecord, the Pydantic model persisted to SQLite
(or an in-memory backend during tests).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from aumai_store import Repository, Store, StoreConfig
from pydantic import BaseModel, Field, model_validator

from aumai_trustforge.models import TrustScore


class TrustRecord(BaseModel):
    """Persisted representation of a single agent trust score result.

    Attributes:
        id: Unique identifier for this trust record (UUID v4 string).
        agent_id: Identifier of the agent that was scored.
        timestamp: UTC ISO-8601 datetime string when the score was computed.
        overall_score: Weighted overall trust score in [0, 1].
        grade: Letter grade derived from the overall score.
        score_json: Full JSON-serialised :class:`~aumai_trustforge.models.TrustScore`.
            When read back from the store the backend may deserialise the JSON
            string into a dict; the validator re-serialises it to ensure this
            field is always a string.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    grade: str = Field(default="F")
    score_json: str = Field(default="{}")

    @model_validator(mode="before")
    @classmethod
    def _coerce_score_json(cls, values: Any) -> Any:
        """Re-serialise score_json when the store returns it as a dict.

        The aumai-store memory backend parses any JSON-string value that
        starts with ``{`` or ``[`` back into a Python object before handing
        it to Pydantic.  This validator ensures ``score_json`` is always a
        ``str`` regardless of the roundtrip.
        """
        if isinstance(values, dict):
            sj = values.get("score_json")
            if sj is not None and not isinstance(sj, str):
                values["score_json"] = json.dumps(sj)
        return values


class TrustStoreConfig(BaseModel):
    """Configuration for :class:`TrustStore`.

    Attributes:
        database_url: Backend connection URL.
        backend: Storage backend — ``"memory"`` or ``"sqlite"``.
        table_prefix: Prefix applied to all table names.
    """

    database_url: str = "sqlite:///aumai_trustforge.db"
    backend: str = "sqlite"
    table_prefix: str = ""

    model_config = {"frozen": False}


class TrustStoreMetrics(BaseModel):
    """Aggregate metrics computed across all stored trust records.

    Attributes:
        total: Total number of records.
        avg_overall_score: Mean overall score across all records (``None``
            when no records exist).
        score_distribution: Count of records per letter grade
            (``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"F"``).
        violation_rate: Fraction of records with grade ``"F"`` (``None``
            when no records exist).
        agents_scored: Number of distinct agent IDs.
    """

    total: int = 0
    avg_overall_score: float | None = None
    score_distribution: dict[str, int] = Field(default_factory=dict)
    violation_rate: float | None = None
    agents_scored: int = 0


class TrustStore:
    """Repository-backed store for agent trust score records.

    Wraps a :class:`~aumai_store.core.Store` and exposes domain-specific
    query methods for trust history and metrics.

    Use :meth:`memory` to create an in-memory instance suitable for unit
    tests.  For production pass a :class:`TrustStoreConfig` pointing at a
    SQLite database.

    Example::

        async with TrustStore.memory() as trust_store:
            record = await trust_store.save_score(trust_score)
            history = await trust_store.get_agent_history("agent-1")
            metrics = await trust_store.get_metrics()
    """

    def __init__(self, store: Store) -> None:
        """Initialise using an existing :class:`~aumai_store.core.Store`.

        Args:
            store: A configured (but not yet necessarily initialised) store.
        """
        self._store: Store = store
        self._repo: Repository[TrustRecord] | None = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def memory(cls) -> "TrustStore":
        """Create an in-memory TrustStore for testing.

        Returns:
            A :class:`TrustStore` backed by
            :class:`~aumai_store.backends.MemoryBackend`.
        """
        return cls(Store.memory())

    @classmethod
    def sqlite(cls, database_url: str = "sqlite:///aumai_trustforge.db") -> "TrustStore":
        """Create a SQLite-backed TrustStore.

        Args:
            database_url: SQLite connection URL, e.g. ``"sqlite:///trust.db"``.

        Returns:
            A :class:`TrustStore` backed by
            :class:`~aumai_store.backends.SQLiteBackend`.
        """
        config = StoreConfig(backend="sqlite", database_url=database_url)
        return cls(Store(config))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the backend connection and ensure the trust table exists.

        Must be called before any data operations.  Idempotent — safe to call
        multiple times.
        """
        await self._store.initialize()
        repo: Repository[TrustRecord] = self._store.repository(TrustRecord)
        await repo.ensure_table()
        self._repo = repo

    async def close(self) -> None:
        """Close the underlying store connection."""
        await self._store.close()

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "TrustStore":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save_score(self, trust_score: TrustScore) -> TrustRecord:
        """Persist a TrustScore and return the saved record.

        Args:
            trust_score: The completed :class:`~aumai_trustforge.models.TrustScore`.

        Returns:
            The persisted :class:`TrustRecord` (with assigned ``id``).

        Raises:
            RuntimeError: If the store has not been initialised.
        """
        self._assert_initialized()
        record = TrustRecord(
            agent_id=trust_score.agent_id,
            timestamp=trust_score.timestamp.isoformat(),
            overall_score=trust_score.overall_score,
            grade=trust_score.grade(),
            score_json=trust_score.model_dump_json(),
        )
        assigned_id = await self._repo.save(record)  # type: ignore[union-attr]
        record = record.model_copy(update={"id": assigned_id})
        return record

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def get_agent_history(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> list[TrustRecord]:
        """Return the N most-recent trust records for an agent, newest first.

        Args:
            agent_id: The agent ID to look up.
            limit: Maximum number of records to return (default ``50``).

        Returns:
            List of :class:`TrustRecord` instances limited to *limit* entries,
            sorted newest first.
        """
        self._assert_initialized()
        records = await self._repo.find(agent_id=agent_id)  # type: ignore[union-attr]
        sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
        return sorted_records[:limit]

    async def get_by_grade(self, grade: str) -> list[TrustRecord]:
        """Return all trust records where the grade matches *grade*.

        Args:
            grade: One of ``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"F"``.

        Returns:
            List of matching :class:`TrustRecord` instances.
        """
        self._assert_initialized()
        return await self._repo.find(grade=grade)  # type: ignore[union-attr]

    async def get_by_id(self, record_id: str) -> TrustRecord | None:
        """Fetch a single trust record by its primary key.

        Args:
            record_id: UUID string assigned during :meth:`save_score`.

        Returns:
            The :class:`TrustRecord`, or ``None`` if not found.
        """
        self._assert_initialized()
        return await self._repo.get(record_id)  # type: ignore[union-attr]

    async def get_all(self) -> list[TrustRecord]:
        """Return every trust record stored in the backend.

        Returns:
            All :class:`TrustRecord` instances.
        """
        self._assert_initialized()
        return await self._repo.find()  # type: ignore[union-attr]

    async def get_violations(self, threshold: float = 0.40) -> list[TrustRecord]:
        """Return records where the overall score is below *threshold*.

        Since the store backend does not support range queries, this method
        fetches all records and filters in Python.

        Args:
            threshold: Score below which a record is considered a violation.
                Default matches the ``AsyncTrustScorer`` default of ``0.40``.

        Returns:
            List of :class:`TrustRecord` instances with
            ``overall_score < threshold``.
        """
        self._assert_initialized()
        all_records = await self.get_all()
        return [r for r in all_records if r.overall_score < threshold]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_metrics(self) -> TrustStoreMetrics:
        """Compute aggregate metrics across all stored trust records.

        Returns:
            A :class:`TrustStoreMetrics` snapshot.
        """
        all_records = await self.get_all()
        if not all_records:
            return TrustStoreMetrics()

        total = len(all_records)
        score_sum = sum(r.overall_score for r in all_records)
        avg_score = score_sum / total

        grade_counts: dict[str, int] = {}
        for record in all_records:
            grade_counts[record.grade] = grade_counts.get(record.grade, 0) + 1

        violation_count = sum(1 for r in all_records if r.grade == "F")
        violation_rate = violation_count / total

        agent_ids = {r.agent_id for r in all_records}

        return TrustStoreMetrics(
            total=total,
            avg_overall_score=round(avg_score, 4),
            score_distribution=grade_counts,
            violation_rate=round(violation_rate, 4),
            agents_scored=len(agent_ids),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        """Raise if :meth:`initialize` has not been called."""
        if self._repo is None:
            raise RuntimeError(
                "TrustStore has not been initialised. "
                "Call await trust_store.initialize() or use it as an async context manager."
            )


__all__ = ["TrustRecord", "TrustStore", "TrustStoreConfig", "TrustStoreMetrics"]
