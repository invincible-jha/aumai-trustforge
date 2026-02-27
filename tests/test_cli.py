"""Tests for aumai_trustforge.cli.

Uses Click's CliRunner for isolated, filesystem-independent invocation.
All filesystem interaction is routed through tmp_path to avoid side effects.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_trustforge.cli import (
    _load_evidence,
    _load_weights,
    _score_bar,
    _score_color,
    main,
)
from aumai_trustforge.models import (
    BehaviorEvidence,
    ProvenanceEvidence,
    TrustScore,
    TrustWeights,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _write_json(path: Path, data: object) -> None:
    """Write *data* as JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_agent_dir(tmp_path: Path, **evidence: object) -> Path:
    """Create a minimal agent directory with the given evidence files.

    Keyword arguments map filename stems to evidence dicts, e.g.::

        _make_agent_dir(tmp_path, provenance={"model_card_present": True})
    """
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()
    for stem, data in evidence.items():
        _write_json(agent_dir / f"{stem}.json", data)
    return agent_dir


# ===========================================================================
# Version flag (pre-existing stub preserved)
# ===========================================================================


def test_cli_version() -> None:
    """Version flag must report 0.1.0."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# ===========================================================================
# score — help text
# ===========================================================================


class TestScoreCommandHelp:
    """score --help should succeed and mention evidence files."""

    def test_score_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--help"])
        assert result.exit_code == 0

    def test_score_help_mentions_evidence_files(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--help"])
        assert "provenance.json" in result.output


# ===========================================================================
# score — text output (default)
# ===========================================================================


class TestScoreCommandTextOutput:
    """score command with --output text (default)."""

    def test_score_empty_agent_dir_exits_zero(self, tmp_path: Path) -> None:
        """An agent dir with no evidence files should use defaults and still exit 0."""
        agent_dir = tmp_path / "empty-agent"
        agent_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0

    def test_score_output_contains_agent_id(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["score", "--agent-dir", str(agent_dir), "--agent-id", "my-special-agent"],
        )
        assert result.exit_code == 0
        assert "my-special-agent" in result.output

    def test_score_default_agent_id_uses_dir_name(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0
        assert agent_dir.name in result.output

    def test_score_output_contains_overall_score(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(
            tmp_path,
            provenance={"model_card_present": True, "license_verified": True},
        )
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0
        assert "Overall" in result.output

    def test_score_output_contains_all_dimension_names(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0
        for dim in ["PROVENANCE", "BEHAVIOR", "CAPABILITY", "SECURITY"]:
            assert dim in result.output

    def test_score_with_full_perfect_evidence_text(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(
            tmp_path,
            provenance={
                "model_card_present": True,
                "license_verified": True,
                "author_verified": True,
                "source_url": "https://example.com",
            },
            behavior={
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "uptime_pct": 100.0,
                "sample_count": 1000,
            },
            capability={
                "claimed_capabilities": ["a", "b"],
                "verified_capabilities": ["a", "b"],
                "verification_method": "benchmark_suite",
            },
            security={
                "sandbox_compliant": True,
                "vulnerability_count": 0,
                "last_audit_date": str(datetime.date.today()),
            },
        )
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0
        assert "Grade" in result.output


# ===========================================================================
# score — JSON output
# ===========================================================================


class TestScoreCommandJsonOutput:
    """score --output json must produce parseable TrustScore JSON."""

    def test_score_json_output_is_valid_json(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["score", "--agent-dir", str(agent_dir), "--output", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "overall_score" in payload

    def test_score_json_output_contains_all_dimension_scores(
        self, tmp_path: Path
    ) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["score", "--agent-dir", str(agent_dir), "--output", "json"]
        )
        payload = json.loads(result.output)
        assert set(payload["dimension_scores"].keys()) == {
            "provenance",
            "behavior",
            "capability",
            "security",
        }

    def test_score_json_overall_score_in_unit_interval(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(
            tmp_path,
            provenance={"model_card_present": True},
        )
        runner = CliRunner()
        result = runner.invoke(
            main, ["score", "--agent-dir", str(agent_dir), "--output", "json"]
        )
        payload = json.loads(result.output)
        assert 0.0 <= payload["overall_score"] <= 1.0

    def test_score_json_agent_id_matches_override(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                "--agent-dir",
                str(agent_dir),
                "--output",
                "json",
                "--agent-id",
                "override-id",
            ],
        )
        payload = json.loads(result.output)
        assert payload["agent_id"] == "override-id"


# ===========================================================================
# score — missing agent dir
# ===========================================================================


class TestScoreCommandMissingAgentDir:
    """score must exit non-zero when --agent-dir does not exist."""

    def test_nonexistent_agent_dir_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["score", "--agent-dir", str(tmp_path / "no-such-dir")]
        )
        assert result.exit_code != 0

    def test_missing_required_flag_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["score"])
        assert result.exit_code != 0


# ===========================================================================
# score — weights file
# ===========================================================================


class TestScoreCommandWeightsFile:
    """score --weights accepts a valid JSON weights file."""

    def test_valid_weights_file_used(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        weights_file = tmp_path / "weights.json"
        _write_json(
            weights_file,
            {"provenance": 0.40, "behavior": 0.30, "capability": 0.20, "security": 0.10},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                "--agent-dir",
                str(agent_dir),
                "--weights",
                str(weights_file),
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "overall_score" in payload

    def test_invalid_weights_file_falls_back_to_equal(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        bad_weights = tmp_path / "bad_weights.json"
        bad_weights.write_text('{"provenance": 0.99, "behavior": 0.99}', encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                "--agent-dir",
                str(agent_dir),
                "--weights",
                str(bad_weights),
                "--output",
                "json",
            ],
        )
        # Should fallback to equal weights and still succeed
        assert result.exit_code == 0

    def test_missing_weights_file_falls_back_to_equal(self, tmp_path: Path) -> None:
        agent_dir = _make_agent_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "score",
                "--agent-dir",
                str(agent_dir),
                "--weights",
                str(tmp_path / "nonexistent.json"),
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0


# ===========================================================================
# score — malformed evidence files
# ===========================================================================


class TestScoreCommandMalformedEvidence:
    """Malformed evidence JSON should be ignored (fallback to defaults)."""

    def test_malformed_provenance_json_falls_back(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "bad-agent"
        agent_dir.mkdir()
        (agent_dir / "provenance.json").write_text("NOT_VALID_JSON", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0

    def test_malformed_behavior_json_falls_back(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "bad-behav-agent"
        agent_dir.mkdir()
        (agent_dir / "behavior.json").write_text("{invalid}", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["score", "--agent-dir", str(agent_dir)])
        assert result.exit_code == 0


# ===========================================================================
# report command
# ===========================================================================


class TestReportCommand:
    """report command reads a saved TrustScore JSON and re-renders it."""

    def _write_trust_score(self, path: Path) -> None:
        ts = TrustScore(agent_id="saved-agent", overall_score=0.75)
        payload = ts.model_dump(mode="json")
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    def test_report_text_output_exits_zero(self, tmp_path: Path) -> None:
        score_file = tmp_path / "score.json"
        self._write_trust_score(score_file)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(score_file)])
        assert result.exit_code == 0

    def test_report_text_contains_agent_id(self, tmp_path: Path) -> None:
        score_file = tmp_path / "score.json"
        self._write_trust_score(score_file)
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(score_file)])
        assert "saved-agent" in result.output

    def test_report_json_output_is_valid(self, tmp_path: Path) -> None:
        score_file = tmp_path / "score.json"
        self._write_trust_score(score_file)
        runner = CliRunner()
        result = runner.invoke(
            main, ["report", "--input", str(score_file), "--format", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["agent_id"] == "saved-agent"
        assert payload["overall_score"] == 0.75

    def test_report_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["report", "--input", str(tmp_path / "nonexistent.json")]
        )
        assert result.exit_code != 0

    def test_report_corrupted_json_exits_nonzero(self, tmp_path: Path) -> None:
        corrupted = tmp_path / "bad.json"
        corrupted.write_text("TOTALLY_BROKEN", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--input", str(corrupted)])
        assert result.exit_code != 0

    def test_report_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["report", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# _load_weights helper
# ===========================================================================


class TestLoadWeights:
    """_load_weights: none -> equal weights, valid file -> custom, bad -> equal."""

    def test_none_returns_equal_weights(self) -> None:
        weights = _load_weights(None)
        assert weights == TrustWeights()

    def test_valid_file_returns_custom_weights(self, tmp_path: Path) -> None:
        weights_file = tmp_path / "w.json"
        _write_json(
            weights_file,
            {"provenance": 0.40, "behavior": 0.30, "capability": 0.20, "security": 0.10},
        )
        weights = _load_weights(str(weights_file))
        assert weights.provenance == 0.40

    def test_nonexistent_file_returns_equal_weights(self, tmp_path: Path) -> None:
        weights = _load_weights(str(tmp_path / "ghost.json"))
        assert weights == TrustWeights()

    def test_invalid_json_file_returns_equal_weights(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{ invalid json", encoding="utf-8")
        weights = _load_weights(str(bad))
        assert weights == TrustWeights()


# ===========================================================================
# _load_evidence helper
# ===========================================================================


class TestLoadEvidence:
    """_load_evidence: missing file -> default, valid -> parsed, bad -> default."""

    def test_missing_file_returns_default_instance(self, tmp_path: Path) -> None:
        evidence = _load_evidence(tmp_path / "missing.json", ProvenanceEvidence)
        assert evidence == ProvenanceEvidence()

    def test_valid_file_returns_parsed_instance(self, tmp_path: Path) -> None:
        path = tmp_path / "provenance.json"
        _write_json(path, {"model_card_present": True, "license_verified": True})
        evidence = _load_evidence(path, ProvenanceEvidence)
        assert evidence.model_card_present is True
        assert evidence.license_verified is True

    def test_malformed_json_returns_default_instance(self, tmp_path: Path) -> None:
        path = tmp_path / "provenance.json"
        path.write_text("BROKEN JSON!", encoding="utf-8")
        evidence = _load_evidence(path, ProvenanceEvidence)
        assert evidence == ProvenanceEvidence()

    def test_invalid_schema_returns_default_instance(self, tmp_path: Path) -> None:
        path = tmp_path / "behavior.json"
        # error_rate must be in [0, 1] — this will fail Pydantic validation
        _write_json(path, {"error_rate": 999.0})
        evidence = _load_evidence(path, BehaviorEvidence)
        assert evidence == BehaviorEvidence()


# ===========================================================================
# _score_bar helper
# ===========================================================================


class TestScoreBar:
    """_score_bar generates ASCII bar of fixed width."""

    def test_zero_score_all_dashes(self) -> None:
        bar = _score_bar(0.0, width=10)
        assert bar == "[" + "-" * 10 + "]"

    def test_full_score_all_hashes(self) -> None:
        bar = _score_bar(1.0, width=10)
        assert bar == "[" + "#" * 10 + "]"

    def test_half_score_half_hashes(self) -> None:
        bar = _score_bar(0.5, width=10)
        assert bar == "[#####-----]"

    def test_bar_total_length_is_correct(self) -> None:
        for width in [5, 10, 20]:
            bar = _score_bar(0.5, width=width)
            # bracket + width chars + bracket
            assert len(bar) == width + 2

    def test_default_width_is_20(self) -> None:
        bar = _score_bar(0.5)
        assert len(bar) == 22  # 2 brackets + 20 chars


# ===========================================================================
# _score_color helper
# ===========================================================================


class TestScoreColor:
    """_score_color returns a valid click color string for any score."""

    @pytest.mark.parametrize(
        ("score", "expected_color"),
        [
            (1.00, "bright_green"),
            (0.85, "bright_green"),
            (0.849, "green"),
            (0.70, "green"),
            (0.699, "yellow"),
            (0.55, "yellow"),
            (0.549, "red"),
            (0.40, "red"),
            (0.399, "bright_red"),
            (0.00, "bright_red"),
        ],
    )
    def test_color_boundaries(self, score: float, expected_color: str) -> None:
        assert _score_color(score) == expected_color

    def test_returns_string_for_any_score_in_range(self) -> None:
        for score in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            color = _score_color(score)
            assert isinstance(color, str)
            assert len(color) > 0
