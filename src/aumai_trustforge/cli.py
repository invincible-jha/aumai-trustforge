"""CLI entry point for aumai-trustforge."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, TypeVar

import click
from pydantic import BaseModel

from aumai_trustforge.core import TrustScorer
from aumai_trustforge.models import (
    BehaviorEvidence,
    CapabilityEvidence,
    ProvenanceEvidence,
    SecurityEvidence,
    TrustDimension,
    TrustScore,
    TrustWeights,
)


@click.group()
@click.version_option(package_name="aumai-trustforge")
def main() -> None:
    """AumAI TrustForge — trust scoring for AI agents across 4 dimensions.

    Use 'aumai-trustforge --help' to see available sub-commands.
    """


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


@main.command("score")
@click.option(
    "--agent-dir",
    "agent_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to agent directory containing evidence files.",
)
@click.option(
    "--weights",
    "weights_file",
    default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="Path to JSON file with custom TrustWeights. Uses equal weights if absent.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
@click.option(
    "--agent-id",
    "agent_id",
    default=None,
    help="Override agent ID (defaults to the agent directory name).",
)
def score_command(
    agent_dir: str,
    weights_file: str | None,
    output_format: str,
    agent_id: str | None,
) -> None:
    """Score an agent's trust across 4 dimensions.

    Reads evidence from JSON files inside *agent_dir*:

    \b
      provenance.json   — ProvenanceEvidence fields
      behavior.json     — BehaviorEvidence fields
      capability.json   — CapabilityEvidence fields
      security.json     — SecurityEvidence fields

    Missing files are treated as empty evidence (lowest possible score).
    """
    agent_path = Path(agent_dir)
    resolved_agent_id = agent_id or agent_path.name

    weights = _load_weights(weights_file)
    scorer = TrustScorer(weights)

    provenance_ev = _load_evidence(agent_path / "provenance.json", ProvenanceEvidence)
    behavior_ev = _load_evidence(agent_path / "behavior.json", BehaviorEvidence)
    capability_ev = _load_evidence(agent_path / "capability.json", CapabilityEvidence)
    security_ev = _load_evidence(agent_path / "security.json", SecurityEvidence)

    prov_score = scorer.score_provenance(provenance_ev)
    behav_score = scorer.score_behavior(behavior_ev)
    cap_score = scorer.score_capability(capability_ev)
    sec_score = scorer.score_security(security_ev)

    trust = scorer.compute_trust(
        prov_score, behav_score, cap_score, sec_score, agent_id=resolved_agent_id
    )

    if output_format == "json":
        click.echo(json.dumps(trust.model_dump(mode="json"), indent=2, default=str))
        return

    _print_trust_report(trust)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@main.command("report")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to a TrustScore JSON file (produced by 'score --output json').",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    show_default=True,
    help="Output format for the report.",
)
def report_command(input_file: str, output_format: str) -> None:
    """Generate a trust report from a saved TrustScore JSON file.

    Reads a previously generated score and formats it for human consumption
    or further processing.
    """
    try:
        raw: Any = json.loads(Path(input_file).read_text(encoding="utf-8"))
        trust = TrustScore.model_validate(raw)
    except Exception as exc:
        click.echo(click.style(f"error loading trust score: {exc}", fg="red"), err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(trust.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_trust_report(trust)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_weights(weights_file: str | None) -> TrustWeights:
    """Load TrustWeights from a JSON file or return equal weights."""
    if weights_file is None:
        return TrustWeights()

    weights_path = Path(weights_file)
    if not weights_path.exists():
        click.echo(
            click.style(
                f"weights file '{weights_file}' not found; using equal weights",
                fg="yellow",
            ),
            err=True,
        )
        return TrustWeights()

    try:
        data: Any = json.loads(weights_path.read_text(encoding="utf-8"))
        return TrustWeights.model_validate(data)
    except Exception as exc:
        click.echo(
            click.style(f"invalid weights file: {exc}; using equal weights", fg="yellow"),
            err=True,
        )
        return TrustWeights()


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_evidence(path: Path, model_class: type[_ModelT]) -> _ModelT:
    """Load evidence from a JSON file, returning a default instance on failure."""
    if not path.exists():
        return model_class()
    try:
        raw_text = path.read_text(encoding="utf-8")
        data: Any = json.loads(raw_text)
        return model_class.model_validate(data)
    except Exception as exc:
        click.echo(
            click.style(
                f"warning: could not load {path.name}: {exc}; using empty evidence",
                fg="yellow",
            ),
            err=True,
        )
        return model_class()


def _print_trust_report(trust: TrustScore) -> None:
    """Print a formatted trust report to stdout."""
    grade = trust.grade()
    grade_color = {
        "A": "bright_green",
        "B": "green",
        "C": "yellow",
        "D": "red",
        "F": "bright_red",
    }.get(grade, "white")

    click.echo()
    click.echo(click.style("=" * 62, fg="cyan"))
    click.echo(click.style("  AumAI TrustForge — Agent Trust Report", fg="cyan", bold=True))
    click.echo(click.style("=" * 62, fg="cyan"))
    click.echo(f"  Agent ID   : {trust.agent_id}")
    click.echo(f"  Timestamp  : {trust.timestamp.isoformat()}")
    click.echo(
        f"  Overall    : "
        + click.style(f"{trust.overall_score:.4f}  (Grade: {grade})", fg=grade_color, bold=True)
    )
    click.echo(click.style("-" * 62, fg="cyan"))
    click.echo("  Dimension Scores:")
    click.echo()

    dimension_order = [d.value for d in TrustDimension]
    for dim_name in dimension_order:
        dim_score = trust.dimension_scores.get(dim_name)
        if dim_score is None:
            continue
        bar = _score_bar(dim_score.score)
        click.echo(
            f"  {dim_name.upper():12s}  {dim_score.score:.4f}  conf={dim_score.confidence:.2f}  "
            + click.style(bar, fg=_score_color(dim_score.score))
        )
        for item in dim_score.evidence:
            click.echo(f"              {click.style('* ' + item, fg='bright_black')}")
        click.echo()

    click.echo(click.style("=" * 62, fg="cyan"))
    click.echo()


def _score_bar(score: float, width: int = 20) -> str:
    """Return a simple ASCII bar representing *score*."""
    filled = round(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _score_color(score: float) -> str:
    """Return a click color name based on the score value."""
    if score >= 0.85:
        return "bright_green"
    if score >= 0.70:
        return "green"
    if score >= 0.55:
        return "yellow"
    if score >= 0.40:
        return "red"
    return "bright_red"


if __name__ == "__main__":
    main()
