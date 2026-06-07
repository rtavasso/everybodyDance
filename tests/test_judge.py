"""Tests for the LLM judge plumbing -- prompt building, response parsing, and the
injectable grader (no API key / no network needed)."""

import json

import pytest

from everybody_dance.judge import (RUBRIC, Scorecard, build_prompt, grade_bundle,
                                    parse_scorecard)


def _canned():
    return {
        "scores": {k: 4 for k, _ in RUBRIC},
        "reasons": {k: "ok" for k, _ in RUBRIC},
        "overall": 4.0, "verdict": "solid", "top_issue": "phantom pads",
    }


def test_prompt_mentions_every_rubric_item_and_metrics():
    p = build_prompt({"coupling": {"score": 0.6}}, {"coupling.score": True})
    for k, _ in RUBRIC:
        assert k in p
    assert "exhibit" in p.lower() and "0.6" in p


def test_parse_plain_json():
    card = parse_scorecard(json.dumps(_canned()))
    assert set(card.scores) == {k for k, _ in RUBRIC}
    assert card.overall == 4.0 and card.top_issue == "phantom pads"
    assert card.mean() == 4.0


def test_parse_tolerates_code_fences_and_prose():
    raw = "Here is my assessment:\n```json\n" + json.dumps(_canned()) + "\n```\nthanks"
    card = parse_scorecard(raw)
    assert card.scores["coupling"] == 4


def test_parse_raises_without_json():
    with pytest.raises(ValueError):
        parse_scorecard("I cannot grade this.")


def test_grade_bundle_with_injected_grader():
    seen = {}

    def fake(prompt, images):
        seen["prompt"], seen["images"] = prompt, images
        return "```json\n" + json.dumps(_canned()) + "\n```"

    card = grade_bundle("PROMPT", ["a.png", "b.png"], grader=fake)
    assert isinstance(card, Scorecard) and card.overall == 4.0
    assert seen["images"] == ["a.png", "b.png"]


def test_scorecard_roundtrips_json():
    card = parse_scorecard(json.dumps(_canned()))
    again = json.loads(card.to_json())
    assert again["scores"]["musicality"] == 4 and again["verdict"] == "solid"
