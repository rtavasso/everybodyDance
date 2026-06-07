"""LLM-as-judge for the observability harness.

An exhibit's notion of "good" is mostly perceptual, so we grade *pictures of the
output* (overlay stills + a piano-roll of the music) plus the objective metrics,
with an LLM scoring them against a fixed rubric. Deterministic replay makes this a
real regression signal: run it before/after a change and watch the scorecard move.

The Claude call is optional and injected, so the harness is testable without a key
(and supports a dry-run that just emits the prompt + image list for offline use).
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# rubric for a creative instrument in a gallery: each scored 1-5 + a reason
RUBRIC: List[tuple] = [
    ("coupling", "Do the music and visuals look clearly CAUSED by the movement? "
                 "Active poses should line up with musical/visual activity."),
    ("liveliness", "Does it feel ALIVE and responsive across the stills -- varied, "
                   "not frozen, repetitive, or laggy?"),
    ("musicality", "From the piano-roll + metrics: is the music coherent, in-key, "
                   "rhythmic, and dynamic (not sparse, random, or monotonous)?"),
    ("gesture_legibility", "When a move is recognised, is its flash/effect distinct "
                           "and readable (a viewer could tell what they did)?"),
    ("visual_aesthetic", "Is the screen beautiful and legible enough for a public "
                         "gallery wall -- composition, contrast, clarity?"),
    ("robustness", "Any signs of trouble -- broken/ jittery skeletons, dead frames, "
                   "clutter, or empty output?"),
]

JUDGE_MODEL = "claude-sonnet-4-6"


@dataclass
class Scorecard:
    scores: Dict[str, int]
    reasons: Dict[str, str]
    overall: float
    verdict: str = ""
    top_issue: str = ""
    raw: str = ""

    def mean(self) -> float:
        return round(sum(self.scores.values()) / max(len(self.scores), 1), 2)

    def to_json(self) -> str:
        return json.dumps({"scores": self.scores, "reasons": self.reasons,
                           "overall": self.overall, "verdict": self.verdict,
                           "top_issue": self.top_issue}, indent=2)


def build_prompt(metrics: Dict, slo: Dict[str, bool]) -> str:
    lines = [
        "You are grading a frame from an interactive ART-EXHIBIT instrument that "
        "turns a visitor's dancing into music + visuals in real time. A stranger "
        "walks up and moves; with no instructions it should feel alive, clearly "
        "caused by their movement, and look good on a gallery wall.",
        "",
        "You are given: (1) a contact sheet of output stills (skeleton + music UI + "
        "any recognised-move flashes), (2) a piano-roll of the music it produced, "
        "(3) objective metrics, (4) automated SLO checks.",
        "",
        "Score each rubric item 1-5 (5=excellent, 1=broken). Be a harsh, specific "
        "critic -- this is a regression signal, not encouragement.",
        "",
        "RUBRIC:",
    ]
    for key, desc in RUBRIC:
        lines.append(f"  - {key}: {desc}")
    lines += ["", "METRICS:", json.dumps(metrics, indent=2),
              "", "SLO CHECKS (true=pass):", json.dumps(slo, indent=2), "",
              "Respond with ONLY a JSON object:",
              '{"scores": {' + ", ".join(f'"{k}": <1-5>' for k, _ in RUBRIC) + "}, "
              '"reasons": {' + ", ".join(f'"{k}": "<one sentence>"' for k, _ in RUBRIC) + "}, "
              '"overall": <1-5 float>, "verdict": "<one line>", '
              '"top_issue": "<the single most important thing to fix>"}']
    return "\n".join(lines)


def parse_scorecard(raw: str) -> Scorecard:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON found in judge response: {raw[:200]!r}")
    d = json.loads(m.group(0))
    scores = {k: int(d["scores"][k]) for k, _ in RUBRIC if k in d.get("scores", {})}
    reasons = {k: str(d.get("reasons", {}).get(k, "")) for k in scores}
    overall = float(d.get("overall", round(sum(scores.values()) / max(len(scores), 1), 2)))
    return Scorecard(scores, reasons, overall, d.get("verdict", ""),
                     d.get("top_issue", ""), raw)


def _anthropic_grader(model: str = JUDGE_MODEL) -> Callable[[str, List[str]], str]:
    """Real grader: sends the prompt + images to Claude. Needs ANTHROPIC_API_KEY."""
    import anthropic  # lazy: optional dependency

    client = anthropic.Anthropic()

    def grade(prompt: str, image_paths: List[str]) -> str:
        content: List[dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            with open(p, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}})
        msg = client.messages.create(model=model, max_tokens=1024,
                                     messages=[{"role": "user", "content": content}])
        return msg.content[0].text

    return grade


def grade_bundle(prompt: str, image_paths: List[str],
                 grader: Optional[Callable[[str, List[str]], str]] = None
                 ) -> Scorecard:
    """Grade an artifact bundle. `grader` is injectable (for tests / dry runs);
    defaults to the Claude vision grader."""
    if grader is None:
        grader = _anthropic_grader()
    return parse_scorecard(grader(prompt, image_paths))
