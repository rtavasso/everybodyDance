# everybodyDance — Exhibit Instrument: Development Kickoff & Design Doc

**Status:** active · **Audience:** the team picking up development · **Owner:** TBD
**This doc directs the work.** Build whatever code is needed to reach the goals in
§6. The hard rule: **every goal must be provably met-or-not by an evaluator that
has only a shell and the ability to view images** (see §4). If you add a goal or a
feature, you must also add the check that proves it.

---

## 1. What we're building

A **creative musical instrument for an art exhibit**. A visitor walks up to a
camera in a gallery; their body movement is turned into music and on-screen
visuals in real time. There are **no instructions, no controllers, no scoring, no
accounts**. It runs **unattended, all day**, for strangers of every body type.

The whole bet rides on three felt qualities, in order:

1. **Caused** — the visitor can tell *their* movement is making the music/visuals.
2. **Alive** — it feels responsive and ever-changing, never frozen or looping.
3. **Beautiful** — it looks and sounds good enough for a gallery wall + speakers.

This is *not* a game (no "match the dance"), *not* a social/UGC product, and *not*
a cloud service. It is one screen, one camera, one speaker system, one room.

## 2. Why this is hard (the problems)

- **P1 — Legible causation.** Generative music easily sounds good *without*
  obviously tracking the dancer. The earlier prototype let the body drive
  macro-knobs while a generator owned the salient notes, so it felt autonomous.
  The fix direction (already started): the body must author *salient, repeatable*
  events (onsets, pitch, recognised moves), not just filters.
- **P2 — Aliveness vs. mush.** Continuous ambient layers keep it from ever being
  silent, but they also play *through stillness* (current `phantom`≈0.67) and can
  turn into a static wall of pads. Caused + alive are in tension with "never
  silent."
- **P3 — Zero-instruction onboarding.** A stranger must "get it" in seconds with
  no text. The screen has to teach by reaction.
- **P4 — Unattended robustness.** Nobody resets it. It must survive empty rooms,
  crowds, occlusion, weird lighting, people walking past, for hours, with no leak
  or crash, and recover on its own.
- **P5 — Real-time on modest hardware.** Motion→sound must feel instant
  (≤ ~80 ms) on a CPU (no usable GPU pose path); audio must never glitch.
- **P6 — Knowing if it's good.** "Good" here is mostly perceptual. We must make it
  measurable so a team can tell, per change, whether they improved it — *the rest
  of this doc exists to solve P6.*

## 3. Foundations already in place (build on these)

| Area | What exists | Where |
|---|---|---|
| Pose | MediaPipe (CPU, auto-downloads weights), normalised skeleton | `everybody_dance/pose.py` |
| Music engine | Continuous body→music, deterministic substrate (scale-safe) | `engine.py`, `substrate.py`, `readout.py` |
| Authoring | "Song builds as you dance" (onsets from hits, pitch from height) | `builder.py` |
| Gestures | 8 deterministic moves → effects + Just-Dance flash; DTW "record a move" | `gestures.py`, `effects.py` |
| Live app | Webcam + skeleton + UI + perf HUD + built-in real-time synth | `tools/live.py`, `rtaudio.py`, `viz.py` |
| **Observability** | **Deterministic replay → pictures + metrics + LLM scorecard** | `tools/grade.py`, `metrics.py`, `judge.py` |
| Tests / env | 56 tests; `uv` project; Win/macOS(Intel+ARM)/Linux launchers | `tests/`, `pyproject.toml`, `scripts/` |

Key property we have deliberately preserved: **determinism (no RNG).** The same
recorded movement replays to byte-identical output. *This is what makes every goal
below LLM-provable and A/B comparison valid. Do not break it without replacing the
verification story.*

## 4. The evaluator model (how goals are proven)

Assume an **evaluating agent (an LLM) with two powers only: run bash, and view
images.** Every acceptance criterion in §6 is one of three types:

- **[BASH]** — a command prints a number / exit code; compare to a threshold.
  Fully automatable, fully reproducible. *Prefer this.*
- **[VISION]** — the agent views a named artifact (a screenshot, contact sheet, or
  piano-roll) and scores it against a rubric; pass = score ≥ threshold. This is
  the LLM-as-judge path (`judge.py`); it is a reproducible *proxy* for perception.
- **[HUMAN]** — genuinely cannot be proven by the agent (true delight, comprehension
  by a real first-timer, latency/throughput on the actual gallery hardware). These
  are explicitly fenced off and require a human/hardware gate. **Minimise them.**

The agent always works over the **fixed corpus** (§7) so results are comparable
across changes. The existing harness already emits the artifacts the agent needs:

```bash
uv run python -m tools.grade --real <session>     # -> out/grade/{contact_sheet,pianoroll}.png, metrics.json
uv run python -m tools.grade --judge              # also produce scorecard.json (needs eval group + key)
uv run pytest -q                                  # correctness + determinism gates
uv run python -m tools.live --record out/live.mp4 # capture the live app for frame inspection
```

**Rule for new work:** a feature isn't "done" until its goal has a [BASH] or
[VISION] check wired into this loop. If you build something whose quality can only
be seen live, add a `--record`/replay path so the agent can view it.

## 5. Guiding principles (constraints on all work)

1. **Deterministic core.** No RNG in the body→output path. (Seeded variation that
   replays identically is fine.)
2. **CPU real-time.** Target commodity gallery hardware. No dependency on a GPU.
3. **Offline-gradable.** Anything that affects "good" must be visible in a replay
   artifact or a metric.
4. **Fail soft, never black/silent for long.** Degrade gracefully; auto-recover.
5. **Zero-config per visitor.** Calibration must be automatic/continuous; no
   per-person setup.
6. **Legibility over cleverness.** When in doubt, make the causation more obvious.

## 6. Design goals & provable acceptance criteria

Each goal: the intent, what to build (direction, not prescription), and the
checks that prove it. Thresholds are starting points — tighten as quality rises.
"Status" reflects the current foundation.

> Many checks reference metric keys from `metrics.json` (e.g. `coupling.score`)
> and judge rubric dimensions from `judge.py` (e.g. `liveliness`). Where a check
> needs a tool that doesn't exist yet, **building that tool is part of the goal.**

### G1 — Legible causation (P1)
**Intent:** a viewer can attribute the sound/visuals to the movement.
**Build:** strengthen body-authored salient events; ensure each musical dimension
is driven by some movement signal; expose the mapping on screen.
**Acceptance:**
- [BASH] `metrics.json`: `coupling.score ≥ 0.5`, `coupling.dead_zone ≤ 0.15`, on
  every corpus session.
- [VISION] judge `coupling ≥ 4/5`; agent confirms in `contact_sheet.png` that
  active poses coincide with musical/visual activity.
- [HUMAN] periodic "guess what I did" test: a human watches output-only and names
  the move; ≥ 70% correct. *(Set up but human-gated.)*
**Status:** coupling.score ≈ 0.60 today; partially met.

### G2 — Aliveness, not phantom or mush (P2)
**Intent:** alive and evolving when someone moves; quiet/settled when they don't.
**Build:** gate ambient voices to presence + recent motion; arrangement that
evolves over time; anti-repetition.
**Acceptance:**
- [BASH] `coupling.phantom ≤ 0.25` (sound during stillness). *Add `phantom` to the
  SLO set in `metrics.py`.*
- [BASH] aliveness proxy: musical density varies over time (windowed density
  std > 0) and `distinct_pitches ≥ 8`; no >8 s identical-output window. *(Add the
  "longest static window" metric.)*
- [VISION] judge `liveliness ≥ 4/5`.
**Status:** phantom **addressed** via opt-in engine `motion_gate` (suppress new
onsets + release held notes when the body is still); `coupling.phantom` added to
the SLO set. Scripted 0.67→0.17, real 0.20→0.10, all SLOs green; proven by
`tests/test_core.py::test_motion_gate_silences_stillness_when_enabled`. Remaining:
the aliveness "longest-static-window" metric, and a "stand still" corpus clip so
the effect is also visible in `pianoroll.png` (the scripted performer never fully
stops, so the roll can't show it).

### G3 — Musicality (P1/P2)
**Intent:** it always sounds musical, never random, sparse, or muddy.
**Build:** sound design + arrangement; density/voice management.
**Acceptance:**
- [BASH] `musicality.in_scale_pct == 100`; `60 ≤ notes_per_min ≤ 600`;
  `dyn_range ≥ 30`.
- [VISION] judge `musicality ≥ 4/5`; agent confirms `pianoroll.png` is neither
  empty nor a wall-of-sound.
**Status:** in-scale 100%; density/mud not yet tuned.

### G4 — Gesture recognition reliability (P1)
**Intent:** recognised moves fire when intended, rarely otherwise, fast.
**Build:** expand/curate the vocabulary; per-move tuning; add jump/stomp via raw
image-space vertical (hip-centring hides them today); reduce false fires.
**Acceptance:**
- [BASH] on the labelled scripted corpus: `recognition.recall ≥ 0.9`,
  `precision ≥ 0.9`, `median_latency_ms ≤ 150`.
- [BASH] on real-dance corpus: no single gesture fires > 20×/min (anti-spam).
  *(Add a per-minute false-fire metric.)*
- [VISION] judge `gesture_legibility ≥ 4/5`.
**Status:** recall 1.0 / precision 0.89 / 67 ms on scripted; `ARMS CROSSED`
over-fires on real dance (≈ 0.34/s) — tune.

### G5 — Gallery-grade visuals (P3)
**Intent:** the screen is beautiful and legible from across a room.
**Build:** a real visual identity beyond the debug stick-figure (trails,
particles, silhouette, color world), projection/large-screen layout, idle/attract
state.
**Acceptance:**
- [VISION] judge `visual_aesthetic ≥ 4/5` on `contact_sheet.png` **and** on
  frames extracted from a `--record` of the live app.
- [BASH] renders at target resolution without dropping below the FPS floor (G7).
- [HUMAN] design review sign-off for the gallery.
**Status:** debug visuals only; large gap.

### G6 — Zero-instruction onboarding (P3)
**Intent:** a first-timer understands within ~5 s, no text.
**Build:** an attract loop for the empty state; immediate, exaggerated reaction on
entry; teach-by-doing.
**Acceptance:**
- [VISION] agent views the empty-room state and the first-2-seconds-after-entry
  frames; rubric "is there an obvious invitation + immediate reaction?" ≥ 4/5.
- [HUMAN] ≥ 5 first-time testers start moving intentionally within 5 s, unprompted.
**Status:** not started.

### G7 — Latency & performance (P5)
**Intent:** motion→sound feels instant; audio never glitches; holds frame rate.
**Build:** measure and instrument live end-to-end latency; optimise to budget.
**Acceptance:**
- [BASH] `timing.headroom_pct ≥ 50` in replay (compute budget).
- [BASH] a latency probe logs end-to-end **motion→sound p95 < 80 ms** and
  **video FPS ≥ 24**; agent asserts from the log. *(Build `tools/latency_probe`
  that emits a machine-readable log from a live or simulated run.)*
- [BASH] audio underrun count == 0 over a 5-min run.
- [HUMAN] confirm on the actual gallery hardware.
**Status:** offline headroom ≈ 99%; **live e2e latency unmeasured** — build the
probe.

### G8 — Unattended robustness (P4)
**Intent:** runs for hours, handles edge cases, recovers itself.
**Build:** presence detection; graceful empty/occluded/crowd handling; continuous
auto-recalibration; watchdog/kiosk wrapper; resource discipline.
**Acceptance:**
- [BASH] `uv run pytest -q` green (incl. determinism tests).
- [BASH] soak test: `tools/soak` replays a ≥ 30-min looped/edge corpus, exits 0,
  RSS growth < 10%, zero exceptions logged. *(Build it.)*
- [BASH] inject empty / occluded / two-person frames → process exits 0 and events
  resume after the person returns.
- [VISION] agent views the "no person" frame → clear, intentional idle state
  (not a frozen/garbage skeleton).
**Status:** `rtaudio` degrades gracefully; no soak/presence harness yet.

### G9 — Multi-person & presence (P4)
**Intent:** sensible behaviour for 0, 1, and many people.
**Build:** decide and implement the policy (e.g., track the most-central/most-active
person; or blend). Define it explicitly.
**Acceptance:**
- [BASH] documented policy + a corpus clip per case; metrics computed without
  error and matching the policy (e.g., switching is bounded, no thrashing — add a
  "subject-switch rate" metric).
- [VISION] judge confirms the on-screen subject matches the intended policy.
**Status:** single-person only (MediaPipe Pose). Decide scope early.

### G10 — Determinism & regression integrity (P6)
**Intent:** keep replay exact so all of the above stays provable.
**Acceptance:**
- [BASH] run any corpus session through `tools.grade` twice → identical
  `session.events_hash` in `metrics.json` (the `timing` block is wall-clock and is
  excluded by design).
- [BASH] determinism unit tests stay green.
**Status:** holds — `events_hash` verified identical across runs. Guard in CI.

## 7. Evaluation corpus (fixed inputs)

All checks run over a versioned corpus so results compare across changes. Start
with what's in-repo and **expand deliberately**; treat the corpus as an asset.

- **Labelled:** the scripted performer (`tools/render_gestures.scripted_performer`,
  emits ground-truth move labels) — drives G4 recall/precision.
- **Real movement:** ≥ 5 varied clips from `data/` (energetic, slow, sparse).
- **Edge cases (to add):** empty room, person enters/exits, occlusion, two people,
  off-to-the-side, low light (once a webcam-capture/record path exists).
- **Real visitor sessions (to add):** record deterministic `.session` files from
  actual webcam use; this is the highest-value corpus and currently missing.

Document the corpus manifest (paths + what each stresses) in `docs/CORPUS.md`.

## 8. Change workflow & A/B protocol

We do **not** need a bespoke A/B tool — the agent does the comparison:

1. **Baseline:** on `main`, run `tools.grade` over the corpus → save bundles.
2. **Candidate:** on the change branch, run the same → new bundles.
3. **Gate (must pass):** every [BASH] SLO in §6 passes on the candidate, and
   `pytest` is green, and determinism holds (G10).
4. **Compare (must not regress):** the agent views baseline vs. candidate
   `contact_sheet.png` / `pianoroll.png` / recorded frames side by side and judges
   per rubric dimension: **better / equal / worse**. A change ships only if no
   dimension regresses and the targeted dimension improves.
5. Record the metrics delta + the judge's verdict in the PR.

## 9. Known issues / starting backlog

1. **~~Phantom pads (G2)~~ — DONE** (opt-in `motion_gate`; phantom 0.67→0.17). See
   the worked example in WORKLOG "Entry 10".
2. **Real-dance coupling below SLO (G1) — now top priority.** On the full
   ambient+gesture pipeline, real-dance `coupling.score ≈ 0.31` (< 0.5 SLO).
   Pre-existing (not caused by the gate, which actually lifts it 0.37→0.44).
   Strengthen body-authored salience and/or revisit the threshold.
3. **`ARMS CROSSED` over-fires (G4)** on real dancing (~0.34/s). Tighten predicate.
3. **Jump/stomp undetectable (G4):** hip-centring removes global vertical motion;
   add a raw image-space vertical signal in the live path.
4. **Live e2e latency unmeasured (G7):** build the probe.
5. **Debug-only visuals (G5):** needs a real gallery visual identity.
6. **No soak/presence/edge harness (G8/G9).**
7. **No real-visitor session corpus (G7 corpus).**

## 10. Non-goals (explicitly out of scope)

Scoring / "match the choreography"; user accounts, sharing, or UGC; cloud
services or networking; mobile/web port; GPU dependency; multi-room/multi-camera;
licensed music. Keep the surface small.

## 11. Milestones (suggested phasing)

- **M1 — Caused & alive:** close G1/G2/G3 to threshold (fix phantom, tune density,
  gate ambience). Provable purely via `grade` metrics + judge.
- **M2 — Gallery-ready surface:** G5 visuals + G6 onboarding + G7 latency probe.
- **M3 — Unattended:** G8 robustness/soak + G9 presence policy.
- **M4 — Real corpus & hardening:** capture real sessions, retune thresholds,
  hardware sign-off.
Each milestone is "done" when its goals' [BASH]/[VISION] checks pass over the
corpus and the [HUMAN] gates (where present) are signed off.

## 12. Risks & open questions

- **R1:** perceptual proxies (LLM judge) may diverge from real visitors —
  calibrate the judge against periodic human ratings; keep [HUMAN] gates on G1/G5/G6.
- **R2:** "never silent" vs. "caused" is a genuine design tension (G2) — the
  presence-gated ambience policy needs an explicit decision.
- **R3:** single-person tracking may be inadequate for a busy gallery (G9) — decide
  scope before M3.
- **R4:** target hardware unspecified — pin it; G7's [HUMAN] gate depends on it.
- **Open:** what is the intended dwell time / session arc per visitor? Should the
  piece build over a visit (like `builder.py`) or be purely momentary? This shapes
  G2/G3 and onboarding.

## 13. Key commands & map

```bash
uv sync                                   # full env (auto-fetches Python 3.10-3.12)
uv run pytest -q                          # correctness + determinism
uv run python -m tools.live --mirror      # the live instrument (webcam)
uv run python -m tools.grade [--real X] [--judge]   # the evaluation harness
```
Engine: `everybody_dance/{pose,features,laban,oscillator,substrate,readout,engine}.py` ·
Authoring/gestures: `{builder,gestures,effects}.py` ·
Live/visual/audio: `tools/live.py`, `everybody_dance/{viz,rtaudio,voices}.py` ·
**Evaluation: `everybody_dance/{metrics,judge}.py`, `tools/grade.py`** ·
History/rationale: `WORKLOG.md`.
