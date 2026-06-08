# Evaluation corpus manifest

The instrument is graded by **deterministic replay** over a fixed set of inputs, so
results compare across code changes (see [`KICKOFF.md`](KICKOFF.md) §4/§7). This
file is that manifest: what each corpus is, what it stresses, and which checks /
goals it feeds. Because the body→output path is deterministic, the **same input
replays to byte-identical output**, which is what makes every check below a real
regression signal.

The harnesses that consume the corpus:

| harness | command | what it proves |
|---|---|---|
| Studio grade | `uv run python -m tools.render_studio [--scripted] [--judge]` | the full multi-stem Studio: coupling, musicality, **aliveness / phantom (G2)**, timbre+FX, gesture actions, timing |
| Legacy grade | `uv run python -m tools.grade [--real X] [--judge]` | the continuous ambient engine + gesture layer (the prior baseline) |
| Gesture render | `uv run python -m tools.render_gestures [--real X]` | recognised moves → effects + flashes over open-ended dance |
| Tests | `uv run pytest -q` | correctness + **determinism** gates for everything above |

---

## 1. Synthetic — `SyntheticPoseSource` (default, in-repo, zero-setup)

A parametric knee-flex-bounce dancer with a scriptable timeline of segments
(`{duration, tempo_hz, energy, openness, asym}`). Needs no camera and no external
data, so it runs in CI on every platform. It is the **default Studio-grade
corpus** (`tools/render_studio.py`) and the backbone of the unit tests.

- **Stresses:** sustained multi-stem groove; tempo/energy/openness changes;
  a deliberate **stillness window** (energy ≈ 0); recovery. A scripted
  **command choreography** force-fires gestures at musical times to exercise the
  discrete actions (fill / drop / breakdown / build / loop record+lock /
  scale-shift / timbre-morph).
- **Feeds:** G1 coupling, G2 **phantom ≤ 0.25 + aliveness** (the stillness window
  is where the phantom gate is measured), G3 musicality (in-scale %, density,
  dynamics), G7 timing headroom, and a multi-stem-activity / distinct-timbre
  check. Determinism gate: two replays → identical event stream.
- **Where:** `everybody_dance/pose.py::SyntheticPoseSource`; driven by
  `tools/render_studio.py`, `run.py --source synthetic --demo`, and most tests.

## 2. Labelled — the scripted performer (in-repo)

A dancer who performs **every built-in move once, in order**, emitting
ground-truth `(t, name)` labels at the instant each move reaches its held pose.
The 10-move vocabulary: `HANDS UP, RAISE LEFT, RAISE RIGHT, T-POSE, SQUAT,
ARMS CROSSED, CLAP, PUNCH, JUMP, STOMP` (JUMP/STOMP via a whole-body vertical
translation that exposes `PoseFrame.root_y`, which hip-centring otherwise hides).

- **Stresses:** gesture recall/precision/latency and false-fire/crosstalk.
- **Feeds:** G4 recognition — `recall ≥ 0.9`, `precision ≥ 0.9`,
  `median_latency_ms ≤ 150` (currently recall 1.0 / precision ≈ 0.91 / ≈ 67 ms),
  and the G4 anti-spam metric (`gesture.max_per_min ≤ 20`).
- **Where:** `tools/render_gestures.py::scripted_performer(with_labels=True)`;
  consumed by `tools/grade.py`, `tools/render_studio.py --scripted`, and
  `tests/test_gestures.py`.

## 3. Edge cases (synthetic, partial — expand)

Reachable today from the synthetic source + frame flags, no external data:

- **Empty room / no person:** `PoseFrame(raw_present=False)` → the Studio enters
  the `attract` phase, the phantom gate suppresses onsets, and the stage shows the
  attract loop. (G6 onboarding, G8 robustness, G9 0-person.)
- **Stillness mid-performance:** the energy≈0 segment of the synthetic script
  (G2 phantom; "stop moving → it goes quiet, never a frozen wall").

**To add** (need a webcam/record path): person enters/exits, occlusion, two
people, off-to-the-side, low light. Tracked in KICKOFF §7/§9.

## 4. Real movement (out-of-repo, fetch on demand)

Genuine bodies, fetched via `bash scripts/get_data.sh` (Git-LFS + model download;
gitignored, so not in the repo). Used with `--real`:

- **LAFAN1 BVH** (Ubisoft mocap) — `dance1/2_subjectN`, 30 fps, full body, 5
  distinct subjects. Drives "distinct per person" + musicality on real dance.
- **MediaPipe `.npz`** — pose estimation on real dance video (noisy, partial
  visibility); the robustness test bed.
- **Dance-with-Melody** — 3D Kinect skeletons (decoded joint layout).

- **Stresses:** distinctness across real bodies; visibility robustness; musicality
  on non-synthetic kinematics.
- **Feeds:** G1/G3 on real movement; G4 anti-spam on real dance.
- **Where:** `everybody_dance/sources.py` (BVH / npz / Kinect loaders);
  `tools/run_dataset.py`, and `--real` on `tools/grade.py` /
  `tools/render_studio.py` / `tools/render_gestures.py`.

## 5. Real-visitor sessions (missing — highest value)

Deterministic `.session` recordings of **actual webcam use** in the gallery. This
is the highest-value corpus and is currently absent (KICKOFF §9). Until it exists,
the synthetic + labelled corpora above stand in, and the [HUMAN]-gated criteria
(true delight, first-timer comprehension, hardware latency) remain human/hardware
gates per KICKOFF §4.

---

### Versioning

Treat the corpus as a versioned asset. When a check's threshold is tightened or a
clip is added, note it here and in `WORKLOG.md` so a metrics delta is always
attributable to either a code change or a corpus change — never ambiguously both.
