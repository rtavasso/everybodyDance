# everybodyDance — music that dances to you

An interactive instrument that watches a body and **generates music tied to how
that person moves** — specific to them, alive, coherent. Not a sonification toy,
not a backing track to dance over. A partner.

> **Building on this?** Start with [`docs/KICKOFF.md`](docs/KICKOFF.md) — the
> development design doc: product vision, the problems to solve, and design goals
> written so each is provably met-or-not by an LLM with bash + screenshots.
>
> **Want the full band?** Jump to **[everybodyDance Studio](#everybodydance-studio--the-full-multi-stem-band)**
> below — five customizable stems (rhythm, pitch, **timbre**, effects), per-stem
> loopers, gesture-driven fills/drops/loops, and a gallery Just-Dance stage.

Body tracking in, music out, where the music is audibly *caused* by the movement
and feels musical the whole time. Timbres are out of scope: the output is
MIDI/OSC into your own synths (fixed timbres). This is about **structure,
rhythm, dynamics, and modulation.**

```
body
  → pose estimation
  → features (by timescale) + Laban Effort
  → entrained clock  +  event/CC streams
  → coherence substrate  (keeps everything musical)
  → MIDI / OSC
  → your synths (fixed timbres)
```

---

## The core idea, and how it's resolved here

Everything hard collapses into one tension — **reactivity** (map body straight
to sound: legible but dead) vs **musicality** (let the music run its own
structure: good, but the coupling is gone). The resolution, implemented here:

> **Decompose movement by timescale, and route each timescale to the musical
> layer that moves at that speed.** Fast motion drives fast musical things; slow
> posture drives slow harmonic things. Each lane is *rate-matched*, so the
> connection reads as tight everywhere.

| timescale | feature | module | musical target |
|-----------|---------|--------|----------------|
| fast (ms) | velocity / accel / jerk / kinetic energy | `features.py` | dynamics, accent, filter cutoff, density |
| mid (beat) | vertical COM bounce → periodicity | `oscillator.py` | **the clock locks to this** |
| slow (phrase) | posture: COM height, openness, symmetry | `features.py` | harmony, register, palette |
| events | stomp, thrust, reversal, freeze | `features.py` | hits, fills, transitions |
| signature | the statistics of how you move | `personalization.py` | the basis for personalization |

**Laban Effort is the intermediate representation** (`laban.py`). We don't map
joints to notes; we map kinematics → Effort (Weight, Time, Space, Flow) → music,
because Effort is *perceptually meaningful* — a human watching would route it the
same way, so the coupling reads as natural for free.

- **Weight** → dynamics + bass presence
- **Time** (sudden/sustained) → articulation, attack
- **Space** (direct/indirect) → harmonic directness vs delay send
- **Flow** (bound/free) → staccato vs legato, reverb send

**The heart — entrain the clock to the body** (`oscillator.py`). An
adaptive-frequency (Hopf) oscillator with dynamic Hebbian frequency learning
(Righetti / Buchli / Ijspeert) is fed the movement-energy carrier (the vertical
COM bounce, which has a clean fundamental — raw kinetic energy peaks *twice* per
bounce). It learns your tempo and locks phase to it. One dial is the whole feel:

> **`--coupling` (0…1)** — low = music follows you tightly (can feel servile);
> high = the oscillator keeps its own groove you push against (dancing with a
> partner who has their own body).

**Coherence substrate — musicality by construction** (`substrate.py`). Fixed
scale/mode + a slowly-moving harmonic field; per-track roles with register &
density bounds; euclidean rhythms on a quantized grid; pitch snapped to the
active scale with chord-tone weighting. The body drives *readout indices and
modulation lanes*; the substrate keeps it in bounds. No looping — the entrained
clock provides meter and phrase without literal periodicity.

**Personalization is required, not a feature** (`personalization.py`). A ~20 s
free-movement calibration captures your range, characteristic tempo, energy
distribution, and Laban signature. Then we (1) normalize so *your* full range
maps to the full musical range — this alone makes it feel alive for everyone —
and (2) bias the substrate from your signature (flowing/slow → ambient legato;
sharp/percussive → rhythmic staccato).

### The open fork, resolved

> *Does tempo follow the body, or stay in a fixed band while the body drives
> only feel and density?*

Both are built; pick with `--tempo`:

- **`--tempo entrain`** (default) — body-led tempo via the adaptive oscillator.
  More truly "dances to you." All the risk lives here (entrainment robustness),
  so it ships with **hysteresis + a rubato/ambient fallback**: when there's no
  clear beat, it stops chasing noise and drifts instead of locking to garbage.
- **`--tempo fixed`** — rock-solid musical clock; the body drives feel, density,
  harmony and dynamics, but not tempo. Looser connection, zero entrainment risk.

---

## Install & run

```bash
pip install -r requirements.txt          # core (numpy) — enough for headless + tests

# Headless self-test on a synthetic dancer. No camera, no MIDI. Prints a
# coupling report you can read in a terminal:
python run.py --source synthetic --backend log --demo

# Run the tests (proves entrainment, scale-safety, stillness, the coupling dial):
pip install pytest && python -m pytest -q
```

### Real use: webcam → Ableton (or any DAW)

```bash
pip install -r requirements-realtime.txt   # mediapipe, opencv, rtmidi, osc

# Calibrate ~20 s of free movement, then play. Body-led tempo, mid coupling:
python run.py --source webcam --backend midi --tempo entrain --coupling 0.4

# Save / reuse a personal profile (skips calibration next time):
python run.py --source webcam --backend midi --profile me.json

# Rock-solid fixed tempo instead:
python run.py --source webcam --backend midi --tempo fixed --fixed-hz 2.0

# OSC instead of MIDI:
python run.py --source webcam --backend osc --osc-host 127.0.0.1 --osc-port 9000
```

`--backend midi` opens a **virtual MIDI port** named `everybodyDance`. In
Ableton/your DAW, enable it as an input and route the channels below to four
instruments (fixed timbres are yours to choose).

### MIDI / OSC map

| channel | role | notes | key CCs |
|---------|------|-------|---------|
| 1 | **bass** | field root, –1 oct | — |
| 2 | **chord / pad** | triad voicing, openness → spread | — |
| 3 | **lead** | scale-snapped, events punch through | CC10 pan ← L/R asymmetry |
| 10 | **drums** | 36 kick · 42 hat · 41/41 toms (fills) | — |
| (global) | modulation | — | CC11 dynamics ← energy · CC74 cutoff ← energy+Weight · CC91 reverb ← Flow · CC93 delay ← Space |

OSC messages: `/note [channel, note, velocity]` and `/cc [channel, num, value]`.

---

## Mappings worth stealing (implemented)

- vertical COM bounce → beat phase / kick (`oscillator` + `readout`)
- kinetic energy → density + dynamics + cutoff
- limb extension / openness → voicing spread, register
- jerk / direction reversal → accent, fill, transition
- L/R asymmetry → stereo placement (CC10) + call/response hook
- **stillness → a held / suspended chord, a breath** — never silence. Systems
  that only respond to motion punish stillness, and stillness is half of dance.

## Where it lives or dies (and what's done about it)

1. **Tempo-entrainment robustness.** Human periodicity is fuzzy and shifts.
   Handled with confidence + hysteresis and a graceful non-periodic fallback to
   rubato/ambient (`oscillator.EntrainedClock`).
2. **Reactive-vs-structural blend, per track.** Drums & bass are quantized to
   the entrained grid (musical, slightly laggy); lead accents and FX run direct
   off kinematics (tight). The split is in `readout.Readout`.
3. **Stillness handling.** Easy to forget; ruins the feel. Explicitly modeled.

## Latency

- One-euro filter (`filters.py`) for low-latency interactive smoothing.
- Smoothing allocated by timescale — fast lane near-raw, slow lane buffers.
- `EntrainedClock.predict_phase()` lets you fire percussion slightly ahead to
  eat output latency (`--coupling` aside, see `EngineConfig.latency_compensation_s`).

## Project layout

```
everybody_dance/
  pose.py            pose sources (synthetic dancer + mediapipe webcam)
  filters.py         one-euro filter, EMA
  features.py        timescale feature stack + movement signature
  laban.py           kinematics → Laban Effort
  oscillator.py      adaptive-frequency clock + rubato fallback  ← the heart
  substrate.py       scales, euclidean rhythm, harmonic field, roles
  readout.py         features + phase → MIDI/CC events
  personalization.py calibration, normalization, signature bias
  output.py          MIDI / OSC / log backends
  engine.py          the one process tying it together
run.py               CLI
tests/test_core.py   headless proofs (entrainment, scale-safety, coupling dial)
```

## everybodyDance Studio — the full multi-stem band

The flagship instrument: **every stem plays at once** and your body shapes them
all. Five stems — **drums, bass, keys, lead, texture** — each with its own
rhythm, pitch, **timbre**, and effects, plus a per-stem **looper**. Recognised
dance moves punch in salient, repeatable changes (fills, drops, breakdowns,
builds, loop record/lock, scale shifts, timbre morphs). It is the synthesis of
the song-builder, the body-looper, and the gesture layer into one polished
package — and unlike the MIDI core above, **timbre and effects are first-class
and customizable** here (a small built-in synth + FX rack).

Three entry points:

```bash
# 1) The gallery instrument: webcam + the Just-Dance stage + zero-setup audio.
uv run python -m tools.studio_live --mirror        # keys 1..0 force any of the 10 moves
# 2) Headless / MIDI / OSC -- drive your DAW with the whole band:
uv run python run.py --mode studio --source webcam --backend midi --coupling 0.4
# 3) Offline, deterministic, gradable -- pictures + WAV + metrics + SLO:
uv run python -m tools.render_studio               # --scripted  --judge  --seconds 24
```

### The Arcade: combos, streaks, the song arc, gold moves, Dance DNA

The Studio plays like a game (Just Dance is the explicit reference), without
ever scoring the *person* — only payoffs, never punishment:

- **Dance DNA** (`identity.py`) — your ~12 s calibration deterministically
  picks *your* sonic world: root note, mode, chord progression, synth kit,
  accent colour, and a generated stage name ("WILD COMET — C# phrygian ·
  circuit kit"). Soft, open movers land in bright, airy, high worlds; hard,
  sharp movers in dark, driven, low ones — and the fine decimals of your
  signature split even similar movers into different keys, chords and kits.
  Same body, same dance → the same identity, every visit. No two visitors
  stand in front of the same instrument.
- **The song arc** (`game.py`) — the band starts as drums + bass and you *earn*
  the rest: dancing heats a meter that advances intro → groove → build → peak
  on bar lines, unlocking keys, then lead, then texture ("KEYS UNLOCKED"
  flashes; locked lanes read *LOCKED — keep dancing*). Unlocks are sticky;
  cooling off softens the arrangement but never takes an instrument away.
- **Combos** — move *sequences* inside a beat window fire payoffs bigger than
  any single move: `SQUAT → JUMP` = **SUPERNOVA** (a one-bar drop into a
  doubled-density slam), `STOMP ×2` = **EARTHQUAKE** (bass drops an octave,
  driven), `CLAP ×3` = **CLAP STORM**, `RAISE L → RAISE R → HANDS UP` =
  **THE WAVE**, `T-POSE → ARMS CROSSED` = **ECLIPSE** (light/dark scale flip),
  `PUNCH ×3` = **KNOCKOUT**. Deterministic: same combo, same payoff, every time.
- **Streak** — recognised moves heat a WARM / FIRE / GOLD meter that decays in
  stillness; higher tiers push velocity and the arc. Commit to the dance and
  the band audibly commits back.
- **Gold moves** — every 8 bars the screen announces a move, then opens a
  one-bar timed window: hit it for a **PERFECT!** riser + build. Misses are
  silent. This is also the zero-instruction tutorial — the screen teaches the
  vocabulary one move at a time.

**And it sounds produced.** The renders are stereo (constant-power per-stem
placement + live pan), the pitched bus is **sidechain-ducked by the kick** (the
classic pump), one global send/return FX pass, and a master bus (block-RMS
compressor + soft limiter) glues it. Pads voice the 7th when your body opens.
The **live** synth routes each stem to the same rich timbre presets (cached per
note), so the gallery instrument sounds like the offline render.

**Proven on real dancers, on film.** `tools/render_session.py` runs a real
dance video end to end — MediaPipe pose estimation → calibration → the full
Studio with the game layer live (the recognizer fires whatever moves the real
dancer actually makes) — and films it: the gallery stage with the original
footage as backdrop, the glowing skeleton tracked **on the dancer**, plus the
stereo render muxed in:

```bash
bash scripts/get_data.sh                       # fetches the dance videos + model
uv run python tools/extract_pose.py data/videos/benchmark_dance.mp4 -o data/poses/benchmark_dance.npz
uv sync --group video                          # bundled ffmpeg for the mux
uv run python -m tools.render_session data/poses/benchmark_dance.npz \
    --video data/videos/benchmark_dance.mp4 --out out/session
```

On the benchmark dancer: identity **SOLAR COMET** (E phrygian · neon kit), the
full arc travelled, all five stems earned, a GOLD streak, an EARTHQUAKE combo,
a gold-move PERFECT — **all ten SLOs pass on real video** (coupling 0.40,
phantom 0.0, in-scale 100 %). Real data also drove three engine hardenings: a
per-move **salience cap** (a move made constantly isn't a salient move), a
**dwell-debounced stillness gate** (a slow transition isn't a freeze), and a
phantom metric judged against the gate's actual state.

**The groove backbone (`groove.py`) — why it sounds like music.** The body
does not micro-manage note placement; it **commits musical structure at
musical boundaries and modulates within it**. Tempo is estimated from your
pulse (bounce-peak intervals) but **latched to an integer BPM** — it re-locks
only after a sustained shift, only at a bar line, and every note is scheduled
at an exact grid time. Drums play **curated groove styles** (four-on-the-
floor / backbeat / breaks / half-time — your Dance DNA picks one) at a
density *level committed per bar* from what you actually danced, slewed one
level per bar; your energy drives the velocities inside the bar, never the
pattern. Bass locks to the kick on chord roots; keys comp on style slots; the
lead is a **motif** your height contour authors — chord tones on strong
beats, stepwise elsewhere — that persists, mutates one note per phrase, and
re-roots with the harmony. And it's **skill-based**: a rhythmic-coherence
score (your bounce peaks vs the latched grid, shown as the GROOVE meter)
gates the richness — dance off the pulse and the kit stays plain and quiet;
lock in and you earn the backbeat, the ghosts, the full motif, the arps, and
the path to PEAK. Provable: `rhythm.*` SLOs (drum onsets ≥95 % within 12 ms
of the grid, bar-vs-modal-groove similarity ≥0.5, ≤4 tempo re-locks).

**The downbeat is yours (`KineticFlux` + `PhaseServo`).** The skeleton is
treated as an audio signal: a rectified multi-joint acceleration envelope
detects your accents like an audio onset detector (the "visual beats" idea),
and a phase-locked loop pulls the grid onto them — tempo stays latched, phase
servos by ≤4 % of a beat per beat (one hard snap at session start, an
integral term that trims tempo bias, and incoherent accents are never
chased). The pulse listener tracks the *dig* — the bottom of your bounce,
where dancers mark time. Hits well above your own norm get an immediate
answered drum accent in the pocket, and make the next beats land harder.
Measured: 82 % of a steady dancer's onsets land on the downbeat (median error
~35 ms); on real video, the tighter dancer's hits own the grid at 61 % while
the syncopated one is deliberately not chased.

**Sonic moves that push further** (all deterministic): a FIRE+ streak blooms
lead notes into ascending chord-tone **arpeggio runs**; reaching PEAK fires
**THE LIFT** — the whole world transposes up two semitones, locked loops
included, and settles back as you cool; the texture lane plays **polyrhythms**
(3/5-pulse euclidean against the 4/4, phasing per bar); fills are budgeted
like a drummer's (about one per phrase) and gesture one-shots quantize to the
next 16th; truly freezing earns a soft falling **wind-down** before the
silence. On screen: a section-tinted aurora, shockwave rings on every big hit,
golden confetti on PERFECT!, and a dancer who literally gilds gold at GOLD
streak.

**Customizable everything.** A declarative `MappingConfig` (JSON) binds movement
signals and gestures to per-stem targets and discrete actions — edit it and the
instrument changes (`tools.studio_live --mapping my_map.json`):

| layer | what you customize | where |
|---|---|---|
| **rhythm** | which movement signal drives each stem's density (core→drums, limbs→lead…) | `mapping.py` → `stems.py` |
| **pitch** | per-stem register/octave + the signal that picks the scale index | `mapping.py`, `substrate.py` |
| **timbre** | per-stem synth preset (osc mix, resonant filter, ADSR, drive) + movement automation | `timbre.py` |
| **effects** | per-stem reverb/delay/drive sends + a global FX rack (delay, reverb, drive, bitcrush) | `timbre.py` |
| **gestures** | which recognised move triggers which action (fill / drop / loop / scale / timbre-morph…) | `mapping.py`, `gestures.py` |

**Caused & alive (the phantom gate).** New note onsets are suppressed when the
body is still or absent — a freeze actually goes quiet (notes ring out via the
note_off scheduler) instead of the grid manufacturing phantom hits. Measured on
the synthetic corpus: ~12 onsets/s in motion → **0 onsets/s during stillness**, so
`coupling.phantom` is 0.0 (the KICKOFF G2 "caused & alive" win).

**Provable.** `tools/render_studio.py` replays a synthetic dancer deterministically
and emits a gradable bundle — a multi-stem piano-roll, a contact sheet of
move-flashes, `metrics.json` + SLO, and a **timbre-rendered stereo `music.wav`**.
The default bundle: coupling ≈ 0.59, phantom 0.0, in-scale 100 % (judged against
the scale active at each note — honest under ECLIPSE/scale shifts), all four
sections visited, all five stems unlocked and active, two combos + a gold-move
PERFECT fired, eight distinct timbres — **all ten SLOs pass** (the seven
originals plus `game.sections_visited`, `game.stems_unlocked`,
`game.combos_fired`). Same dance + same moves → byte-identical events and audio
(determinism is preserved end to end).

The gallery screen (`stage.py`, drawn live by `tools/studio_live.py` and offline
for grading) shows a mood-coloured glowing dancer with motion trails + beat
particles, a per-stem band (VU, loop-onset rings, REC/LOOP/MUTE badges, the
current timbre, locked-lane states), the section tracker + streak meter, the
gold-move challenge card, your Dance-DNA identity card, Just-Dance "gold-move"
gesture cards with onboarding prompts, and a self-animating attract loop for
the empty room.

Modules: `everybody_dance/{studio,stems,timbre,mapping,stage,game,identity}.py`,
`tools/{render_studio,studio_live}.py`. Corpus + checks: [`docs/CORPUS.md`](docs/CORPUS.md).

## Song-builder (dance continuously, the song builds itself)

No controls, no pedal. Instruments are authored in a fixed order and the phases
auto-advance to the clock; you just keep dancing and a chorus of looping skeleton
"ghosts" accumulates as the song builds. Each instrument gets two **deterministic,
body-authored** passes — the fix for "the body only drives macro-knobs":

- **RHYTHM**: your hits (sharp extremity strikes) place the note onsets on the
  grid — *you* choose placement (no Euclidean pattern, no RNG).
- **PITCH**: the rhythm loops back; your whole-body height (crouch→rise) sets each
  onset's pitch as it replays (snapped to scale, but your contour — no chord-tone
  nudge, no RNG). Drums map height → kick/snare/hat.

Same dance → byte-identical song (it's deterministic), so the instrument is
learnable and repeatable.

```bash
# Offline: render the 4-panel looping-skeleton visualization + the WAV:
python -m tools.render_build data/bvh/dance1_subject1.bvh --kind bvh
```

Modules: `everybody_dance/builder.py` (SongBuilder), `tools/render_build.py`.

### Gesture layer (specific moves -> effects + flash, à la Just Dance)

Open-ended dancing keeps shaping the music, **and** specific recognised moves
punch in their own one-shot effect with a Just-Dance-style flash (the body glows,
the move is named, a colour vignette frames the screen). This is the *salient,
learnable, repeatable* control the continuous mapping lacked — same move, same
effect, every time (no RNG).

Built-in moves: HANDS UP (riser), RAISE LEFT / RIGHT (lead accents), T-POSE
(drop), SQUAT (bass drop), ARMS CROSSED (downlifter), CLAP (snare), PUNCH (chord
stab). All effect notes are snapped to the active scale. You can also record your
own move as a DTW template (`gestures.record_template`) — the same
reference-matching idea Just Dance uses.

```bash
python -m tools.render_gestures                 # scripted demo of every move
python -m tools.render_gestures --real data/bvh/dance1_subject1.bvh   # on a real dance
```
The gesture layer is also live in the sandbox below. Modules:
`everybody_dance/gestures.py`, `everybody_dance/effects.py`.

Note on tracking: the skeleton is hip-centred, so pure vertical translation
(a jump) isn't recoverable from pose alone — the built-ins use body *shape* and
limb motion instead.

### Observability harness (pictures + metrics + LLM judge)

How a dev knows a change made it *better*, for an art-exhibit instrument where
"good" is mostly perceptual. Because replay is deterministic, the same recorded
movement runs through any code version and produces a comparable scorecard.

```bash
python -m tools.grade                      # scripted session -> bundle (dry run)
python -m tools.grade --real data/bvh/dance1_subject1.bvh
python -m tools.grade --judge              # also call Claude to grade (uv sync --group eval; ANTHROPIC_API_KEY)
```

It replays the movement, then writes to `out/grade/`: **pictures of the output**
(`contact_sheet.png` of overlay stills with move-flashes, `pianoroll.png` of the
music) and **objective metrics** (`metrics.json`) across all four dimensions:

- **coupling** — body→music correlation matrix, plus *dead-zones* (move, no
  sound) and *phantoms* (sound, no move).
- **musicality** — in-scale %, density, dynamic range, distinct pitches.
- **recognition** — gesture recall / precision / latency vs. labels.
- **timing** — per-frame compute headroom vs. the frame budget.

These are checked against exhibit **SLOs** (CI pass/fail), and an **LLM judge**
grades the pictures + metrics against a gallery rubric (coupling, liveliness,
musicality, gesture legibility, visual aesthetic, robustness) into a
`scorecard.json`. Modules: `everybody_dance/metrics.py`,
`everybody_dance/judge.py`, `tools/grade.py`.

### Live sandbox (webcam + on-screen pose/UI/perf + sound, no DAW)

The easiest way to play with it and give feedback. One window shows your camera
with the tracked skeleton, the song-builder UI (2×2 instrument grid + looping
ghosts), the pitch-height meter, the loop timeline, and a live FPS / inference /
draw HUD. A **built-in real-time synth** plays the song as it builds — no MIDI
port, DAW, or soundfont needed.

**Recommended — [uv](https://docs.astral.sh/uv/) (any OS; manages Python too):**
```bash
uv sync                              # creates .venv from pyproject/uv.lock
uv run python -m tools.live --mirror # run the sandbox (no venv activation needed)
```
`uv sync` installs the full real-time stack and, if you don't have a compatible
interpreter, **downloads one for you** (the project pins Python 3.10–3.12, the
range with Intel-mac MediaPipe wheels) — so Intel-mac users don't need to install
Python manually. `uv run pytest` runs the tests.

**Windows (one shot — sets up a venv and runs):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\sandbox_windows.ps1
# pass through any flags, e.g.:  ... sandbox_windows.ps1 --mirror --calibrate 10
```

**macOS (Intel & Apple Silicon) and Linux (one shot):**
```bash
bash scripts/sandbox_unix.sh --mirror      # any tools.live flags pass through
```

**Any OS (manual):**
```bash
pip install -r requirements-realtime.txt
python -m tools.live              # --mirror  --camera 1  --calibrate 10  --no-audio
```

Platform notes:
- **Intel Macs (x86_64):** Google's last MediaPipe wheel for Intel is `0.10.20`,
  which only builds for **Python 3.10–3.12** — so use one of those (e.g.
  `brew install python@3.12`, then `PYTHON=python3.12 bash scripts/sandbox_unix.sh`).
  `requirements-realtime.txt` already pins `0.10.20` on Intel macOS automatically.
  Apple Silicon / Linux / Windows just get the latest MediaPipe.
- **macOS camera permission:** first run, macOS asks your terminal for camera
  access (System Settings → Privacy & Security → Camera). If the feed is black,
  enable it there and re-run.
- **Linux audio:** the `sounddevice` wheel bundles PortAudio on macOS/Windows; on
  Linux install the system lib (`sudo apt install libportaudio2`) or run
  `--no-audio` (the sandbox runs fine either way — audio degrades to silent).

First run downloads the MediaPipe pose weights automatically (cached after). It
runs on **CPU** (real-time at `--complexity 1`) on every platform — there is no
GPU/Metal path via the pip wheels (Intel Mac included), and none is needed. Flow:
dance freely during the short calibration,
then keep dancing — drums→bass→keys→lead author themselves (hits place the
rhythm, crouch/rise sets the pitch). Keys: `q` quit · `r` restart song · `c`
re-calibrate · `SPACE` pause · `--record out.mp4` to capture the window.

After a session, jot notes in `FEEDBACK.md` and I'll iterate.

## Live body-looper (build loops, lock them, then dance the gestalt)

Inspired by Imogen Heap's Mi.Mu gloves, but whole-body. Build each instrument's
loop with your body, commit it with a pedal, layer the next one; once everything
is locked, the dance modulates the whole arrangement.

```
RECORD drums → [pedal] → RECORD bass → [pedal] → RECORD chord → [pedal] → RECORD lead → [pedal]
   tempo entrains            tempo LOCKS on first commit; committed loops play back
   to your bounce            while you record the next track
→ LOCKED ENSEMBLE → PERFORM: the dance modulates the gestalt
     continuous : energy/posture → master filter (CC74), intensity, density
     gestures   : thrust→fill · stomp→drop (mute melody) · freeze→breakdown · open-up→build
```

- **Commit / advance**: foot pedal, space, or `c`. `u` = undo (re-record the
  current take), `r` = reset, `q` = quit. (A pedal that emits a keypress works.)
- Tempo locks on the first commit so every loop shares one grid; loops are
  step-quantised to the entrained grid and stay in scale.

```bash
# Live: webcam in, MIDI out. Calibrate ~20s, then loop with the pedal.
python run.py --mode loop --source webcam --backend midi

# Offline: build the loop from a recorded dance with a scripted pedal, render WAV
# (hear drums → +bass → +chord → +lead → perform):
python -m tools.run_looper data/bvh/dance1_subject1.bvh --kind bvh \
    --rec-seconds 7 --perform-seconds 22 --out out/looper.wav
```

Modules: `everybody_dance/looper.py` (loop station + perform modulation),
`everybody_dance/controls.py` (pedal/keyboard + scripted control),
`tools/run_looper.py` (offline render).

## Running on real data

The instrument has been iterated against real bodies, not just the synthetic
dancer. See `WORKLOG.md` for the full debugging trail.

```bash
bash scripts/get_data.sh          # LAFAN1 dance mocap + MediaPipe model + clips

# Pose estimation on a real video -> keypoints:
python tools/extract_pose.py data/videos/bolt-detection.mp4 -o data/poses/bolt.npz

# Run the instrument across dancers: per-person OUTPUT signature, a distinctness
# matrix, and a rendered WAV per dancer (so musicality can be heard):
python -m tools.run_dataset 'data/bvh/dance*.bvh' --max-seconds 50      # mocap
python -m tools.run_dataset 'data/poses/*.npz' --kind npz               # pose-est

# Specificity acceptance probes (stop -> stops; one body part -> one thing):
python -m tools.probe_specificity
```

Real-data modules: `everybody_dance/sources.py` (BVH + npz pose sources),
`tools/bvh.py` (BVH parser + FK), `tools/extract_pose.py` (MediaPipe Tasks API),
`tools/synth.py` (events → WAV), `tools/run_dataset.py`, `tools/probe_specificity.py`.

What the iteration established (all verified, see `tests/test_specificity.py`):
- **distinct per person** — different dancers land in different palettes/scales,
  tempos and densities; the music is calibrated to each body;
- **musical** — in-scale throughout, tempo octave-folded into a musical band,
  real dynamic range in the rendered audio;
- **specific** — stop moving and the music stops (a held breath chord remains);
  move one body part and one thing moves (beat ← core, melody ← limbs).

## Status & next steps

This is the hand-built prototype the brief calls for: one Python process, runs
headless so the only question that matters early — *does the coupling read?* —
is answered fast (and in CI). Deferred on purpose: the learned per-person
embedding / decoder (the heavy path), depth-camera 3D, and a live GUI for the
coupling dial. The adaptive Hopf oscillator has a small steady-state tempo bias
(~3%); fine for feel, tunable if you want metronomic accuracy.
