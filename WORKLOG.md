# WORKLOG — everybodyDance, real-data iteration

Goal: feed **real pose data from multiple people** through the engine and iterate
until the output is (a) **distinct per person**, (b) **musical**, (c) **specific**
— stop moving → music stops; move one body part → one thing moves in audio.

Status legend: ✅ working · ⚠️ partial · ❌ broken · ⏳ in progress

---

## Plan
1. ⏳ Acquire real multi-person pose data (real videos → MediaPipe, or a public
   keypoint dataset).
2. ⏳ File-based PoseSource adapter (map external joint convention → ours).
3. ⏳ Run engine per dancer; collect output stats; render audio so musicality is
   actually audible.
4. ⏳ Evaluate the three criteria with metrics + listening.
5. ⏳ Iterate on the model.

## Environment / network
- pypi ✅, github ✅, googleapis reachable (400 on bare host), huggingface 403.
- numpy + pytest installed. mediapipe + opencv-python-headless installing.

---

## What works (baseline, synthetic)
- Pipeline runs end-to-end headless; 14 tests pass.
- Clock entrains to vertical-COM bounce; tracks tempo changes; rubato fallback.
- All melodic output in-scale; stillness → suspended chord.

## What doesn't / unknown going in
- Never tested on real, noisy pose data (jitter, missing joints, camera scale).
- "Distinct per person" unproven on real bodies.
- "Move one body part → one audio thing" not yet isolated/verified.
- No way to actually *hear* output yet (no synth/render).

---

## Scoreboard (live)
- ✅ Real DANCE data: LAFAN1 BVH (Ubisoft, LFS) — `dance1/2_subjectN`, 30fps,
  full body, no occlusion, 5 distinct subjects. BVH parser + FK done.
- ✅ Real POSE-ESTIMATION data: MediaPipe on 3 intel videos (noisy, partial).
- ✅ Engine runs on both; offline synth renders events → WAV (audible).
- ✅ **Musical tempo** via octave folding (72–100 BPM across dancers).
- ✅ **Distinct per user**: 3 palettes in use (rhythmic/phrygian,
  ambient/pentatonic, neutral/dorian), tempo 72–100, density 2.7–6.1 n/s,
  drum_frac 0.16–0.48; distinctness matrix mean ~5, all pairs separated.
- ✅ **Effort no longer saturates** — per-person percentile normalization of all
  Effort drivers (W/T/S/F now span their ranges and discriminate).
- ✅ **Specific — stop → music stops**: 9→0 note-ons/s, held breath chord remains
  (verified on probe + regression test).
- ✅ **Specific — one body part → one thing**: core/limb energy split routes the
  beat to hips/legs and the melody to arms/hands; wrist-only ⇒ lead+pan move,
  drums/bass quiet (verified on probe + regression test).
- ✅ **Numerical robustness**: oscillator sub-steps + state clamp + input guard,
  so noisy/low-fps pose data no longer blows it up to NaN.
- 16/16 tests pass (incl. 2 specificity acceptance tests).

### Key insights logged
- Energy ∝ velocity² peaks twice/bounce → drive the clock with vertical
  COM-above-feet (a clean fundamental), not raw kinetic energy.
- Hip-centring cancels a rigid bob → measure bounce above the feet.
- Human periodicity is often a slow ~0.6 Hz sway → octave-fold the felt tempo
  into a musical band while staying phase-locked.
- Specificity is *relative to calibration*: calibrate on full-body range, then a
  small partial motion gives a small, localised response. Per-person
  normalization must NOT be done on the partial motion itself.

## Entry 2 — better full-body dance data (higher-quality pose estimation)
Old data was weak: intel surveillance clips (28–78% detection, ankles 0.1–0.5).
Hunted for full-body dance with clean pose estimation. Network limits: CDNs 403,
google.github.io/AIST++ 403, AIST++ GCS bucket gone (NoSuchBucket), KTH/Motorica
behind SharePoint. Reachable: github raw / codeload / LFS, and the github trees
API (intermittently). Found three good sources:

- ✅ **Real dance VIDEO → MediaPipe** (`PrashantSaikia/...`): `benchmark_dance`,
  `right_dance` — portrait, single dancer filling frame. **100% detection, full
  body** (hips/shoulders 1.0, wrists ~0.9, ankles ~0.9). Huge jump over intel.
  (`wrong_dance` is landscape with feet cropped → ankles 0.1; excluded.)
- ✅ **"Dance with Melody" 3D Kinect pose** (`Music-to-dance-motion-synthesis`):
  61 dance clips, each `skeletons.json` (23-joint 3D). Joint layout undocumented;
  **decoded from bone-length topology + height + motion** and validated (rigid
  bones: torso 45±1.9, arm 30±2.8, thigh 48±5.2; L/R + head>sh>hip>knee>ankle
  all consistent). Loader: `sources.load_dance_skeleton_source`, map `MELODY_MAP`.
- (kept) LAFAN1 mocap — 5 genuinely distinct dance subjects.

### Evaluation insight (this is the iteration)
- On benchmark/right_dance the bounce/feet signals are now reliable end-to-end —
  the clean pose removes the NaN-inducing jumps the noisy npz had.
- The 6 Dance-with-Melody clips come out **homogeneous** (all phrygian/rhythmic,
  72–77 BPM, near-identical signatures) because they're the same performer/style
  — the system is being *faithful*: similar movers → similar music. LAFAN1's
  genuinely different subjects still span 3 palettes and 72–103 BPM. So
  distinctness tracks real mover differences, not noise.
- Cross-modality check: torso-normalization makes KE/Effort comparable across
  capture systems (mocap cm, Kinect cm, MediaPipe image), so the same thresholds
  generalize — verified by all three sources producing sensible, in-scale music.

## Entry 3 — live body-looper (Imogen-Heap-style, whole body)
New performance mode layered on the instrument. Decisions (asked the user):
commit = out-of-band pedal/key (clean timing, no fragile gesture recog);
perform = both continuous + structural gestures.

Architecture: `controls.py` (pedal/keyboard + ScriptedControl for headless),
`looper.py` (LoopStation). Phase machine RECORD(drums→bass→chord→lead)→PERFORM.
- entrained clock provides the grid; tempo LOCKS on first commit so loops align.
- loops are step-quantised buffers per role; committed loops replay
  deterministically while the next track records; future tracks stay silent.
- PERFORM: continuous master CC (filter/expression/mod from energy+posture),
  intensity scales note velocity, hats density-gated by energy; gestures →
  thrust=fill, stomp=drop, freeze=breakdown.

Bugs found & fixed while iterating on real dance:
- Recording during the calm intro → near-empty loops (nf.core_energy ~0). Fix:
  `tools/run_looper.py` auto-finds the liveliest window and calibrates on it
  (mirrors live use: calibrate on your actual movement, then loop).
- Bass loop empty: density `0.6·core·(max4)` rounded to 0. Fix: while RECORDING a
  track you're deliberately performing it, so density uses overall energy + a
  floor (every take lays down notes); core/limb specificity stays in the
  continuous instrument and in perform-mode. Bumped bass max density.
Result: clean build-up drums→+bass→+chord→+lead→perform; 138/138 notes in scale;
827 master-CC modulation events + fills in perform. 23/23 tests pass (added
tests/test_looper.py: phases, capture/replay, tempo lock, track isolation,
perform modulation, undo). Live path wired: `run.py --mode loop`.

## Entry 4 — song-builder ("dance continuously, the song builds")
Reframe of the looper after the red-team (which showed the body controlled
macro-knobs, not salient events: pitch/onset owned by scale+Euclid+RNG). Design
converged via UX discussion -> "auto-advance" makes instrument selection
unnecessary; deterministic order, and each committed stem becomes a looping
skeleton ghost. The fix for the fatal critique:
- RHYTHM pass: the body's HITS (extremity accel peaks, per-person threshold +
  refractory) place onsets on the grid -- you choose placement (no Euclid/RNG).
- PITCH pass: the rhythm loops; whole-body HEIGHT (crouch->rise = nf.com_height,
  already calibrated) sets each onset's pitch, snapped to scale but no chord-tone
  nudge, no RNG. Drums use height -> kick/snare/hat.
- Deterministic end-to-end: `test_determinism_same_dance_same_song` proves the
  same dance yields a byte-identical song -> learnable/repeatable (the red-team's
  repeatability critique addressed).
- `builder.py` SongBuilder (FSM: count-in -> per instrument [rhythm,pitch] ->
  loop; captures a looping skeleton ghost per stem). `tools/render_build.py`
  renders the 4-panel "chorus of skeletons" overlay + the WAV. `run.py --mode
  build` is the live audio path.
- Validated on real LAFAN1 dance: 166/166 notes in scale; all 4 stems authored
  and looping. 28/28 tests pass (added tests/test_builder.py).
Open: this is offline-on-recorded-dance; real validation needs a live dancer
intentionally placing hits/heights. Perform-phase gestalt modulation not yet
ported into builder.

## Entry 5 — live sandbox (webcam + on-screen pose/UI/perf + zero-setup audio)
For the user to actually play with the builder and report feedback on Windows.
- Facts established: MediaPipe `solutions.pose` auto-downloads/caches weights and
  runs CPU (XNNPACK); no GPU path via pip on Windows, and none needed (real-time
  at complexity 1). Documented honestly rather than promising GPU.
- `everybody_dance/voices.py`: shared synth voices (single source of truth);
  tools/synth.py now imports `render_note` from it.
- `everybody_dance/rtaudio.py`: RealtimeSynth backend via sounddevice -- renders
  each note_on to a finite buffer, mixes in the audio callback, soft-clips. Zero
  external setup (no DAW/MIDI/soundfont); degrades to a silent no-op if
  sounddevice/PortAudio is absent. Mixer split out as `_mix` so it's unit-tested
  without a device.
- `everybody_dance/viz.py`: shared drawing (skeleton, instrument panels, timeline,
  pitch meter); `compose_offline` (render_build now reuses it) + `compose_live`
  (camera + 2x2 grid + HUD).
- `everybody_dance/calib.py`: shared calibration (batch + streaming
  BuildCalibrator) returning profile, octave-folded tempo, per-person hit thresh.
- `tools/live.py`: the sandbox -- webcam capture, MediaPipe, calibrate-then-build
  FSM, single window with FPS/infer/draw HUD, real-time audio, keys
  q/r/c/SPACE + --record/--mirror/--no-audio.
- `scripts/sandbox_windows.ps1` (venv + install + run), requirements-realtime
  (+sounddevice), README sandbox section, FEEDBACK.md template.
- Could NOT test camera/audio/GUI in this headless Linux box; verified everything
  underneath headlessly: imports, voices (finite/bounded/deterministic), rt mixer
  (mix/drain/polyphony-cap/panic/disabled-noop), fold_tempo, compose_live frame,
  offline render parity. 35/35 tests pass (added tests/test_sandbox.py).
Open: live latency/feel and pose stability need the user's real run + FEEDBACK.md;
skeletons can overflow panel bottoms (cosmetic clip TODO).

## Entry 6 — Intel-mac (x86_64) support for the live sandbox
Code is platform-neutral; the issue is packaging. Verified against PyPI:
- MediaPipe latest (0.10.35) ships NO Intel-mac wheel (only macosx_11_0_arm64,
  linux x86_64, win). Last version with an x86_64 macOS wheel is 0.10.20
  (cp310/311/312). OpenCV is fine on Intel mac (x86_64 wheels through cp313).
- requirements-realtime.txt: env-marker split -- `mediapipe==0.10.20` on
  `sys_platform=='darwin' and platform_machine=='x86_64'`, else `mediapipe>=0.10`.
  Used De Morgan complement (`!=...or...!=`) because PEP 508 markers have no
  `not(...)`; verified mutually exclusive + exhaustive across the 4 platforms.
- scripts/sandbox_unix.sh: macOS (Intel+Silicon) + Linux launcher; on Intel mac
  it checks Python is 3.10-3.12 and errors with a fix (PYTHON=python3.12 ...);
  prints the macOS camera-permission hint.
- README platform notes (Intel-mac Python pin, camera TCC, CPU-only everywhere).
Couldn't run on a Mac from here; validated shell syntax + marker resolution.

## Entry 7 — uv support (pyproject + lockfile)
Project had only requirements*.txt; added pyproject.toml (hatchling, package
everybody_dance, console script `everybodydance`) + uv.lock so `uv sync` works.
- deps include the full real-time stack so `uv sync` -> ready-to-run sandbox;
  pytest in a default dependency-group. Carried the Intel-mac MediaPipe marker
  split into [project.dependencies].
- requires-python ">=3.10,<3.13": the cross-platform range with Intel-mac wheels;
  lets uv auto-download a compatible Python (Intel-mac users skip the brew step).
- uv unified mediapipe to 0.10.20 across all platforms in the universal lock
  (satisfies both specs, wheels exist everywhere) -> one reproducible version.
- Verified here: uv lock (42 pkgs), uv sync, `uv run pytest` 35/35, `uv run`
  offline render. sounddevice needs system libportaudio on Linux (wheel bundles
  it on mac/win); rtaudio already degrades to silent. README: uv as recommended
  path + Linux-audio note.

## Entry 8 — gesture layer (Just-Dance-style moves -> effects + flash)
How Just Dance works: it's template matching against a *time-synced authored
reference* (Wii = right-hand accelerometer signature; Kinect/phone = pose/accel),
scoring how well you match the expected move in each window ("gold moves"). Not
open-vocabulary recognition.
What we built (the salient/learnable/repeatable triggers the red-team wanted),
layered OVER the continuous mapping so open-ended dance still drives music:
- gestures.py: GestureRecognizer with StaticPose/HeuristicMotion predicates +
  DTWGesture (record-a-move template matching, the Just-Dance analog). All
  deterministic. Library: HANDS UP, RAISE L/R, T-POSE, SQUAT, ARMS CROSSED,
  CLAP, PUNCH.
  KEY design constraint discovered: skeleton is hip-centred + torso-normalised,
  so (a) global translation is gone -> "jump" is unrecoverable here (dropped;
  needs raw image-y), and (b) thresholds are torso units (standing verticality
  ~3, not 0.4). Squat uses an auto-calibrated verticality ratio (person-indep).
  Tightened PUNCH (one-arm horizontal jab) + CLAP (meet-not-cross) to kill
  crosstalk with T-pose/hands-up/arms-crossed.
- effects.py: EffectEngine maps move -> in-scale one-shot (snare/riser/drop/
  stab/...) via the Substrate + a fading Flash. note_on(dur)+future note_off so
  both rt + offline backends work.
- viz.draw_flashes: body-glow + vignette + big move name. compose_live draws it.
- tools/render_gestures.py: scripted-performer demo (all 8 moves, clean
  cause->effect) + ambient engine + WAV; --real <bvh> runs on real dance.
- Wired into tools/live.py (gesture layer active during build phase).
Validated: scripted performer fires all 8, no crosstalk, deterministic; real
131s BVH fired 75 across all types; demo: 85 ambient + 23 fx notes, all fx
in-scale. 42/42 tests (added tests/test_gestures.py incl. DTW).
Open: ARMS CROSSED a bit trigger-happy on real dance (45/131s); jump/stomp need
raw image-y in the live path; DTW "teach a move" key not yet bound in live.

## Entry 9 — observability harness (pictures + metrics + LLM judge)
Direction set: product = creative instrument for an ART EXHIBIT (no scoring/UGC);
measure ALL four kinds of good; evaluator = "LLM grades + takes pictures of
outputs". Built on the determinism we banked earlier: same movement replays to
identical output -> a real regression signal.
- metrics.py: coupling (body->music correlation matrix + dead_zone/phantom),
  musicality (in-scale/density/dynamics), recognition (recall/precision/latency
  vs labels), timing (compute headroom). SLO dict + check_slo for CI pass/fail.
- judge.py: gallery rubric (coupling, liveliness, musicality, gesture_legibility,
  visual_aesthetic, robustness), prompt builder, response parser, Scorecard, and
  an injectable Claude vision grader (lazy anthropic; key via env; dry-run/test
  friendly).
- viz.py: pianoroll (music made visible) + contact_sheet (tiled overlay stills) =
  the "pictures of outputs".
- tools/grade.py: replay (ambient engine + gesture layer) -> artifacts
  (contact_sheet.png, pianoroll.png, metrics.json, prompt.txt) -> judge ->
  scorecard.json. scripted_performer now emits ground-truth labels for recall.
- requirements-eval.txt + pyproject [dependency-groups] eval = anthropic; relocked.
Validated on scripted session: coupling 0.60, in-scale 100%, recall 1.0 /
precision 0.89 / latency 67ms, compute headroom 99% -- all 4 SLOs pass. The
harness surfaced phantom=0.67 (ambient pads sustain through stillness); judging
the bundle as the LLM independently flagged the same issue -> loop works. 56/56
tests (added test_metrics.py, test_judge.py).
Open: live e2e latency still needs the HUD path; multi-version A/B (replay through
engine A vs B -> two bundles -> diff/prefer) is a thin layer on top, not built;
no real webcam session corpus yet (uses dance corpus + scripted performer).

## Entry 10 — everybodyDance Studio (multi-stem, customizable, gesture-driven band)
Built the flagship "polished package" the brief asked for: highly customizable
rhythm/pitch/timbre/effects across **multiple stems with loopers**, an intuitive
Just-Dance-inspired UI, and gesture/dance recognition as the precise control
surface. Developed as a new layer ON TOP of the existing primitives (substrate,
oscillator, features, laban, personalization, gestures, effects) so nothing in the
prior modes broke and determinism + offline-gradability were preserved.

Built via a multi-agent workflow in waves (each module landed green before the
next depended on it):
- **timbre.py** — customizable per-stem synth (`TimbrePreset`: osc mix, resonant
  one-pole filter, ADSR, drive, noise; detuned unison) + `FXRack` (feedback delay,
  Schroeder reverb, drive, bitcrush) + an 11-preset palette. This brings **timbre
  and effects into scope** (the MIDI core deliberately left them out). numpy-only,
  deterministic (seeded noise). 11 tests.
- **mapping.py** — the customizability layer: a declarative, JSON-serializable
  `MappingConfig` (per-stem `*_src` signal bindings + ~10 gesture→action bindings +
  mood→scale) and a pure `MappingResolver`. `default()` ships a coherent 5-stem
  Just-Dance mapping. 11 tests.
- **gestures expansion** — JUMP/STOMP via a new `PoseFrame.root_y` (raw hip-center
  vertical in torso units, populated by the sources; the hip-centred skeleton
  otherwise hides global translation). Tightened ARMS CROSSED (was over-firing).
  `build_gestures(names)`/`GESTURE_LIBRARY` registry so a config can enable a
  named subset. Vocabulary now 10 moves; scripted_performer + grade updated;
  grade recall 1.0 / precision 0.91 / 67 ms.
- **stems.py + studio.py** — the engine. One `Studio.step()` runs the whole band:
  pose → features+Effort → entrained clock → a flat signal dict → `MappingResolver`
  → per-stem continuous targets on five stems (drums/bass/keys/lead/texture) → in-
  scale, step-quantised MIDI. Recognised moves AND force-fired `commands` route
  through the bindings to actions: fill/drop/breakdown/build, fx one-shots,
  per-stem loop record/lock/toggle/clear, scale_shift, timbre_morph, stem
  mute/solo. Returns a `StudioUI` (for the stage) + per-stem `automation` (for the
  renderer). 25 tests.
- **stage.py + tools/studio_live.py** — the gallery screen (mood-coloured glowing
  dancer + motion trails + beat particles, per-stem band with VU/loop-rings/badges,
  gold-move cards + onboarding prompts, attract loop) and the live webcam app
  (MediaPipe with root_y, calibrate→perform, RealtimeSynth audio, keys 1..0 force
  moves). compose_stage is headless-capable (camera=None) so the grader can view
  frames. 9 tests.
- **tools/render_studio.py** — the offline gradable harness: deterministic replay
  of a synthetic dancer (groove → change → a real stillness window → recovery) with
  a scripted command choreography exercising every gesture action; emits a
  timbre-rendered WAV, a multi-stem piano-roll (colours by stem tag; handles the
  5th/texture stem viz.pianoroll can't), a contact sheet, metrics.json + SLO. 6
  tests.
- **metrics.py** — added `liveliness` (density variation + longest silence),
  `gesture_spam` (per-minute fire rate), the `coupling.phantom`/`liveliness.density_std`/
  `gesture.max_per_min` SLOs, and strict `>`/`<` operators in `check_slo`.
- **run.py --mode studio** — a headless/MIDI/OSC path for the band; **docs/CORPUS.md**
  — the corpus manifest (KICKOFF §7).

The headline fix (KICKOFF G2, the top backlog item): the **phantom gate**. New
note onsets are suppressed when energy < `motion_floor` or no body is present
(notes ring out via the note_off scheduler; one soft pad revoice on entering
stillness). Result on the synthetic corpus: ~12 onsets/s in motion → **0
onsets/s during stillness**, `coupling.phantom` 0.0 (was ≈0.67 on the ambient
engine). A freeze now actually goes quiet instead of a static wall of pads.

Validation (synthetic corpus, default 24–30 s bundle): coupling ≈ 0.59,
phantom 0.0, in-scale 100 %, liveliness.density_std ≈ 7–9, all five stems active
(drums/bass/keys/lead/texture), six distinct timbres in use — **all seven SLOs
pass**. Determinism (G10): two replays → byte-identical events AND audio; only the
wall-clock `timing` block differs (a perf measurement, as in the legacy grade).
Full suite **124 passed** (was 56). Eyeballed the stage perform/attract frames and
the piano-roll/contact-sheet — gallery-grade, clean cause→effect.

Open: live feel + pose stability on real hardware (studio_live can't run in this
headless box — needs a real webcam session); a timbre-aware *real-time* synth
(the rich timbres render offline; live audio uses the simpler RealtimeSynth
voices); a real-visitor `.session` corpus (KICKOFF §9); render_studio `--scripted`
precision (0.71) is looser than grade's (0.91) — grade remains the recognition
authority.

## Entry 11 — the Arcade update (game layer + Dance DNA + the sound upgrade)
Direction from the owner: make it FEEL LIKE A GAME (Just Dance), make each
person's output feel unique to them, and make it sound good. Three new layers
on the Studio, all deterministic, all wired into the gradable harness:

- **game.py — the game layer.** (1) COMBOS: named move sequences in beat
  windows fire big repeatable payoffs (SQUAT→JUMP = SUPERNOVA drop-slam,
  STOMP×2 = EARTHQUAKE octave-down bass, CLAP×3 = CLAP STORM, RAISE L→R→HANDS
  UP = THE WAVE, T-POSE→ARMS CROSSED = ECLIPSE light/dark flip, PUNCH×3 =
  KNOCKOUT); longest match wins, history clears on fire. (2) STREAK: moves heat
  a meter (WARM/FIRE/GOLD) that decays in stillness; tier pushes note velocity
  and the arc. (3) SONG ARC: heat-driven sections (intro→groove→build→peak)
  that UNLOCK stems as they're earned — the song starts drums+bass and grows
  into the full band; unlocks are sticky (never punish), transitions land on
  bar lines, every unlock is announced on the stage. (4) GOLD MOVES: every 8
  bars a specific move is announced, then a 1-bar timed window; hit = PERFECT
  payoff. Doubles as the zero-instruction tutorial.
- **identity.py — Dance DNA.** The calibration signature deterministically
  picks a whole sonic world: root note, mode, chord progression
  (substrate.PROGRESSIONS, curated per scale), synth kit (5 curated ensembles
  over 7 new presets), accent colour, and a generated stage name (adjective
  reflects the movement palette, so "WILD COMET" is *earned*). Coarse choices
  are legible (soft/open → ambient lydian high; hard/sharp → rhythmic phrygian
  low); fine choices CRC the signature decimals so similar movers still split.
  Same calibration → byte-identical identity.
- **the sound upgrade.** render_studio's WAV is now STEREO (constant-power
  per-stem placement + live pan automation), kick-SIDECHAINED (the classic
  pump; pitched bus ducks 0.45 on every kick), one send/return FX pass, and a
  MASTER BUS (block-RMS compressor + tanh soft limit). Verified: L/R corr 0.62,
  RMS arc 0.07→0.24→0 (stillness)→0.26, crest 5.4. Pads voice the 7th when the
  body is open. LIVE audio: RealtimeSynth can now route stem tags to the rich
  timbre presets (cached on quantised note keys, so each costs synthesis once)
  — the live band finally sounds like the offline render; studio_live syncs
  presets from automation so timbre_morph is audible live.
- **stage.py** draws the new state: section tracker + streak meter (top
  centre), the gold-move challenge card (announce/NOW! + draining bar), the
  Dance-DNA identity card (bottom left), locked lanes ("LOCKED — keep
  dancing"), and unlock/combo/PERFECT toasts ride the existing flash pipeline.
- **provable.** metrics gains `game.*` SLOs (sections_visited ≥ 3,
  stems_unlocked ≥ 5, combos_fired ≥ 1) + segment-aware in-scale% (notes judged
  against the scale active at their onset — honest under ECLIPSE/scale_shift).
  The corpus choreography now fires a CLAP STORM, an EARTHQUAKE, and a JUMP
  timed into the first gold window. Default 32 s bundle: ALL TEN SLOs pass
  (coupling 0.59, phantom 0.0, in-scale 100 %, 4 sections, 5 stems, 2 combos,
  1 gold hit, 8 distinct timbres). Determinism holds end to end.
- 147/147 tests (was 124; +test_game.py, +test_identity.py). The KICKOFF
  "not a game" non-goal is superseded by the owner's direction — there is
  still no scoring/judgement of the *person*, only payoffs; misses are silent.

Open: live feel of unlock pacing + gold windows needs a real webcam session
(can't run cameras here); first render of a long pad live costs ~tens of ms
before its cache entry exists (pre-warm if it ever audibly hiccups).

## Entry 12 — real video end to end + the session film + sonic/visual push
Direction: prove it on REAL dancers (video -> pose estimation -> the full
instrument), film the result, and push sonic + visual creativity further.

- **Real data acquired + extracted.** Fetched the two full-body dance videos
  (benchmark_dance / right_dance, 406x718@30, 21.3 s, single dancer) + the
  MediaPipe pose_landmarker model; `tools/extract_pose.py` -> npz at **100 %
  detection** (wrists ≥ .85, ankles ≥ .90).
- **tools/render_session.py — the session film.** npz/BVH -> calibrate ->
  full Studio (game + identity live, recognizer firing from the REAL moves, no
  scripted commands) -> every frame through the Stage with the original video
  as backdrop and the **glowing skeleton drawn in image space ON the dancer**
  (the visible proof pose estimation drives everything) -> stereo WAV -> muxed
  session.mp4 (imageio-ffmpeg, optional `video` dep group) + metrics/pianoroll.
- **What real data immediately broke (and the fixes, all in the engine):**
  1. *Gesture spam*: PUNCH fired 42x/21 s (energetic arms trip the velocity
     predicate constantly) -> the **salience cap**: per-move rolling limits
     (burst 4/14 s — keeps x3 combos; sustained 6/40 s ≈ 9/min). A move made
     constantly isn't a salient move; the continuous layer already expresses it.
  2. *Stillness-gate flicker*: real dancers dip under the energy floor between
     moves ~20x/min, re-firing the entering-stillness one-shots -> **dwell
     debounce** (`still_dwell_s` 0.4 s to engage, instant release).
  3. *Phantom metric honesty*: now judged against the **gate's actual state**
     (binned >90 % gated), excluding each engagement bin (which hosts the
     sanctioned freeze acknowledgement) and excluding fx one-shots (move-caused
     by construction — a punch from a standing dancer is causation, not
     phantom). Corpus keeps a real still window so the gate stays proven.
- **Results on real dancers:** benchmark_dance — identity SOLAR COMET
  (E phrygian/neon), full arc, 5/5 stems earned, GOLD streak, EARTHQUAKE combo,
  1/1 gold-move hit, coupling 0.40, phantom 0.0, in-scale 100 %: **all ten
  SLOs pass on real video.** right_dance (same performer, 2nd take) — TURBO
  PULSE (D phrygian), SUPERNOVA combo, 9/10 (coupling 0.319 vs 0.35 — honest
  marginal miss on a flatter-energy take). Similar identities for the same
  body = the system being faithful.
- **Sonic push:** FIRE+ streak turns lead notes into ascending chord-tone
  **arpeggio runs**; **THE LIFT** — reaching PEAK transposes the whole world
  +2 semitones (locked loop buffers transposed with it; undone on cool-down);
  **polyrhythmic texture** (3- or 5-pulse euclidean against the 4/4, rotating
  per bar); KNOCKOUT = three ascending chord stabs; **freeze wind-down** (a
  soft falling run the moment you truly freeze); 32nd-note **hat ratchets**
  inside fills. All deterministic; in-scale stays 100 % via the segment-aware
  metric (tonic changes are part of the timeline).
- **Visual push (stage.py):** section-tinted **aurora** background (heat-
  brightened, rolling bands); **shockwave rings** from the dancer on every big
  flash; deterministic golden-angle **confetti** on PERFECT!; lane note-sparks;
  the dancer **gilds gold** at GOLD streak. Session-film mode: compose(draw_
  body=False, cam_gain=0.85) — the real camera + image-space glow IS the
  dancer.
- 150/150 tests (salience cap, dwell debounce, peak lift round-trip). Corpus
  bundle still 10/10 SLOs, byte-deterministic.

Open: right_dance coupling 0.319 (threshold 0.35) — flat-energy takes correlate
less; consider energy-derivative coupling features. Session films render ~2x
realtime offline; live stage cost of aurora/shockwaves unmeasured on gallery
hardware (G7 probe still open).

## Log
(newest at bottom)

### Entry 0 — setup
- Probed network, kicked off mediapipe/opencv install, wrote this worklog.

### Entry 1 — real pose data acquired ✅ (with caveats)
- Network: many CDNs 403 (proxy allowlist); huggingface 403; **github raw +
  codeload work**; api.github.com **rate-limited** (shared IP) so no
  search/trees discovery — must use exact known URLs.
- mediapipe 0.10.35 dropped `mp.solutions`; uses **Tasks API**
  (`vision.PoseLandmarker`) + a `.task` model. Downloaded `pose_landmarker_full`
  from storage.googleapis.com. Needed system libs `libgles2 libegl1` (apt).
- Source: `intel-iot-devkit/sample-videos` (raw). 3 distinct real people:
  - bolt-detection (Usain Bolt running) — detected **28%**, legs poor.
  - face-demographics-walking (woman walking to camera) — **37%**.
  - face-demographics-walking-and-pause — **78%**, has real stop/start → good
    for the stillness test.
- ⚠️ PROBLEM: these are surveillance/sports clips → **poor lower-body
  visibility** (ankles 0.12–0.47, hips 0.29–0.76). My bounce signal = COM
  height *above the feet*, so it will break when feet aren't seen.
  Couldn't find full-body dance clips (API rate-limited, CDNs blocked).
- DECISION: (1) make engine **visibility-robust** (real improvement anyway);
  (2) treat these as the real test bed for distinctness + stillness;
  (3) verify "one body part → one audio thing" with a controlled single-joint
  probe on a real skeleton; (4) add an offline **synth→WAV** so musicality is
  actually audible, not just stats.
