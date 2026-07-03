# everybodyDance V2 — Assessment & a Better Approach

**Status:** proposal · **Scope:** a critical review of how the current system turns
dance into music, and a concrete redesign that widens the body's authorship of the
music without repeating the failures the WORKLOG already paid for.

---

## Part 1 — What the system actually does today

Strip away the marketing and the pipeline is:

```
pose (13 joints, hip-centred, torso-normalised)
  → ~14 scalar signals/frame  (energy, openness, height, 4 Laban efforts, …)  [features.py, laban.py]
  → tempo/phase: bounce-peak intervals → integer-BPM latch; accent PLL       [groove.py: RhythmCoherence, TempoLatch, KineticFlux, PhaseServo]
  → per-bar COMMIT: density level 0–3 per stem, slewed ±1/bar, skill-gated   [studio.py:_on_bar]
  → CONTENT: curated template libraries indexed by those levels
       drums: 4 styles × 4 levels (16 hand-written bars)                     [groove.py: STYLES]
       bass:  locked to the kick, root/fifth/octave                          [groove.py: bass_plan]
       keys:  pad or style-slot stabs                                        [groove.py: keys_plan]
       lead:  motif from 10 rhythm templates + the height contour            [groove.py: MotifWriter]
       harmony: a fixed 4-chord loop cycling with loop position              [substrate.py, studio.py:_on_step]
  → velocity/pan/cutoff modulated continuously by the body
  → 10 hard-coded gestures + combos fire discrete payoffs                    [gestures.py, game.py]
  → key/mode/progression/kit chosen once at calibration ("Dance DNA")        [identity.py]
```

Two details of that last stage matter for the assessment. The *coarse* identity
layer (weight/jerk/openness → palette/mode/register) is honestly body-derived.
The *fine* layer is not: tonic, progression, kit, style and stage name are
picked by slicing a **CRC-32 hash of the calibration statistics' decimals**
(`identity._dna`: `dna % 8`, `dna // 32`, `dna // 320`, …). Two movers differing
in the fourth decimal of `energy_std` get different chord progressions for no
perceptual reason. Per-visitor uniqueness is real, but it is partly
*manufactured novelty*, not derived character.

The development history (WORKLOG entries 13–14) is the essential context. The
project has already traversed both ends of the reactivity–musicality spectrum:

- **v1 — the body micro-manages notes** (entrained Hopf clock + per-step euclidean
  recomputation + pitch from instantaneous height). Owner verdict: *"terrible —
  drums spamming off-beat, no coherence, melody random."* A beat only exists if
  it repeats.
- **v2 — templates + modulation** (the current architecture). First owner verdict:
  *"sounds completely decoupled from movement — no downbeat on significant
  movements"* — fixed by the PhaseServo. What remains is musical and on-beat,
  but the body's authorship has been narrowed to what couldn't break the music.

That narrowing is the point to understand precisely. **The dancer's total
information channel into the music today is roughly:** an integer BPM + grid
phase; five density levels (0–3, slewed, so ~1 bit/bar/stem); note velocities;
one scalar height contour sampled at motif-compose time; ten discrete gestures
(salience-capped); and a one-time identity draw at calibration. Everything else —
the drum patterns, the chord loop, the comp rhythms, the motif grammar — is
**pre-authored content the dance selects among**, from a library of a few dozen
bars total.

## Part 2 — Merits: what is genuinely right (and must be kept)

An honest review has to start with the fact that several hard lessons here are
*correct* and expensive to relearn:

1. **Commit structure at musical boundaries, modulate within it** (`groove.py`).
   The single most important principle in the codebase. Direct sonification of
   continuous movement was tried and failed. Any richer generator must still
   emit *bar plans at bar lines*, never per-frame note decisions.
2. **The body as an audio signal** (`KineticFlux` + `PhaseServo` + trough-based
   pulse detection). Onset-envelope + PLL beat alignment is well-engineered and
   *measured*: 82 % of a steady dancer's onsets land on the downbeat (~35 ms
   median), incoherent accents are deliberately not chased, and an integral term
   trims tempo bias. Keep wholesale.
3. **The tempo latch** (integer BPM, re-lock only at bar lines after sustained
   deviation). The Hopf oscillator — still marketed as "the heart" in the README —
   is in truth a cold-start fallback (it wandered 64–124 BPM on a clean 2 Hz
   bounce; WORKLOG entry 13). The latch is the real heart. The README should say so.
4. **Perceptual-causality engineering**: the phantom gate (stillness actually goes
   quiet, notes ring out), the dwell-debounced stillness gate, per-move salience
   caps, the fill budget, accent answers snapped to the nearest 16th. This is
   taste, earned on real video, that a naive redesign would lose.
5. **Determinism + replay + SLO harness + LLM judge**. Same dance → byte-identical
   song, and every perceptual claim has a [BASH]/[VISION] check. This is an
   unusually strong evaluation story and it is the reason the project can be
   iterated at all. Non-negotiable to preserve.
6. **Per-person normalization** (your range → the full musical range) and the
   skill loop (`RhythmCoherence` gating richness) — both align incentives with
   legibility.

## Part 3 — Where "music generated to match the dancing" breaks down

### W1 — The representation bottleneck
Everything the system knows about *how you move* is ~14 hand-computed scalars
plus 10 threshold-based gesture detectors. Shape, style, which limb, spatial
pattern, phrasing, quality of gesture — collapsed to energy/openness/height and
75 lines of hand-tuned Laban arithmetic (`laban.py`). Consequence: **two visibly
different dances with similar energy statistics produce essentially the same
music.** A krumper and a waacker with matching intensity land in the same drum
level, the same velocities, nearly the same motif. The system responds to *how
much* you dance, barely to *how* you dance.

### W2 — The content ceiling
16 drum bars, ~20 curated chord loops, 10 motif rhythm templates. The "Dance
DNA" combinatorial space (tonic × mode × progression × kit × style) makes
*identities* distinct, but within a session the band has a few dozen bars of
vocabulary. For an exhibit this reads as coherence; for a **game** (the stated
goal) it is a replayability ceiling: session three sounds like session one.

### W3 — Harmony is not danced
The chord loop cycles mechanically with loop position
(`set_field_index((step*n)//loop_steps)`). The body's entire harmonic channel is
ECLIPSE (scale flip) and THE LIFT (+2 transpose at PEAK). No tension/release, no
cadence, no harmonic response to a choreographic climax. Harmony is the layer
where "generated to match" is currently *false* — it is a fixed loop with a
per-visitor skin.

### W4 — The dancer's own structure is ignored
Dancers phrase (8-counts), repeat motifs, build and release. Nothing in the
pipeline detects that. The song arc is a heat meter integrating energy
(`SongArc.update_frame`); phrase structure is a fixed 4-bar cycle. The strongest
available causality signal at the phrase timescale — **"when I repeat my move,
the music brings back its hook"** — is unexploited. (The MotifWriter persists a
motif, but its persistence is clocked by bar counters, not by the dancer's
recurrence.)

### W5 — Magic-number fragility
`Z_ON = 2.2`, `PROMINENCE = 0.015`, `motion_floor = 0.06`, `drive_ref = 0.02`,
per-gesture geometry thresholds (`_punch` alone carries six), streak/arc heat
constants, `biased_config` cutoffs "set against observed real-mocap scales" —
tuned against one synthetic dancer and two real videos, which are also the
corpus the SLOs are tuned to pass on (`right_dance` sits at coupling 0.319
against a 0.35 threshold — the tuning set's own edge). The normalizer absorbs
some variance, but robustness across bodies, cameras and frame rates rests on
constants with n≈3 evidence. Related code-health issues compound it: gesture
debouncing is **split-brain** (per-move `cooldown` in `gestures.py`, the real
anti-spam salience cap in `studio.py` — neither is authoritative);
`MappingResolver.signal` silently resolves a typo'd source to `0.0`, muting a
stem with no diagnostic; JUMP/STOMP silently never fire on sources that don't
populate `root_y`; and the DTW "record your own move" path is unused in the
shipped live app (no key binds it), so a whole advertised capability is
decorative.

### W6 — The evaluation harness proves consistency, not quality
The harness is the project's best asset, but several of its green lights are
circular: `in_scale_pct ≥ 99` measures the substrate's snapper (which cannot
emit out-of-scale notes), not musicality; `on_grid_pct` measures drum onsets
against **the grid the engine itself latched** — self-consistency, not "in time
to a listener"; `bar_similarity ≥ 0.5` rewards repetition and therefore cannot
distinguish groove from monotony; `coupling.score` is Pearson correlation and
cannot distinguish expressive response from trivially proportional response.
The LLM judge grades a contact sheet and a piano-roll — **it never hears the
audio**, so "sounds good" is graded from pictures. And `check_slo` skips
missing keys, so a bundle can pass by simply not emitting a metric. The WORKLOG
knows this: every genuinely musical verdict in it came from a human listening
("owner verdict: terrible…"), not from the SLOs, which were green at the time.

### W7 — The game is a veneer on an installation
The KICKOFF explicitly said "not a game," then relaxed it mid-flight; the game
layer (arc/streak/combos/gold moves) is payoff-only meters bolted onto the
instrument. There is no core loop of challenge → performance → judgment →
progression, because judging the *person* was (rightly, for a gallery) banned.
But the user-stated goal is an **interactive game based on Just Dance, inverted**:
the music matches the dancing. That inversion deserves its own loop, not Just
Dance's HUD on top of an ambient instrument.

### The honest kernel, hiding in the offline mode
One module realizes the thesis literally: `builder.py`. In the song-builder,
your extremity hits **place the onsets** (no euclidean pattern, no RNG) and your
crouch→rise contour **sets each onset's pitch** as the rhythm replays. That is
the body genuinely authoring notes — and it is confined to the offline render
path, flagged in the WORKLOG as never validated live. The trajectory of the
project has been: every time authorship was widened it sounded bad, and each fix
(entries 4, 13, 14) moved authority away from the body toward curated structure
— because *unstructured* authorship is what sounded bad. The design conclusion
V2 draws: don't shrink the channel, **structure it** — let the body author rich
content through a representation that guarantees musical form, instead of either
raw scalars (v1) or template indices (today).

## Part 4 — The better approach: the dance is the score

Keep the groove spine (latch, servo, commit-at-boundaries, phantom gate,
substrate guardrails, determinism). Replace the three weakest organs — movement
representation, content generation, structural form — and rebuild the game on
top of what they enable.

```
pose stream
  ├─ groove spine (KEEP): KineticFlux → PhaseServo → TempoLatch → bar grid
  ├─ A. MOTION ENCODER (new): skeleton windows → embedding zₜ + move events + effort axes
  ├─ B. STRUCTURE LISTENER (new): self-similarity over zₜ → phrases, motif recurrences, novelty
  ├─ C. CONDITIONED GENERATOR (new): (zphrase, level, style vec, section, seed)
  │       → bar plans (drums/bass/keys/lead) + chord transitions,
  │       snapped by the substrate, slewed/skill-gated as today
  └─ D. GAME LOOP (rebuilt): author-your-hook verses / re-perform-your-hook choruses
```

### A. Motion encoder — replace scalars-and-thresholds with a learned embedding

- A small skeleton encoder (ST-GCN/AGCN-class or a temporal ConvNet, 1–5 M
  params) over 1–2 s windows of the existing 13-joint stream, hopped at ~10 Hz.
  These models are tiny; int8 ONNX inference is comfortably CPU-real-time, and
  the consumer runs **at bar boundaries**, off the per-frame hot path — exactly
  the commit-at-boundaries discipline.
- Train self-supervised (masked-motion / contrastive) on **AIST++** (10 M frames
  of multi-genre dance with 3D keypoints, music-paired) plus the repo's own
  deterministic session corpus. Small supervised heads on top:
  - **move recognition** (superset of the current 10; per-move confidence) —
    replaces threshold gestures; the DTW template path stays for user-recorded
    moves and as the no-model fallback;
  - **effort axes** (weight/time/space/flow regression, distilled from the
    current heuristics + a small hand-labelled set) — replaces `laban.py`'s
    arithmetic with something that generalises across bodies;
  - the raw **embedding zₜ** for similarity and conditioning.
- Fallback tier: no model file present → current heuristics. Fail-soft, per the
  KICKOFF's own principles.

This directly attacks **W1** and **W5**: the thresholds stop being load-bearing,
and "how you move" finally has more dimensions than "how much."

### B. Structure listener — mirror the dancer's own form (no ML required)

Maintain an online **self-similarity matrix** over the session's embeddings
(cosine similarity; with Phase 0 it can run on the existing scalar feature
vector and already work passably):

- **Phrase boundaries**: novelty-kernel peaks on the SSM diagonal (the standard
  audio structure-analysis trick, applied to motion), snapped to the nearest bar
  → cadence scheduling, section-change proposals feeding `SongArc` (which stops
  being a pure heat meter).
- **Motif recurrence**: when the current window matches a past window above
  threshold → *the dancer repeated themselves* → recall the musical material
  generated the first time. Concretely: `MotifWriter` becomes a **motif bank**
  keyed by movement-cluster id; recurrence retrieves and re-roots the hook
  instead of composing fresh.
- **Novelty**: a sustained low-similarity stretch (new movement vocabulary)
  licenses new material — the generator's conditioning changes, and the band
  audibly turns a corner *because you did*.

This attacks **W4** with the highest causality-per-engineering-hour in the whole
plan: "I repeat my move and my hook comes back" is legible to a first-time
visitor within one chorus, and it is provable offline (a scripted
repeat-your-move probe with a hook-recall SLO).

### C. Conditioned generation — the template library becomes a prior, not the product

Replace fixed content indexed by density with a **small conditional symbolic
generator emitting exactly the data structures `groove.py` already commits**
(16-slot drum bars, bass plans, comp slots, motif slots):

- **Model**: a tiny bar-level transformer (or, first iteration, a factored
  Markov/grammar model) trained on the Groove MIDI Dataset (drums) and
  Lakh/Hooktheory-class symbolic data (bass/keys/melody), with conditioning
  tokens for intensity level, style vector, section, and a summary of the last
  phrase's motion embedding.
- **Determinism**: sampling seeded by `(session_seed, bar_index,
  hash(conditioning))` → same dance still yields the byte-identical song.
  Replay/eval story intact (G10 survives).
- **Guardrails unchanged**: generated pitches pass through the substrate's
  scale/chord snapping; generated bars pass through the same ±1 level slew, the
  skill gate, and the fill budget. The curated `STYLES` stay in the box as the
  degraded-mode tier and as training-prior anchors.
- **Harmony becomes danced** (fixes **W3**): replace the position-cycled loop with
  a functional-harmony transition model per mode (a weighted automaton is
  enough), whose *tension* input comes from the motion effort axes and
  contraction/expansion, and whose *cadences* land on the structure listener's
  phrase ends. Dance DNA seeds the transition weights — per-visitor uniqueness
  deepens (your world has harmonic habits, not just a key).
- **Melody**: motifs conceived from a multi-channel contour (height + dominant-hand
  trajectory + effort), stored in the motif bank (B), developed by the generator
  rather than one-slot mutation.

This attacks **W2** and **W3**: content variety scales with the model instead of
the hand-written library, while the substrate + commit discipline keep the
v1 failure mode (random-sounding output) structurally impossible.

### D. The game, rebuilt on authorship — the Just Dance inversion made explicit

Just Dance's loop is: *the game prescribes moves; you are judged against them.*
The honest inversion is not "remove the judging" (that's the current veneer) —
it is: **you author the choreography, then the game asks for it back.**

- **Verse (author)**: free dance. The system builds hooks from your movement
  motifs (B + C) and *shows you what it learned* — "THAT'S YOUR HOOK" with the
  ghost-skeleton replay of the move that made it.
- **Chorus (perform)**: the game presents *your own move* as the challenge card
  (Just Dance-style pictogram = your recorded ghost) and opens a timed window.
  Match quality = embedding/DTW similarity to *your own template* — judged
  against yourself, so the no-body-shaming line holds — and it directly gates
  how gloriously the hook returns (full band vs. thinned recall).
- Streaks, combos, gold moves stay as arcade seasoning; the currently-unused
  DTW template machinery (`gestures.record_template` — advertised but never
  bound in the live app) becomes load-bearing, auto-fed by the motif bank
  instead of manual recording. And the verse's authorship passes are
  `builder.py`'s two body-authored passes (hits place onsets, contour sets
  pitch) promoted from the offline mode into the live loop — structured this
  time by the bar-commit discipline and the motif bank rather than raw.
- **Session artifact**: the deterministic render + choreography ghost = a
  shareable "your song" — and, crucially, every session becomes a new corpus
  file, closing the KICKOFF's "no real visitor sessions" gap (its own
  highest-value missing asset) as a side effect of people playing.

This gives the project an actual game loop — challenge, performance, judgment
(against yourself), progression (your song grows) — while making the
music-from-dance causality *the* mechanic rather than an ambient property.

## Part 5 — Evaluation: extend the harness to measure what V2 claims

Keep every existing SLO. Add:

| new check | what it proves | type |
|---|---|---|
| **Retrieval coupling**: given a rendered session, a simple classifier must match it to the correct motion sequence among N decoys (score ≫ 1/N) | the music is *specific to* the dance, not merely correlated with dancing-in-general — the thing the current correlation metrics cannot see | [BASH] |
| **Beat Alignment Score** (AIST++ metric) between movement onsets and musical beats | grid alignment, comparable to published dance-music work | [BASH] |
| **Hook-recall probe**: scripted performer repeats a motif at t₀ and t₁ → the bar plans at t₁ must recall t₀'s material above a similarity floor | W4 is actually fixed | [BASH] |
| **Cross-session variety**: distribution distance between generated content across corpus sessions (the template library scores ≈0 by construction) | W2 is actually fixed | [BASH] |
| **Cross-dancer specificity**: swap two dancers' sessions through the same seed → generated content must differ beyond key/kit | W1 is actually fixed | [BASH] |
| **The judge hears**: grade the rendered `music.wav` (audio-capable judge model, plus objective audio features — the beat-spectrum check already exists), not just stills and a piano-roll | closes the W6 hole where "sounds good" was graded from pictures | [VISION→AUDIO] |
| **SLO manifests**: each bundle declares the metric keys it must emit; `check_slo` fails on a missing key instead of skipping it | a bundle can no longer pass by not reporting | [BASH] |

## Part 6 — Phasing (each phase ships with SLOs green)

- **Phase 0 — no ML, immediate wins.** Structure listener (B) on the existing
  scalar features; harmony transition automaton with movement-driven tension;
  motif bank keyed by feature-space clusters; hook-recall + retrieval SLOs.
  Pure NumPy, deterministic, CPU-trivial. This alone converts the weakest
  claims (W3, W4) and is the recommended first PR.
- **Phase 1 — the encoder.** Import/train the skeleton encoder; swap gestures
  and Laban to learned heads with heuristic fallback; re-cluster the motif bank
  on embeddings; add cross-dancer specificity SLO.
- **Phase 2 — the generator.** Conditional bar-plan model behind the substrate
  guardrails; curated STYLES demoted to fallback tier; variety SLO.
- **Phase 3 — the game loop.** Verse/chorus authorship mechanic; session
  artifacts as corpus; retune thresholds on the newly-abundant real sessions.

## Part 7 — Risks, honestly

- **CPU budget (P5/G7).** Mitigations: models are 1–5 M params int8; inference
  scheduled at bar boundaries (≥2 s apart at 120 BPM) with the current frame
  loop untouched; hard fallback tier. The latency probe the KICKOFF already
  demands (G7, still unbuilt) becomes a prerequisite, not an afterthought.
- **Determinism vs. floating point.** Pin the ONNX runtime and quantize
  conditioning before hashing; integer-seeded sampling. G10's double-run test
  is the tripwire and stays in CI.
- **Data licensing.** AIST++ / Groove MIDI are research-licensed — fine for the
  prototype; an exhibit/commercial build needs a data pass (or distillation
  onto self-collected sessions, which Phase 3 generates organically).
- **Model failure modes.** A generator can emit a bad bar; the substrate snap +
  slew + skill gate bound the damage to "plain," never "wrong," and the
  fallback tier bounds it to "today's behavior."
- **The v1 ghost.** The single biggest process risk is re-learning entry 13's
  lesson the hard way. The design rule that guards it: *no component may make a
  per-frame note decision; all content is committed at bar lines.* Every new
  module above obeys it.

---

*Companion reading: `docs/KICKOFF.md` (the goals and evaluator model this design
inherits), `WORKLOG.md` entries 13–14 (the failures this design is built not to
repeat).*
