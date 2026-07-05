# V2 ML Stack — models, data, and the training plan

**Status:** proposal (companion to [`V2_DESIGN.md`](V2_DESIGN.md)) · **Verified against the
model landscape as of July 2026.**

The brief, sharpened: *genuinely fun, sounds good, personal to how you dance and
who you are, with game elements.* Five models cover it. Three can start from
released checkpoints; two we train ourselves (both are single-GPU-scale, not
lab-scale). Everything keeps the V2 rule — no per-frame note decisions; ML runs
at window/bar boundaries; the deterministic symbolic layer remains the ground
truth for replay and eval.

One decision up front: **target hardware moves from "any CPU" to an Apple
Silicon Mac (even a Mac mini) or a consumer NVIDIA GPU.** The current DSP synth
stays as the CPU-only fallback tier, so nothing existing breaks. This is the
price of "sounds good," and in 2026 it is a cheap price.

---

## The five models

### M1 — Motion encoder: *how you dance*, finally represented

The replacement for the ~14 hand-computed scalars and 10 threshold gestures.

- **Architecture:** ST-GCN/CTR-GCN-class graph conv or a small motion
  transformer over the existing 13-joint MediaPipe stream. 2–10 M params, int8
  ONNX. Runs on 1–2 s windows at ~10 Hz hop — real-time on CPU, trivial on the
  new target hardware.
- **Pretraining (self-supervised):** masked-motion prediction + contrastive
  views, on a pooled skeleton corpus (below), with aggressive augmentation
  matched to our deployment reality: MediaPipe jitter, joint dropout/occlusion,
  variable frame rate, camera angle — the model must survive a webcam, not a
  mocap stage.
- **Heads (small, supervised/distilled):**
  1. **Embedding `z_t`** — the currency of the whole system: self-similarity
     (structure listener), motif clustering, conditioning for generation,
     move-match scoring.
  2. **Move recognition** — supervised on labeled sets, optionally
     text-aligned (T-MOR-style skeleton↔text contrastive alignment) so the game
     can *name* moves it was never explicitly taught (open-vocabulary flash
     cards instead of ten hard-coded predicates).
  3. **Style/effort axes** — regression distilled from the current Laban
     heuristics plus a small hand-labeled set; replaces `laban.py`'s
     arithmetic with something that generalizes across bodies.
  4. **Tracking-quality confidence** — gates everything downstream (fail-soft).
- **Fallback tier:** current heuristic gestures + Laban, selected automatically
  when no model file is present.

### M2 — Choreo-musical style translator: *who you are* → your sound

The replacement for the CRC-32 "Dance DNA" hash. Maps a session's motion-style
embedding into **music-style space** — concretely, into (a) MusicCoCa embedding
space (the joint music–text space Magenta RealTime uses for steering) and (b)
discrete style tokens for the symbolic generator.

- **Architecture:** a small projection/MLP (or shallow transformer over
  per-phrase embeddings) trained contrastively on **paired dance–music data**:
  AIST++ and FineDance clips give (motion, music) pairs across street-dance and
  choreography genres — the ChoreoMaster insight, done with 2026 tools.
- **What it buys:** identity becomes a *continuous, perceptually meaningful*
  vector — a lyrical, flowing mover genuinely lands in a different sonic world
  than a popper, because the mapping was learned from how real dance styles
  pair with real music, not from hash arithmetic on calibration decimals.
  Deterministic (the encoder is deterministic), so same body + same dance →
  same world still holds.
- **Persistence = "who you are":** a local player profile — an EMA of session
  style vectors plus explicit taste nudges (a genre picker, skip/keep feedback)
  — so your world *deepens across visits* instead of being re-drawn from a
  20 s calibration. This is the personalization the current system promises
  and can't deliver.

### M3 — Groove & symbolic generator: the band's brain

Three small models, all emitting the bar-plan data structures `groove.py`
already commits, all seeded → deterministic replay preserved, all passing the
existing substrate/slew/skill-gate guardrails.

1. **Drums — a GrooVAE-class seq2seq ("tap2drum + humanize").** This is the
   single best model-fit in the whole project: it maps a *sparse onset
   pattern* to a full expressive drum bar (velocities, microtiming, feel) in a
   drummer's style. Our `KineticFlux` already emits exactly that sparse onset
   pattern from the body. **Your hits, interpreted by a real drummer** — the
   magic moment, and Magenta released trained checkpoints; retraining from
   scratch is hours, not days (the dataset is 13 h).
2. **Harmony — a chord-progression LM with tension conditioning.** First
   iteration: a weighted functional-harmony automaton (no training). Second: a
   tiny transformer trained on open progression corpora, conditioned on mode,
   style token (M2), and a movement-derived tension signal; cadences scheduled
   by the structure listener's phrase ends.
3. **Melody/motif — a motif continuation model.** Small transformer,
   conditioned on chord, style token, and the multi-channel body contour
   (height + dominant-hand trajectory + effort); develops the motif bank's
   hooks instead of `MotifWriter`'s one-slot mutation.

### M4 — Neural audio renderer: *sounds good* (the ambitious bet)

**Magenta RealTime 2** — the open-weights live music model (Apache-2.0 code,
**CC-BY-4.0 weights**, i.e. commercially usable with credit):

- `mrt2_small` (230 M) streams **in real time on any Apple Silicon Mac**;
  `mrt2_base` (2.4 B) needs Pro/Max chips; both run offline on any NVIDIA GPU.
- Steerable continuously by **text prompts, audio examples, and MIDI**, with
  frame-level control at ~200 ms response — and its style space is MusicCoCa,
  which is exactly where M2 projects your personal style vector. *Your dance
  style literally becomes the prompt.*
- **Two-tier audio architecture** (this is the key design move, and it is the
  project's own timescale thesis applied to rendering):
  - **Tight tier — the existing DSP synth**: drums, bass, accent answers, gesture
    one-shots. Sample-accurate, <80 ms motion→sound, deterministic. The
    causality-critical layer stays exactly as engineered today.
  - **Rich tier — Magenta RT 2**: pads, texture, atmosphere, transitions,
    the "produced" feel. Steered by the symbolic layer's chords/section/BPM
    (MIDI + style embedding); its ~200 ms–2 s response time is *appropriate*
    for the slow musical layers it renders. Fast lane → DSP, slow lane →
    neural: rate-matched, like everything else in the system.
- **Determinism note:** neural audio on GPU is not bitwise-reproducible across
  devices. The symbolic event stream remains the deterministic ground truth
  (SLOs/eval unchanged); the neural render is presentation. Offline renders
  can pin seeds + backend for near-identical output.
- **Fallback tier:** no model / CPU-only → today's `timbre.py` synth.

### M5 — Move-match scorer: the game's judge

For the verse/chorus loop (re-perform *your own* move) and gold-move
challenges:

- **Base:** cosine similarity in M1 embedding space + the existing whitened-DTW
  as a calibrated second opinion. Zero training needed to start.
- **Better:** a tiny scorer head trained *without human labels*: positives are
  augmented re-performances of a clip (time-warp, mirror, noise, partial
  occlusion), hard negatives are other moves from the same session/corpus.
  Output calibrated to PERFECT / GOOD / OK bands. Judged against your own
  template → the no-body-shaming line holds by construction.

### What we deliberately do NOT need

- **End-to-end dance-to-music research models** (D2M-GAN / LORIS / RVQ
  dance-to-music pipelines): offline, uncontrollable, unstable quality, and
  they erase the deterministic symbolic layer the whole eval story rests on.
  We take their *lesson* (learn the choreo-musical correspondence — M2) and
  skip their architecture.
- **Cloud music-gen APIs:** latency, cost, no offline gallery mode, output
  licensing friction. Everything above runs on the box.
- **Video models:** MediaPipe pose already streams on CPU; the gap was never
  pose estimation.

---

## Data

| dataset | what for | size | license |
|---|---|---|---|
| **AIST++** | M1 pretrain + M2 pairs (street-dance genres ↔ music) | 1.4 k seqs / 10.1 M frames, 10 genres, 30 dancers, music-paired | research — audit before commercial |
| **FineDance** | M1 pretrain + M2 pairs (choreography, 22 genres) | ~14.6 h mocap, music-paired | research |
| **AMASS / BABEL, HumanML3D, NTU RGB+D 120** | M1 self-supervised pretraining bulk + move-label supervision | 100 k+ clips combined | academic/research |
| **Groove MIDI Dataset + E-GMD** | M3 drums (tap2drum, humanization) | 13.6 h + 444 h, real drummers | **CC-BY 4.0** ✅ |
| **Chordonomicon** | M3 harmony LM | 666 k progressions | **open** ✅ |
| **POP909, Hooktheory/TheoryTab** | M3 melody/harmony fine-tune | 909 songs / ~20 k excerpts | research / restricted — optional |
| **Lakh MIDI (+ GigaMIDI)** | M3 melody pretraining | 176 k+ MIDI files | research/mixed |
| **Magenta RT 2 weights (incl. MusicCoCa, SpectroStream)** | M4 | 230 M / 2.4 B | **CC-BY 4.0** ✅ |
| **Our own sessions** (the flywheel — see below) | everything, forever | grows per play | ours, with consent UX |

**The licensing picture is honest and workable:** the three most load-bearing
external components (Groove MIDI family, Chordonomicon, Magenta RT 2 weights)
are already permissive. The research-licensed motion sets are fine for the
prototype; a commercial release needs either a data audit or the flywheel:

**The data flywheel is the game design.** Every session already produces a
deterministic `.session` replay (the KICKOFF's own "highest-value corpus,
currently missing"). The game loop *labels it as a side effect of play*:
gold-move windows are weak labels for move recognition; re-perform scores are
similarity labels for M5; skip/keep feedback labels M2's taste mapping. Ship
the game → collect (with consent) → fine-tune M1/M2/M5 on real players in real
rooms → the personalization gets visibly better with the player base. This is
the moat none of the datasets provide.

## Training budget (single-GPU-scale, deliberately)

| model | from scratch? | compute |
|---|---|---|
| M1 encoder + heads | yes (self-sup pretrain) | days on one A100/4090 — skeleton data is tiny |
| M2 style translator | yes | hours (it's a projection trained on ~15 k pairs) |
| M3 drums | **released GrooVAE checkpoints** (TF1-era; a PyTorch retrain is hours on GMD) | hours |
| M3 harmony/melody | automaton first (no training) → small transformers | days |
| M4 Magenta RT 2 | **released**; optional later LoRA-style style-tuning | none to start |
| M5 scorer | embedding-space to start; tiny head later | hours |

## Phasing — fun first, training later

- **P0 — zero-training "wow" spike (weeks, not months):**
  `KineticFlux` onsets → GrooVAE tap2drum → *your hits come back as a real
  drummer's groove*, on the tight tier; Magenta RT 2 small as the rich tier,
  steered by a text prompt derived from the coarse (honest) identity layer +
  section/BPM/chords from the existing engine. Two released checkpoints, both
  CC-licensed, bolted onto the pipeline as it stands. This alone transforms
  how the instrument sounds and feels, and proves the two-tier architecture.
- **P1 — M1:** pretrain the encoder; swap gestures/Laban to learned heads with
  heuristic fallback; structure listener moves onto embeddings.
- **P2 — M2:** train the style translator on AIST++/FineDance pairs; retire the
  CRC hash; persistent player profiles.
- **P3 — M3:** harmony automaton → learned chord LM; motif model; motif bank.
- **P4 — M5 + the game loop:** verse/chorus authorship mechanic, calibrated
  scoring, consent UX, the flywheel starts turning.

Each phase keeps `render_studio` SLOs green on the symbolic stream, adds its
own probe (V2_DESIGN Part 5), and keeps a no-model fallback tier so the
CPU-only build never dies.

## Risks specific to the ML stack

- **Domain gap: mocap-trained encoder vs. webcam skeletons.** Mitigation is in
  the augmentation pipeline (MediaPipe-noise simulation) + fine-tuning on the
  flywheel's real sessions; the confidence head gates degraded tracking.
- **Magenta RT 2 musical coherence under MIDI steering** — it's a
  texture/vibe engine, not a precision sequencer; that's why drums/bass stay
  on the tight tier and the neural tier only renders the layers where ~200 ms
  is musically acceptable. The P0 spike exists to validate exactly this split
  before anything is trained.
- **Two audio worlds blending** (DSP tier + neural tier must sound like one
  band): shared BPM/key from the latch/substrate, sidechain the neural bed
  under the DSP kick (the master-bus glue already exists), and let the neural
  tier own reverb/space so the DSP tier sits inside *its* room.
- **Bitwise determinism** ends at the neural audio boundary — documented above;
  eval stays symbolic; G10's double-run test unchanged.
