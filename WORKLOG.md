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
