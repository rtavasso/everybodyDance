#!/usr/bin/env python3
"""Offline gradability harness for the Studio -- the deterministic replay that
PROVES the Studio is good (KICKOFF S4: every goal provable by an evaluator with
a shell + the ability to view images).

A synthetic dancer is driven, frame by frame, through the multi-stem Studio with
a deterministic *command choreography* (gestures force-fired at musical times).
The emitted note stream is rendered to a *timbre-aware* WAV (so timbre_morph and
the per-stem cutoff/drive/gain/reverb/delay automation are audible), the run is
photographed (skeleton + flash stills + a 5-stem piano-roll), and the four kinds
of "good" are quantified (coupling / musicality / liveliness / timing) and gated
by the SLOs in metrics.py. Like grade.py, it is a real regression signal: same
dance + same commands replays byte-identically.

    uv run python -m tools.render_studio                 # default synthetic corpus
    uv run python -m tools.render_studio --scripted      # over scripted_performer
    uv run python -m tools.render_studio --judge         # also call Claude (needs key)
    uv run python -m tools.render_studio --seconds 24 --out out/studio

Bundle (in --out, default out/studio/): contact_sheet.png, pianoroll.png,
music.wav, metrics.json, prompt.txt, and (with --judge) scorecard.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import wave
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from everybody_dance import metrics as M
from everybody_dance import stems as stemlib
from everybody_dance import timbre as T
from everybody_dance.effects import Flash
from everybody_dance.features import FeatureExtractor
from everybody_dance.gestures import GESTURE_LIBRARY
from everybody_dance.judge import build_prompt, grade_bundle
from everybody_dance.mapping import GestureBinding, MappingConfig
from everybody_dance.output import LogBackend
from everybody_dance.pose import SyntheticPoseSource
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.studio import Studio, StudioConfig
from everybody_dance.voices import SR
from tools.render_gestures import XR, YR, make_profile, scripted_performer

# Drum pitch -> percussive preset (inverse of stems.DRUM_PIECE: 36/42/38/41).
DRUM_PRESET: Dict[int, str] = {
    stemlib.DRUM_PIECE["kick"]: "analog_kick",   # 36
    stemlib.DRUM_PIECE["hat"]: "noise_hat",      # 42
    stemlib.DRUM_PIECE["snare"]: "snare",        # 38
    stemlib.DRUM_PIECE["tom"]: "tom",            # 41
}
# fx one-shots carry their own channel; pick a preset by channel for pitched hits.
CH_PRESET: Dict[int, str] = {2: "warm_keys", 3: "supersaw_lead", 1: "sub_bass",
                             4: "glass_pad"}


# -- the default synthetic corpus ------------------------------------------

def corpus_layout(seconds: float) -> dict:
    """The four-phase corpus timeline scaled to fit `seconds`: groove ->
    tempo/energy/openness change -> a CLEAR stillness window (energy ~0) ->
    recovery. Returns the segment durations plus the stillness window bounds, so
    a genuine low-energy window ALWAYS exists (even at the short test duration)
    -- that's what makes coupling.phantom (G2) a meaningful, passing metric. The
    minimum length keeps each phase long enough to land low-energy bins."""
    seconds = max(float(seconds), 8.0)
    # Proportions: groove 0.30, change 0.30, stillness 0.22, recovery 0.18.
    groove = round(0.30 * seconds, 3)
    change = round(0.30 * seconds, 3)
    still = round(0.22 * seconds, 3)
    recover = round(seconds - groove - change - still, 3)
    return {"groove": groove, "change": change, "still": still,
            "recover": recover,
            "still_start": groove + change,
            "still_end": groove + change + still}


def corpus_mapping() -> MappingConfig:
    """The shipped default mapping plus one extra binding the corpus uses:
    PUNCH -> loop_clear(bass). A locked loop deliberately keeps replaying through
    the stillness gate (that's a feature: the dancer chose it), so the corpus
    clears the bass loop just before the stillness window -- otherwise the looped
    bass would sound during stillness and inflate coupling.phantom. Demonstrates
    the loop_clear action on top of arm/lock/replay."""
    cfg = MappingConfig.default()
    cfg.gestures.append(GestureBinding("PUNCH", "loop_clear", target="bass"))
    return cfg


def studio_script(seconds: float) -> List[dict]:
    """SyntheticPoseSource segments for :func:`corpus_layout`. The stillness
    segment is genuinely motionless (energy 0) so its bins carry ~no onsets."""
    L = corpus_layout(seconds)
    return [
        dict(duration=L["groove"], tempo_hz=2.0, energy=1.0, openness=1.0, asym=0.0),
        # tempo/energy/openness change: faster, punchier, narrower.
        dict(duration=L["change"], tempo_hz=2.6, energy=1.25, openness=0.55, asym=0.25),
        # a clear stillness window (energy ~0 -> the phantom gate engages).
        dict(duration=L["still"], tempo_hz=2.0, energy=0.0, openness=0.3, asym=0.0),
        # recovery: the body comes back, the band re-blooms.
        dict(duration=L["recover"] + 1.0, tempo_hz=2.1, energy=1.1, openness=0.9,
             asym=-0.1),
    ]


def command_choreography(fps: float, seconds: float) -> Dict[int, List[str]]:
    """Deterministic gesture injection keyed to the corpus phases (no RNG),
    demonstrating the gesture->action features end to end and keeping it
    musical: a fill, a bass loop (arm->lock), a lead timbre morph, a dark scale
    shift -- all before the stillness window -- then a breakdown, an EARTHQUAKE
    combo (STOMP x2), and a build during recovery. Also exercises the game
    layer: a CLAP x3 CLAP STORM combo mid-change, and a JUMP timed to land in
    the first gold-move window (bar 5 at the entrained ~120 BPM). Times scale
    with the layout so they always land in their intended phase."""
    L = corpus_layout(seconds)
    g, ss, se = L["groove"], L["still_start"], L["still_end"]
    raw: List[Tuple[float, List[str]]] = [
        (0.40 * g, ["JUMP"]),             # fill: a guaranteed drum burst
        (0.80 * g, ["SQUAT"]),            # loop_record bass: arm
        (g + 0.20 * L["change"], ["RAISE RIGHT"]),   # timbre_morph lead -> glass_pad
        (g + 0.25 * L["change"], ["JUMP"]),          # aimed at gold window (bar 5)
        (g + 0.40 * L["change"], ["SQUAT"]),         # loop_record bass: lock loop
        (g + 0.55 * L["change"], ["CLAP"]),          # CLAP x3 -> CLAP STORM combo
        (g + 0.62 * L["change"], ["CLAP"]),
        (g + 0.69 * L["change"], ["CLAP"]),
        (g + 0.80 * L["change"], ["ARMS CROSSED"]),  # scale_shift -> dark
        (ss - 0.6, ["PUNCH"]),            # loop_clear bass before the stillness
        (se + 0.20 * L["recover"], ["T-POSE"]),      # breakdown after stillness
        (se + 0.45 * L["recover"], ["STOMP"]),       # STOMP x2 -> EARTHQUAKE
        (se + 0.45 * L["recover"] + 0.4, ["STOMP"]),
        (se + 0.70 * L["recover"], ["HANDS UP"]),    # build / riser
    ]
    out: Dict[int, List[str]] = {}
    for t, moves in raw:
        if 0.0 <= t < seconds:
            out.setdefault(int(round(t * fps)), []).extend(moves)
    return out


def synthetic_corpus(seconds: float, fps: float) -> Tuple[np.ndarray, float, bool]:
    """An (T,13,3) sequence from SyntheticPoseSource over :func:`studio_script`.
    flip_y=False: SyntheticPoseSource already emits normalised (y-up) frames, so
    we wrap its xyz back through ArrayPoseSource without re-flipping (mirrors how
    grade.py treats scripted skeletons)."""
    src = SyntheticPoseSource(fps=fps, duration=max(seconds, 8.0),
                              script=studio_script(seconds))
    xyz = np.stack([fr.xyz for fr in src.frames()])
    return xyz, fps, False


# -- the replay -------------------------------------------------------------

def replay(xyz: np.ndarray, fps: float, flip: bool, *, scripted: bool,
           mapping: Optional[MappingConfig] = None,
           live_end_s: Optional[float] = None,
           commands_at: Optional[Dict[int, List[str]]] = None) -> dict:
    """Drive the synthetic dancer through the Studio, capturing everything the
    gradable bundle needs. Calibrate the Profile on the LIVELY part of the dance
    (skip the stillness window) so normalisation isn't poisoned by zero-energy
    frames, then run the Studio over the full sequence."""
    dur = len(xyz) / fps
    commands_at = commands_at or {}

    # Profile from the lively portion only (drop the stillness window if any), so
    # the per-person normalisation isn't poisoned by the zero-energy frames.
    live_end = len(xyz)
    if live_end_s is not None:
        live_end = min(len(xyz), int(live_end_s * fps))
    lively = xyz[:max(live_end, int(2 * fps))]
    profile = make_profile(
        ArrayPoseSource(lively, fps=fps, flip_y=flip).frames(), fps)

    be = LogBackend()
    studio = Studio(be, profile, mapping=mapping or MappingConfig.default(),
                    cfg=StudioConfig())

    # A parallel FeatureExtractor pass for the movement series the coupling
    # metric needs (read the same way grade.py reads them off the features).
    fe = FeatureExtractor(fps)

    frames = list(ArrayPoseSource(xyz, fps=fps, flip_y=flip).frames())
    autom_timeline: List[Tuple[float, dict]] = []
    per_frame: List[Tuple[np.ndarray, List[Flash]]] = []
    fired: List[Tuple[float, str]] = []
    proc_ms: List[float] = []
    mv = {k: [] for k in ("energy", "core", "limb", "open", "comh", "norm_e", "gated", "groove")}

    for i, fr in enumerate(frames):
        cmds = list(commands_at.get(i, []))
        t0 = time.perf_counter()
        out = studio.step(fr, cmds)
        proc_ms.append((time.perf_counter() - t0) * 1000.0)

        f = fe.update(fr)
        mv["energy"].append(f.energy_env)
        mv["core"].append(f.core_energy_env)
        mv["limb"].append(f.limb_energy_env)
        mv["open"].append(f.openness)
        mv["comh"].append(f.com_height)
        mv["norm_e"].append(float(out.ui.energy))
        mv["gated"].append(1.0 if out.ui.still else 0.0)
        mv["groove"].append(float(studio.coherence.score))

        autom_timeline.append((fr.t, out.automation))
        per_frame.append((out.ui.live_xyz, [Flash(fl.name, fl.color, fl.t0,
                                                  fl.ttl, fl.big)
                                            for fl in out.ui.flashes]))
        # Record every recognised/forced move that flashed this frame (those
        # newly-added flashes whose t0 == this frame's t). Game flashes (combo
        # names, PERFECT!, unlock toasts) are not moves -- keep them out of the
        # recognition/spam metrics -- and an fx re-flash of the same move name
        # at the same t is one fire, not two.
        seen = set()
        for fl in out.ui.flashes:
            if (abs(fl.t0 - fr.t) < 1e-9 and fl.name in GESTURE_LIBRARY
                    and fl.name not in seen):
                seen.add(fl.name)
                fired.append((fr.t, fl.name))
    studio.panic()

    return dict(studio=studio, events=list(be.events), fired=fired,
                per_frame=per_frame, proc_ms=proc_ms, mv=mv, fps=fps, dur=dur,
                autom=autom_timeline, sub=studio.sub,
                segments=list(studio.key_log))


# -- timbre-aware WAV render ------------------------------------------------

def _autom_at(autom_timeline: List[Tuple[float, dict]], t: float) -> dict:
    """The automation snapshot whose timestamp is closest to (<=) t."""
    snap = autom_timeline[0][1] if autom_timeline else {}
    for ft, a in autom_timeline:
        if ft <= t + 1e-9:
            snap = a
        else:
            break
    return snap


def _preset_for(ev, stem_snap: dict) -> str:
    """Which TimbrePreset renders this note_on, honouring timbre_morph."""
    tag = ev.tag
    if tag == "drums" or ev.channel == 10:
        return DRUM_PRESET.get(ev.a, "analog_kick")
    if tag in stemlib.STEM_ORDER:
        a = stem_snap.get(tag)
        if a and a.get("timbre") in T.PRESETS:
            return a["timbre"]
    # fx-tagged (or anything else) pitched one-shot: pick by channel.
    return CH_PRESET.get(ev.channel, "warm_keys")


# Fixed per-stem stereo placement layered onto the live pan automation, so the
# mix has width even when the dancer stands centred. Drums/bass stay anchored.
STEM_SPREAD = {"drums": 0.0, "bass": 0.0, "keys": -0.18, "lead": 0.22,
               "texture": 0.30, "fx": 0.12}


def _sidechain(pitched: np.ndarray, kick_times: List[float], sr: int,
               depth: float = 0.45, release_s: float = 0.25) -> np.ndarray:
    """Classic pump: duck the pitched bus on every kick, recovering over
    `release_s`. Deterministic, and the single biggest 'sounds produced' win --
    the kick gets pocket and the low-mids stop fighting it."""
    env = np.ones(pitched.shape[0])
    n_rel = int(release_s * sr)
    if n_rel <= 0 or not kick_times:
        return pitched
    tau = np.arange(n_rel) / sr
    curve = 1.0 - depth * np.exp(-tau / (release_s / 3.0))
    for tk in kick_times:
        i0 = int(tk * sr)
        i1 = min(i0 + n_rel, env.shape[0])
        if 0 <= i0 < env.shape[0]:
            env[i0:i1] = np.minimum(env[i0:i1], curve[: i1 - i0])
    return pitched * env[:, None]


def _master_bus(x: np.ndarray, sr: int, thr: float = 0.35, ratio: float = 3.0,
                block: int = 1024) -> np.ndarray:
    """A gentle block-RMS bus compressor (fast attack, slow release) plus a
    tanh soft limiter: glues the stems and tames combo-slam peaks without
    audible distortion. Deterministic (block arithmetic only)."""
    n = x.shape[0]
    nb = max((n + block - 1) // block, 1)
    mono = np.abs(x).max(axis=1)
    rms = np.sqrt(np.mean(
        np.pad(mono, (0, nb * block - n)).reshape(nb, block) ** 2, axis=1))
    gain_b = np.ones(nb)
    over = rms > thr
    gain_b[over] = (thr / rms[over]) ** (1.0 - 1.0 / ratio)
    g, sm = 1.0, np.empty(nb)
    for i in range(nb):                       # ~46 ms blocks: tiny loop
        a = 0.6 if gain_b[i] < g else 0.12    # attack down fast, release up slow
        g += a * (gain_b[i] - g)
        sm[i] = g
    gain = np.repeat(sm, block)[:n]
    y = x * gain[:, None]
    return np.tanh(y * 1.25) / np.tanh(1.25)


def render_wav(events: List, autom_timeline: List[Tuple[float, dict]],
               out_path: Optional[str], dur: float, tail: float = 1.5
               ) -> np.ndarray:
    """Sum every note_on (its timbre + the per-stem automation at its time) into
    a STEREO mix -- constant-power panned per stem, kick-sidechained, one global
    reverb/delay send, then a master bus (compressor + soft limit) -- and (if
    out_path) write a stereo int16 WAV. Returns the float (n, 2) mix so
    callers/tests can inspect it without writing files."""
    ons = [e for e in events if e.kind == "note_on"]
    end = (max((e.t for e in ons), default=0.0) + tail) if ons else max(dur, 1.0)
    n_total = int(end * SR) + SR
    drums = np.zeros((n_total, 2), dtype=np.float64)
    pitched = np.zeros((n_total, 2), dtype=np.float64)
    kick_times: List[float] = []

    for ev in ons:
        snap = _autom_at(autom_timeline, ev.t)
        name = _preset_for(ev, snap)
        preset = T.PRESETS[name]
        a = snap.get(ev.tag, {}) if ev.tag in stemlib.STEM_ORDER else {}
        cutoff01 = float(a.get("cutoff", 0.5))
        drive01 = float(a.get("drive", 0.2))
        gain01 = float(a.get("gain", 1.0)) if a else 1.0
        buf = T.render(preset, ev.a, ev.dur or 0.25, ev.b,
                       cutoff_mul=0.3 + 1.7 * cutoff01,
                       drive_mul=0.5 + 1.5 * drive01)
        buf = buf.astype(np.float64) * (0.3 + 0.7 * gain01)
        i0 = int(ev.t * SR)
        i1 = i0 + len(buf)
        if i1 > n_total:
            continue
        # constant-power pan: live automation + the stem's fixed spread.
        pan = float(np.clip(float(a.get("pan", 0.0))
                            + STEM_SPREAD.get(ev.tag, 0.0), -1.0, 1.0))
        th = (pan + 1.0) * np.pi / 4.0
        bus = drums if ev.channel == 10 else pitched
        bus[i0:i1, 0] += buf * np.cos(th)
        bus[i0:i1, 1] += buf * np.sin(th)
        if ev.channel == 10 and ev.a == stemlib.DRUM_PIECE["kick"]:
            kick_times.append(ev.t)

    pitched = _sidechain(pitched, kick_times, SR)

    # ONE global FX send (median of the stems' automation), computed on the
    # mono sum and returned equally to both channels -- a classic send/return.
    revs = [a.get("reverb", 0.0) for _, snap in autom_timeline
            for a in snap.values()]
    dels = [a.get("delay", 0.0) for _, snap in autom_timeline
            for a in snap.values()]
    R = float(np.clip(np.median(revs), 0.0, 0.6)) if revs else 0.0
    D = float(np.clip(np.median(dels), 0.0, 0.5)) if dels else 0.0
    mono = pitched.mean(axis=1)
    wet = T.FXRack().process(mono, reverb=R, delay=D).astype(np.float64) - mono
    mix = drums + pitched + wet[:, None]

    mixed = _master_bus(mix, SR)
    peak = float(np.max(np.abs(mixed))) + 1e-9
    out = (mixed / peak * 0.95).astype(np.float32)
    if out_path is not None:
        pcm = (out * 32767).astype(np.int16)
        with wave.open(out_path, "w") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
    return out


# -- pictures ---------------------------------------------------------------

def _frame_img(xyz, flashes, t, W=420, H=420):
    from everybody_dance import viz
    img = np.full((H, W, 3), 20, np.uint8)
    glow = flashes[-1].color if flashes else (235, 235, 235)
    viz.draw_skeleton(img, xyz, 10, 10, W - 20, H - 20, glow, 4 if flashes else 2,
                      xr=XR, yr=YR)
    viz.draw_flashes(img, flashes, t)
    import cv2
    cv2.putText(img, f"{t:4.1f}s", (10, H - 10), viz.FONT, 0.5, (150, 150, 150), 1)
    return img


def pick_frames(r, k=6):
    pf, fps = r["per_frame"], r["fps"]
    idxs = {int(t * fps) for t, _ in r["fired"]}          # frames where moves fired
    idxs |= set(np.linspace(0, len(pf) - 1, k, dtype=int))
    return sorted(i for i in idxs if 0 <= i < len(pf))[:max(k, 9)]


def multistem_pianoroll(events, dur, W=1000, H=400):
    """A 5-stem piano-roll: viz.pianoroll only knows channels 1/2/3/10 and would
    KeyError on texture (ch 4), so colour notes by ``ev.tag`` via stems.STEM_COLOR
    and show all five pitched stems on one staff + a drum lane. The [VISION]
    artifact a judge reads for musicality / coupling / liveliness."""
    import cv2
    from everybody_dance import viz
    img = np.full((H, W, 3), 26, np.uint8)
    x0, y0, pw, ph = 56, 16, W - 72, H - 86

    # Pair note_on/off into spans per (channel, pitch); colour by the on's tag.
    notes, drums, active = [], [], {}
    for e in sorted(events, key=lambda x: (x.t, x.kind == "note_on")):
        if e.kind not in ("note_on", "note_off"):
            continue
        if e.channel == 10:
            if e.kind == "note_on":
                drums.append((e.t, e.b, e.tag))
            continue
        key = (e.channel, e.a)
        if e.kind == "note_on":
            active[key] = (e.t, e.b, e.tag)
        elif key in active:
            t0, v, tag = active.pop(key)
            notes.append((e.a, t0, e.t - t0, v, tag))
    for (ch, a), (t0, v, tag) in active.items():
        notes.append((a, t0, dur - t0, v, tag))

    ps = [n[0] for n in notes] or [60]
    lo, hi = min(ps) - 1, max(ps) + 1
    span = max(hi - lo, 1)
    for k in range(0, span + 1, 3):
        y = int(y0 + ph - k / span * ph)
        cv2.line(img, (x0, y), (x0 + pw, y), (40, 40, 40), 1)
    for a, t0, d, v, tag in notes:
        x = int(x0 + t0 / max(dur, 1e-6) * pw)
        xe = int(x0 + (t0 + d) / max(dur, 1e-6) * pw)
        y = int(y0 + ph - (a - lo) / span * ph)
        base = stemlib.STEM_COLOR.get(tag, (200, 200, 200))
        col = tuple(int(c * (0.4 + 0.6 * v / 127)) for c in base)
        cv2.rectangle(img, (x, y - 4), (max(xe, x + 2), y + 4), col, -1)

    # drum lane
    yd = H - 36
    cv2.putText(img, "drums", (4, yd + 4), viz.FONT, 0.4,
                stemlib.STEM_COLOR["drums"], 1)
    for t0, v, tag in drums:
        x = int(x0 + t0 / max(dur, 1e-6) * pw)
        cv2.line(img, (x, yd - 8), (x, yd + 8), stemlib.STEM_COLOR["drums"], 1)

    # legend: the five lane colours
    lx = x0
    for name in stemlib.STEM_ORDER:
        cv2.putText(img, name, (lx, H - 8), viz.FONT, 0.4,
                    stemlib.STEM_COLOR[name], 1)
        lx += int((W - x0 - 20) / len(stemlib.STEM_ORDER))
    cv2.putText(img, "pitch", (4, y0 + 12), viz.FONT, 0.4, (150, 150, 150), 1)
    cv2.putText(img, f"0 - {dur:.0f}s", (x0, y0 + 12), viz.FONT, 0.4,
                (150, 150, 150), 1)
    return img


# -- metrics ----------------------------------------------------------------

def compute_metrics(r, labels) -> dict:
    dur, fps = r["dur"], r["fps"]
    events = r["events"]
    nb = max(int(dur / 0.25), 4)
    times = [i / fps for i in range(len(r["proc_ms"]))]
    move = {"energy": M._bin(times, r["mv"]["energy"], nb, dur),
            "limb": M._bin(times, r["mv"]["limb"], nb, dur),
            "core": M._bin(times, r["mv"]["core"], nb, dur),
            "height": M._bin(times, r["mv"]["comh"], nb, dur),
            "openness": M._bin(times, r["mv"]["open"], nb, dur)}

    ons = [(e.t, 1.0) for e in events if e.kind == "note_on"]
    vel = [(e.t, e.b) for e in events if e.kind == "note_on"]
    pit = [(e.t, e.a) for e in events if e.kind == "note_on" and e.channel != 10]
    counts = np.zeros(nb)
    for t, _ in ons:
        counts[min(int(t / dur * nb), nb - 1)] += 1
    music = {"density": counts,
             "velocity": M._bin([t for t, _ in vel], [v for _, v in vel], nb, dur),
             "pitch": M._bin([t for t, _ in pit], [p for _, p in pit], nb, dur)}
    # phantom is judged against the gate's ACTUAL state: bins that sat fully
    # under the stillness gate (>90% of frames), EXCLUDING each engagement bin
    # -- the design sanctions exactly one freeze acknowledgement there (the
    # wind-down + breath chord, caused by the freeze itself). Onsets in any
    # later gated bin are true leakage: sound while the gate held.
    gb = M._bin(times, r["mv"]["gated"], nb, dur) > 0.9
    edge = gb & ~np.roll(gb, 1)
    if gb.size:
        edge[0] = gb[0]
    still_mask = gb & ~edge
    # fx one-shots are by construction move-caused (a punch from a standing
    # dancer is causation, not phantom) -- exclude them from the leak check.
    counts_nonfx = np.zeros(nb)
    for e in events:
        if e.kind == "note_on" and e.tag != "fx":
            counts_nonfx[min(int(e.t / dur * nb), nb - 1)] += 1

    # rhythm: the groove must be PROVABLY tight. on_grid = drum onsets within
    # 12 ms of a logged grid-emission time; bar_similarity = mean Jaccard of
    # consecutive bars' drum slot patterns (a beat exists only if it repeats).
    g = r["studio"]
    grid = list(g.grid_log)
    # the committed groove only: accent answers (tag 'accent') are deliberate
    # body punctuation, not pattern.
    drum_ts = [e.t for e in events if e.kind == "note_on" and e.channel == 10
               and e.tag != "accent"]
    if grid and drum_ts:
        gt = np.array([t for _, t in grid])
        gs = np.array([s for s, _ in grid])
        idx = np.clip(np.searchsorted(gt, drum_ts), 1, len(gt) - 1)
        near = np.where(np.abs(gt[idx] - drum_ts) < np.abs(gt[idx - 1] - drum_ts),
                        idx, idx - 1)
        err = np.abs(gt[near] - np.array(drum_ts))
        on_grid_pct = float(np.mean(err <= 0.012) * 100.0)
        bars: dict = {}
        for k in gs[near]:
            bars.setdefault(int(k) // 16, set()).add(int(k) % 16)
        # similarity to the MODAL bar: fills/breakdowns are sanctioned
        # per-gesture variation, so the claim is "there is a home groove the
        # bars orbit", not "every bar equals the last".
        pats = [frozenset(v) for v in bars.values()]
        if pats:
            modal = max(sorted(pats, key=sorted), key=pats.count)
            sims = [len(p & modal) / max(len(p | modal), 1) for p in pats]
            bar_similarity = float(np.mean(sims))
        else:
            bar_similarity = 0.0
    else:
        on_grid_pct, bar_similarity = 0.0, 0.0
    # downbeat alignment: do the dancer's SIGNIFICANT movements (strong
    # kinetic-flux onsets) coincide with the grid's 8ths? This is the felt
    # "my hit owns the beat" property the phase servo exists for.
    beat_s = 60.0 / g.latch.bpm
    eighths = np.array([t for s, t in grid if s % 2 == 0]) if grid else np.array([])
    strong = [t for t, w in g.onset_log if w >= 0.4]
    if eighths.size and strong:
        errs = []
        for t in strong:
            i = int(np.clip(np.searchsorted(eighths, t), 1, len(eighths) - 1))
            errs.append(min(abs(eighths[i] - t), abs(eighths[i - 1] - t)) / beat_s)
        errs = np.array(errs)
        on_pulse_pct = float(np.mean(errs <= 0.125) * 100.0)
        pulse_err_med = float(np.median(errs))
    else:
        on_pulse_pct, pulse_err_med = 0.0, 0.5
    rhythm = {"bpm": float(g.latch.bpm), "tempo_relocks": int(g.latch.relocks),
              "on_grid_pct": round(on_grid_pct, 1),
              "bar_similarity": round(bar_similarity, 3),
              "on_pulse_pct": round(on_pulse_pct, 1),
              "pulse_err_med_beats": round(pulse_err_med, 3),
              "n_onsets": len(g.onset_log),
              "groove_score_mean": round(float(np.mean(r["mv"]["groove"])), 3)
              if r["mv"].get("groove") else None,
              "style": g.style, "swing": g.swing}

    sub = r["sub"]
    # per-stem activity (note_on counts per stem tag) -> proves multi-stem alive.
    per_stem = Counter(e.tag for e in events
                       if e.kind == "note_on" and e.tag in stemlib.STEM_ORDER)
    per_stem = {name: int(per_stem.get(name, 0)) for name in stemlib.STEM_ORDER}
    # distinct timbres actually used in the render.
    autom = r["autom"]
    used_timbres = set()
    for e in events:
        if e.kind != "note_on":
            continue
        used_timbres.add(_preset_for(e, _autom_at(autom, e.t)))

    st = r["studio"]
    game = {
        "sections_visited": len(st.arc.sections_visited),
        "sections": list(st.arc.sections_visited),
        "stems_unlocked": len(st.arc.unlocked),
        "combos_fired": len(st.combo_log),
        "combos": [name for _, name in st.combo_log],
        "max_streak_tier": int(st.max_streak_tier),
        "gold_prompts": int(st.gold.prompts),
        "gold_hits": int(st.gold.hits),
    }
    if st.identity is not None:
        game["identity"] = {"name": st.identity.name, "root": st.identity.root_name,
                            "scale": st.identity.scale,
                            "kit": st.identity.kit_name,
                            "progression": list(st.identity.progression)}

    # coarser bins (~0.75 s) for the dead-zone check: a committed groove at a
    # low level is legitimately sparse on the 0.25 s grid.
    nb2 = max(int(dur / 0.75), 2)
    de = M._bin(times, r["mv"]["energy"], nb2, dur)
    dd = np.zeros(nb2)
    for e in events:
        if e.kind == "note_on":
            dd[min(int(e.t / dur * nb2), nb2 - 1)] += 1

    cpl = M.coupling(move, music, still_mask=still_mask,
                     still_density=counts_nonfx,
                     dead_energy=de, dead_density=dd)
    # Timescale-honest coupling (the project thesis: each timescale routed to
    # the layer that moves at that speed). Density is now COMMITTED structure
    # that the body drives at bar rate -- correlate it on ~1.5 s bins; the
    # fast lanes (velocity, pitch) stay fine-grained. The score averages each
    # musical dimension at its own timescale.
    nbar = max(int(dur / 1.5), 4)
    move_bar = {k: M._bin(times, r["mv"][kk], nbar, dur)
                for k, kk in (("energy", "energy"), ("limb", "limb"),
                              ("core", "core"), ("height", "comh"),
                              ("openness", "open"))}
    dens_bar = np.zeros(nbar)
    for e in events:
        if e.kind == "note_on":
            dens_bar[min(int(e.t / dur * nbar), nbar - 1)] += 1
    cpl["matrix"]["density_bar"] = {k: round(M._corr(v, dens_bar), 3)
                                    for k, v in move_bar.items()}
    # the kick's velocity lane, masked to bars that contain kicks: its pattern
    # base is constant, so this series IS the body's dynamics modulation (the
    # most audible coupling channel of the committed-groove design).
    kcnt, ksum = np.zeros(nbar), np.zeros(nbar)
    for e in events:
        if e.kind == "note_on" and e.channel == 10 and e.a == 36:
            b = min(int(e.t / dur * nbar), nbar - 1)
            kcnt[b] += 1
            ksum[b] += e.b
    kmask = kcnt > 0
    if kmask.sum() >= 4:
        kvb = ksum[kmask] / kcnt[kmask]
        cpl["matrix"]["kick_vel_bar"] = {
            k: round(M._corr(v[kmask], kvb), 3) for k, v in move_bar.items()}
    per_music = {m: max((abs(v) for v in row.values()), default=0.0)
                 for m, row in cpl["matrix"].items() if m != "density"}
    cpl["score"] = round(float(np.mean(list(per_music.values()))), 3)

    return {
        "coupling": cpl,
        "musicality": M.musicality(events, sub.cfg.tonic, sub.cfg.scale, dur / 60,
                                   segments=r.get("segments")),
        "liveliness": M.liveliness(events, dur),
        "recognition": M.recognition(r["fired"], labels),
        "gesture": M.gesture_spam(r["fired"], dur),
        "timing": M.timing(r["proc_ms"], fps),
        "rhythm": rhythm,
        "game": game,
        "stems": {"per_stem_onsets": per_stem,
                  "active_stems": int(sum(1 for v in per_stem.values() if v > 0)),
                  "distinct_timbres": int(len(used_timbres)),
                  "timbres_used": sorted(used_timbres)},
        "session": {"duration_s": round(dur, 1), "frames": len(r["proc_ms"]),
                    "total_notes": int(sum(per_stem.values())
                                       + sum(1 for e in events if e.kind == "note_on"
                                             and e.tag not in stemlib.STEM_ORDER))},
    }


def flatten(metrics) -> dict:
    out = {
        "coupling.score": metrics["coupling"]["score"],
        "coupling.dead_zone": metrics["coupling"]["dead_zone"],
        "coupling.phantom": metrics["coupling"]["phantom"],
        "musicality.in_scale_pct": metrics["musicality"]["in_scale_pct"],
        "liveliness.density_std": metrics["liveliness"]["density_std"],
        "gesture.max_per_min": metrics["gesture"]["max_per_min"],
        "timing.headroom_pct": metrics["timing"].get("headroom_pct", 0.0),
    }
    if "game" in metrics:
        out["game.sections_visited"] = float(metrics["game"]["sections_visited"])
        out["game.stems_unlocked"] = float(metrics["game"]["stems_unlocked"])
        out["game.combos_fired"] = float(metrics["game"]["combos_fired"])
    if "rhythm" in metrics:
        out["rhythm.on_grid_pct"] = float(metrics["rhythm"]["on_grid_pct"])
        out["rhythm.bar_similarity"] = float(metrics["rhythm"]["bar_similarity"])
        out["rhythm.tempo_relocks"] = float(metrics["rhythm"]["tempo_relocks"])
    return out


# -- top-level pipeline (factored so the test can skip PNG writes) ----------

def run_pipeline(seconds: float, scripted: bool, out: Optional[str],
                 write_images: bool = True) -> dict:
    """Run the whole replay+render+metrics pipeline. Returns a dict with
    `metrics`, `slo`, `flat`, `mix`, `replay`, `prompt`. Writes the bundle into
    `out` when given; with `write_images=False`, skips the cv2 PNG writes (so the
    test can get metrics/wav fast and headlessly)."""
    if scripted:
        xyz, labels = scripted_performer(with_labels=True)
        fps, flip = 30.0, False
        mapping = MappingConfig.default()
        commands_at = None       # scripted moves fire bindings on their own
        live_end_s = None        # no dedicated stillness window to skip
    else:
        fps, flip = 30.0, False
        xyz, fps, flip = synthetic_corpus(seconds, fps)
        labels = None
        mapping = corpus_mapping()
        commands_at = command_choreography(fps, len(xyz) / fps)
        live_end_s = corpus_layout(len(xyz) / fps)["still_start"]

    r = replay(xyz, fps, flip, scripted=scripted, mapping=mapping,
               live_end_s=live_end_s, commands_at=commands_at)
    metrics = compute_metrics(r, labels)
    flat = flatten(metrics)
    slo = M.check_slo(flat)
    prompt = build_prompt(metrics, slo)

    wav_path = None
    if out is not None:
        os.makedirs(out, exist_ok=True)
        wav_path = os.path.join(out, "music.wav")
    mix = render_wav(r["events"], r["autom"], wav_path, r["dur"])

    if out is not None:
        json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
        open(os.path.join(out, "prompt.txt"), "w").write(prompt)
        if write_images:
            import cv2
            from everybody_dance import viz
            sheet = viz.contact_sheet([_frame_img(*r["per_frame"][i], i / fps)
                                       for i in pick_frames(r)])
            roll = multistem_pianoroll(r["events"], r["dur"])
            cv2.imwrite(os.path.join(out, "contact_sheet.png"), sheet)
            cv2.imwrite(os.path.join(out, "pianoroll.png"), roll)

    return dict(metrics=metrics, slo=slo, flat=flat, mix=mix, replay=r,
                prompt=prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", action="store_true",
                    help="drive scripted_performer (labelled recognition)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="out/studio")
    ap.add_argument("--judge", action="store_true",
                    help="actually call Claude (needs key)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    res = run_pipeline(args.seconds, args.scripted, args.out, write_images=True)
    metrics, slo, flat = res["metrics"], res["slo"], res["flat"]

    print("METRICS:", json.dumps(flat, indent=2))
    print("SLO:", json.dumps(slo))
    print("PER-STEM:", json.dumps(metrics["stems"]["per_stem_onsets"]),
          "| distinct timbres:", metrics["stems"]["distinct_timbres"],
          metrics["stems"]["timbres_used"])
    print(f"bundle -> {args.out}/ (contact_sheet.png, pianoroll.png, music.wav, "
          "metrics.json, prompt.txt)")

    if args.judge:
        from everybody_dance.judge import _anthropic_grader
        imgs = [os.path.join(args.out, "contact_sheet.png"),
                os.path.join(args.out, "pianoroll.png")]
        grader = _anthropic_grader(args.model) if args.model else None
        card = grade_bundle(res["prompt"], imgs, grader)
        open(os.path.join(args.out, "scorecard.json"), "w").write(card.to_json())
        print("\nSCORECARD:\n" + card.to_json())
    else:
        print("\n(dry run -- pass --judge to call Claude, or grade by hand)")


if __name__ == "__main__":
    main()
