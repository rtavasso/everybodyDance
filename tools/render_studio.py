#!/usr/bin/env python3
"""Render the everybodyDance *studio* offline: a dance authors a customizable
multi-stem loop (rhythm from body hits / euclid, pitch from body height, per-stem
timbre + FX from a preset), which is rendered to audio through the synth + FX
engines, while a Just-Dance-style UI is rendered to video (skeleton, gesture
flashes, score/combo, move-prompt lane, per-stem beat-grid rack).

    python -m tools.render_studio --preset house
    python -m tools.render_studio data/bvh/dance1_subject1.bvh --preset synthwave
"""

import argparse
import os
import wave

import numpy as np

from everybody_dance.calib import BuildCalibrator, fold_tempo
from everybody_dance.effects import STYLE, DEFAULT_STYLE, Flash
from everybody_dance.features import FeatureExtractor
from everybody_dance.fx import apply_fx
from everybody_dance.gestures import GestureRecognizer, default_gestures
from everybody_dance.pose import JOINT_INDEX, PoseFrame
from everybody_dance.presets import get_preset
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.stems import StemRack
from everybody_dance.studio_types import SR, MovePrompt, StudioUIState
from everybody_dance.synth_engine import render_note
from everybody_dance.studio_ui import compose_studio
from tools.render_build import energetic_window
from tools.run_dataset import _load

FOOT = [JOINT_INDEX[j] for j in ("l_ankle", "r_ankle")]
HAND = [JOINT_INDEX[j] for j in ("l_wrist", "r_wrist")]


def _peaks(sig, thr, dt, refractory=0.12):
    """Indices where sig has a peak above thr with a refractory gap."""
    out, last = [], -1e9
    for i, v in enumerate(sig):
        if v > thr and (i * dt - last) >= refractory:
            out.append(i)
            last = i * dt
    return out


def author(rack, feats, fps, bpm):
    """Author every stem from the movement: euclid stems get their pattern; body
    stems get onsets from limb hits (feet -> drums/bass, hands -> melodic)."""
    foot = np.array([f.accel_mag[FOOT].max() for f in feats])
    hand = np.array([f.accel_mag[HAND].max() for f in feats])
    comh = np.array([f.com_height for f in feats])
    lo, hi = np.percentile(comh, 5), np.percentile(comh, 95)
    comh01 = np.clip((comh - lo) / (hi - lo + 1e-6), 0, 1)
    dt = 1.0 / fps
    foot_hits = _peaks(foot, np.percentile(foot, 70), dt)
    hand_hits = _peaks(hand, np.percentile(hand, 78), dt)

    for i, st in enumerate(rack.states):
        cfg = st.config
        if cfg.rhythm.mode == "euclid":
            rack.author_euclid(i)
            continue
        sd = rack.step_dur(bpm, i)
        total = cfg.rhythm.total_steps
        melodic = cfg.channel != 10 and cfg.name not in ("bass", "drone", "pulse")
        hits = hand_hits if melodic else foot_hits
        strength = hand if melodic else foot
        for fi in hits:
            t = fi * dt
            step = int(round(t / sd)) % total
            vel = int(np.clip(60 + 30 * (strength[fi] / (strength.max() + 1e-6)), 40, 120))
            rack.record_onset(i, step, vel, float(comh01[fi]))
    return comh01


def write_wav(buf, path):
    pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/bvh/dance1_subject1.bvh")
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--preset", default="house")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out", default="out/studio.mp4")
    ap.add_argument("--wav", default="out/studio.wav")
    args = ap.parse_args()
    import cv2

    src = energetic_window(_load(args.path, args.kind, None), args.seconds)
    # resample movement to the render fps
    fps = args.fps
    step = max(1, int(round(src.fps / fps)))
    xyz = src.xyz_seq[::step]
    fps = src.fps / step
    frames = list(ArrayPoseSource(xyz, fps=fps, flip_y=src.flip_y).frames())

    # calibrate tempo + features
    cal = BuildCalibrator(fps)
    fe = FeatureExtractor(fps)
    feats = []
    for fr in frames:
        cal.observe(fr)
        feats.append(fe.update(fr))
    prof, tempo_hz, _ = cal.finalize()
    bpm = fold_tempo(tempo_hz) * 60.0

    rack = StemRack(get_preset(args.preset))
    comh01 = author(rack, feats, fps, bpm)

    # ---- audio: render the loop, repeated to cover the clip ----
    loop_secs = rack.states[0].config.rhythm.total_steps * rack.step_dur(bpm, 0)
    dur = len(frames) / fps
    cycles = max(2, int(np.ceil(dur / loop_secs)) + 1)
    mix = rack.render(SR, bpm, render_note, apply_fx, cycles=cycles)
    os.makedirs(os.path.dirname(args.wav) or ".", exist_ok=True)
    write_wav(mix[:int(dur * SR)], args.wav)

    # ---- video: the Just-Dance studio UI over the performance ----
    rec = GestureRecognizer(default_gestures(), fps=fps)
    total = rack.states[0].config.rhythm.total_steps
    sd = rack.step_dur(bpm, 0)
    fired = []
    for i, (fr, f) in enumerate(zip(frames, feats)):
        for ev in rec.update(fr, f):
            fired.append((fr.t, ev.name))
    prompts_all = [MovePrompt(n, due_t=t, lead_s=2.0, hit=True) for t, n in fired]

    W, H = 1280, 720
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    emax = max((f.energy_env for f in feats), default=1.0) + 1e-6
    score, combo, last_fire, fire_set = 0, 0, -1e9, set(j for j in range(len(fired)))
    flashes = []
    fi = 0
    for i, (fr, f) in enumerate(zip(frames, feats)):
        t = fr.t
        # advance the score from any gestures that fired at/just before this frame
        while fi < len(fired) and fired[fi][0] <= t:
            name = fired[fi][1]
            combo = combo + 1 if (t - last_fire) < 3.0 else 1
            score += 100 * combo
            last_fire = t
            col = STYLE.get(name, DEFAULT_STYLE)[0]
            flashes.append(Flash(name, col, t, 0.5, STYLE.get(name, DEFAULT_STYLE)[1]))
            fi += 1
        flashes = [fl for fl in flashes if fl.alpha(t) > 0]
        playhead = int((t % loop_secs) / sd) % total
        state = StudioUIState(
            stems=rack.states, active_stem=int(t / 2.0) % len(rack.states),
            phase="perform", playhead=playhead, bpm=bpm,
            pitch01=float(comh01[i]), flashes=flashes,
            prompts=[p for p in prompts_all if -0.4 <= p.due_t - t <= p.lead_s],
            score=score, combo=combo,
            rating="PERFECT" if (t - last_fire) < 0.6 else "",
            meters={"energy": float(f.energy_env / emax),
                    "pitch": float(comh01[i]),
                    "cutoff": float(np.mean([s.config.timbre.cutoff for s in rack.states]))},
            live_xyz=fr.xyz, present=True)
        vw.write(compose_studio(state, W, H, t=t))
    vw.release()

    print(f"preset={args.preset} bpm={bpm:.0f} loop={loop_secs:.1f}s")
    print("onsets/stem:", {s.config.name: s.n_onsets for s in rack.states})
    print("gestures:", len(fired), "final score:", score)
    print(f"rendered -> {args.out}  (+ {args.wav})")


if __name__ == "__main__":
    main()
