#!/usr/bin/env python3
"""Render the gesture layer over open-ended dancing.

Open-ended movement drives the continuous ambient engine (always on); specific
recognised moves punch in one-shot effects with a Just-Dance-style flash. Outputs
an overlay MP4 (skeleton + flashes + gesture log + timeline) and a WAV that mixes
the ambient music with the gesture effects.

    python -m tools.render_gestures                       # scripted move demo
    python -m tools.render_gestures --real data/bvh/dance1_subject1.bvh
"""

import argparse
import os
from collections import Counter, deque

import numpy as np

from everybody_dance.effects import STYLE, DEFAULT_STYLE, EffectEngine, Flash
from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.features import FeatureExtractor
from everybody_dance.gestures import GestureRecognizer, default_gestures
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.personalization import Calibrator
from everybody_dance.pose import JOINTS, PoseFrame
from everybody_dance.sources import ArrayPoseSource
from everybody_dance import viz
from tools.synth import render as render_wav

XR, YR = (-1.9, 1.9), (-2.1, 1.7)     # skeleton fit for full-body incl. T-pose
FPS = 30.0

NEUTRAL = dict(nose=(0, 1.35, 0), l_shoulder=(-.25, 1, 0), r_shoulder=(.25, 1, 0),
               l_elbow=(-.45, .55, 0), r_elbow=(.45, .55, 0),
               l_wrist=(-.5, .1, 0), r_wrist=(.5, .1, 0),
               l_hip=(-.15, 0, 0), r_hip=(.15, 0, 0),
               l_knee=(-.17, -1, 0), r_knee=(.17, -1, 0),
               l_ankle=(-.18, -2, 0), r_ankle=(.18, -2, 0))


def _pose(**ov):
    p = dict(NEUTRAL)
    p.update(ov)
    return np.array([p[j] for j in JOINTS], float)


def scripted_performer(fps=FPS):
    """A dancer who performs every built-in move in turn (clean cause->effect)."""
    NEU = _pose()
    moves = [
        ("HANDS UP", _pose(l_wrist=(-.3, 1.6, 0), r_wrist=(.3, 1.6, 0),
                           l_elbow=(-.35, 1.2, 0), r_elbow=(.35, 1.2, 0))),
        ("RAISE LEFT", _pose(l_wrist=(-.3, 1.6, 0), l_elbow=(-.35, 1.2, 0))),
        ("RAISE RIGHT", _pose(r_wrist=(.3, 1.6, 0), r_elbow=(.35, 1.2, 0))),
        ("T-POSE", _pose(l_wrist=(-1.6, 1, 0), r_wrist=(1.6, 1, 0),
                         l_elbow=(-.9, 1, 0), r_elbow=(.9, 1, 0))),
        ("SQUAT", _pose(nose=(0, 1.25, 0), l_shoulder=(-.25, .9, 0), r_shoulder=(.25, .9, 0),
                        l_elbow=(-.45, .45, 0), r_elbow=(.45, .45, 0),
                        l_wrist=(-.5, 0, 0), r_wrist=(.5, 0, 0),
                        l_knee=(-.2, -.55, 0), r_knee=(.2, -.55, 0),
                        l_ankle=(-.2, -1.1, 0), r_ankle=(.2, -1.1, 0))),
        ("ARMS CROSSED", _pose(l_wrist=(.25, .9, 0), r_wrist=(-.25, .9, 0),
                               l_elbow=(-.1, .7, 0), r_elbow=(.1, .7, 0))),
        ("CLAP", _pose(l_wrist=(-.05, .6, .2), r_wrist=(.05, .6, .2),
                       l_elbow=(-.2, .5, .1), r_elbow=(.2, .5, .1))),
        ("PUNCH", _pose(l_wrist=(-.35, .9, 0), r_wrist=(1.3, .95, .3),
                        l_elbow=(-.4, .7, 0), r_elbow=(.7, .9, .1))),
    ]
    guard = _pose(l_wrist=(-.35, .9, 0), r_wrist=(.35, .9, 0),
                  l_elbow=(-.4, .7, 0), r_elbow=(.4, .7, 0))
    seq = []
    lerp = lambda a, b, n: [a + (b - a) * k / max(n - 1, 1) for k in range(n)]
    hold = lambda p, s: seq.extend([p] * int(s * fps))
    move = lambda a, b, s: seq.extend(lerp(a, b, int(s * fps)))
    hold(NEU, 1.6)
    for name, target in moves:
        if name == "PUNCH":
            move(NEU, guard, .3)
            move(guard, target, .14)
        else:
            move(NEU, target, .18)
        hold(target, .55)
        move(target, NEU, .2)
        hold(NEU, .55)
    seq = np.stack(seq)
    # a gentle groove so the ambient engine has something to chew on: upper-body
    # sway + a foot bob. Both preserve the relative geometry the predicates use.
    from everybody_dance.pose import JOINT_INDEX
    up = [JOINT_INDEX[j] for j in ("nose", "l_shoulder", "r_shoulder",
                                   "l_elbow", "r_elbow", "l_wrist", "r_wrist")]
    feet = [JOINT_INDEX[j] for j in ("l_knee", "r_knee", "l_ankle", "r_ankle")]
    t = np.arange(len(seq)) / fps
    seq[:, up, 0] += (0.12 * np.sin(2 * np.pi * 1.6 * t))[:, None]
    seq[:, feet, 1] += (0.08 * np.sin(2 * np.pi * 2.0 * t))[:, None]
    return seq


def make_profile(src_frames, fps):
    fe, lab, clk, cal = FeatureExtractor(fps), LabanEstimator(), EntrainedClock(), Calibrator()
    for fr in src_frames:
        f = fe.update(fr)
        cal.observe(f, lab.update(f, f.dt), clk.update(float(f.bounce), f.dt).tempo_hz)
    return cal.finalize(signature=fe.signature.summary())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="BVH/npz path; omit for scripted demo")
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="out/gestures.mp4")
    ap.add_argument("--wav", default="out/gestures.wav")
    args = ap.parse_args()
    import cv2

    if args.real:
        from tools.render_build import energetic_window
        from tools.run_dataset import _load
        src = energetic_window(_load(args.real, args.kind, None), args.seconds)
        xyz, fps, flip = src.xyz_seq, src.fps, src.flip_y
    else:
        xyz, fps, flip = scripted_performer(), FPS, False

    profile = make_profile(ArrayPoseSource(xyz, fps=fps, flip_y=flip).frames(), fps)

    # Pass A: open-ended movement -> continuous ambient engine.
    be_amb = LogBackend()
    eng = Engine(ArrayPoseSource(xyz, fps=fps, flip_y=flip), be_amb,
                 EngineConfig(fps=fps, calibrate_s=0.0), profile=profile)
    eng.run()

    # Pass B: the gesture layer (same frames), effects keyed to the same scale.
    be_fx = LogBackend()
    fx = EffectEngine(be_fx, eng.substrate)
    fe = FeatureExtractor(fps)
    rec = GestureRecognizer(default_gestures(), fps=fps)
    # the scripted skeletons are already normalised; re-run through the source so
    # the drawn skeleton matches what the recogniser saw.
    frames = list(ArrayPoseSource(xyz, fps=fps, flip_y=flip).frames())
    per_frame, log = [], deque(maxlen=11)
    for frame in frames:
        f = fe.update(frame)
        for ev in rec.update(frame, f):
            fx.trigger(ev.name, frame.t)
            log.appendleft((frame.t, ev.name))
        per_frame.append((frame.xyz, [Flash(fl.name, fl.color, fl.t0, fl.ttl, fl.big)
                                      for fl in fx.active_flashes(frame.t)], list(log)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render_wav(be_amb.events + be_fx.events, args.wav)

    W, H = 1280, 720
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    dur = len(per_frame) / fps
    for i, (x, flashes, lg) in enumerate(per_frame):
        t = i / fps
        canvas = np.full((H, W, 3), 20, np.uint8)
        cv2.putText(canvas, "open-ended dance -> ambient music (always on)   +   "
                    "specific moves -> effects", (16, 28), viz.FONT, 0.6, (200, 200, 200), 1)
        glow = flashes[-1].color if flashes else (235, 235, 235)
        viz.draw_skeleton(canvas, x, 360, 70, 560, 600, glow, 5 if flashes else 3,
                          xr=XR, yr=YR)
        viz.draw_flashes(canvas, flashes, t)
        cv2.putText(canvas, "moves", (24, 86), viz.FONT, 0.6, (150, 150, 150), 1)
        for k, (tt, name) in enumerate(lg):
            col = STYLE.get(name, DEFAULT_STYLE)[0]
            cv2.putText(canvas, f"{tt:5.1f}s  {name}", (24, 116 + k * 26),
                        viz.FONT, 0.5, tuple(int(c * (1 - 0.05 * k)) for c in col), 1)
        y0 = H - 40
        cv2.line(canvas, (24, y0), (W - 24, y0), (60, 60, 60), 1)
        for tt, name in fx.log:
            sx = 24 + int(tt / dur * (W - 48))
            cv2.line(canvas, (sx, y0 - 10), (sx, y0 + 10), STYLE.get(name, DEFAULT_STYLE)[0], 2)
        px = 24 + int(t / dur * (W - 48))
        cv2.line(canvas, (px, y0 - 16), (px, y0 + 16), (255, 255, 255), 1)
        vw.write(canvas)
    vw.release()

    print("gestures:", dict(Counter(n for _, n in fx.log)), "total", len(fx.log))
    print("ambient notes:", sum(e.kind == "note_on" for e in be_amb.events),
          "| fx notes:", sum(e.kind == "note_on" for e in be_fx.events))
    print(f"rendered -> {args.out}  (+ {args.wav})")


if __name__ == "__main__":
    main()
