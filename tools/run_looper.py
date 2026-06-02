#!/usr/bin/env python3
"""Drive the body-looper offline on a recorded dance, with a scripted 'pedal'
that commits each instrument in turn, then renders the result to a WAV so the
loop build-up (drums -> +bass -> +chord -> +lead -> perform) can be heard.
"""

import argparse
import os

import numpy as np

from everybody_dance.controls import Command, ScriptedControl
from everybody_dance.features import FeatureExtractor
from everybody_dance.looper import LoopStation, LooperConfig
from everybody_dance.output import LogBackend
from everybody_dance.sources import ArrayPoseSource
from tools.run_dataset import _load, calibrate_full
from tools.synth import render


def energetic_window(src, seconds):
    """Pick the liveliest span of `seconds` -- you'd loop while actually dancing,
    not during a calm intro (and live, you calibrate on that same movement)."""
    fe = FeatureExtractor(src.fps)
    vals = np.array([(f.core_energy_env + f.limb_energy_env) for f in
                     (fe.update(fr) for fr in src.frames())])
    W = int(seconds * src.fps)
    if len(vals) <= W:
        return src
    smooth = np.convolve(vals, np.ones(W) / W, "valid")
    start = int(np.argmax(smooth))
    seg = src.xyz_seq[start:start + W]
    return ArrayPoseSource(seg, fps=src.fps, flip_y=src.flip_y, name=src.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--rec-seconds", type=float, default=8.0,
                    help="recording time per instrument before the pedal commits")
    ap.add_argument("--perform-seconds", type=float, default=16.0)
    ap.add_argument("--coupling", type=float, default=0.4)
    ap.add_argument("--out", default="out/looper.wav")
    args = ap.parse_args()

    order = ["drums", "bass", "chord", "lead"]
    total = args.rec_seconds * len(order) + args.perform_seconds
    src = _load(args.path, args.kind, None)            # load full clip
    src = energetic_window(src, total)                  # loop while actually dancing
    profile = calibrate_full(src)                       # calibrate on that movement

    # scripted pedal: commit each instrument after rec_seconds of recording
    schedule = [(args.rec_seconds * (i + 1), Command.COMMIT) for i in range(len(order))]
    control = ScriptedControl(schedule)

    be = LogBackend()
    station = LoopStation(be, profile,
                          LooperConfig(record_order=order, coupling=args.coupling))
    station.run(src, control)

    # report what played in each phase
    print(f"source={src.name}  tempo~{station.st.bpm:.0f}bpm  "
          f"palette={station.sub.cfg.palette} scale={station.sub.cfg.scale}")
    print("commit schedule (s):", [round(t, 1) for t, _ in schedule])
    from collections import Counter
    for lo, hi, label in [
        (0, args.rec_seconds, "REC drums"),
        (args.rec_seconds, 2 * args.rec_seconds, "REC bass (+drums loop)"),
        (2 * args.rec_seconds, 3 * args.rec_seconds, "REC chord (+drums,bass)"),
        (3 * args.rec_seconds, 4 * args.rec_seconds, "REC lead (+all)"),
        (4 * args.rec_seconds, total, "PERFORM (locked + dance modulates)"),
    ]:
        roles = Counter(e.tag.split(".")[0] for e in be.events
                        if e.kind == "note_on" and lo <= e.t < hi)
        print(f"  {label:<38} {dict(roles)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render(be.events, args.out)
    print(f"rendered -> {args.out}")


if __name__ == "__main__":
    main()
