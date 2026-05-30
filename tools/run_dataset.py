#!/usr/bin/env python3
"""Run the instrument on a set of real pose sequences and report:
  * a per-dancer OUTPUT signature (tempo, density, role mix, register, effort)
  * a pairwise distinctness matrix (are different people audibly different?)
  * a rendered WAV per dancer (so musicality can be heard)
"""

import argparse
import glob
import os
import statistics as st
from collections import Counter

import numpy as np

from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.features import FeatureExtractor
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.personalization import Calibrator
from everybody_dance.sources import (load_bvh_source, load_dance_skeleton_source,
                                      load_npz_source)
from tools.synth import render


def _load(path, kind, max_seconds):
    if kind == "bvh":
        return load_bvh_source(path, max_seconds=max_seconds)
    if kind == "melody":
        cfg = os.path.join(os.path.dirname(path), "config.json")
        return load_dance_skeleton_source(path, config=cfg)
    return load_npz_source(path)


def calibrate_full(src):
    """Build a Profile from the WHOLE clip. These mocap clips aren't staged
    'move freely' sessions, so a prefix isn't representative; live use has a
    dedicated 20 s calibration that is."""
    fe = FeatureExtractor(fps=src.fps)
    lab = LabanEstimator()
    clk = EntrainedClock()
    cal = Calibrator()
    for fr in src.frames():
        f = fe.update(fr)
        d = lab.update(f, f.dt)
        c = clk.update(float(f.bounce), f.dt)
        cal.observe(f, d, c.tempo_hz)
    return cal.finalize(signature=fe.signature.summary())


def run_one(path, kind, max_seconds, calibrate, coupling, out_dir):
    src = _load(path, kind, max_seconds)
    fps = src.fps
    profile = calibrate_full(src)              # representative calibration
    be = LogBackend()
    eng = Engine(src, be, EngineConfig(fps=fps, calibrate_s=0,
                                       coupling=coupling), profile=profile)
    eng.run()

    traces = eng.traces
    locked = [t for t in traces if t.locked]
    notes = [e for e in be.events if e.kind == "note_on"]
    role = Counter(e.tag.split(".")[0] for e in notes)
    n = max(len(notes), 1)
    pitched = [e.a for e in notes if e.channel != 10]
    vels = [e.b for e in notes]
    eff = {k: st.mean([t.effort[k] for t in traces]) for k in
           ("weight", "time", "space", "flow")} if traces else {}
    play_s = (traces[-1].t - traces[0].t) if len(traces) > 1 else 1.0

    sig = {
        "name": src.name,
        "scale": eng.substrate.cfg.scale,
        "palette": eng.substrate.cfg.palette,
        "bpm": st.median([t.bpm for t in locked]) if locked else 0.0,
        "notes_per_s": len(notes) / play_s,
        "drum_frac": role.get("drums", 0) / n,
        "lead_frac": role.get("lead", 0) / n,
        "bass_frac": role.get("bass", 0) / n,
        "mean_pitch": st.mean(pitched) if pitched else 0.0,
        "mean_vel": st.mean(vels) if vels else 0.0,
        "energy": st.mean([t.energy for t in traces]) if traces else 0.0,
        **{f"eff_{k}": v for k, v in eff.items()},
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        wav = os.path.join(out_dir, f"{src.name}.wav")
        try:
            render(be.events, wav)
            sig["wav"] = wav
        except ValueError:
            sig["wav"] = None
    return sig


# features used for the distinctness distance (z-scored across dancers)
DIST_KEYS = ["bpm", "notes_per_s", "drum_frac", "lead_frac", "bass_frac",
             "mean_pitch", "mean_vel", "energy", "eff_weight", "eff_time",
             "eff_space", "eff_flow"]


def distinctness(sigs):
    M = np.array([[s.get(k, 0.0) for k in DIST_KEYS] for s in sigs], dtype=float)
    mu = M.mean(0)
    sd = M.std(0) + 1e-9
    Z = (M - mu) / sd
    N = len(sigs)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(Z[i] - Z[j])
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="bvh/npz files or globs")
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--max-seconds", type=float, default=45)
    ap.add_argument("--calibrate", type=float, default=15)
    ap.add_argument("--coupling", type=float, default=0.4)
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    sigs = [run_one(f, args.kind, args.max_seconds, args.calibrate,
                    args.coupling, args.out_dir) for f in files]

    cols = ["name", "scale", "palette", "bpm", "notes_per_s", "drum_frac",
            "lead_frac", "bass_frac", "mean_pitch", "mean_vel", "energy",
            "eff_weight", "eff_time", "eff_space", "eff_flow"]
    print("\n=== per-dancer OUTPUT signatures ===")
    print(f"{'name':<22}{'scale':<8}{'pal':<10}{'bpm':>5}{'n/s':>6}"
          f"{'drum':>6}{'lead':>6}{'bass':>6}{'pitch':>7}{'vel':>5}"
          f"{'enrg':>6}{'W':>5}{'T':>5}{'S':>5}{'F':>5}")
    for s in sigs:
        print(f"{s['name'][:21]:<22}{s['scale']:<8}{s['palette']:<10}"
              f"{s['bpm']:>5.0f}{s['notes_per_s']:>6.1f}{s['drum_frac']:>6.2f}"
              f"{s['lead_frac']:>6.2f}{s['bass_frac']:>6.2f}{s['mean_pitch']:>7.1f}"
              f"{s['mean_vel']:>5.0f}{s['energy']:>6.2f}{s['eff_weight']:>5.2f}"
              f"{s['eff_time']:>5.2f}{s['eff_space']:>5.2f}{s['eff_flow']:>5.2f}")

    if len(sigs) > 1:
        D = distinctness(sigs)
        print("\n=== distinctness matrix (z-scored euclidean; bigger=more different) ===")
        names = [s["name"][:10] for s in sigs]
        print(" " * 12 + "".join(f"{n:>11}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:<12}" + "".join(f"{D[i, j]:>11.2f}" for j in range(len(names))))
        off = D[~np.eye(len(sigs), dtype=bool)]
        print(f"\nmean off-diagonal distance: {off.mean():.2f} "
              f"(min {off[off>0].min():.2f})")


if __name__ == "__main__":
    main()
