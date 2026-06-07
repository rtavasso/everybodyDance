#!/usr/bin/env python3
"""Observability harness: replay a movement deterministically, take pictures of
the output + compute metrics, then have an LLM judge grade it.

    python -m tools.grade                          # scripted session, dry-run bundle
    python -m tools.grade --real data/bvh/dance1_subject1.bvh
    python -m tools.grade --judge                  # actually call Claude (needs key)

Outputs, in --out (default out/grade/): contact_sheet.png, pianoroll.png,
metrics.json, prompt.txt, and (with --judge) scorecard.json. Because replay is
deterministic, running this before/after a change is a real regression signal.
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np

from everybody_dance import metrics as M
from everybody_dance import viz
from everybody_dance.effects import EffectEngine, Flash
from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.features import FeatureExtractor
from everybody_dance.gestures import GestureRecognizer, default_gestures
from everybody_dance.judge import build_prompt, grade_bundle
from everybody_dance.output import LogBackend
from everybody_dance.pose import PoseFrame
from everybody_dance.sources import ArrayPoseSource
from tools.render_gestures import XR, YR, make_profile, scripted_performer


def _frame_img(xyz, flashes, t, W=420, H=420):
    img = np.full((H, W, 3), 20, np.uint8)
    glow = flashes[-1].color if flashes else (235, 235, 235)
    viz.draw_skeleton(img, xyz, 10, 10, W - 20, H - 20, glow, 4 if flashes else 2,
                      xr=XR, yr=YR)
    viz.draw_flashes(img, flashes, t)
    cv2_putt(img, f"{t:4.1f}s", (10, H - 10))
    return img


def cv2_putt(img, s, org):
    import cv2
    cv2.putText(img, s, org, viz.FONT, 0.5, (150, 150, 150), 1)


def replay(xyz, fps, flip):
    """Deterministic replay: ambient engine + gesture layer over the movement."""
    profile = make_profile(ArrayPoseSource(xyz, fps=fps, flip_y=flip).frames(), fps)
    be_amb = LogBackend()
    eng = Engine(ArrayPoseSource(xyz, fps=fps, flip_y=flip), be_amb,
                 EngineConfig(fps=fps, calibrate_s=0.0), profile=profile)
    eng.run()

    be_fx = LogBackend()
    fx = EffectEngine(be_fx, eng.substrate)
    fe = FeatureExtractor(fps)
    rec = GestureRecognizer(default_gestures(), fps=fps)
    frames = list(ArrayPoseSource(xyz, fps=fps, flip_y=flip).frames())
    per_frame, fired, proc_ms = [], [], []
    mv = {k: [] for k in ("energy", "core", "limb", "vert", "open", "comh")}
    for fr in frames:
        t0 = time.perf_counter()
        f = fe.update(fr)
        for ev in rec.update(fr, f):
            fx.trigger(ev.name, fr.t)
            fired.append((fr.t, ev.name))
        proc_ms.append((time.perf_counter() - t0) * 1000)
        per_frame.append((fr.xyz, [Flash(*(fl.name, fl.color, fl.t0, fl.ttl, fl.big))
                                   for fl in fx.active_flashes(fr.t)]))
        mv["energy"].append(f.energy_env); mv["core"].append(f.core_energy_env)
        mv["limb"].append(f.limb_energy_env); mv["vert"].append(f.verticality)
        mv["open"].append(f.openness); mv["comh"].append(f.com_height)
    return dict(eng=eng, events=be_amb.events + be_fx.events, fired=fired,
                per_frame=per_frame, proc_ms=proc_ms, mv=mv, fps=fps,
                dur=len(frames) / fps, sub=eng.substrate)


def compute_metrics(r, labels):
    dur, fps = r["dur"], r["fps"]
    nb = max(int(dur / 0.25), 4)
    times = [i / fps for i in range(len(r["proc_ms"]))]
    move = {"energy": M._bin(times, r["mv"]["energy"], nb, dur),
            "limb": M._bin(times, r["mv"]["limb"], nb, dur),
            "core": M._bin(times, r["mv"]["core"], nb, dur),
            "height": M._bin(times, r["mv"]["comh"], nb, dur),
            "openness": M._bin(times, r["mv"]["open"], nb, dur)}
    ons = [(e.t, 1.0) for e in r["events"] if e.kind == "note_on"]
    vel = [(e.t, e.b) for e in r["events"] if e.kind == "note_on"]
    pit = [(e.t, e.a) for e in r["events"] if e.kind == "note_on" and e.channel != 10]
    music = {"density": M._bin([t for t, _ in ons], [1] * len(ons), nb, dur)
             if ons else np.zeros(nb),
             "velocity": M._bin([t for t, _ in vel], [v for _, v in vel], nb, dur),
             "pitch": M._bin([t for t, _ in pit], [p for _, p in pit], nb, dur)}
    # density as counts per bin
    music["density"] = M._bin([t for t, _ in ons], [1] * len(ons), nb, dur) * 0
    counts = np.zeros(nb)
    for t, _ in ons:
        counts[min(int(t / dur * nb), nb - 1)] += 1
    music["density"] = counts
    sub = r["sub"]
    return {
        "coupling": M.coupling(move, music),
        "musicality": M.musicality(r["events"], sub.cfg.tonic, sub.cfg.scale, dur / 60),
        "recognition": M.recognition(r["fired"], labels),
        "timing": M.timing(r["proc_ms"], fps),
        "session": {"duration_s": round(dur, 1), "frames": len(r["proc_ms"])},
    }


def pick_frames(r, k=6):
    pf, dur, fps = r["per_frame"], r["dur"], r["fps"]
    idxs = {int(t * fps) for t, _ in r["fired"]}          # frames where moves fired
    idxs |= set(np.linspace(0, len(pf) - 1, k, dtype=int))
    return sorted(i for i in idxs if 0 <= i < len(pf))[:max(k, 8)]


def flatten(metrics):
    return {"coupling.score": metrics["coupling"]["score"],
            "coupling.dead_zone": metrics["coupling"]["dead_zone"],
            "musicality.in_scale_pct": metrics["musicality"]["in_scale_pct"],
            "timing.headroom_pct": metrics["timing"].get("headroom_pct", 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--kind", choices=["bvh", "npz", "melody"], default="bvh")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--out", default="out/grade")
    ap.add_argument("--judge", action="store_true", help="actually call Claude (needs key)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    import cv2

    labels = None
    if args.real:
        from tools.render_build import energetic_window
        from tools.run_dataset import _load
        src = energetic_window(_load(args.real, args.kind, None), args.seconds)
        xyz, fps, flip = src.xyz_seq, src.fps, src.flip_y
    else:
        xyz, labels = scripted_performer(with_labels=True)
        fps, flip = 30.0, False

    r = replay(xyz, fps, flip)
    metrics = compute_metrics(r, labels)
    slo = M.check_slo(flatten(metrics))

    os.makedirs(args.out, exist_ok=True)
    sheet = viz.contact_sheet([_frame_img(*r["per_frame"][i], i / fps)
                               for i in pick_frames(r)])
    roll = viz.pianoroll(r["events"], r["dur"])
    cv2.imwrite(f"{args.out}/contact_sheet.png", sheet)
    cv2.imwrite(f"{args.out}/pianoroll.png", roll)
    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
    prompt = build_prompt(metrics, slo)
    open(f"{args.out}/prompt.txt", "w").write(prompt)
    imgs = [f"{args.out}/contact_sheet.png", f"{args.out}/pianoroll.png"]

    print("METRICS:", json.dumps(flatten(metrics), indent=2))
    print("SLO:", json.dumps(slo))
    print(f"bundle -> {args.out}/ (contact_sheet.png, pianoroll.png, metrics.json, prompt.txt)")

    if args.judge:
        from everybody_dance.judge import _anthropic_grader
        grader = _anthropic_grader(args.model) if args.model else None
        card = grade_bundle(prompt, imgs, grader)
        open(f"{args.out}/scorecard.json", "w").write(card.to_json())
        print("\nSCORECARD:\n" + card.to_json())
    else:
        print("\n(dry run -- pass --judge to call Claude, or grade the bundle by hand)")


if __name__ == "__main__":
    main()
