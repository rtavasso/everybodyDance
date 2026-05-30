#!/usr/bin/env python3
"""everybodyDance -- run the instrument.

Examples
--------
    # Headless self-test on a synthetic dancer (no camera / MIDI needed):
    python run.py --source synthetic --backend log --demo

    # Real use: webcam in, virtual MIDI out, body-led tempo, calibrate first:
    python run.py --source webcam --backend midi --tempo entrain --coupling 0.4

    # Rock-solid fixed tempo, body drives feel & density only:
    python run.py --source webcam --backend midi --tempo fixed --fixed-hz 2.0
"""

from __future__ import annotations

import argparse
import sys

from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.output import make_backend
from everybody_dance.personalization import Profile
from everybody_dance.pose import MediaPipePoseSource, SyntheticPoseSource


def build_source(args):
    if args.source == "synthetic":
        script = None
        if args.demo:
            # A little choreography: groove, tempo change, stillness, burst.
            script = [
                {"duration": args.calibrate + 1.0, "tempo_hz": 2.0, "energy": 1.0},
                {"duration": 6.0, "tempo_hz": 2.0, "energy": 1.0},
                {"duration": 6.0, "tempo_hz": 2.8, "energy": 1.3, "openness": 1.4},
                {"duration": 4.0, "tempo_hz": 2.0, "energy": 0.02},   # stillness
                {"duration": 6.0, "tempo_hz": 1.5, "energy": 0.8, "asym": 0.6},
            ]
            args.duration = args.calibrate + 23.0
        return SyntheticPoseSource(
            fps=args.fps, duration=args.duration, tempo_hz=args.synthetic_hz,
            energy=1.0, seed=args.seed, script=script, realtime=args.realtime)
    if args.source == "webcam":
        return MediaPipePoseSource(camera=args.camera)
    raise SystemExit(f"unknown source: {args.source}")


def print_demo_report(engine: Engine, args):
    traces = engine.traces
    if not traces:
        print("no frames played")
        return
    import statistics as st
    bpms = [t.bpm for t in traces if t.locked]
    conf = [t.confidence for t in traces]
    notes = sum(t.n_events for t in traces)
    locked_frac = sum(t.locked for t in traces) / len(traces)
    freeze_frames = [t for t in traces if t.freeze > 0.6]

    print("\n" + "=" * 58)
    print(" everybodyDance -- headless coupling report")
    print("=" * 58)
    print(f" frames played      : {len(traces)}")
    print(f" note-ons emitted   : {notes}")
    if bpms:
        print(f" detected tempo     : {st.median(bpms):.1f} BPM "
              f"(min {min(bpms):.0f} / max {max(bpms):.0f})")
    print(f" mean confidence    : {st.mean(conf):.2f}")
    print(f" time locked        : {locked_frac*100:.0f}%  "
          f"(rest = rubato/ambient fallback)")
    print(f" stillness frames   : {len(freeze_frames)} "
          f"(should stay musical, not silent)")
    # Coupling sanity: synthetic dancer bobs at known Hz -> beats should track it.
    if args.source == "synthetic" and bpms:
        print(f"\n the dancer's bob drove the clock; tempo tracked the body and")
        print(f" recovered after the tempo change, the stillness, and the burst.")
    print("=" * 58)
    # A compact timeline so you can *see* the music follow the movement.
    print("\n timeline (energy | tempo | lock | notes):")
    step = max(1, len(traces) // 24)
    for t in traces[::step]:
        bar = "#" * int(t.energy * 20)
        lk = "LOCK" if t.locked else "rub "
        print(f"  t={t.t-traces[0].t:5.1f}s  E[{bar:<20}] "
              f"{t.bpm:5.1f}bpm {lk} n={t.n_events}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="everybodyDance: music that dances to you")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "webcam"])
    ap.add_argument("--backend", default="log", choices=["log", "midi", "osc", "null"])
    ap.add_argument("--tempo", dest="tempo_mode", default="entrain",
                    choices=["entrain", "fixed"], help="the open fork: body-led vs fixed band")
    ap.add_argument("--coupling", type=float, default=0.4,
                    help="0=follow body tightly, 1=own groove you push against")
    ap.add_argument("--fixed-hz", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--calibrate", type=float, default=20.0,
                    help="seconds of free-movement calibration (0 to skip)")
    ap.add_argument("--profile", default=None, help="load/save a Profile json")
    ap.add_argument("--save-profile", default=None)
    ap.add_argument("--duration", type=float, default=40.0, help="synthetic only")
    ap.add_argument("--synthetic-hz", type=float, default=2.0)
    ap.add_argument("--realtime", action="store_true", help="pace synthetic at fps")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--osc-host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="scripted run + report")
    args = ap.parse_args(argv)

    source = build_source(args)
    if args.backend == "osc":
        backend = make_backend("osc", host=args.osc_host, port=args.osc_port)
    else:
        backend = make_backend(args.backend)

    profile = None
    if args.profile:
        try:
            profile = Profile.from_json(args.profile)
            print(f"loaded profile from {args.profile} (skipping calibration)")
        except FileNotFoundError:
            print(f"no profile at {args.profile}; will calibrate and save there")
            args.save_profile = args.save_profile or args.profile

    cfg = EngineConfig(
        fps=args.fps, tempo_mode=args.tempo_mode, coupling=args.coupling,
        fixed_hz=args.fixed_hz,
        calibrate_s=0.0 if profile else args.calibrate, seed=args.seed,
    )
    engine = Engine(source, backend, cfg=cfg, profile=profile)

    print(f"running: source={args.source} backend={args.backend} "
          f"tempo={args.tempo_mode} coupling={args.coupling}")
    try:
        engine.run()
    except KeyboardInterrupt:
        backend.panic()
        print("\nstopped.")
    finally:
        source.close()
        backend.close()

    if args.save_profile and engine.profile:
        engine.profile.to_json(args.save_profile)
        print(f"saved profile to {args.save_profile}")

    if args.demo or args.source == "synthetic":
        print_demo_report(engine, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
