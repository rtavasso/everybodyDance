#!/usr/bin/env python3
"""Specificity probes on a REAL skeleton (a LAFAN1 frame):

  A) move, then STOP  -> music should stop (only a held/breath chord remains).
  B) move ONE body part (right wrist) -> ONE thing should move in the audio
     (lead + stereo pan), not the whole band.

Prints event counts per window so the localisation is measurable, not vibes.
"""

import numpy as np

from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.output import LogBackend
from everybody_dance.sources import ArrayPoseSource, load_bvh_source
from everybody_dance.features import FeatureExtractor
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.personalization import Calibrator
from everybody_dance.pose import JOINT_INDEX

FPS = 30.0


def base_pose():
    src = load_bvh_source("data/bvh/dance1_subject1.bvh", max_seconds=10)
    return src.xyz_seq[120].copy()      # a mid, upright-ish real pose (raw, y-up)


def calib(src):
    fe, lab, clk, cal = (FeatureExtractor(src.fps), LabanEstimator(),
                         EntrainedClock(), Calibrator())
    for fr in src.frames():
        f = fe.update(fr)
        cal.observe(f, lab.update(f, f.dt), clk.update(float(f.bounce), f.dt).tempo_hz)
    return cal.finalize(signature=fe.signature.summary())


def full_body_profile():
    """Calibrate on real full-body dance. Specificity is *relative* to this:
    once the system knows your full range, a small partial motion should give a
    small, localised response."""
    src = load_bvh_source("data/bvh/dance1_subject1.bvh", max_seconds=50)
    return calib(src)


def run(xyz_seq, name, profile):
    src = ArrayPoseSource(xyz_seq, fps=FPS, flip_y=False, name=name)
    be = LogBackend()
    Engine(src, be, EngineConfig(fps=FPS, calibrate_s=0), profile=profile).run()
    return be


def counts(be, t0, t1, tag_prefix=""):
    return sum(1 for e in be.events if e.kind == "note_on" and t0 <= e.t < t1
               and e.tag.startswith(tag_prefix))


def probe_stop():
    base = base_pose()
    T_move, T_stop = int(18 * FPS), int(8 * FPS)
    seq = np.zeros((T_move + T_stop, 13, 3))
    rng = np.random.default_rng(0)
    # whole-body groove: vertical knee-flex bounce + arm swing
    for i in range(T_move):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base.copy()
        bob = 6.0 * np.sin(ph)
        x[:, 1] -= bob
        for a in ("l_ankle", "r_ankle"):
            x[JOINT_INDEX[a], 1] = base[JOINT_INDEX[a], 1]   # feet planted
        for w, sgn in (("l_wrist", 1), ("r_wrist", -1)):
            x[JOINT_INDEX[w], 1] += 12 * sgn * np.sin(ph)
        seq[i] = x + rng.normal(0, 0.2, (13, 3))
    for i in range(T_stop):
        seq[T_move + i] = base + rng.normal(0, 0.05, (13, 3))   # essentially still
    be = run(seq, "stop", PROFILE)
    move_s, stop_s = T_move / FPS, T_stop / FPS
    mv = counts(be, 2, move_s) / (move_s - 2)
    stp = counts(be, move_s + 1, move_s + stop_s) / (stop_s - 1)
    print("\n[A] move -> STOP")
    print(f"   moving : {mv:5.2f} note-ons/s")
    print(f"   stopped: {stp:5.2f} note-ons/s   "
          f"(drums={counts(be, move_s+1, move_s+stop_s, 'drums')}, "
          f"lead={counts(be, move_s+1, move_s+stop_s, 'lead')}, "
          f"sus-chord={counts(be, move_s+1, move_s+stop_s, 'chord.sus')})")
    print(f"   -> music {'STOPS' if stp < 0.3 * mv else 'does NOT stop'} when the body stops")


def probe_one_part():
    base = base_pose()
    T = int(24 * FPS)
    seq = np.zeros((T, 13, 3))
    rng = np.random.default_rng(1)
    wi = JOINT_INDEX["r_wrist"]
    for i in range(T):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base + rng.normal(0, 0.05, (13, 3))   # body essentially still
        x[wi, 0] += 18 * np.sin(ph)               # ONLY the right wrist moves
        x[wi, 1] += 10 * np.sin(ph * 0.7)
        seq[i] = x
    be = run(seq, "one_part", PROFILE)
    dur = T / FPS
    drums = counts(be, 2, dur, "drums")
    bass = counts(be, 2, dur, "bass")
    lead = counts(be, 2, dur, "lead")
    pans = [e.b for e in be.events if e.kind == "cc" and e.a == 10]
    print("\n[B] move ONE body part (right wrist)")
    print(f"   lead={lead}  drums={drums}  bass={bass}  "
          f"pan range={min(pans) if pans else 64}..{max(pans) if pans else 64}")
    print(f"   -> {'localised (lead/pan dominate, band quiet)' if lead > 3 * max(drums, 1) or drums < 5 else 'NOT localised'}")


if __name__ == "__main__":
    PROFILE = full_body_profile()
    probe_stop()
    probe_one_part()
