"""Specificity tests -- the explicit acceptance criteria:

  * stop moving  -> the music stops (only a held breath chord remains)
  * move one body part -> one thing moves in the audio (lead, not the whole band)

Self-contained: builds pose arrays directly, calibrates on a full-body groove
(via the engine's own calibration prefix), then plays a probe. No external data.
"""

import numpy as np

from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.output import LogBackend
from everybody_dance.pose import JOINT_INDEX, JOINTS
from everybody_dance.sources import ArrayPoseSource

FPS = 30.0


def neutral():
    p = {
        "nose": (0, 1.6, 0), "l_shoulder": (-0.2, 1.4, 0), "r_shoulder": (0.2, 1.4, 0),
        "l_elbow": (-0.3, 1.1, 0), "r_elbow": (0.3, 1.1, 0),
        "l_wrist": (-0.35, 0.85, 0), "r_wrist": (0.35, 0.85, 0),
        "l_hip": (-0.12, 0.9, 0), "r_hip": (0.12, 0.9, 0),
        "l_knee": (-0.13, 0.45, 0), "r_knee": (0.13, 0.45, 0),
        "l_ankle": (-0.14, 0.0, 0), "r_ankle": (0.14, 0.0, 0),
    }
    return np.array([p[j] for j in JOINTS], dtype=float)


def groove_frames(n, rng, amp=1.0):
    base = neutral()
    out = np.zeros((n, 13, 3))
    for i in range(n):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base.copy()
        x[:, 1] -= 0.12 * amp * np.sin(ph)                 # bounce
        for a in ("l_ankle", "r_ankle"):
            x[JOINT_INDEX[a], 1] = base[JOINT_INDEX[a], 1]  # feet planted
        for w, s in (("l_wrist", 1), ("r_wrist", -1)):
            x[JOINT_INDEX[w], 1] += 0.25 * amp * s * np.sin(ph)
        out[i] = x + rng.normal(0, 0.004, (13, 3))
    return out


def still_frames(n, rng):
    return neutral()[None] + rng.normal(0, 0.002, (n, 13, 3))


def run(seq, calib_s):
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    be = LogBackend()
    Engine(src, be, EngineConfig(fps=FPS, calibrate_s=calib_s)).run()
    return be


def _count(be, t0, t1, prefix=""):
    return sum(1 for e in be.events if e.kind == "note_on" and t0 <= e.t < t1
               and e.tag.startswith(prefix))


def test_stop_moving_stops_the_music():
    rng = np.random.default_rng(0)
    calib = int(15 * FPS)
    play = int(8 * FPS)
    stop = int(6 * FPS)
    seq = np.concatenate([groove_frames(calib + play, rng), still_frames(stop, rng)])
    be = run(seq, calib_s=15)
    t_play0, t_stop0 = 15.0, 15.0 + play / FPS
    moving = _count(be, t_play0, t_stop0) / (t_stop0 - t_play0)
    stopped = _count(be, t_stop0 + 1, t_stop0 + stop / FPS) / (stop / FPS - 1)
    assert moving > 1.0
    assert stopped < 0.25 * moving           # the music stops
    # ...but a held breath chord is sounding (not dead silence)
    assert _count(be, t_stop0, t_stop0 + stop / FPS, "chord") >= 1


def test_one_body_part_moves_one_thing():
    rng = np.random.default_rng(1)
    calib = int(15 * FPS)
    play = int(10 * FPS)
    base = neutral()
    wi = JOINT_INDEX["r_wrist"]
    probe = np.zeros((play, 13, 3))
    for i in range(play):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base + rng.normal(0, 0.002, (13, 3))
        x[wi, 0] += 0.35 * np.sin(ph)            # ONLY the right wrist
        x[wi, 1] += 0.2 * np.sin(ph * 0.7)
        probe[i] = x
    seq = np.concatenate([groove_frames(calib, rng), probe])
    be = run(seq, calib_s=15)
    t0 = 15.0
    lead = _count(be, t0, t0 + play / FPS, "lead")
    drums = _count(be, t0, t0 + play / FPS, "drums")
    bass = _count(be, t0, t0 + play / FPS, "bass")
    # the melody (limbs) responds; the beat (core) stays quiet
    assert lead > 4
    assert lead > 2 * (drums + bass)
    # stereo pan deflects with the one-sided motion
    pans = [e.b for e in be.events if e.kind == "cc" and e.a == 10 and e.t > t0]
    assert pans and (max(pans) - min(pans)) > 20
