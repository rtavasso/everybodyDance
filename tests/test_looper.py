"""Tests for the live body-looper: phase machine, loop capture/replay, tempo
lock, track isolation while recording, and perform-mode modulation.

Self-contained (synthetic full-body groove), so it runs in CI.
"""

import numpy as np

from everybody_dance.controls import Command, ScriptedControl
from everybody_dance.features import FeatureExtractor
from everybody_dance.laban import LabanEstimator
from everybody_dance.looper import LoopStation, LooperConfig
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.personalization import Calibrator
from everybody_dance.pose import JOINT_INDEX, JOINTS
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.substrate import SCALES

FPS = 30.0


def _neutral():
    p = {"nose": (0, 1.6, 0), "l_shoulder": (-0.2, 1.4, 0), "r_shoulder": (0.2, 1.4, 0),
         "l_elbow": (-0.3, 1.1, 0), "r_elbow": (0.3, 1.1, 0),
         "l_wrist": (-0.35, 0.85, 0), "r_wrist": (0.35, 0.85, 0),
         "l_hip": (-0.12, 0.9, 0), "r_hip": (0.12, 0.9, 0),
         "l_knee": (-0.13, 0.45, 0), "r_knee": (0.13, 0.45, 0),
         "l_ankle": (-0.14, 0.0, 0), "r_ankle": (0.14, 0.0, 0)}
    return np.array([p[j] for j in JOINTS], dtype=float)


def _groove(n, seed=0):
    base = _neutral()
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 13, 3))
    for i in range(n):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base.copy()
        x[:, 1] -= 0.12 * np.sin(ph)
        for a in ("l_ankle", "r_ankle"):
            x[JOINT_INDEX[a], 1] = base[JOINT_INDEX[a], 1]
        for w, s in (("l_wrist", 1), ("r_wrist", -1)):
            x[JOINT_INDEX[w], 1] += 0.3 * s * np.sin(ph)
            x[JOINT_INDEX[w], 0] += 0.15 * s * np.cos(ph)
        out[i] = x + rng.normal(0, 0.004, (13, 3))
    return out


def _profile(seq):
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    fe, lab, clk, cal = FeatureExtractor(FPS), LabanEstimator(), EntrainedClock(), Calibrator()
    for fr in src.frames():
        f = fe.update(fr)
        cal.observe(f, lab.update(f, f.dt), clk.update(float(f.bounce), f.dt).tempo_hz)
    return cal.finalize(signature=fe.signature.summary())


def _run(rec=6.0, perform=8.0, commits=4):
    total = int((rec * commits + perform) * FPS)
    seq = _groove(total)
    prof = _profile(seq)
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    be = LogBackend()
    st = LoopStation(be, prof, LooperConfig())
    sched = [(rec * (i + 1), Command.COMMIT) for i in range(commits)]
    st.run(src, ScriptedControl(sched))
    return st, be, rec


def test_phases_advance_and_lock():
    st, be, rec = _run()
    assert st.st.phase == "perform"
    assert st.st.committed == ["drums", "bass", "chord", "lead"]
    assert st.st.tempo_locked and st.clock.tempo_mode == "fixed"


def test_committed_loops_are_captured_and_replay():
    st, be, rec = _run()
    # every instrument left a non-empty loop buffer
    for role in ("drums", "bass", "chord", "lead"):
        hits = sum(len(s) for s in st.buffers[role])
        assert hits > 0, f"{role} loop empty"


def test_future_tracks_silent_while_recording():
    st, be, rec = _run()
    # during the bass take (rec..2*rec) nothing from chord/lead should sound yet
    win = [e for e in be.events if e.kind == "note_on" and rec <= e.t < 2 * rec]
    assert win
    assert not any(e.tag.startswith(("chord", "lead")) for e in win)
    # drums (already committed) DO play
    assert any(e.tag.startswith("drums") for e in win)


def test_perform_modulates_the_gestalt():
    st, be, rec = _run()
    t_perform = rec * 4
    mcc = [e for e in be.events if e.kind == "cc" and e.channel == 16
           and e.a in (74, 11, 1) and e.t >= t_perform]
    assert len(mcc) > 10        # continuous global modulation is happening


def test_loop_output_in_scale():
    st, be, rec = _run()
    tonic = st.sub.cfg.tonic
    scale = set(SCALES[st.sub.cfg.scale])
    mel = [e for e in be.events if e.kind == "note_on" and e.channel != 10]
    assert mel and all((e.a - tonic) % 12 in scale for e in mel)


def test_undo_clears_active_take():
    seq = _groove(int(10 * FPS))
    prof = _profile(seq)
    be = LogBackend()
    st = LoopStation(be, prof, LooperConfig())
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    undone = False
    for fr in src.frames():
        if fr.t >= 5.0 and not undone:        # record ~5s, then hit undo
            assert sum(len(s) for s in st.buffers["drums"]) > 0   # had a take
            st.step(fr, [Command.UNDO])
            undone = True
            break
        st.step(fr, [])
    # undo cleared the active take, still on drums to re-record
    assert st.st.active == "drums"
    assert sum(len(s) for s in st.buffers["drums"]) == 0
