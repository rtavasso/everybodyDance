"""Tests for the everybodyDance prototype.

These run headless on the synthetic dancer -- the whole point of that source is
that the coupling can be evaluated in CI without a camera or MIDI hardware.
"""

import numpy as np
import pytest

from everybody_dance.engine import Engine, EngineConfig
from everybody_dance.features import FeatureExtractor
from everybody_dance.filters import OneEuroFilter
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.pose import SyntheticPoseSource
from everybody_dance.readout import Readout
from everybody_dance.substrate import SCALES, Substrate, SubstrateConfig, euclidean


# --- substrate primitives -------------------------------------------------

def test_euclidean_distributes_pulses():
    assert sum(euclidean(4, 16)) == 4
    assert euclidean(4, 16) == [1, 0, 0, 0] * 4
    assert euclidean(0, 8) == [0] * 8
    assert euclidean(8, 8) == [1] * 8
    # classic tresillo-ish even spread
    assert sum(euclidean(3, 8)) == 3


def test_snap_in_range_and_in_scale():
    sub = Substrate(SubstrateConfig.default())
    role = sub.cfg.roles["lead"]
    scale = set(SCALES[sub.cfg.scale])
    for v in np.linspace(0, 1, 50):
        n = sub.snap(float(v), role)
        assert role.lo <= n <= role.hi
        assert (n - sub.cfg.tonic) % 12 in scale


# --- one-euro filter ------------------------------------------------------

def test_one_euro_tracks_and_smooths():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    dt = 1 / 60
    # constant step: should converge to the value
    out = None
    for _ in range(120):
        out = f(np.array([5.0]), dt)
    assert abs(out[0] - 5.0) < 0.1
    # noise on a constant signal is attenuated
    f2 = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(300):
        vals.append(f2(np.array([rng.normal(0, 1.0)]), dt)[0])
    assert np.std(vals[50:]) < 0.6  # smoother than the raw unit-variance input


# --- oscillator entrainment ----------------------------------------------

@pytest.mark.parametrize("hz", [1.5, 2.0, 3.0])
def test_oscillator_entrains_to_frequency(hz):
    clk = EntrainedClock(tempo_mode="entrain", init_hz=2.0, coupling=0.3)
    dt = 1 / 60
    st = None
    for i in range(2400):
        sig = 0.1 * np.sin(2 * np.pi * hz * i * dt)
        st = clk.update(sig, dt)
    # within ~8% (adaptive Hopf has a small steady-state bias)
    assert abs(st.tempo_hz - hz) / hz < 0.08
    assert st.confidence > 0.7
    assert st.locked


def test_oscillator_falls_back_to_rubato_on_noise():
    clk = EntrainedClock(tempo_mode="entrain", coupling=0.3)
    rng = np.random.default_rng(1)
    dt = 1 / 60
    st = None
    for _ in range(1800):
        st = clk.update(rng.normal(0, 0.002), dt)  # no periodic content
    assert not st.locked          # dropped out of lock
    assert st.confidence < 0.3


def test_fixed_mode_holds_tempo():
    clk = EntrainedClock(tempo_mode="fixed", fixed_hz=2.0, coupling=0.5)
    dt = 1 / 60
    for i in range(600):
        st = clk.update(0.1 * np.sin(2 * np.pi * 3.0 * i * dt), dt)
    assert abs(st.tempo_hz - 2.0) < 1e-6   # ignores the body's 3 Hz


# --- features -------------------------------------------------------------

def test_bounce_oscillates_with_the_beat():
    src = SyntheticPoseSource(fps=30, duration=6, tempo_hz=2.0)
    fe = FeatureExtractor(fps=30)
    bounce = [fe.update(fr).bounce for fr in src.frames()]
    bounce = np.array(bounce)
    assert np.ptp(bounce) > 0.05         # the knee-flex bounce is visible
    assert bounce.std() > 0.01


def test_freeze_detected_on_stillness():
    src = SyntheticPoseSource(fps=30, duration=4, tempo_hz=2.0, energy=0.0)
    fe = FeatureExtractor(fps=30)
    last = None
    for fr in src.frames():
        last = fe.update(fr)
    assert last.events.get("freeze", 0.0) > 0.5


# --- end to end: the coupling reads --------------------------------------

def _run(script, coupling=0.4, calibrate=10, seed=0, **cfg):
    src = SyntheticPoseSource(fps=30, duration=sum(s["duration"] for s in script),
                              script=script, seed=seed)
    be = LogBackend()
    eng = Engine(src, be, EngineConfig(calibrate_s=calibrate, coupling=coupling,
                                       seed=seed, **cfg))
    eng.run()
    return eng, be


def test_motion_gate_silences_stillness_when_enabled():
    """Exhibit option: with motion_gate, stillness goes quiet; default stays
    musical (the never-silent behaviour above is preserved)."""
    script = [{"duration": 10, "tempo_hz": 2.0, "energy": 1.0},
              {"duration": 4, "tempo_hz": 2.0, "energy": 1.0},
              {"duration": 5, "tempo_hz": 2.0, "energy": 0.004}]   # stillness
    _, be_off = _run(script, calibrate=10)
    _, be_on = _run(script, calibrate=10, motion_gate=True)
    t_end = max(e.t for e in be_off.events)
    tail = lambda be: [e for e in be.events
                       if e.kind == "note_on" and e.t > t_end - 3]
    assert len(tail(be_on)) == 0          # gated quiet during stillness
    assert len(tail(be_off)) > 0          # default keeps a held chord going


def test_tempo_tracks_the_body():
    # slow groove then fast groove; the detected tempo should rise with the body.
    # Avoid an exact-octave change (1.5->3.0), which invites subharmonic lock.
    eng, be = _run([
        {"duration": 10, "tempo_hz": 1.5, "energy": 1.0},
        {"duration": 8, "tempo_hz": 1.5, "energy": 1.0},
        {"duration": 10, "tempo_hz": 2.5, "energy": 1.0},
    ], calibrate=10)
    slow = [t.bpm for t in eng.traces if t.t < eng.traces[0].t + 7 and t.locked]
    fast = [t.bpm for t in eng.traces if t.t > eng.traces[0].t + 14 and t.locked]
    assert slow and fast
    assert np.median(fast) > np.median(slow) + 20   # clock followed the body


def test_output_is_in_scale_and_never_silent_on_stillness():
    eng, be = _run([
        {"duration": 10, "tempo_hz": 2.0, "energy": 1.0},
        {"duration": 5, "tempo_hz": 2.0, "energy": 1.0},
        {"duration": 5, "tempo_hz": 2.0, "energy": 0.005},   # stillness
    ], calibrate=10)
    scale = set(SCALES[eng.substrate.cfg.scale])
    tonic = eng.substrate.cfg.tonic
    melodic = [e for e in be.events if e.kind == "note_on" and e.channel != 10]
    assert melodic
    assert all((e.a - tonic) % 12 in scale for e in melodic)
    # stillness must produce a held/suspended chord, not silence
    assert any(e.tag == "chord.sus" and e.kind == "note_on" for e in be.events)


def test_coupling_dial_changes_behaviour():
    # Different coupling should yield a different phase trajectory (the dial works).
    e_lo, _ = _run([{"duration": 16, "tempo_hz": 2.0, "energy": 1.0}],
                   coupling=0.0, calibrate=8)
    e_hi, _ = _run([{"duration": 16, "tempo_hz": 2.0, "energy": 1.0}],
                   coupling=0.95, calibrate=8)
    p_lo = np.array([t.phase for t in e_lo.traces])
    p_hi = np.array([t.phase for t in e_hi.traces])
    n = min(len(p_lo), len(p_hi))
    assert np.mean(np.abs(p_lo[:n] - p_hi[:n])) > 1e-3


def test_no_hanging_notes_after_run():
    eng, be = _run([{"duration": 14, "tempo_hz": 2.0, "energy": 1.0}], calibrate=8)
    # panic at end emits all-notes-off on every channel
    assert any(e.kind == "cc" and e.a == 123 for e in be.events)


def test_source_joint_maps_complete():
    # the decoded BVH / Kinect / MediaPipe maps must cover all 13 joints uniquely
    from everybody_dance.pose import JOINTS
    from everybody_dance.sources import BVH_MAP, MELODY_MAP, MP_SUBSET
    for m in (BVH_MAP, MELODY_MAP, MP_SUBSET):
        assert set(m.keys()) == set(JOINTS)
        assert len(set(m.values())) == len(JOINTS)   # no joint mapped twice
