"""Tests for the Studio engine: deterministic multi-stem performance, the
phantom/stillness gate, the body-looper, gesture actions (scale shift, timbre
morph, stem mute), and the StudioOut/StudioUI contract.

Self-contained synthetic groove (no cv2/audio/data), mirroring test_builder /
test_looper. Commands are force-fired move names, which is both the keyboard/
pedal path and how these tests inject gestures deterministically.
"""

import numpy as np

from everybody_dance.features import FeatureExtractor
from everybody_dance.laban import LabanEstimator
from everybody_dance.oscillator import EntrainedClock
from everybody_dance.output import LogBackend
from everybody_dance.personalization import Calibrator
from everybody_dance.pose import JOINT_INDEX, JOINTS
from everybody_dance.sources import ArrayPoseSource
from everybody_dance.studio import Studio, StudioConfig, StudioOut, StudioUI, StemUI
from everybody_dance.substrate import SCALES

FPS = 30.0
AUTOM_KEYS = {"cutoff", "drive", "gain", "pan", "reverb", "delay", "timbre"}


def _neutral():
    p = {"nose": (0, 1.6, 0), "l_shoulder": (-0.2, 1.4, 0), "r_shoulder": (0.2, 1.4, 0),
         "l_elbow": (-0.3, 1.1, 0), "r_elbow": (0.3, 1.1, 0),
         "l_wrist": (-0.35, 0.85, 0), "r_wrist": (0.35, 0.85, 0),
         "l_hip": (-0.12, 0.9, 0), "r_hip": (0.12, 0.9, 0),
         "l_knee": (-0.13, 0.45, 0), "r_knee": (0.13, 0.45, 0),
         "l_ankle": (-0.14, 0.0, 0), "r_ankle": (0.14, 0.0, 0)}
    return np.array([p[j] for j in JOINTS], dtype=float)


def _groove(n, seed=0, energy=1.0):
    """A full-body groove: a knee-flex bounce (the beat carrier) plus wide arm
    swings (the limb energy that drives the lead). The knee-flex keeps the hips,
    shoulders and ankles fixed, so the COM-above-feet bounces (the clock locks)
    while the body root (root_y) stays put -- no false JUMP/STOMP firings, mirror-
    ing how the shipped SyntheticPoseSource bounces."""
    base = _neutral()
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 13, 3))
    for i in range(n):
        ph = 2 * np.pi * 2.0 * i / FPS
        x = base.copy()
        for k in ("l_knee", "r_knee"):
            x[JOINT_INDEX[k], 1] = base[JOINT_INDEX[k], 1] \
                - 0.12 * energy * max(0.0, np.sin(ph))
        for w, e, s in (("l_wrist", "l_elbow", 1), ("r_wrist", "r_elbow", -1)):
            x[JOINT_INDEX[w], 1] += 0.45 * energy * s * np.sin(ph)
            x[JOINT_INDEX[w], 0] += 0.30 * energy * s * np.cos(ph)
            x[JOINT_INDEX[e], 1] += 0.22 * energy * s * np.sin(ph)
        out[i] = x + rng.normal(0, 0.004, (13, 3))
    return out


def _still(n):
    """A held neutral pose -> stillness (no motion -> freeze, energy ~0)."""
    base = _neutral()
    return np.repeat(base[None], n, axis=0)


def _profile(seq):
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    fe, lab, clk, cal = FeatureExtractor(FPS), LabanEstimator(), EntrainedClock(), Calibrator()
    for fr in src.frames():
        f = fe.update(fr)
        cal.observe(f, lab.update(f, f.dt), clk.update(float(f.bounce), f.dt).tempo_hz)
    return cal.finalize(signature=fe.signature.summary())


def _run(seq, commands_at=None, cfg=None):
    """Run the studio over `seq`; `commands_at` maps frame index -> [moves]."""
    commands_at = commands_at or {}
    prof = _profile(seq)
    be = LogBackend()
    st = Studio(be, prof, cfg=cfg)
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    last = None
    for i, fr in enumerate(src.frames()):
        last = st.step(fr, commands_at.get(i, []))
    st.panic()
    return st, be, last


def _onsets(be):
    return [(e.channel, e.a, round(e.t, 3), e.tag)
            for e in be.events if e.kind == "note_on"]


# -- contract / structure --------------------------------------------------

def test_returns_studio_out_with_full_ui_and_automation():
    st, be, out = _run(_groove(int(16 * FPS)))
    assert isinstance(out, StudioOut)
    assert isinstance(out.ui, StudioUI)
    # one StemUI per stem, in stem order, with sane ranges
    assert [s.name for s in out.ui.stems] == ["drums", "bass", "keys", "lead", "texture"]
    for s in out.ui.stems:
        assert isinstance(s, StemUI)
        assert 0.0 <= s.level <= 1.0 and 0.0 <= s.density <= 1.0
        assert 0.0 <= s.cutoff <= 1.0
        assert len(s.onsets) == out.ui.loop_steps and s.timbre
    # automation: an entry per stem with all required keys
    assert set(out.automation) == {"drums", "bass", "keys", "lead", "texture"}
    for name, a in out.automation.items():
        assert set(a) == AUTOM_KEYS
        assert -1.0 <= a["pan"] <= 1.0 and isinstance(a["timbre"], str)
    assert out.ui.phase == "perform" and out.ui.present
    assert out.ui.loop_steps == st.cfg.loop_steps


def test_exposes_substrate_like_engine():
    st, be, out = _run(_groove(int(8 * FPS)))
    assert st.sub is not None and hasattr(st.sub, "cfg")
    assert st.sub.cfg.scale in SCALES


# -- determinism -----------------------------------------------------------

def test_determinism_same_dance_same_commands():
    seq = _groove(int(20 * FPS))
    cmds = {int(8 * FPS): ["JUMP"], int(12 * FPS): ["SQUAT"]}
    _, be1, _ = _run(seq, cmds)
    _, be2, _ = _run(seq, cmds)
    assert _onsets(be1) == _onsets(be2)            # byte-identical note stream


# -- multi-stem + channels + scale -----------------------------------------

def test_multi_stem_tags_and_channels():
    st, be, out = _run(_groove(int(24 * FPS)))
    tags = {e.tag for e in be.events if e.kind == "note_on"}
    stem_tags = {t for t in tags if t in
                 {"drums", "bass", "keys", "lead", "texture"}}
    assert len(stem_tags) >= 4                  # contract: >= 4 distinct stems
    chan = {"drums": 10, "bass": 1, "keys": 2, "lead": 3, "texture": 4}
    for e in be.events:
        if e.kind == "note_on" and e.tag in chan:
            assert e.channel == chan[e.tag]


def test_fill_adds_percussion_and_breakdown_strips_low_end():
    seq = _groove(int(24 * FPS))
    st0, be0, _ = _run(seq)
    # JUMP -> fill on drums over ~1 bar: extra percussion onsets.
    st1, be1, _ = _run(seq, {int(t * FPS): ["JUMP"] for t in (8, 10, 12)})

    def cnt(be, tag, a, b):
        return sum(1 for e in be.events if e.kind == "note_on"
                   and e.tag == tag and a <= e.t < b)
    assert cnt(be1, "drums", 8, 13) > cnt(be0, "drums", 8, 13)
    # T-POSE -> breakdown: strip drums/bass to sparse briefly.
    st2, be2, _ = _run(seq, {int(8 * FPS): ["T-POSE"]})
    assert cnt(be2, "drums", 8.0, 9.4) + cnt(be2, "bass", 8.0, 9.4) == 0
    assert cnt(be0, "drums", 8.0, 9.4) > 0      # baseline had drums there


def test_pitched_stems_in_scale():
    st, be, out = _run(_groove(int(24 * FPS)))
    tonic = st.sub.cfg.tonic
    scale = set(SCALES[st.sub.cfg.scale])
    mel = [e for e in be.events if e.kind == "note_on"
           and e.tag in ("bass", "keys", "lead", "texture")]
    assert mel and all((e.a - tonic) % 12 in scale for e in mel)


# -- the phantom / stillness gate ------------------------------------------

def test_phantom_gate_suppresses_onsets_during_stillness():
    seq = np.concatenate([_groove(int(14 * FPS)), _still(int(8 * FPS))])
    st, be, out = _run(seq)
    t_split = 14.0
    motion = [e for e in be.events if e.kind == "note_on" and e.t < t_split]
    # ignore the first ~0.5s of the stillness window (held notes/pad allowance)
    still = [e for e in be.events if e.kind == "note_on" and e.t > t_split + 0.8]
    assert motion, "expected notes during motion"
    motion_rate = len(motion) / t_split
    still_rate = len(still) / (seq.shape[0] / FPS - (t_split + 0.8))
    assert still_rate < 0.2 * motion_rate          # near-zero onsets when still
    assert out.ui.phase == "perform" or out.ui.energy < st.cfg.motion_floor


def test_no_body_gates_onsets():
    n = int(10 * FPS)
    seq = _groove(n)
    present = np.ones(n, dtype=bool)
    present[int(6 * FPS):] = False                  # body leaves
    prof = _profile(seq)
    be = LogBackend()
    st = Studio(be, prof)
    src = ArrayPoseSource(seq, fps=FPS, present=present, flip_y=False)
    out = None
    for fr in src.frames():
        out = st.step(fr)
    st.panic()
    absent = [e for e in be.events if e.kind == "note_on" and e.t > 6.5]
    assert len(absent) <= 2                         # essentially no new onsets
    assert out.ui.phase == "attract" and not out.ui.present


# -- looper ----------------------------------------------------------------

def test_looper_record_lock_replay_and_clear():
    seq = _groove(int(28 * FPS))
    # SQUAT is bound to loop_record on bass. Fire once to record, again to lock.
    cmds = {int(6 * FPS): ["SQUAT"], int(16 * FPS): ["SQUAT"]}
    st, be, out = _run(seq, cmds)
    bass = st.rack.get("bass")
    assert bass.looping and sum(len(s) for s in bass.buffer) > 0
    # the locked loop keeps replaying after the lock
    bass_after = [e for e in be.events if e.kind == "note_on"
                  and e.tag == "bass" and e.t > 16.5 / 1.0]
    assert bass_after, "locked bass loop should keep sounding"
    # loop_clear empties it
    st._do_action(_binding("loop_clear", "bass"), out.ui.t)
    assert not bass.looping and sum(len(s) for s in bass.buffer) == 0


def test_loop_replay_is_deterministic():
    seq = _groove(int(24 * FPS))
    cmds = {int(6 * FPS): ["SQUAT"], int(14 * FPS): ["SQUAT"]}
    _, be1, _ = _run(seq, cmds)
    _, be2, _ = _run(seq, cmds)
    b1 = [(e.a, round(e.t, 3)) for e in be1.events
          if e.kind == "note_on" and e.tag == "bass"]
    b2 = [(e.a, round(e.t, 3)) for e in be2.events
          if e.kind == "note_on" and e.tag == "bass"]
    assert b1 == b2 and b1


# -- gesture actions -------------------------------------------------------

def test_scale_shift_changes_scale():
    seq = _groove(int(12 * FPS))
    st, be, out = _run(seq, {int(6 * FPS): ["ARMS CROSSED"]})
    # ARMS CROSSED -> scale_shift to "dark" -> phrygian
    assert st.sub.cfg.scale == "phrygian"
    assert out.ui.mood == "phrygian"
    # and notes after the shift are in the new scale
    tonic = st.sub.cfg.tonic
    late = [e for e in be.events if e.kind == "note_on" and e.t > 7.0
            and e.tag in ("bass", "keys", "lead", "texture")]
    assert all((e.a - tonic) % 12 in set(SCALES["phrygian"]) for e in late)


def test_timbre_morph_reflected_in_automation():
    seq = _groove(int(12 * FPS))
    st, be, out = _run(seq, {int(6 * FPS): ["RAISE RIGHT"]})
    # RAISE RIGHT -> timbre_morph lead -> glass_pad
    assert out.automation["lead"]["timbre"] == "glass_pad"
    assert st.rack.get("lead").timbre == "glass_pad"


def test_stem_mute_silences_that_stem():
    # add a stem_mute binding so we can exercise it deterministically
    from everybody_dance.mapping import MappingConfig, GestureBinding
    cfg = MappingConfig.default()
    cfg.gestures.append(GestureBinding("CLAP", "stem_mute", target="drums"))
    seq = _groove(int(20 * FPS))
    prof = _profile(seq)
    be = LogBackend()
    st = Studio(be, prof, mapping=cfg)
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    for i, fr in enumerate(src.frames()):
        st.step(fr, ["CLAP"] if i == int(8 * FPS) else [])
    st.panic()
    drums_before = [e for e in be.events if e.kind == "note_on"
                    and e.tag == "drums" and e.t < 8.0]
    drums_after = [e for e in be.events if e.kind == "note_on"
                   and e.tag == "drums" and e.t > 9.0]
    assert drums_before and not drums_after        # muted after the gesture


def test_forced_command_always_flashes():
    seq = _groove(int(8 * FPS))
    prof = _profile(seq)
    be = LogBackend()
    st = Studio(be, prof)
    src = ArrayPoseSource(seq, fps=FPS, flip_y=False)
    fire = int(4 * FPS)
    flashed = False
    for i, fr in enumerate(src.frames()):
        out = st.step(fr, ["T-POSE"] if i == fire else [])
        if i == fire:
            # the studio exposes active flashes so the stage can render any move
            flashed = any(f.name == "T-POSE" for f in out.ui.flashes)
    st.panic()
    assert flashed


def test_fx_gesture_emits_fx_tagged_notes():
    seq = _groove(int(10 * FPS))
    st, be, out = _run(seq, {int(5 * FPS): ["CLAP"]})   # CLAP -> fx snare on drums
    assert any(e.kind == "note_on" and e.tag == "fx" for e in be.events)


# -- no hanging notes ------------------------------------------------------

def test_no_hanging_notes():
    st, be, out = _run(_groove(int(20 * FPS)))
    from collections import Counter
    on = Counter((e.channel, e.a) for e in be.events if e.kind == "note_on")
    off = Counter((e.channel, e.a) for e in be.events if e.kind == "note_off")
    # every note_on (channel,pitch) is matched by >= as many note_offs (panic
    # adds all-notes-off cc too); no key has more ons than offs left hanging.
    for key, n_on in on.items():
        assert off.get(key, 0) >= n_on, f"hanging note {key}: {n_on} on, {off.get(key,0)} off"


# -- helpers ---------------------------------------------------------------

def _binding(action, target, params=None):
    from everybody_dance.mapping import GestureBinding
    return GestureBinding("X", action, target=target, params=params or {})
