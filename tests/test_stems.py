"""Tests for the per-stem mechanics + step-quantised loopers.

Pure unit tests over stems.py: the StemRack wires five stems to substrate roles
with the right channels/colours, looper capture/replay/clear behave like the
LoopStation buffers, solo-aware muting works, and generation stays in-scale and
honours register/density. No cv2, no audio, no pose pipeline -- a Substrate is
all that's needed.
"""

import numpy as np

from everybody_dance.mapping import StemParams
from everybody_dance.output import MusicEvent
from everybody_dance import stems as stemlib
from everybody_dance.substrate import SCALES, Substrate, SubstrateConfig

STEPS = 32  # loop_steps for the tests


def _sub():
    sub = Substrate(SubstrateConfig.default())
    sub.cfg.steps_per_beat = 4
    sub.cfg.beats_per_bar = 4
    return sub


def _rack():
    return stemlib.StemRack(_sub(), STEPS, timbres={
        "drums": "analog_kick", "bass": "sub_bass", "keys": "warm_keys",
        "lead": "supersaw_lead", "texture": "glass_pad"})


def test_rack_wires_five_stems_channels_and_colours():
    rack = _rack()
    names = [s.name for s in rack]
    assert names == ["drums", "bass", "keys", "lead", "texture"]
    chans = {s.name: s.channel for s in rack}
    assert chans == {"drums": 10, "bass": 1, "keys": 2, "lead": 3, "texture": 4}
    for s in rack:
        assert isinstance(s.color, tuple) and len(s.color) == 3
        assert s.timbre  # each carries a preset name


def test_keys_uses_chord_role_and_texture_role_added():
    sub = _sub()
    rack = stemlib.StemRack(sub, STEPS)
    assert rack.get("keys").role is sub.cfg.roles["chord"]
    assert "texture" in sub.cfg.roles                # added on top
    tex = sub.cfg.roles["texture"]
    assert tex.legato and tex.max_density <= 4 and tex.hi - tex.lo >= 12


def test_looper_capture_replay_and_clear():
    rack = _rack()
    bass = rack.get("bass")
    assert sum(len(s) for s in bass.buffer) == 0
    # first fire arms recording (clears + captures)
    bass.start_record()
    assert bass.recording and not bass.looping
    evs = [MusicEvent("note_on", 1, 40, 90, dur=0.3, tag="bass")]
    bass.capture(3, evs)
    assert sum(len(s) for s in bass.buffer) == 1
    # while recording (not yet looping) replay is silent
    assert bass.replay(3) == []
    # second fire locks it -> now it replays that slot deterministically
    bass.start_record()
    assert bass.looping and not bass.recording
    rep = bass.replay(3)
    assert len(rep) == 1 and rep[0].a == 40 and rep[0] is not evs[0]   # a copy
    # clear empties it
    bass.clear_loop()
    assert not bass.looping and sum(len(s) for s in bass.buffer) == 0


def test_capture_only_while_recording():
    rack = _rack()
    s = rack.get("lead")
    s.capture(0, [MusicEvent("note_on", 3, 70, 90, dur=0.2, tag="lead")])
    assert sum(len(x) for x in s.buffer) == 0        # not armed -> ignored


def test_solo_aware_audibility():
    rack = _rack()
    drums, lead = rack.get("drums"), rack.get("lead")
    assert rack.audible(drums) and rack.audible(lead)
    drums.muted = True
    assert not rack.audible(drums)
    drums.muted = False
    lead.soloed = True
    assert rack.audible(lead) and not rack.audible(drums)   # solo silences others


def test_generate_in_scale_and_in_register():
    sub = _sub()
    rack = stemlib.StemRack(sub, STEPS)
    scale = set(SCALES[sub.cfg.scale])
    tonic = sub.cfg.tonic
    sig = {"energy": 0.9, "weight": 0.7, "flow": 0.5, "com_height": 0.6,
           "openness": 0.6}
    for name in ("bass", "keys", "lead", "texture"):
        stem = rack.get(name)
        stem.params = StemParams(name, density=1.0, pitch=0.7,
                                 register_octave=stem.params.register_octave)
        produced = False
        for step in range(STEPS):
            for ev in stemlib.generate(stem, sub, step, sub.cfg.steps_per_beat *
                                       sub.cfg.beats_per_bar, sig, 0.5, 1.0):
                produced = True
                assert (ev.a - tonic) % 12 in scale, f"{name} off-scale {ev.a}"
                assert stem.role.lo <= ev.a <= stem.role.hi
                assert ev.tag == name and ev.channel == stem.channel
        assert produced, f"{name} produced no notes at full density"


def test_drum_generate_uses_gm_pieces():
    sub = _sub()
    rack = stemlib.StemRack(sub, STEPS)
    drums = rack.get("drums")
    drums.params = StemParams("drums", density=1.0)
    pieces = set()
    for step in range(STEPS):
        for ev in stemlib.generate(drums, sub, step, sub.cfg.steps_per_beat *
                                   sub.cfg.beats_per_bar, {"energy": 1.0,
                                   "weight": 0.5}, 0.5, 1.0):
            assert ev.channel == 10 and ev.tag == "drums"
            pieces.add(ev.a)
    assert pieces and pieces <= set(stemlib.DRUM_PIECE.values())


def test_revoice_pad_in_scale():
    sub = _sub()
    rack = stemlib.StemRack(sub, STEPS)
    scale = set(SCALES[sub.cfg.scale])
    tonic = sub.cfg.tonic
    for name in ("keys", "texture"):
        stem = rack.get(name)
        evs = stemlib.revoice_pad(stem, sub, {"openness": 0.7, "weight": 0.5,
                                              "com_height": 0.5}, 0.5)
        assert evs
        for ev in evs:
            assert (ev.a - tonic) % 12 in scale
            assert stem.role.lo <= ev.a <= stem.role.hi
            assert ev.channel == stem.channel and ev.tag == name


def test_density_zero_generates_nothing():
    sub = _sub()
    rack = stemlib.StemRack(sub, STEPS)
    bass = rack.get("bass")
    bass.params = StemParams("bass", density=0.0)
    total = sum(len(stemlib.generate(bass, sub, step, 16,
                {"energy": 0.0, "weight": 0.0}, 0.5, 0.0)) for step in range(STEPS))
    assert total == 0
