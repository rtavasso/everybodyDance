"""Tests for the mapping (customizability) layer: the default preset round-trips
through JSON struct-identically, the resolver is a pure function of a plain
signal dict that always produces in-range params (even with missing keys), the
``const`` / ``one_minus:`` source conventions work, and every gesture binding in
the default config names a supported action.
"""

import pytest

from everybody_dance.mapping import (ACTIONS, MappingConfig, MappingResolver,
                                     StemMapping, StemParams)
from everybody_dance.substrate import SCALES

# Per-field documented ranges for StemParams (closed intervals).
RANGES = {
    "density": (0.0, 1.0), "pitch": (0.0, 1.0), "cutoff": (0.0, 1.0),
    "resonance": (0.0, 1.0), "reverb": (0.0, 1.0), "delay": (0.0, 1.0),
    "drive": (0.0, 1.0), "gain": (0.0, 1.0), "pan": (-1.0, 1.0),
}

# A spread of signal dicts, including empty / missing keys / out-of-range.
SIGNALS = [
    {},                                                  # nothing -> defaults
    {"energy": 1.0, "core_energy": 1.0, "limb_energy": 1.0,
     "openness": 1.0, "com_height": 1.0, "weight": 1.0, "time": 1.0,
     "space": 1.0, "flow": 1.0, "asymmetry_lr": 1.0},    # all hot
    {"asymmetry_lr": -1.0, "flow": 0.0},                 # negative pan, dark
    {"energy": 5.0, "asymmetry_lr": -9.0, "weight": -3.0},  # out of range
    {"core_energy": 0.5, "openness": 0.3, "thrust": 0.7},  # partial
]


def _assert_in_range(p: StemParams):
    for field_name, (lo, hi) in RANGES.items():
        v = getattr(p, field_name)
        assert lo <= v <= hi, f"{p.stem}.{field_name}={v} out of [{lo},{hi}]"
    assert isinstance(p.register_octave, int)


def test_default_json_roundtrip(tmp_path):
    cfg = MappingConfig.default()
    path = str(tmp_path / "mapping.json")
    cfg.to_json(path)
    back = MappingConfig.from_json(path)
    # struct-identical: dataclass dicts match exactly
    assert back.to_dict() == cfg.to_dict()
    # ...and re-serializing yields byte-identical JSON
    path2 = str(tmp_path / "mapping2.json")
    back.to_json(path2)
    assert open(path).read() == open(path2).read()


def test_resolve_continuous_one_param_per_stem_all_in_range():
    cfg = MappingConfig.default()
    res = MappingResolver(cfg)
    stem_names = [m.stem for m in cfg.stems]
    for sig in SIGNALS:
        out = res.resolve_continuous(sig)
        assert list(out.keys()) == stem_names      # one StemParams per stem
        for name, p in out.items():
            assert isinstance(p, StemParams) and p.stem == name
            _assert_in_range(p)


def test_missing_keys_default_to_zero_no_crash():
    res = MappingResolver(MappingConfig.default())
    out = res.resolve_continuous({})                # every src unknown
    for p in out.values():
        _assert_in_range(p)                         # no crash, all in range
        # plain-key density defaults to 0 when its signal is absent
        assert p.density == 0.0
    # drums.cutoff is a plain key ("energy") -> 0; texture.cutoff is
    # one_minus:flow -> 1.0 (1 - 0). Both correct, both in range.
    assert out["drums"].cutoff == 0.0
    assert out["texture"].cutoff == 1.0


def test_signal_const_and_one_minus():
    res = MappingResolver(MappingConfig.default())
    assert res.signal({}, "const") == 1.0
    assert res.signal({"flow": 0.25}, "one_minus:flow") == 0.75
    assert res.signal({}, "one_minus:flow") == 1.0     # missing -> 1-0
    assert res.signal({"energy": 0.4}, "energy") == 0.4
    assert res.signal({}, "bogus_key") == 0.0          # unknown -> 0.0


def test_one_minus_used_by_texture_cutoff():
    res = MappingResolver(MappingConfig.default())
    bright = res.resolve_continuous({"flow": 0.0})["texture"].cutoff
    dark = res.resolve_continuous({"flow": 1.0})["texture"].cutoff
    assert bright == 1.0 and dark == 0.0               # cutoff = one_minus:flow


def test_resolve_gesture_matches_and_empty():
    res = MappingResolver(MappingConfig.default())
    bound = res.resolve_gesture("T-POSE")
    assert bound and all(g.move == "T-POSE" for g in bound)
    assert res.resolve_gesture("NOT A MOVE") == []


def test_resolve_gesture_can_return_several():
    cfg = MappingConfig.default()
    cfg.gestures.append(
        type(cfg.gestures[0])("CLAP", "stem_solo", target="bass"))
    res = MappingResolver(cfg)
    assert len(res.resolve_gesture("CLAP")) == 2       # two bindings same move


def test_default_actions_all_supported():
    for g in MappingConfig.default().gestures:
        assert g.action in ACTIONS, g.action
        assert g.valid()


def test_default_scale_shift_targets_a_known_mood():
    cfg = MappingConfig.default()
    for g in cfg.gestures:
        if g.action == "scale_shift":
            assert g.target in cfg.moods
            assert cfg.moods[g.target] in SCALES        # mood -> real scale


def test_default_sources_all_resolve():
    cfg = MappingConfig.default()
    res = MappingResolver(cfg)
    sig = {k: 0.5 for k in (
        "energy", "core_energy", "limb_energy", "openness", "com_height",
        "weight", "time", "space", "flow", "asymmetry_lr",
        "thrust", "stomp", "reversal", "freeze")}
    for m in cfg.stems:
        for src in (m.density_src, m.pitch_src, m.cutoff_src, m.reverb_src,
                    m.delay_src, m.drive_src, m.gain_src, m.pan_src):
            res.signal(sig, src)                        # must not raise


def test_default_has_five_distinct_stems_with_timbres():
    cfg = MappingConfig.default()
    assert [m.stem for m in cfg.stems] == ["drums", "bass", "keys", "lead", "texture"]
    timbres = [m.timbre for m in cfg.stems]
    assert all(timbres) and len(set(timbres)) == 5      # each a distinct preset
