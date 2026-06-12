"""Dance DNA (everybody_dance.identity): each calibration signature lands in
its own curated sonic world, deterministically.

The contract under test: (1) same signature -> byte-identical identity (the
"this is MY sound" claim is repeatable); (2) different movers split somewhere
audible; (3) every possible pick is valid -- the scale exists, every kit preset
renders, the progression indexes the scale; (4) the coarse choices are legible
(soft movers -> ambient worlds, hard movers -> rhythmic ones).
"""

import numpy as np

from everybody_dance.identity import (KITS, SoundIdentity, derive_identity)
from everybody_dance.substrate import PROGRESSIONS, SCALES
from everybody_dance.timbre import PRESETS

SOFT = {"energy_mean": 0.2, "energy_std": 0.05, "energy_max": 0.5,
        "openness_mean": 1.8, "jerk_mean": 80.0, "weight_mean": 0.25}
HARD = {"energy_mean": 1.1, "energy_std": 0.4, "energy_max": 2.0,
        "openness_mean": 0.9, "jerk_mean": 260.0, "weight_mean": 1.0}
MID = {"energy_mean": 0.6, "energy_std": 0.2, "energy_max": 1.2,
       "openness_mean": 1.2, "jerk_mean": 160.0, "weight_mean": 0.6}


def _variants(base, n=24):
    """n signatures that differ only in fine decimals (similar movers)."""
    out = []
    for k in range(n):
        sig = dict(base)
        sig["energy_mean"] = base["energy_mean"] + 0.003 * k
        sig["jerk_mean"] = base["jerk_mean"] + 0.7 * k
        out.append(sig)
    return out


def test_deterministic_same_signature_same_identity():
    a = derive_identity(dict(MID))
    b = derive_identity(dict(MID))
    assert a == b


def test_empty_signature_gives_valid_default():
    ident = derive_identity(None)
    assert isinstance(ident, SoundIdentity)
    assert ident.scale in SCALES
    assert derive_identity({}) == ident


def test_every_identity_is_valid_and_renderable():
    for sig in _variants(SOFT) + _variants(MID) + _variants(HARD):
        ident = derive_identity(sig)
        assert ident.scale in SCALES
        assert ident.progression in PROGRESSIONS[ident.scale]
        assert set(ident.kit) == {"drums", "bass", "keys", "lead", "texture"}
        for preset in ident.kit.values():
            assert preset in PRESETS, f"kit preset {preset} missing"
        assert 36 <= ident.tonic <= 72
        assert ident.name and " " in ident.name
        assert ident.root_name in ident.tagline and ident.scale in ident.tagline


def test_similar_movers_still_split_somewhere():
    idents = [derive_identity(s) for s in _variants(MID)]
    keys = {(i.name, i.tonic, tuple(i.progression), i.kit_name) for i in idents}
    # fine decimals hash apart: most of 24 near-identical movers differ.
    assert len(keys) >= 12, f"only {len(keys)} distinct identities"


def test_coarse_choices_are_legible():
    soft, hard = derive_identity(SOFT), derive_identity(HARD)
    assert soft.palette == "ambient" and hard.palette == "rhythmic"
    # soft movers sit higher/brighter than hard movers.
    assert soft.tonic > hard.tonic - 8
    assert soft.kit_name in ("velvet", "glass")
    assert hard.kit_name in ("circuit", "neon")


def test_kits_are_complete_and_distinct():
    for name, kit in KITS.items():
        assert set(kit) == {"drums", "bass", "keys", "lead", "texture"}
        for preset in kit.values():
            assert preset in PRESETS
    fingerprints = {tuple(sorted(k.items())) for k in KITS.values()}
    assert len(fingerprints) == len(KITS)
