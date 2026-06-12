"""Dance DNA -- a sound identity derived from *how this person moves*.

Personalisation already normalises ranges and biases the substrate; this layer
goes further: the calibration signature deterministically picks a whole sonic
world -- root note, mode, chord progression, synth kit, accent colour, and a
generated stage name -- so two visitors never stand in front of the same
instrument. The mapping is two-layered on purpose:

  * the COARSE choices are legible (energetic, sharp movers land in dark, low,
    driven worlds; soft, open movers in bright, high, airy ones), so the
    identity feels *caused* by the dance, not assigned;
  * the FINE choices (root, progression, kit variant, name) hash the decimals
    of the signature, so even two similar movers split into different keys,
    chords, presets and names.

Everything is deterministic (CRC of the rounded stats -- stable across runs and
platforms; no Python ``hash()``, which is salted) and every option is curated,
so any identity sounds intentional. Same calibration -> same identity, which is
what makes "this is MY sound" a real, repeatable claim.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .substrate import PROGRESSIONS, SCALES

# Pitch-class names for the identity card (sharps; concise for the screen).
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Coherent synth ensembles. Each kit is a curated combination that mixes well;
# the identity picks one whole kit, never a random per-stem grab bag.
KITS: Dict[str, Dict[str, str]] = {
    "neon": {"drums": "analog_kick", "bass": "reese_bass", "keys": "warm_keys",
             "lead": "supersaw_lead", "texture": "glass_pad"},
    "velvet": {"drums": "analog_kick", "bass": "velvet_bass",
               "keys": "velvet_keys", "lead": "pluck_lead", "texture": "tape_pad"},
    "glass": {"drums": "analog_kick", "bass": "sub_bass", "keys": "music_box",
              "lead": "pluck_lead", "texture": "glass_pad"},
    "circuit": {"drums": "analog_kick", "bass": "acid_bass", "keys": "warm_keys",
                "lead": "chip_lead", "texture": "glass_pad"},
    "midnight": {"drums": "analog_kick", "bass": "velvet_bass",
                 "keys": "warm_keys", "lead": "brass_stab", "texture": "tape_pad"},
}

# Kit -> BGR accent colour for the identity card / stage accents.
KIT_COLOR: Dict[str, Tuple[int, int, int]] = {
    "neon": (255, 120, 220), "velvet": (140, 170, 255), "glass": (255, 240, 170),
    "circuit": (120, 255, 170), "midnight": (255, 180, 120),
}

# Movement palette -> which kits suit it, which scales suit it, where the root
# sits. Gentle movers get higher/brighter worlds, energetic ones lower/darker
# (mirrors substrate.biased_config so the two layers never disagree in spirit).
_PALETTE_KITS = {"ambient": ["velvet", "glass"], "neutral": ["neon", "midnight"],
                 "rhythmic": ["circuit", "neon"]}
_PALETTE_SCALES = {"ambient": ["lydian", "major_pentatonic", "major"],
                   "neutral": ["dorian", "minor", "mixolydian"],
                   "rhythmic": ["phrygian", "minor_pentatonic", "dorian"]}
_PALETTE_ROOT_LO = {"ambient": 57, "neutral": 53, "rhythmic": 48}

# Stage-name vocabulary: the adjective reflects the movement palette (so the
# name reads as earned), the noun carries the fine-grained uniqueness.
_ADJ = {"ambient": ["LUNAR", "VELVET", "MISTY", "OPAL", "DRIFTING"],
        "neutral": ["GOLDEN", "NEON", "COSMIC", "JADE", "RETRO"],
        "rhythmic": ["ELECTRIC", "CRIMSON", "WILD", "TURBO", "SOLAR"]}
_NOUN = ["FOX", "COMET", "ORCHID", "TIGER", "MIRAGE", "PULSE",
         "CANYON", "BLOOM", "RAVEN", "ENGINE", "NOVA", "DELTA"]


@dataclass
class SoundIdentity:
    """One visitor's sonic world, fully determined by their calibration."""
    name: str = "FIRST LIGHT"
    palette: str = "neutral"
    tonic: int = 57
    root_name: str = "A"
    scale: str = "minor"
    progression: List[int] = field(default_factory=lambda: [0, 5, 3, 4])
    kit_name: str = "neon"
    kit: Dict[str, str] = field(default_factory=lambda: dict(KITS["neon"]))
    color: Tuple[int, int, int] = (255, 120, 220)

    @property
    def tagline(self) -> str:
        return f"{self.root_name} {self.scale} - {self.kit_name} kit"


def _dna(signature: dict) -> int:
    """A stable 32-bit digest of the signature's fine structure.

    Rounded to 4 decimals so float noise below perceptual relevance doesn't
    flip choices, then CRC'd (deterministic across runs/platforms)."""
    keys = ("energy_mean", "energy_std", "energy_max", "openness_mean",
            "jerk_mean", "weight_mean")
    s = ",".join(f"{float(signature.get(k, 0.0) or 0.0):.4f}" for k in keys)
    return zlib.crc32(s.encode())


def _palette(signature: dict) -> str:
    """Same thresholds as substrate.biased_config, so the coarse identity and
    the substrate bias always agree about what kind of mover this is."""
    weight = signature.get("weight_mean", signature.get("energy_mean", 0.5))
    jerk = signature.get("jerk_mean", 100.0)
    if weight < 0.4 and jerk < 150:
        return "ambient"
    if weight > 0.85 or jerk > 200:
        return "rhythmic"
    return "neutral"


def derive_identity(signature: dict | None) -> SoundIdentity:
    """Signature stats -> a complete, curated SoundIdentity. Deterministic."""
    sig = dict(signature or {})
    if not sig:
        return SoundIdentity()
    dna = _dna(sig)
    palette = _palette(sig)

    # Coarse, legible picks: openness brightens the mode within the palette.
    openness = float(sig.get("openness_mean", 1.0) or 1.0)
    open01 = float(np.clip((openness - 0.5) / 1.5, 0.0, 1.0))
    scales = _PALETTE_SCALES[palette]
    scale = scales[0 if open01 > 0.6 else (1 if open01 > 0.3 else 2)]

    # Fine, hash-split picks: root / progression / kit / name. Different digit
    # mixes per choice so similar movers still split somewhere audible.
    tonic = _PALETTE_ROOT_LO[palette] + (dna % 8)
    progression = list(PROGRESSIONS[scale][(dna // 8) % len(PROGRESSIONS[scale])])
    kits = _PALETTE_KITS[palette]
    kit_name = kits[(dna // 32) % len(kits)]
    name = (_ADJ[palette][(dna // 64) % len(_ADJ[palette])] + " "
            + _NOUN[(dna // 320) % len(_NOUN)])

    assert scale in SCALES
    return SoundIdentity(
        name=name, palette=palette, tonic=int(tonic),
        root_name=_PC_NAMES[tonic % 12], scale=scale, progression=progression,
        kit_name=kit_name, kit=dict(KITS[kit_name]), color=KIT_COLOR[kit_name])
