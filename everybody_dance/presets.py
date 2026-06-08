"""Preset system for the studio — named, fully-customizable stem kits, plus
JSON (de)serialisation so visitors/curators can save and load their own sound.

A preset is just a list of StemConfig (rhythm + pitch + timbre + fx + mapping +
colour per stem). Everything is plain dataclasses from studio_types, so a preset
round-trips losslessly through JSON.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, List

from .studio_types import (FxConfig, Mapping, PitchConfig, RhythmConfig, StemConfig,
                           TimbreConfig, default_stem_configs)

# ---- (de)serialisation ---------------------------------------------------

_SUB = {"rhythm": RhythmConfig, "pitch": PitchConfig, "timbre": TimbreConfig,
        "fx": FxConfig, "mapping": Mapping}


def stem_to_dict(s: StemConfig) -> dict:
    return dataclasses.asdict(s)


def stem_from_dict(d: dict) -> StemConfig:
    d = dict(d)
    for key, cls in _SUB.items():
        if key in d and isinstance(d[key], dict):
            fields = {f.name for f in dataclasses.fields(cls)}
            d[key] = cls(**{k: v for k, v in d[key].items() if k in fields})
    if isinstance(d.get("color"), list):
        d["color"] = tuple(d["color"])
    for sub in ("rhythm", "pitch"):
        cfg = d.get(sub)
        if cfg is not None and isinstance(getattr(cfg, "register", None), list):
            cfg.register = tuple(cfg.register)
    fields = {f.name for f in dataclasses.fields(StemConfig)}
    return StemConfig(**{k: v for k, v in d.items() if k in fields})


def save_preset(configs: List[StemConfig], path: str) -> None:
    json.dump([stem_to_dict(c) for c in configs], open(path, "w"), indent=2)


def load_preset(path: str) -> List[StemConfig]:
    return [stem_from_dict(d) for d in json.load(open(path))]


# ---- built-in kits -------------------------------------------------------

def _house() -> List[StemConfig]:
    return [
        StemConfig("drums", 10, color=(90, 130, 255),
                   rhythm=RhythmConfig("euclid", 16, 2, pulses=4, gate=0.4),
                   timbre=TimbreConfig("kick", decay=0.16, gain=0.95)),
        StemConfig("bass", 1, color=(90, 220, 130),
                   rhythm=RhythmConfig("body", 16, 2, gate=0.6),
                   timbre=TimbreConfig("bass", cutoff=0.5, gain=0.85),
                   pitch=PitchConfig("height", register=(36, 52)),
                   fx=FxConfig(drive=0.2)),
        StemConfig("keys", 2, color=(240, 180, 70),
                   rhythm=RhythmConfig("euclid", 16, 2, pulses=7, gate=0.9),
                   timbre=TimbreConfig("saw", attack=0.01, decay=0.2, gain=0.4),
                   pitch=PitchConfig("height", register=(55, 79)),
                   fx=FxConfig(delay_send=0.25, reverb_send=0.2)),
        StemConfig("lead", 3, color=(210, 110, 240),
                   rhythm=RhythmConfig("body", 16, 2, gate=0.5),
                   timbre=TimbreConfig("pluck", decay=0.25, gain=0.6),
                   pitch=PitchConfig("height", register=(60, 84)),
                   fx=FxConfig(delay_send=0.35, delay_time=0.375, delay_feedback=0.4)),
    ]


def _ambient() -> List[StemConfig]:
    return [
        StemConfig("pulse", 10, color=(120, 160, 255),
                   rhythm=RhythmConfig("euclid", 16, 4, pulses=2, gate=0.3),
                   timbre=TimbreConfig("hat", gain=0.5)),
        StemConfig("drone", 1, color=(120, 210, 160),
                   rhythm=RhythmConfig("euclid", 16, 4, pulses=1, gate=1.0),
                   timbre=TimbreConfig("pad", attack=0.4, release=1.2, gain=0.5),
                   pitch=PitchConfig("height", register=(36, 55)),
                   fx=FxConfig(reverb_send=0.5, reverb_size=0.8)),
        StemConfig("pad", 2, color=(245, 190, 90),
                   rhythm=RhythmConfig("euclid", 16, 4, pulses=3, gate=1.0),
                   timbre=TimbreConfig("pad", attack=0.3, release=1.0, gain=0.4),
                   pitch=PitchConfig("height", register=(55, 79)),
                   fx=FxConfig(reverb_send=0.6, reverb_size=0.85)),
        StemConfig("bells", 3, color=(210, 130, 245),
                   rhythm=RhythmConfig("body", 16, 4, gate=0.6),
                   timbre=TimbreConfig("bell", decay=0.8, gain=0.5),
                   pitch=PitchConfig("height", register=(67, 91)),
                   fx=FxConfig(delay_send=0.4, delay_feedback=0.5, reverb_send=0.4)),
    ]


def _synthwave() -> List[StemConfig]:
    return [
        StemConfig("drums", 10, color=(100, 120, 255),
                   rhythm=RhythmConfig("body", 16, 2, gate=0.4),
                   timbre=TimbreConfig("snare", gain=0.9)),
        StemConfig("bass", 1, color=(100, 230, 140),
                   rhythm=RhythmConfig("euclid", 16, 2, pulses=8, gate=0.7),
                   timbre=TimbreConfig("saw", cutoff=0.45, gain=0.8),
                   pitch=PitchConfig("height", register=(36, 52)),
                   fx=FxConfig(drive=0.3)),
        StemConfig("pad", 2, color=(245, 185, 80),
                   rhythm=RhythmConfig("euclid", 16, 2, pulses=2, gate=1.0),
                   timbre=TimbreConfig("pad", attack=0.15, release=0.6, gain=0.4),
                   pitch=PitchConfig("height", register=(55, 79)),
                   fx=FxConfig(reverb_send=0.4, reverb_size=0.7)),
        StemConfig("lead", 3, color=(210, 120, 245),
                   rhythm=RhythmConfig("body", 16, 2, gate=0.6),
                   timbre=TimbreConfig("square", decay=0.3, gain=0.55),
                   pitch=PitchConfig("height", register=(60, 84)),
                   fx=FxConfig(delay_send=0.4, delay_time=0.5, delay_feedback=0.45, drive=0.15)),
    ]


PRESETS: Dict[str, callable] = {
    "default": default_stem_configs,
    "house": _house,
    "ambient": _ambient,
    "synthwave": _synthwave,
}


def get_preset(name: str) -> List[StemConfig]:
    if name in PRESETS:
        return PRESETS[name]()
    return load_preset(name)        # treat as a path to a JSON preset
