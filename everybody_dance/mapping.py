"""Mapping -- the customizability layer between body and music.

Everything else in the instrument fixes *how* movement becomes sound. This is
where a user (or a preset) declares *what maps to what*: which signal drives a
stem's density, which gesture drops the beat, which pose darkens the scale. It
is a declarative, JSON-serializable config plus a pure resolver -- no engine,
no state, no body objects. The studio passes in a plain signal dict and gets
back per-stem target parameters and discrete actions, so this layer stays
decoupled and trivially testable.

Signal sources are strings resolved against that dict:
  * a plain key        -> its value (unknown -> 0.0)
  * ``"one_minus:k"``  -> ``1 - sig[k]`` (e.g. cutoff from ``one_minus:flow``)
  * ``"const"``        -> 1.0 (scaled by the field's gain/base)

Available continuous keys (floats): energy, core_energy, limb_energy, openness,
com_height, weight, time, space, flow (0..1); asymmetry_lr (-1..1); and event
impulses thrust, stomp, reversal, freeze (0..1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

# Continuous signal keys the studio promises to provide (for reference/validation).
SIGNAL_KEYS = (
    "energy", "core_energy", "limb_energy", "openness", "com_height", "weight",
    "time", "space", "flow", "asymmetry_lr",
    "thrust", "stomp", "reversal", "freeze",
)

# Discrete-action vocabulary. The studio implements the behaviour; we model and
# validate the names so a config can't bind a typo'd action.
ACTIONS = frozenset({
    "fx",            # one-shot musical effect on target stem
    "fill",          # drum/percussion fill
    "drop",          # the drop -- cut + reintroduce
    "breakdown",     # strip back to sparse texture
    "build",         # riser / tension build
    "loop_record",   # start/stop recording target stem's looper
    "loop_toggle",   # mute/unmute target loop
    "loop_clear",    # clear target loop
    "scale_shift",   # change mood/scale to target
    "timbre_morph",  # swap target stem timbre to params['to']
    "stem_mute",     # mute target stem
    "stem_solo",     # solo target stem
})


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


# -- stem mapping ----------------------------------------------------------

@dataclass
class StemMapping:
    """Declares how one stem reacts to the live signals.

    Each ``*_src`` is a source string (plain key / ``one_minus:k`` / ``const``)
    resolved per frame; gains/bases scale the resolved value before clamping.
    """
    stem: str
    density_src: str = "energy"
    density_gain: float = 1.0
    pitch_src: str = "com_height"
    register_octave: int = 0
    cutoff_src: str = "energy"
    reverb_src: str = "flow"
    delay_src: str = "space"
    drive_src: str = "weight"
    gain_src: str = "const"
    gain_base: float = 1.0
    pan_src: str = "asymmetry_lr"
    timbre: str = ""
    mute: bool = False


@dataclass
class StemParams:
    """Resolved per-frame targets for one stem, every field clamped to range.

    Ranges: density 0..1, pitch 0..1, register_octave int, cutoff 0..1,
    resonance 0..1, reverb 0..1, delay 0..1, drive 0..1, gain 0..1, pan -1..1.
    """
    stem: str
    density: float = 0.0
    pitch: float = 0.0
    register_octave: int = 0
    cutoff: float = 0.0
    resonance: float = 0.3
    reverb: float = 0.0
    delay: float = 0.0
    drive: float = 0.0
    gain: float = 1.0
    pan: float = 0.0
    timbre: str = ""
    mute: bool = False


# -- gesture binding -------------------------------------------------------

@dataclass
class GestureBinding:
    """Binds a recognised move to a discrete action on a target."""
    move: str
    action: str
    target: str = ""
    params: dict = field(default_factory=dict)

    def valid(self) -> bool:
        return self.action in ACTIONS


# -- config ----------------------------------------------------------------

@dataclass
class MappingConfig:
    """The whole customizability layer: per-stem reactions + gesture bindings.

    ``moods`` maps a mood name to a scale name from ``substrate.SCALES`` so a
    ``scale_shift`` action's target can be resolved to an actual scale.
    """
    stems: List[StemMapping] = field(default_factory=list)
    gestures: List[GestureBinding] = field(default_factory=list)
    moods: Dict[str, str] = field(default_factory=dict)

    # -- serialization (mirrors personalization.Profile) -------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MappingConfig":
        return MappingConfig(
            stems=[StemMapping(**s) for s in d.get("stems", [])],
            gestures=[GestureBinding(**g) for g in d.get("gestures", [])],
            moods=dict(d.get("moods", {})),
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_json(path: str) -> "MappingConfig":
        with open(path) as fh:
            return MappingConfig.from_dict(json.load(fh))

    # -- the shipped preset ------------------------------------------------

    @staticmethod
    def default() -> "MappingConfig":
        """A coherent, musically-sensible 5-stem Just-Dance-style mapping.

        Distinct sources per stem (drums off the core, lead off the limbs,
        texture off openness/flow...) and ~10 expressive gesture bindings.
        """
        stems = [
            # Drums: the beat comes from the CORE; weight drives drive/grit.
            StemMapping("drums", density_src="core_energy", density_gain=1.0,
                        pitch_src="const", cutoff_src="energy",
                        reverb_src="const", delay_src="const",
                        drive_src="weight", pan_src="const",
                        timbre="analog_kick"),
            # Bass: also core-locked, sits low; height nudges its register up.
            StemMapping("bass", density_src="core_energy", density_gain=0.8,
                        pitch_src="com_height", register_octave=-1,
                        cutoff_src="weight", reverb_src="const",
                        delay_src="const", drive_src="weight",
                        pan_src="const", timbre="sub_bass"),
            # Keys/pad: openness opens the voicing, flow washes the reverb.
            StemMapping("keys", density_src="time", pitch_src="openness",
                        cutoff_src="openness", reverb_src="flow",
                        delay_src="space", drive_src="const",
                        pan_src="asymmetry_lr", timbre="warm_keys"),
            # Lead: the melody comes from the LIMBS; height picks the register.
            StemMapping("lead", density_src="limb_energy", pitch_src="com_height",
                        register_octave=1, cutoff_src="energy",
                        reverb_src="flow", delay_src="space",
                        drive_src="thrust", pan_src="asymmetry_lr",
                        timbre="supersaw_lead"),
            # Texture: airy, indirect; cutoff from openness, brightness opens as
            # flow frees up. Sits wide and quiet under everything.
            StemMapping("texture", density_src="openness", pitch_src="openness",
                        cutoff_src="one_minus:flow", reverb_src="flow",
                        delay_src="space", drive_src="const",
                        gain_base=0.7, pan_src="asymmetry_lr",
                        timbre="glass_pad"),
        ]
        gestures = [
            GestureBinding("JUMP", "fill", target="drums"),
            GestureBinding("STOMP", "drop", target="drums"),
            GestureBinding("T-POSE", "breakdown"),
            GestureBinding("HANDS UP", "build"),
            GestureBinding("SQUAT", "loop_record", target="bass"),
            GestureBinding("CLAP", "fx", target="drums", params={"hit": "snare"}),
            GestureBinding("PUNCH", "fx", target="lead", params={"hit": "stab"}),
            GestureBinding("ARMS CROSSED", "scale_shift", target="dark"),
            GestureBinding("RAISE LEFT", "loop_toggle", target="keys"),
            GestureBinding("RAISE RIGHT", "timbre_morph", target="lead",
                           params={"to": "glass_pad"}),
        ]
        moods = {
            "bright": "lydian",
            "warm": "major",
            "neutral": "dorian",
            "dark": "phrygian",
            "moody": "minor",
        }
        return MappingConfig(stems=stems, gestures=gestures, moods=moods)


# -- resolver --------------------------------------------------------------

class MappingResolver:
    """Pure function of a signal dict: resolves a config to per-frame targets.

    Holds no state beyond the config; given the same config and signal it always
    returns the same params, so it's trivially testable and replayable.
    """

    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg

    def signal(self, sig: dict, src: str) -> float:
        """Resolve a source string to a float, defaulting safely to 0.0.

        ``const`` -> 1.0 (a multiplicative identity the gain/base scales);
        ``one_minus:k`` -> ``1 - sig[k]``; otherwise a plain key lookup.
        """
        if src == "const":
            return 1.0
        if src.startswith("one_minus:"):
            return 1.0 - float(sig.get(src[len("one_minus:"):], 0.0))
        return float(sig.get(src, 0.0))

    def _params(self, sig: dict, m: StemMapping) -> StemParams:
        return StemParams(
            stem=m.stem,
            density=_clip(self.signal(sig, m.density_src) * m.density_gain, 0.0, 1.0),
            pitch=_clip(self.signal(sig, m.pitch_src), 0.0, 1.0),
            register_octave=int(m.register_octave),
            cutoff=_clip(self.signal(sig, m.cutoff_src), 0.0, 1.0),
            resonance=0.3,
            reverb=_clip(self.signal(sig, m.reverb_src), 0.0, 1.0),
            delay=_clip(self.signal(sig, m.delay_src), 0.0, 1.0),
            drive=_clip(self.signal(sig, m.drive_src), 0.0, 1.0),
            gain=_clip(self.signal(sig, m.gain_src) * m.gain_base, 0.0, 1.0),
            pan=_clip(self.signal(sig, m.pan_src), -1.0, 1.0),
            timbre=m.timbre,
            mute=m.mute,
        )

    def resolve_continuous(self, sig: dict) -> Dict[str, StemParams]:
        """One fully-clamped :class:`StemParams` per stem, keyed by stem name."""
        return {m.stem: self._params(sig, m) for m in self.cfg.stems}

    def resolve_gesture(self, move: str) -> List[GestureBinding]:
        """All bindings whose ``move`` matches (could be several), or ``[]``."""
        return [g for g in self.cfg.gestures if g.move == move]
