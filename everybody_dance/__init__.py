"""everybodyDance -- music that dances to you.

Body tracking in, music out, where the music is audibly *caused* by the
movement and stays musical the whole time. Unique per person.

Pipeline:
    body -> pose -> features (by timescale) + Laban Effort
         -> entrained clock + event/CC streams
         -> coherence substrate -> MIDI/OSC -> your synths (fixed timbres)
"""

from .engine import Engine, EngineConfig, FrameTrace
from .oscillator import EntrainedClock, ClockState
from .output import make_backend, LogBackend, MusicEvent
from .pose import MediaPipePoseSource, SyntheticPoseSource
from .personalization import Profile

__all__ = [
    "Engine", "EngineConfig", "FrameTrace",
    "EntrainedClock", "ClockState",
    "make_backend", "LogBackend", "MusicEvent",
    "MediaPipePoseSource", "SyntheticPoseSource",
    "Profile",
]

__version__ = "0.1.0"
