"""Output backends: MIDI / OSC into your synths (fixed timbres), plus a
console/log backend so the instrument runs and can be evaluated headless.

The rest of the system speaks in :class:`MusicEvent` objects; a backend just
renders them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MusicEvent:
    kind: str                 # "note_on" | "note_off" | "cc"
    channel: int              # 1..16
    a: int                    # note / cc number
    b: int                    # velocity / cc value
    t: float = 0.0
    dur: Optional[float] = None   # for note_on with scheduled note_off
    tag: str = ""             # provenance, e.g. "drums.kick", "lead", "fx.delay"


class Backend:
    def send(self, ev: MusicEvent) -> None:
        raise NotImplementedError

    def panic(self) -> None:
        for ch in range(1, 17):
            self.send(MusicEvent("cc", ch, 123, 0))  # all notes off

    def close(self) -> None:  # pragma: no cover
        pass


class NullBackend(Backend):
    def send(self, ev: MusicEvent) -> None:
        pass


class LogBackend(Backend):
    """Records every event. Used by tests and the headless demo."""

    def __init__(self):
        self.events: List[MusicEvent] = []

    def send(self, ev: MusicEvent) -> None:
        self.events.append(ev)

    def notes(self, tag_prefix: str = "") -> List[MusicEvent]:
        return [e for e in self.events
                if e.kind == "note_on" and e.tag.startswith(tag_prefix)]


class RtMidiBackend(Backend):  # pragma: no cover - needs hardware
    """Real MIDI out via python-rtmidi (lazy import)."""

    def __init__(self, port_name: str = "everybodyDance", virtual: bool = True,
                 port_index: Optional[int] = None):
        import rtmidi
        self._midi = rtmidi.MidiOut()
        ports = self._midi.get_ports()
        if port_index is not None:
            self._midi.open_port(port_index)
        elif virtual:
            self._midi.open_virtual_port(port_name)
        elif ports:
            self._midi.open_port(0)
        else:
            self._midi.open_virtual_port(port_name)

    def send(self, ev: MusicEvent) -> None:
        ch = (ev.channel - 1) & 0x0F
        if ev.kind == "note_on":
            self._midi.send_message([0x90 | ch, ev.a & 0x7F, ev.b & 0x7F])
        elif ev.kind == "note_off":
            self._midi.send_message([0x80 | ch, ev.a & 0x7F, 0])
        elif ev.kind == "cc":
            self._midi.send_message([0xB0 | ch, ev.a & 0x7F, ev.b & 0x7F])

    def close(self) -> None:
        self.panic()
        self._midi.close_port()


class OscBackend(Backend):  # pragma: no cover - needs a receiver
    """OSC out via python-osc (lazy import). Sends /note and /cc messages."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        from pythonosc.udp_client import SimpleUDPClient
        self._client = SimpleUDPClient(host, port)

    def send(self, ev: MusicEvent) -> None:
        if ev.kind in ("note_on", "note_off"):
            vel = ev.b if ev.kind == "note_on" else 0
            self._client.send_message("/note", [ev.channel, ev.a, vel])
        elif ev.kind == "cc":
            self._client.send_message("/cc", [ev.channel, ev.a, ev.b])


def make_backend(name: str, **kw) -> Backend:
    name = (name or "log").lower()
    if name in ("log", "console"):
        return LogBackend()
    if name == "null":
        return NullBackend()
    if name == "midi":
        return RtMidiBackend(**kw)
    if name == "osc":
        return OscBackend(**kw)
    raise ValueError(f"unknown backend: {name}")
