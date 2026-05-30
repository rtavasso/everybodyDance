"""The entrained clock -- the thing that makes the music dance to *you*.

An adaptive-frequency oscillator (Righetti, Buchli, Ijspeert 2006) is fed the
movement-energy envelope. Via dynamic Hebbian learning it discovers the body's
tempo and locks phase to it. The implementation is the adaptive Hopf oscillator:

    r  = sqrt(x^2 + y^2)
    F  = teach(t) - x                      # perturbation = input minus output
    x' = (mu - r^2) x - omega y + eps F
    y' = (mu - r^2) y + omega x
    omega' = -eta F (y / r)                # Hebbian frequency adaptation

The limit cycle gives a clean phase (atan2(y, x)); omega converges to a
frequency component present in the teaching signal.

The soul is *mutual entrainment*, exposed as one dial -- `coupling` in [0,1]:
  coupling -> 0 : phase is pulled hard onto the body. Tight, can feel servile.
  coupling -> 1 : oscillator free-runs on its learned tempo, its own groove you
                  push against -- "dancing with a partner who has their own body".

Robustness: tempo confidence is tracked with hysteresis. When the body has no
clear beat, we fall back to a slow rubato clock instead of chasing noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class ClockState:
    phase: float          # beat phase in [0, 2pi)
    tempo_hz: float       # current beat frequency
    bpm: float
    confidence: float     # 0..1, how locked we are
    locked: bool          # False => rubato/ambient fallback engaged
    amplitude: float      # oscillator limit-cycle radius
    beat: bool            # True on the frame the phase crossed a beat boundary
    step: int             # current step within the bar (for euclidean grids)
    step_advanced: bool   # True on the frame a new grid step began


class EntrainedClock:
    def __init__(self,
                 tempo_mode: str = "entrain",      # "entrain" | "fixed"
                 init_hz: float = 2.0,
                 min_hz: float = 0.6,
                 max_hz: float = 4.0,
                 coupling: float = 0.4,
                 mu: float = 1.0,
                 eps: float = 2.5,                 # input coupling into limit cycle
                 eta: float = 6.0,                 # frequency learning rate
                 steps_per_beat: int = 4,
                 fixed_hz: float = 2.0,
                 grace_s: float = 2.5,             # spin-up before rubato is allowed
                 drive_ref: float = 0.02,          # raw modulation that counts as a beat
                 musical_lo_hz: float = 1.0,       # fold the felt tempo into this band
                 musical_hi_hz: float = 2.3):
        self.tempo_mode = tempo_mode
        self.min_w = TWO_PI * min_hz
        self.max_w = TWO_PI * max_hz
        self.mus_lo = TWO_PI * musical_lo_hz
        self.mus_hi = TWO_PI * musical_hi_hz
        self._mult = 1                            # tempo-octave multiplier (>=1)
        self.coupling = float(np.clip(coupling, 0.0, 1.0))
        self.mu = mu
        self.eps = eps
        self.eta = eta
        self.steps_per_beat = steps_per_beat
        self.fixed_w = TWO_PI * fixed_hz
        self.grace_s = grace_s
        self.drive_ref = drive_ref

        self.x = np.sqrt(mu)
        self.y = 0.0
        self.w = TWO_PI * init_hz
        self.phase = 0.0
        self._step = 0
        self._last_beat_phase = 0.0

        # confidence / hysteresis state
        self._conf = 0.0
        self._locked = True
        self._env_mean = 0.0
        self._rms = 0.0           # running RMS of the (DC-removed) drive
        self._below_t = 0.0       # dwell timers for hysteresis
        self._above_t = 0.0
        self._t = 0.0
        self._w_slow = self.w     # slow-following w, for frequency stability

    def set_coupling(self, c: float) -> None:
        self.coupling = float(np.clip(c, 0.0, 1.0))

    def _update_confidence(self, dt: float):
        # drive: is there real periodic modulation to lock onto?
        drive = float(np.clip(np.sqrt(self._rms) / self.drive_ref, 0, 1))
        # stability: has the learned frequency settled?
        af = 1.0 - np.exp(-dt / 1.0)
        self._w_slow = (1 - af) * self._w_slow + af * self.w
        rel = abs(self.w - self._w_slow) / max(self.w, 1e-6)
        stability = float(np.clip(1.0 - 8.0 * rel, 0.3, 1.0))
        target = drive * stability
        ac = 1.0 - np.exp(-dt / 0.5)
        self._conf = (1 - ac) * self._conf + ac * target
        # Hysteresis with dwell: enter rubato only after sustained low confidence,
        # re-lock only after sustained recovery. Never drop during the grace window.
        if self._conf < 0.25:
            self._below_t += dt
            self._above_t = 0.0
        elif self._conf > 0.45:
            self._above_t += dt
            self._below_t = 0.0
        else:
            self._below_t = self._above_t = 0.0
        if self._locked and self._t > self.grace_s and self._below_t > 0.5:
            self._locked = False
        elif not self._locked and self._above_t > 0.3:
            self._locked = True

    def update(self, drive_signal: float, dt: float) -> ClockState:
        """`drive_signal` is the body's beat carrier -- the vertical COM bounce
        (a clean fundamental), not raw kinetic energy (which peaks twice/cycle)."""
        dt = max(dt, 1e-3)
        self._t += dt
        if not np.isfinite(drive_signal):     # guard against bad pose frames
            drive_signal = self._env_mean
        # Remove slow DC so we entrain to *modulation*, and track its RMS.
        am = 1.0 - np.exp(-dt / 1.5)
        self._env_mean = (1 - am) * self._env_mean + am * drive_signal
        teach = drive_signal - self._env_mean
        ar = 1.0 - np.exp(-dt / 0.7)
        self._rms = (1 - ar) * self._rms + ar * teach * teach
        # Normalise the teaching signal to ~unit amplitude so entrainment works
        # regardless of body scale (and pairs with per-person calibration).
        teach_n = float(np.clip(teach / (np.sqrt(self._rms) + 1e-6), -3.0, 3.0))

        self._update_confidence(dt)

        # Sub-step the integration: explicit Euler on the Hopf oscillator is
        # unstable when dt is large (e.g. 12 fps pose data) or the state is big.
        # Small internal steps keep it bounded; we also clamp the state below.
        n = max(1, int(np.ceil(dt / 0.034)))
        sdt = dt / n
        for _ in range(n):
            if self.tempo_mode == "fixed":
                self.w = self.fixed_w
                self._integrate_fixed(sdt)
                locked = True
            elif not self._locked:
                self._integrate_rubato(sdt)
                locked = False
            else:
                self._integrate_adaptive(teach_n, sdt)
                locked = True
            self._sanitize()

        return self._emit(locked, dt)

    def _sanitize(self):
        """Keep the oscillator state finite and bounded (robust to noisy input)."""
        R = 3.0 * np.sqrt(self.mu)
        if not (np.isfinite(self.x) and np.isfinite(self.y)):
            self.x, self.y = np.sqrt(self.mu), 0.0
        self.x = float(np.clip(self.x, -R, R))
        self.y = float(np.clip(self.y, -R, R))
        if not np.isfinite(self.phase):
            self.phase = 0.0

    # --- integrators -------------------------------------------------------

    def _integrate_adaptive(self, teach: float, dt: float):
        x, y, w = self.x, self.y, self.w
        r2 = x * x + y * y
        r = np.sqrt(r2) + 1e-9
        if r < 0.1 * np.sqrt(self.mu):
            # Re-seed the limit cycle (e.g. coming back from rubato).
            x = np.sqrt(self.mu)
            y = 0.0
            r2 = self.mu
            r = np.sqrt(self.mu)
        F = teach - x
        dx = (self.mu - r2) * x - w * y + self.eps * F
        dy = (self.mu - r2) * y + w * x
        dw = -self.eta * F * (y / r)
        self.x = x + dx * dt
        self.y = y + dy * dt
        self.w = float(np.clip(w + dw * dt, self.min_w, self.max_w))
        # Phase: blend the oscillator's own phase advance with a pull toward the
        # internal limit-cycle phase. `coupling` sets how autonomous we are.
        osc_phase = np.arctan2(self.y, self.x) % TWO_PI
        free = (self.phase + self.w * dt) % TWO_PI
        pull = _ang_lerp(free, osc_phase, 1.0 - self.coupling)
        self.phase = pull % TWO_PI

    def _integrate_fixed(self, dt: float):
        self.phase = (self.phase + self.w * dt) % TWO_PI
        # keep a token limit cycle alive for amplitude reporting
        self.x, self.y = np.cos(self.phase), np.sin(self.phase)

    def _integrate_rubato(self, dt: float):
        # Slow, freely drifting clock for ambient/non-periodic passages.
        rubato_w = max(self.min_w * 0.5, self.w * 0.5)
        self.phase = (self.phase + rubato_w * dt) % TWO_PI
        # let the limit cycle decay toward rest so confidence stays honest
        self.x *= np.exp(-dt / 1.0)
        self.y *= np.exp(-dt / 1.0)

    # --- emit --------------------------------------------------------------

    def _emit(self, locked: bool, dt: float) -> ClockState:
        w = self.w if self.tempo_mode != "fixed" else self.fixed_w
        # Tempo-octave folding: human periodicity is often a slow sway (~0.6 Hz).
        # Express the felt tempo at an integer multiple of the body's fundamental
        # so it lands in a musical band while staying phase-locked to the body.
        mult = self._mult
        while w * mult < self.mus_lo and mult < 4:
            mult *= 2
        while w * mult > self.mus_hi and mult > 1:
            mult //= 2
        self._mult = mult

        # Musical phase runs at mult beats per body cycle (integer => clean wraps).
        mphase = (self.phase * mult) % TWO_PI
        beat = mphase < self._last_beat_phase       # wrapped past 0 => new beat
        self._last_beat_phase = mphase
        step = int(mphase / TWO_PI * self.steps_per_beat)
        step_advanced = step != self._step
        self._step = step
        tempo_hz = w * mult / TWO_PI
        return ClockState(
            phase=mphase,
            tempo_hz=tempo_hz,
            bpm=tempo_hz * 60.0,
            confidence=self._conf,
            locked=locked,
            amplitude=float(np.hypot(self.x, self.y)),
            beat=bool(beat),
            step=step,
            step_advanced=bool(step_advanced),
        )

    def predict_phase(self, lookahead_s: float) -> float:
        """Phase we expect `lookahead_s` from now -- fire percussion early to
        eat output latency."""
        return (self.phase + self.w * lookahead_s) % TWO_PI


def _ang_lerp(a: float, b: float, t: float) -> float:
    """Interpolate angle a -> b by t along the shortest arc."""
    d = (b - a + np.pi) % TWO_PI - np.pi
    return a + t * d
