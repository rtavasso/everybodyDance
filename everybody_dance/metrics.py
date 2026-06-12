"""Objective metrics for the observability harness.

These quantify the four "kinds of good" as far as numbers honestly can, on a
deterministic replay of a movement session:

  * coupling     -- does the music track the movement? (correlation matrix +
                    dead-zones / phantoms)
  * musicality   -- in-scale %, density, dynamics, range
  * recognition  -- gesture recall/precision/latency (when labels are available)
  * timing       -- per-frame compute headroom vs the frame budget

They feed the LLM judge as context and double as CI pass/fail signals. Anything
perceptual ("does it look alive / beautiful") is left to the judge + humans.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .substrate import SCALES


def _bin(times: Sequence[float], vals: Sequence[float], n_bins: int, dur: float):
    out = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for t, v in zip(times, vals):
        b = min(int(t / dur * n_bins), n_bins - 1)
        out[b] += v
        cnt[b] += 1
    return np.where(cnt > 0, out / np.maximum(cnt, 1), 0.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def coupling(move: Dict[str, np.ndarray], music: Dict[str, np.ndarray]) -> Dict:
    """Correlation matrix between movement and music feature series + summary."""
    matrix = {m: {k: round(_corr(move[k], music[m]), 3) for k in move} for m in music}
    # each musical dimension should be explained by *some* movement signal
    per_music = {m: max((abs(v) for v in row.values()), default=0.0)
                 for m, row in matrix.items()}
    score = round(float(np.mean(list(per_music.values()))) if per_music else 0.0, 3)

    energy = move.get("energy", np.zeros(1))
    density = music.get("density", np.zeros(1))
    hi = energy > np.percentile(energy, 66) if energy.size else np.array([False])
    lo = energy < np.percentile(energy, 10) if energy.size else np.array([False])
    dead = float(np.mean(density[hi] == 0)) if hi.any() else 0.0      # move, no music
    phantom = float(np.mean(density[lo] > 0)) if lo.any() else 0.0    # music, no move
    return {"matrix": matrix, "score": score, "dead_zone": round(dead, 3),
            "phantom": round(phantom, 3)}


def musicality(events, tonic: int, scale: str, minutes: float,
               segments: Optional[List[Tuple[float, int, str]]] = None) -> Dict:
    """``segments`` optionally gives the (t_start, tonic, scale) timeline when
    the key changed mid-session (scale_shift / ECLIPSE): each note is then
    judged against the scale that was active at its onset, which is the honest
    version of in-scale% for a session with deliberate key changes."""
    ons = [e for e in events if e.kind == "note_on"]
    pitched = [e for e in ons if e.channel != 10]
    if not ons:
        return {"notes": 0, "in_scale_pct": 0.0, "notes_per_min": 0.0,
                "distinct_pitches": 0, "dyn_range": 0, "mean_velocity": 0.0}

    def key_at(t: float) -> Tuple[int, set]:
        tn, deg = tonic, set(SCALES[scale])
        for ts, stn, ssc in (segments or []):
            if ts <= t + 1e-9:
                tn, deg = stn, set(SCALES[ssc])
            else:
                break
        return tn, deg

    in_scale = 0
    for e in pitched:
        tn, deg = key_at(e.t)
        in_scale += (e.a - tn) % 12 in deg
    vels = [e.b for e in ons]
    return {
        "notes": len(ons),
        "in_scale_pct": round(100 * in_scale / max(len(pitched), 1), 1),
        "notes_per_min": round(len(ons) / max(minutes, 1e-6), 1),
        "distinct_pitches": len({e.a for e in pitched}),
        "dyn_range": int(max(vels) - min(vels)),
        "mean_velocity": round(float(np.mean(vels)), 1),
    }


def recognition(fired: List[Tuple[float, str]],
                labels: Optional[List[Tuple[float, str]]],
                tol: float = 0.6) -> Dict:
    """Recall/precision/latency vs labelled (t, name) ground truth, if given."""
    if not labels:
        from collections import Counter
        return {"labelled": False, "fires": dict(Counter(n for _, n in fired)),
                "total": len(fired)}
    used = [False] * len(fired)
    tp, lat = 0, []
    for lt, ln in labels:
        for i, (ft, fn) in enumerate(fired):
            if not used[i] and fn == ln and abs(ft - lt) <= tol:
                used[i] = True
                tp += 1
                lat.append(ft - lt)
                break
    fp = used.count(False)
    recall = tp / max(len(labels), 1)
    precision = tp / max(len(fired), 1) if fired else 0.0
    return {"labelled": True, "recall": round(recall, 3), "precision": round(precision, 3),
            "tp": tp, "fp": fp, "n_labels": len(labels),
            "median_latency_ms": round(1000 * float(np.median(lat)), 1) if lat else None}


def timing(process_ms: Sequence[float], fps: float) -> Dict:
    if not len(process_ms):
        return {}
    p = np.asarray(process_ms)
    budget = 1000.0 / fps
    return {"frame_budget_ms": round(budget, 1),
            "compute_ms_p50": round(float(np.percentile(p, 50)), 2),
            "compute_ms_p95": round(float(np.percentile(p, 95)), 2),
            "headroom_pct": round(100 * (1 - np.percentile(p, 95) / budget), 1),
            "note": "offline compute only; add camera+audio buffers for live e2e"}


def liveliness(events, dur: float, window_s: float = 1.0) -> Dict:
    """Aliveness proxies (KICKOFF G2): does the musical density vary over time, and
    how long is the longest onset-free stretch? A live arrangement varies (std > 0)
    and never freezes into one long static block. (A *gated* stillness window is
    expected to be onset-free -- that lowers `phantom`, the better signal -- so
    `longest_silence_s` is informational, not a hard fail.)"""
    ons = sorted(e.t for e in events if e.kind == "note_on")
    if not ons or dur <= 0:
        return {"density_std": 0.0, "mean_density": 0.0,
                "longest_silence_s": round(float(dur), 2)}
    nb = max(int(dur / window_s), 4)
    counts = np.zeros(nb)
    for t in ons:
        counts[min(int(t / dur * nb), nb - 1)] += 1
    gaps = np.diff([0.0] + ons + [dur])
    return {"density_std": round(float(counts.std()), 3),
            "mean_density": round(float(counts.mean()), 3),
            "longest_silence_s": round(float(gaps.max()), 2)}


def gesture_spam(fired: List[Tuple[float, str]], dur: float) -> Dict:
    """Per-minute fire rate per move (KICKOFF G4 anti-spam: no single gesture should
    fire > ~20x/min on real dance)."""
    from collections import Counter
    minutes = max(dur / 60.0, 1e-6)
    per = {n: round(k / minutes, 2) for n, k in Counter(n for _, n in fired).items()}
    return {"max_per_min": round(max(per.values()), 2) if per else 0.0,
            "per_gesture_per_min": per, "total": len(fired)}


# default exhibit SLOs -> CI pass/fail. Operators: >=, <=, >, <.
# (check_slo skips keys a harness doesn't emit, so the game.* SLOs only gate
# the Studio bundle -- the legacy ambient-engine grade has no game layer.)
SLO = {
    "coupling.score": (">=", 0.35),
    "coupling.dead_zone": ("<=", 0.25),
    "coupling.phantom": ("<=", 0.25),          # G2: little/no sound during stillness
    "musicality.in_scale_pct": (">=", 99.0),
    "liveliness.density_std": (">", 0.0),       # G2: the arrangement varies over time
    "gesture.max_per_min": ("<=", 20.0),        # G4: no single move spams
    "timing.headroom_pct": (">=", 0.0),
    "game.sections_visited": (">=", 3.0),       # the song arc actually travels
    "game.stems_unlocked": (">=", 5.0),         # the full band gets earned
    "game.combos_fired": (">=", 1.0),           # combos are reachable + fire
}

_OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t,
        ">": lambda v, t: v > t, "<": lambda v, t: v < t}


def check_slo(flat: Dict[str, float]) -> Dict[str, bool]:
    out = {}
    for key, (op, thr) in SLO.items():
        v = flat.get(key)
        if v is None:
            continue
        out[key] = bool(_OPS[op](v, thr))
    return out
