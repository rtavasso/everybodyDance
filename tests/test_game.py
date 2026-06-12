"""The game layer (everybody_dance.game) -- combos, streak, song arc, gold
moves -- as pure units AND integrated through the Studio.

Everything here is deterministic: combos fire on exact move sequences inside
beat windows, the arc unlocks stems in a fixed order as heat is earned, the
gold schedule is a pure function of the bar counter. The integration tests
drive the Studio with force-fired commands (the same path as the live keys),
mirroring test_studio.
"""

import sys

import numpy as np

from everybody_dance.game import (COMBOS, GOLD_ROTATION, SECTIONS,
                                  ComboTracker, GoldMoveGame, SongArc,
                                  StreakMeter)
from everybody_dance.studio import Studio, StudioConfig
from everybody_dance.output import LogBackend
from everybody_dance.sources import ArrayPoseSource

from test_studio import FPS, _groove, _profile, _run

BEAT = 0.5   # 120 BPM


# -- combos ------------------------------------------------------------------

def test_combo_fires_on_sequence_within_window():
    ct = ComboTracker()
    assert ct.observe("SQUAT", 1.0, BEAT) is None
    hit = ct.observe("JUMP", 1.8, BEAT)
    assert hit is not None and hit.name == "SUPERNOVA"


def test_combo_misses_when_too_slow():
    ct = ComboTracker()
    ct.observe("SQUAT", 1.0, BEAT)
    assert ct.observe("JUMP", 1.0 + 5 * BEAT, BEAT) is None   # > 4-beat window


def test_longest_combo_wins_and_history_clears():
    ct = ComboTracker()
    ct.observe("PUNCH", 1.0, BEAT)
    ct.observe("PUNCH", 1.4, BEAT)
    hit = ct.observe("PUNCH", 1.8, BEAT)
    assert hit is not None and hit.name == "KNOCKOUT"
    # history cleared: the next punch starts a fresh count.
    assert ct.observe("PUNCH", 2.2, BEAT) is None


def test_combo_refires_after_clear():
    ct = ComboTracker()
    for k in range(3):
        first = ct.observe("CLAP", 1.0 + 0.3 * k, BEAT)
    assert first.name == "CLAP STORM"
    for k in range(3):
        second = ct.observe("CLAP", 3.0 + 0.3 * k, BEAT)
    assert second.name == "CLAP STORM"


def test_interleaved_moves_break_a_sequence():
    ct = ComboTracker()
    ct.observe("SQUAT", 1.0, BEAT)
    ct.observe("CLAP", 1.3, BEAT)
    assert ct.observe("JUMP", 1.6, BEAT) is None


# -- streak --------------------------------------------------------------------

def test_streak_rises_with_moves_and_decays_in_stillness():
    s = StreakMeter()
    assert s.tier == 0
    for _ in range(5):
        s.observe_move()
    assert s.heat > 0.8 and s.tier == 3 and s.tier_name == "GOLD"
    for _ in range(600):                      # 20 s of stillness at 30 fps
        s.update(1 / 30.0, energy=0.0)
    assert s.heat == 0.0 and s.tier == 0


def test_streak_energy_sustains_more_than_silence():
    a, b = StreakMeter(), StreakMeter()
    a.heat = b.heat = 0.6
    for _ in range(90):
        a.update(1 / 30.0, energy=1.0)
        b.update(1 / 30.0, energy=0.0)
    assert a.heat > b.heat


# -- song arc -------------------------------------------------------------------

def _heat_arc(arc, level, bars=1, dt=1 / 30.0, frames_per_bar=60):
    for _ in range(bars):
        for _ in range(frames_per_bar):
            arc.update_frame(dt, energy=level, streak_level=level)
        arc.on_bar()


def test_arc_unlocks_in_order_and_sticks():
    arc = SongArc()
    assert arc.section == "intro"
    assert arc.is_unlocked("drums") and arc.is_unlocked("bass")
    assert not arc.is_unlocked("keys")
    order = []
    for _ in range(8):
        _heat_arc(arc, 1.0)
        for s in ("keys", "lead", "texture"):
            if arc.is_unlocked(s) and s not in order:
                order.append(s)
    assert order == ["keys", "lead", "texture"]
    assert arc.section == "peak"
    assert arc.sections_visited == SECTIONS
    # cooling off lowers the section but never re-locks a stem.
    _heat_arc(arc, 0.0, bars=6)
    assert arc.section != "peak"
    assert arc.is_unlocked("texture")


def test_arc_density_mul_defaults_to_one():
    arc = SongArc()
    arc.section = "groove"
    assert arc.density_mul("lead") == 1.0


# -- gold moves -------------------------------------------------------------------

def test_gold_schedule_announce_window_idle():
    g = GoldMoveGame(start_bar=4, period_bars=8)
    for bar in range(4):
        g.on_bar(bar)
        assert g.state == "idle"
    g.on_bar(4)
    assert g.state == "announce" and g.move == GOLD_ROTATION[0]
    g.on_bar(5)
    assert g.state == "window"
    g.on_bar(6)                                # window passed: silent miss
    assert g.state == "idle" and g.move is None and g.hits == 0
    g.on_bar(12)                               # next prompt, next move
    assert g.state == "announce" and g.move == GOLD_ROTATION[1]


def test_gold_hit_only_counts_in_window():
    g = GoldMoveGame(start_bar=4, period_bars=8)
    g.on_bar(4)
    assert not g.observe(GOLD_ROTATION[0], 8.0)     # announce: not yet
    g.on_bar(5)
    assert not g.observe("CLAP", 10.0)              # wrong move
    assert g.observe(GOLD_ROTATION[0], 10.5)        # the move, in the window
    assert g.hits == 1 and g.state == "idle"


# -- studio integration ------------------------------------------------------------

def test_studio_combo_fires_payoff_and_flash():
    n = int(10 * FPS)
    # three CLAPs in close succession ~4 s in -> CLAP STORM
    st, be, last = _run(_groove(n), commands_at={120: ["CLAP"], 126: ["CLAP"],
                                                 132: ["CLAP"]})
    assert [name for _, name in st.combo_log] == ["CLAP STORM"]
    # fx one-shots from the payoff exist (the snare hits are channel 10 fx).
    assert any(e.tag == "fx" for e in be.events if e.kind == "note_on")


def test_studio_locked_stems_stay_silent_until_unlocked():
    n = int(12 * FPS)
    prof = _profile(_groove(n))
    be = LogBackend()
    st = Studio(be, prof)
    unlock_t = {}
    for fr in ArrayPoseSource(_groove(n), fps=FPS, flip_y=False).frames():
        out = st.step(fr, [])
        for s in out.ui.stems:
            if not s.locked and s.name not in unlock_t:
                unlock_t[s.name] = fr.t
    st.panic()
    assert unlock_t["drums"] == 0.0 and unlock_t["bass"] == 0.0
    assert unlock_t["keys"] > 0.5                  # earned, not given
    # no stem makes a sound before its unlock time (one frame of slack: the
    # unlock is observed at frame time, events carry exact grid times).
    for e in be.events:
        if e.kind == "note_on" and e.tag in unlock_t:
            assert e.t >= unlock_t[e.tag] - 1.5 / FPS, \
                (e.tag, e.t, unlock_t[e.tag])


def test_studio_game_ui_snapshot_complete():
    st, be, last = _run(_groove(int(6 * FPS)))
    g = last.ui.game
    assert g is not None
    assert g.section in SECTIONS and g.sections == SECTIONS
    assert 0.0 <= g.streak <= 1.0 and 0.0 <= g.heat <= 1.0
    assert set(g.unlocked) == {"drums", "bass", "keys", "lead", "texture"}
    assert g.identity_name and g.identity_tagline


def test_studio_game_off_is_clean_legacy_mode():
    cfg = StudioConfig(game=False, use_identity=False)
    st, be, last = _run(_groove(int(6 * FPS)), cfg=cfg)
    assert last.ui.game is None
    assert st.identity is None
    assert not any(s.locked for s in last.ui.stems)


def test_studio_identity_applied_to_substrate_and_kit():
    st, be, last = _run(_groove(int(4 * FPS)))
    ident = st.identity
    assert ident is not None
    assert st.sub.cfg.tonic == ident.tonic
    assert st.sub.cfg.scale == ident.scale
    assert st.sub.cfg.harmonic_field == ident.progression
    for stem in st.rack:
        assert stem.timbre == ident.kit[stem.name]


def test_salience_cap_drops_constant_spam_but_keeps_triples():
    n = int(20 * FPS)
    # CLAP hammered twice a second for 20 s (a real dancer's incidental claps).
    cmds = {i: ["CLAP"] for i in range(0, n, 15)}
    st, be, last = _run(_groove(n), commands_at=cmds)
    accepted = st._move_times["CLAP"]
    # burst rule: never more than 4 accepted in any rolling 14 s window...
    for k in range(len(accepted)):
        win = [x for x in accepted if accepted[k] - 14.0 <= x <= accepted[k]]
        assert len(win) <= 4
    # ...and the x3 burst still got through fast enough to fire the combo.
    assert "CLAP STORM" in [name for _, name in st.combo_log]


def test_still_dwell_ignores_brief_dips_but_catches_freezes():
    from test_studio import _still
    n1, nd, n2 = int(6 * FPS), int(6 * FPS), int(3 * FPS)
    groove = _groove(n1 + nd + n2)
    brief = np.concatenate([groove[:n1], _still(6), groove[n1 + 6:]])  # 0.2 s dip
    real = np.concatenate([groove[:n1], _still(nd), groove[:n2]])      # 6 s freeze
    st_b, be_b, _ = _run(brief)
    st_r, be_r, _ = _run(real)
    # a 0.2 s dip never engages the gate -> no FREEZE wind-down fires;
    # a real freeze fires it exactly once.
    freezes_b = sum(1 for t, n in st_b.fx.log if n == "FREEZE")
    freezes_r = sum(1 for t, n in st_r.fx.log if n == "FREEZE")
    assert freezes_b == 0
    assert freezes_r == 1


def test_peak_lift_transposes_and_returns():
    n = int(20 * FPS)
    prof = _profile(_groove(n))
    be = LogBackend()
    st = Studio(be, prof)
    base = st.sub.cfg.tonic
    # energetic groove + named moves (PEAK deliberately needs the move push),
    # then a long cool-down so the arc falls back out of peak.
    seq = np.concatenate([_groove(n), _groove(int(14 * FPS), energy=0.05)])
    moves = {i: ["JUMP" if (i // 45) % 2 == 0 else "CLAP"]
             for i in range(30, n, 45)}
    lifted_seen = False
    for i, fr in enumerate(ArrayPoseSource(seq, fps=FPS, flip_y=False).frames()):
        st.step(fr, moves.get(i, []))
        if st.arc.section == "peak":
            lifted_seen = True
            assert st.sub.cfg.tonic == base + 2     # THE LIFT is in effect
    st.panic()
    assert lifted_seen
    # cooled all the way down -> the lift has been undone.
    assert st.arc.section != "peak" and st.sub.cfg.tonic == base


def test_studio_determinism_with_game_on():
    cmds = {60: ["CLAP"], 66: ["CLAP"], 72: ["CLAP"], 150: ["STOMP"],
            156: ["STOMP"]}
    n = int(8 * FPS)
    _, be1, _ = _run(_groove(n), commands_at=cmds)
    _, be2, _ = _run(_groove(n), commands_at=cmds)
    ev1 = [(e.kind, e.channel, e.a, e.b, round(e.t, 6)) for e in be1.events]
    ev2 = [(e.kind, e.channel, e.a, e.b, round(e.t, 6)) for e in be2.events]
    assert ev1 == ev2 and ev1
