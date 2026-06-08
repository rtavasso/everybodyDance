"""Tests for everybody_dance.stems — the multi-stem looper rack.

Uses stub synth/fx so no dependency on synth_engine or fx modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from everybody_dance.stems import euclid, snap_pitch, StemRack, _drum_pitch
from everybody_dance.studio_types import (
    FxConfig,
    Mapping,
    Note,
    PitchConfig,
    RhythmConfig,
    StemConfig,
    TimbreConfig,
    SR,
)
from everybody_dance.substrate import SCALES
from everybody_dance.output import MusicEvent

# ---------------------------------------------------------------------------
# Stubs — no dependency on synth_engine or fx.
# ---------------------------------------------------------------------------

def stub_synth(pitch, dur_s, vel, timbre, sr):
    """Return a flat buffer of 0.1 so energy is predictable."""
    return np.ones(max(1, int(dur_s * sr)), dtype=np.float32) * 0.1


def stub_fx(buf, fx, sr, bpm):
    """Identity: return buffer unchanged."""
    return buf


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _simple_cfg(name="s", channel=1, steps=16, bars=1, pulses=4,
                swing=0.0, gate=0.9, scale="minor", tonic=57,
                register=(48, 72), mode="height"):
    return StemConfig(
        name, channel,
        rhythm=RhythmConfig(steps=steps, bars=bars, pulses=pulses,
                            swing=swing, gate=gate),
        pitch=PitchConfig(mode=mode, scale=scale, tonic=tonic,
                          register=register, quantize=True),
        timbre=TimbreConfig(),
        fx=FxConfig(),
    )


# ---------------------------------------------------------------------------
# euclid
# ---------------------------------------------------------------------------

class TestEuclid:
    def test_hit_count_16_4(self):
        pat = euclid(16, 4)
        assert len(pat) == 16
        assert sum(pat) == 4

    def test_hit_count_8_3(self):
        pat = euclid(8, 3)
        assert len(pat) == 8
        assert sum(pat) == 3

    def test_8_3_known_pattern(self):
        # Bjorklund(8,3): hits at positions 0, 3, 6
        pat = euclid(8, 3)
        assert pat == [True, False, False, True, False, False, True, False]

    def test_zero_pulses(self):
        pat = euclid(8, 0)
        assert all(not p for p in pat)
        assert len(pat) == 8

    def test_full_pulses(self):
        pat = euclid(8, 8)
        assert all(pat)

    def test_rotation_shifts_pattern(self):
        base = euclid(8, 3, 0)
        rotated = euclid(8, 3, 2)
        assert sum(rotated) == 3
        assert rotated != base

    def test_returns_list_of_bool(self):
        pat = euclid(8, 3)
        assert isinstance(pat, list)
        assert all(isinstance(p, bool) for p in pat)


# ---------------------------------------------------------------------------
# snap_pitch
# ---------------------------------------------------------------------------

class TestSnapPitch:
    """snap_pitch must stay in register and, when quantize=True, in scale."""

    def _minor_intervals(self):
        return set(SCALES["minor"])

    def test_in_register_height_mode(self):
        pc = PitchConfig(mode="height", scale="minor", tonic=57,
                         register=(48, 72), quantize=True)
        for h in np.linspace(0.0, 1.0, 21):
            n = snap_pitch(float(h), pc)
            assert 48 <= n <= 72, f"h={h} -> note={n} out of register"

    def test_in_scale_minor(self):
        pc = PitchConfig(mode="height", scale="minor", tonic=57,
                         register=(48, 72), quantize=True)
        intervals = self._minor_intervals()
        for h in np.linspace(0.0, 1.0, 11):
            n = snap_pitch(float(h), pc)
            pc_rel = (n - 57) % 12
            assert pc_rel in intervals, \
                f"h={h} -> note={n} (pc_rel={pc_rel}) not in minor scale"

    def test_no_quantize_stays_in_register(self):
        pc = PitchConfig(mode="height", scale="minor", tonic=57,
                         register=(48, 72), quantize=False)
        for h in [0.0, 0.5, 1.0]:
            n = snap_pitch(float(h), pc)
            assert 48 <= n <= 72

    def test_fixed_mode_uses_fixed_degree(self):
        # degree 0 -> tonic (57), in register (48,72)
        pc = PitchConfig(mode="fixed", scale="minor", tonic=57,
                         register=(48, 72), fixed_degree=0)
        n = snap_pitch(0.5, pc)
        assert n == 57

    def test_fixed_mode_degree2(self):
        # minor intervals: [0,2,3,5,7,8,10]; degree 2 -> interval 3 -> 57+3=60
        pc = PitchConfig(mode="fixed", scale="minor", tonic=57,
                         register=(48, 72), fixed_degree=2)
        n = snap_pitch(0.5, pc)
        assert n == 60

    def test_boundary_heights(self):
        pc = PitchConfig(mode="height", scale="major", tonic=60,
                         register=(48, 84), quantize=True)
        n_lo = snap_pitch(0.0, pc)
        n_hi = snap_pitch(1.0, pc)
        assert 48 <= n_lo <= 84
        assert 48 <= n_hi <= 84

    def test_different_scales(self):
        for scale in ("major", "dorian", "minor_pentatonic", "lydian"):
            pc = PitchConfig(mode="height", scale=scale, tonic=57,
                             register=(48, 72), quantize=True)
            intervals = set(SCALES[scale])
            n = snap_pitch(0.5, pc)
            assert (n - 57) % 12 in intervals, \
                f"scale={scale} note={n} not in scale"


# ---------------------------------------------------------------------------
# StemRack construction
# ---------------------------------------------------------------------------

class TestStemRackInit:
    def test_loop_length_equals_total_steps(self):
        cfg = _simple_cfg(steps=16, bars=2)
        rack = StemRack([cfg])
        assert len(rack.states[0].loop) == 32  # 16 * 2

    def test_loop_starts_empty(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        assert all(n is None for n in rack.states[0].loop)
        assert rack.states[0].n_onsets == 0

    def test_multiple_stems(self):
        cfgs = [_simple_cfg(f"s{i}") for i in range(4)]
        rack = StemRack(cfgs)
        assert len(rack.states) == 4


# ---------------------------------------------------------------------------
# author_euclid
# ---------------------------------------------------------------------------

class TestAuthorEuclid:
    def test_fills_correct_number_of_onsets(self):
        cfg = _simple_cfg(steps=16, bars=1, pulses=4)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        pat = euclid(16, 4, 0)
        assert rack.states[0].n_onsets == sum(pat)

    def test_onset_positions_match_pattern(self):
        cfg = _simple_cfg(steps=8, bars=1, pulses=3)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        pat = euclid(8, 3, 0)
        for step, hit in enumerate(pat):
            note = rack.states[0].loop[step]
            if hit:
                assert note is not None, f"step {step} should have a note"
            else:
                assert note is None, f"step {step} should be empty"

    def test_pitches_in_register(self):
        cfg = _simple_cfg(steps=16, bars=1, pulses=6,
                          register=(48, 72), mode="height")
        rack = StemRack([cfg])
        rack.author_euclid(0)
        for note in rack.states[0].loop:
            if note is not None:
                assert 48 <= note.pitch <= 72

    def test_tag_is_euclid(self):
        cfg = _simple_cfg(steps=8, bars=1, pulses=3)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        for note in rack.states[0].loop:
            if note is not None:
                assert note.tag == "euclid"

    def test_clears_previous_loop(self):
        cfg = _simple_cfg(steps=8, bars=1, pulses=3)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        first_count = rack.states[0].n_onsets
        rack.record_onset(0, 1, 100, 0.5)  # add extra note
        rack.author_euclid(0)  # should reset
        assert rack.states[0].n_onsets == first_count


# ---------------------------------------------------------------------------
# record_onset
# ---------------------------------------------------------------------------

class TestRecordOnset:
    def test_places_note_at_step(self):
        cfg = _simple_cfg(steps=16, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 5, 100, 0.5)
        assert rack.states[0].loop[5] is not None

    def test_pitch_from_height(self):
        cfg = _simple_cfg(steps=16, bars=1, register=(48, 72))
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.0)
        rack.record_onset(0, 1, 100, 1.0)
        p_lo = rack.states[0].loop[0].pitch
        p_hi = rack.states[0].loop[1].pitch
        assert p_lo <= p_hi, "higher height should give >= pitch"

    def test_overdub_replaces_note(self):
        cfg = _simple_cfg(steps=16, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 3, 100, 0.2)
        rack.record_onset(0, 3, 80, 0.8)
        note = rack.states[0].loop[3]
        assert note.vel == 80

    def test_velocity_clamped(self):
        cfg = _simple_cfg(steps=16, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 200, 0.5)  # over 127
        assert rack.states[0].loop[0].vel == 127

    def test_step_wraps_around(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 10, 100, 0.5)  # 10 % 8 = 2
        assert rack.states[0].loop[2] is not None

    def test_drums_kick_height(self):
        drum_cfg = StemConfig("drums", 10,
            rhythm=RhythmConfig(steps=16, bars=1, gate=0.4),
            pitch=PitchConfig(mode="fixed"),
            timbre=TimbreConfig(instrument="kick"))
        rack = StemRack([drum_cfg])
        rack.record_onset(0, 0, 100, 0.1)  # kick range
        assert rack.states[0].loop[0].pitch == 36  # GM kick

    def test_drums_snare_height(self):
        drum_cfg = StemConfig("drums", 10,
            rhythm=RhythmConfig(steps=16, bars=1, gate=0.4),
            pitch=PitchConfig(mode="fixed"),
            timbre=TimbreConfig(instrument="snare"))
        rack = StemRack([drum_cfg])
        rack.record_onset(0, 0, 100, 0.55)  # snare range
        assert rack.states[0].loop[0].pitch == 38  # GM snare

    def test_drums_hat_height(self):
        drum_cfg = StemConfig("drums", 10,
            rhythm=RhythmConfig(steps=16, bars=1, gate=0.4),
            pitch=PitchConfig(mode="fixed"),
            timbre=TimbreConfig(instrument="hat"))
        rack = StemRack([drum_cfg])
        rack.record_onset(0, 0, 100, 0.85)  # hat range
        assert rack.states[0].loop[0].pitch == 42  # GM hi-hat


# ---------------------------------------------------------------------------
# clear / mute / solo
# ---------------------------------------------------------------------------

class TestClearMuteSolo:
    def test_clear_removes_all_notes(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        rack.clear(0)
        assert rack.states[0].n_onsets == 0
        assert all(n is None for n in rack.states[0].loop)

    def test_set_muted(self):
        cfg = _simple_cfg()
        rack = StemRack([cfg])
        rack.set_muted(0, True)
        assert rack.states[0].muted is True
        rack.set_muted(0, False)
        assert rack.states[0].muted is False

    def test_set_solo(self):
        cfg = _simple_cfg()
        rack = StemRack([cfg])
        rack.set_solo(0, True)
        assert rack.states[0].solo is True
        rack.set_solo(0, False)
        assert rack.states[0].solo is False

    def test_mute_silences_stem_in_render(self):
        # Stem with a note, muted -> render should be silent
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        rack.set_muted(0, True)
        mix = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)
        assert np.max(np.abs(mix)) == pytest.approx(0.0)

    def test_unmuted_stem_audible(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        mix = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)
        assert np.max(np.abs(mix)) > 0.0

    def test_solo_excludes_non_solo_stems(self):
        cfg1 = _simple_cfg("s1", steps=8, bars=1)
        cfg2 = _simple_cfg("s2", steps=8, bars=1)
        rack = StemRack([cfg1, cfg2])
        rack.record_onset(0, 0, 100, 0.5)   # stem 0 note at step 0
        rack.record_onset(1, 4, 100, 0.5)   # stem 1 note at step 4

        # Solo stem 0 — same as muting stem 1
        rack.set_solo(0, True)
        mix_solo = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)

        rack.set_solo(0, False)
        rack.set_muted(1, True)
        mix_muted = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)

        assert np.allclose(mix_solo, mix_muted)


# ---------------------------------------------------------------------------
# apply_mapping
# ---------------------------------------------------------------------------

class TestApplyMapping:
    def test_writes_fx_field(self):
        cfg = _simple_cfg()
        cfg.mapping = Mapping(bindings={"openness": "fx.filter_cutoff"})
        rack = StemRack([cfg])
        rack.apply_mapping(0, {"openness": 0.3})
        assert rack.states[0].config.fx.filter_cutoff == pytest.approx(0.3)

    def test_writes_rhythm_field(self):
        cfg = _simple_cfg()
        cfg.mapping = Mapping(bindings={"energy": "rhythm.density"})
        rack = StemRack([cfg])
        rack.apply_mapping(0, {"energy": 0.8})
        assert rack.states[0].config.rhythm.density == pytest.approx(0.8)

    def test_writes_timbre_field(self):
        cfg = _simple_cfg()
        cfg.mapping = Mapping(bindings={"brightness": "timbre.cutoff"})
        rack = StemRack([cfg])
        rack.apply_mapping(0, {"brightness": 0.6})
        assert rack.states[0].config.timbre.cutoff == pytest.approx(0.6)

    def test_clamps_to_0_1(self):
        cfg = _simple_cfg()
        cfg.mapping = Mapping(bindings={"x": "fx.delay_send"})
        rack = StemRack([cfg])
        # signal already clamped to [0,1] inside apply_mapping
        rack.apply_mapping(0, {"x": 1.5})
        assert rack.states[0].config.fx.delay_send == pytest.approx(1.0)

    def test_missing_signal_ignored(self):
        cfg = _simple_cfg()
        original_cutoff = cfg.fx.filter_cutoff
        cfg.mapping = Mapping(bindings={"absent": "fx.filter_cutoff"})
        rack = StemRack([cfg])
        rack.apply_mapping(0, {})  # signal not present
        assert rack.states[0].config.fx.filter_cutoff == pytest.approx(original_cutoff)

    def test_unknown_path_ignored(self):
        cfg = _simple_cfg()
        cfg.mapping = Mapping(bindings={"x": "nonexistent.field"})
        rack = StemRack([cfg])
        # should not raise
        rack.apply_mapping(0, {"x": 0.5})


# ---------------------------------------------------------------------------
# step_dur / swing
# ---------------------------------------------------------------------------

class TestStepDur:
    def test_step_dur_formula(self):
        # step_dur = (60/bpm) * (4/steps)
        cfg = _simple_cfg(steps=16, bars=1)
        rack = StemRack([cfg])
        bpm = 120.0
        expected = (60.0 / bpm) * (4.0 / 16)
        assert rack.step_dur(bpm) == pytest.approx(expected)

    def test_even_step_not_swung(self):
        cfg = _simple_cfg(steps=16, bars=1, swing=0.5)
        rack = StemRack([cfg])
        base = rack.step_dur(120.0)
        t0 = rack._step_time(0, base, 0.5)
        assert t0 == pytest.approx(0.0)

    def test_odd_step_swung_later(self):
        cfg = _simple_cfg(steps=16, bars=1, swing=0.5)
        rack = StemRack([cfg])
        base = rack.step_dur(120.0)
        t1 = rack._step_time(1, base, 0.5)
        expected = base + 0.5 * base * 0.5
        assert t1 == pytest.approx(expected)

    def test_even_step2_not_swung(self):
        cfg = _simple_cfg(steps=16, bars=1, swing=0.5)
        rack = StemRack([cfg])
        base = rack.step_dur(120.0)
        t2 = rack._step_time(2, base, 0.5)
        assert t2 == pytest.approx(2 * base)

    def test_zero_swing_no_push(self):
        cfg = _simple_cfg(steps=16, bars=1, swing=0.0)
        rack = StemRack([cfg])
        base = rack.step_dur(120.0)
        t1 = rack._step_time(1, base, 0.0)
        assert t1 == pytest.approx(base)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

class TestRender:
    def test_returns_float32_array(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        mix = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)
        assert isinstance(mix, np.ndarray)
        assert mix.dtype == np.float32

    def test_finite_output(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        mix = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=2)
        assert np.all(np.isfinite(mix))

    def test_peak_bounded(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        mix = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=2)
        assert np.max(np.abs(mix)) <= 0.96

    def test_length_approx_cycles_times_loop(self):
        cfg = _simple_cfg(steps=16, bars=2)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        bpm = 120.0
        cycles = 2
        mix = rack.render(SR, bpm, stub_synth, stub_fx, cycles=cycles)
        r = cfg.rhythm
        base_dur = (60.0 / bpm) * (4.0 / r.steps)
        loop_s = r.total_steps * base_dur
        expected_min = int(loop_s * cycles * SR)
        expected_max = int(loop_s * cycles * SR) + SR  # +1 s headroom
        assert expected_min <= len(mix) <= expected_max

    def test_empty_loop_gives_silence(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        mix = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)
        assert np.max(np.abs(mix)) == pytest.approx(0.0)

    def test_fx_fn_longer_buffer_handled(self):
        """FX fn may return a longer buffer (simulating reverb tail)."""
        def fx_with_tail(buf, fx, sr, bpm):
            return np.concatenate([buf, np.zeros(sr, dtype=np.float32)])

        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        mix = rack.render(SR, 120.0, stub_synth, fx_with_tail, cycles=1)
        assert np.all(np.isfinite(mix))

    def test_determinism(self):
        cfg = _simple_cfg(steps=16, bars=1)
        rack = StemRack([cfg])
        rack.author_euclid(0)
        mix1 = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=2)
        mix2 = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=2)
        assert np.array_equal(mix1, mix2)

    def test_two_stems_louder_than_one_pre_normalize(self):
        """Before tanh normalization, two stems produce more energy.
        We detect this by checking the pre-normalize sum; here we verify that
        muting one stem does not equal the full mix (two stems differ from one).
        """
        cfg1 = _simple_cfg("s1", steps=8, bars=1)
        cfg2 = _simple_cfg("s2", steps=8, bars=1)
        rack = StemRack([cfg1, cfg2])
        rack.record_onset(0, 0, 100, 0.5)  # stem0 step 0
        rack.record_onset(1, 4, 100, 0.5)  # stem1 step 4 (different time slot)

        rack.set_muted(1, True)
        mix_one = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)

        rack.set_muted(1, False)
        mix_two = rack.render(SR, 120.0, stub_synth, stub_fx, cycles=1)

        # With different step positions they should differ
        assert not np.array_equal(mix_one, mix_two)


# ---------------------------------------------------------------------------
# to_events
# ---------------------------------------------------------------------------

class TestToEvents:
    def test_returns_list_of_music_events(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        events = rack.to_events(120.0, cycles=1)
        assert isinstance(events, list)
        assert all(isinstance(e, MusicEvent) for e in events)

    def test_paired_note_on_off(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        events = rack.to_events(120.0, cycles=1)
        note_ons = [e for e in events if e.kind == "note_on"]
        note_offs = [e for e in events if e.kind == "note_off"]
        assert len(note_ons) == len(note_offs) == 1

    def test_note_off_after_note_on(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        events = rack.to_events(120.0, cycles=1)
        note_on = next(e for e in events if e.kind == "note_on")
        note_off = next(e for e in events if e.kind == "note_off")
        assert note_off.t > note_on.t

    def test_pitch_matches_between_on_off(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 3, 100, 0.5)
        events = rack.to_events(120.0, cycles=1)
        note_on = next(e for e in events if e.kind == "note_on")
        note_off = next(e for e in events if e.kind == "note_off")
        assert note_on.a == note_off.a  # same pitch

    def test_cycles_multiplies_events(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        events1 = rack.to_events(120.0, cycles=1)
        events2 = rack.to_events(120.0, cycles=2)
        assert len(events2) == 2 * len(events1)

    def test_events_sorted_by_time(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        rack.record_onset(0, 4, 100, 0.5)
        events = rack.to_events(120.0, cycles=2)
        times = [e.t for e in events]
        assert times == sorted(times)

    def test_tag_matches_stem_name(self):
        cfg = _simple_cfg("lead", steps=8, bars=1)
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.5)
        events = rack.to_events(120.0, cycles=1)
        for e in events:
            assert e.tag == "lead"

    def test_channel_matches_config(self):
        cfg = StemConfig("drums", 10,
            rhythm=RhythmConfig(steps=8, bars=1),
            pitch=PitchConfig(mode="fixed"))
        rack = StemRack([cfg])
        rack.record_onset(0, 0, 100, 0.1)
        events = rack.to_events(120.0, cycles=1)
        for e in events:
            assert e.channel == 10

    def test_no_notes_no_events(self):
        cfg = _simple_cfg(steps=8, bars=1)
        rack = StemRack([cfg])
        events = rack.to_events(120.0, cycles=2)
        assert events == []


# ---------------------------------------------------------------------------
# Drum pitch map
# ---------------------------------------------------------------------------

class TestDrumPitch:
    def test_kick_low_height(self):
        assert _drum_pitch(0.0) == 36
        assert _drum_pitch(0.2) == 36

    def test_snare_mid_height(self):
        assert _drum_pitch(0.5) == 38
        assert _drum_pitch(0.6) == 38

    def test_hat_high_height(self):
        assert _drum_pitch(0.8) == 42
        assert _drum_pitch(1.0) == 42
