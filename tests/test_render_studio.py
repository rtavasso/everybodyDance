"""CI-safe tests for the Studio gradability harness (tools/render_studio).

Headless, no --judge, short duration for speed. Proves the end-to-end replay ->
timbre-aware WAV -> metrics + SLO pipeline runs, is deterministic, exercises the
multi-stem band, and -- the key G2 proof -- that coupling.phantom passes (the
stillness gate works through the WHOLE pipeline) on the default synthetic corpus.

The pipeline is factored so these tests get metrics + the in-memory float mix
without forcing any cv2 PNG writes (write_images=False, out=None).
"""

import numpy as np

from tools import render_studio as R

SECONDS = 10.0   # short but long enough that the corpus stillness window exists


def _run():
    """Default synthetic corpus, in-memory only (no PNG/WAV files written)."""
    return R.run_pipeline(SECONDS, scripted=False, out=None, write_images=False)


def _note_stream(res):
    return [(e.channel, e.a, round(e.t, 4), e.tag)
            for e in res["replay"]["events"] if e.kind == "note_on"]


def test_pipeline_runs_and_mix_is_finite_nonsilent():
    res = _run()
    mix = res["mix"]
    # the in-memory mix stands in for music.wav: finite, bounded, and audible.
    assert mix.size > 0
    assert np.isfinite(mix).all()
    assert float(np.abs(mix).max()) <= 1.0          # normalised, not clipping out
    assert float(np.abs(mix).max()) > 0.0           # not silent


def test_music_wav_written_and_nonempty(tmp_path):
    import wave
    out = tmp_path / "bundle"
    R.run_pipeline(SECONDS, scripted=False, out=str(out), write_images=False)
    wav = out / "music.wav"
    assert wav.exists()
    with wave.open(str(wav)) as w:
        assert w.getnframes() > 0
        assert w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.isfinite(data).all() and int(np.abs(data).max()) > 0


def test_metrics_have_required_keys_and_in_scale_100():
    res = _run()
    m = res["metrics"]
    for key in ("coupling", "musicality", "liveliness", "timing"):
        assert key in m, f"missing metrics block: {key}"
    # all Studio output is scale-snapped -> 100% in-scale.
    assert m["musicality"]["in_scale_pct"] == 100.0
    # the SLO dict is produced and is all-bool.
    assert res["slo"] and all(isinstance(v, bool) for v in res["slo"].values())


def test_phantom_passes_the_stillness_gate():
    """G2 proof: with a real stillness window in the corpus, the gate keeps the
    arrangement quiet -> coupling.phantom stays at/under the SLO threshold."""
    res = _run()
    phantom = res["flat"]["coupling.phantom"]
    assert phantom <= 0.25, f"phantom gate leaked: {phantom}"
    assert res["slo"]["coupling.phantom"] is True


def test_determinism_identical_note_streams():
    a = _note_stream(_run())
    b = _note_stream(_run())
    assert a == b and a, "replay should be byte-identical and non-empty"


def test_multi_stem_covers_at_least_four_stems():
    res = _run()
    stem_tags = {e.tag for e in res["replay"]["events"]
                 if e.kind == "note_on" and e.tag in
                 ("drums", "bass", "keys", "lead", "texture")}
    assert len(stem_tags) >= 4, f"only {stem_tags} active"
    # the band is genuinely multi-timbral too.
    assert res["metrics"]["stems"]["distinct_timbres"] >= 4
