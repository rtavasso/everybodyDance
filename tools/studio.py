#!/usr/bin/env python3
"""everybodyDance STUDIO (live): the polished webcam instrument.

Dance in front of the camera; the studio auto-advances through the stems, loops
what you author (rhythm from your hits, pitch from your height), lets recognised
moves punch in effects + score, and plays it all back through the built-in synth
+ FX -- with the Just-Dance-style UI. Pick any preset for a different sound.

    python -m tools.studio --preset house --mirror
    python -m tools.studio --preset ambient --no-audio

Keys (focus the window): q quit · n next stem · m mute active · c re-calibrate ·
SPACE pause · r re-record active stem.
"""

import argparse
import threading
import time

import numpy as np

from everybody_dance.calib import BuildCalibrator, fold_tempo
from everybody_dance.effects import STYLE, DEFAULT_STYLE, Flash
from everybody_dance.features import FeatureExtractor
from everybody_dance.fx import apply_fx
from everybody_dance.gestures import GestureRecognizer, default_gestures
from everybody_dance.pose import JOINT_INDEX, JOINTS, MP_INDEX, PoseFrame, _normalise
from everybody_dance.presets import get_preset
from everybody_dance.stems import StemRack
from everybody_dance.studio_types import SR, MovePrompt, StudioUIState
from everybody_dance.studio_ui import attract_screen, compose_studio
from everybody_dance.synth_engine import render_note

FOOT = [JOINT_INDEX[j] for j in ("l_ankle", "r_ankle")]
HAND = [JOINT_INDEX[j] for j in ("l_wrist", "r_wrist")]


class LoopPlayer:
    """Plays a mono loop buffer on repeat; swap the buffer when stems change.
    Degrades to a silent no-op if sounddevice/PortAudio is unavailable."""

    def __init__(self, sr=SR):
        self.sr, self.ok, self.error = sr, False, ""
        self._buf = np.zeros(1, np.float32)
        self._pos = 0
        self._lock = threading.Lock()
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=sr, channels=1, blocksize=512,
                                           dtype="float32", callback=self._cb)
            self._stream.start()
            self.ok = True
        except Exception as e:
            self._stream = None
            self.error = f"{type(e).__name__}: {e}"

    def set_loop(self, buf):
        with self._lock:
            self._buf = buf.astype(np.float32) if len(buf) else np.zeros(1, np.float32)
            self._pos = self._pos % len(self._buf)

    def _cb(self, outdata, frames, time_info, status):  # pragma: no cover
        with self._lock:
            buf, n = self._buf, len(self._buf)
            idx = (self._pos + np.arange(frames)) % n
            outdata[:, 0] = buf[idx]
            self._pos = (self._pos + frames) % n

    def close(self):  # pragma: no cover
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass


def landmarks(lm, w, h):
    xyz = np.array([[lm[MP_INDEX[j]].x, lm[MP_INDEX[j]].y, lm[MP_INDEX[j]].z]
                    for j in JOINTS], float)
    vis = np.array([lm[MP_INDEX[j]].visibility for j in JOINTS])
    px = np.column_stack([xyz[:, 0] * w, xyz[:, 1] * h])
    return _normalise(xyz), vis, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="house")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calibrate", type=float, default=12.0)
    ap.add_argument("--bars-per-stem", type=float, default=4.0)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}; try --camera 1")
    pose = mp.solutions.pose.Pose(model_complexity=args.complexity,
                                  smooth_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    player = None if args.no_audio else LoopPlayer()
    rack = StemRack(get_preset(args.preset))
    fe = FeatureExtractor()
    rec = GestureRecognizer(default_gestures())
    cal = BuildCalibrator()

    st = {"phase": "calib", "t0": time.time(), "bpm": 100.0, "active": 0,
          "foot_thr": 12.0, "hand_thr": 12.0, "comlo": 0.0, "comhi": 1.0,
          "score": 0, "combo": 0, "last_fire": -1e9, "paused": False,
          "stem_start": 0.0, "flashes": [], "_foot_ref": 0.0, "_hand_ref": 0.0,
          "prompts": []}

    def rerender():
        if player is None:
            return
        try:
            player.set_loop(rack.render(SR, st["bpm"], render_note, apply_fx, cycles=2))
        except Exception:
            pass

    def begin_perform():
        for i in range(len(rack.states)):
            if rack.states[i].config.rhythm.mode == "euclid":
                rack.author_euclid(i)
        st["phase"], st["active"], st["stem_start"] = "record", 0, time.time()

    win = "everybodyDance studio"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps_ema, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            present = bool(res.pose_landmarks)
            xyz = vis = px = None
            if present:
                xyz, vis, px = landmarks(res.pose_landmarks.landmark, w, h)
            pf = PoseFrame(t=now, xyz=xyz if present else np.zeros((13, 3)),
                           visibility=vis if present else np.zeros(13), raw_present=present)
            f = fe.update(pf)
            comh01 = float(np.clip((f.com_height - st["comlo"]) /
                                   (st["comhi"] - st["comlo"] + 1e-6), 0, 1))

            if st["phase"] == "calib":
                if present and not st["paused"]:
                    cal.observe(pf)
                left = max(0.0, args.calibrate - (now - st["t0"]))
                if left <= 0 and cal.n > 10:
                    prof, tempo, _ = cal.finalize()
                    st["bpm"] = fold_tempo(tempo) * 60.0
                    comh = np.array([0.0, 1.0])  # ranges refined live below
                    begin_perform()
                rerender_due = False
                state = StudioUIState(stems=rack.states, phase="record_rhythm",
                                      bpm=st["bpm"], pitch01=comh01,
                                      live_xyz=xyz, present=present, bars_left=int(left),
                                      meters={"energy": min(1.0, f.energy_env)})
            else:
                # gestures -> flashes + score (always live)
                for ev in rec.update(pf, f):
                    st["combo"] = st["combo"] + 1 if (now - st["last_fire"]) < 3 else 1
                    st["score"] += 100 * st["combo"]; st["last_fire"] = now
                    c = STYLE.get(ev.name, DEFAULT_STYLE)
                    st["flashes"].append(Flash(ev.name, c[0], now, 0.5, c[1]))
                st["flashes"] = [fl for fl in st["flashes"] if fl.alpha(now) > 0]

                # live body mapping (openness/energy -> mapped params)
                signals = {"energy": min(1.0, f.energy_env), "openness": min(1.0, f.openness),
                           "limb_energy": min(1.0, f.limb_energy_env)}
                for i in range(len(rack.states)):
                    if rack.states[i].config.mapping.bindings:
                        rack.apply_mapping(i, signals)

                sd0 = rack.step_dur(st["bpm"], 0)
                total = rack.states[0].config.rhythm.total_steps
                loop_secs = total * sd0
                phase = "record" if st["phase"] == "record" else "perform"

                if st["phase"] == "record" and present and not st["paused"]:
                    # record onsets into the active stem from limb hits
                    cfg = rack.states[st["active"]].config
                    melodic = cfg.channel != 10 and cfg.name not in ("bass", "drone", "pulse")
                    sig = float(f.accel_mag[HAND].max() if melodic else f.accel_mag[FOOT].max())
                    ref = "_hand_ref" if melodic else "_foot_ref"
                    st[ref] = max(sig, st[ref] * 0.995)
                    if cfg.rhythm.mode == "body" and sig > 0.55 * st[ref] and sig > 4:
                        step = int(round((now - st["stem_start"]) / sd0)) % total
                        rack.record_onset(st["active"], step,
                                          int(np.clip(60 + 40 * comh01, 40, 120)), comh01)
                        rerender()
                    if (now - st["stem_start"]) >= args.bars_per_stem * sd0 * (total // 2):
                        st["active"] += 1
                        st["stem_start"] = now
                        if st["active"] >= len(rack.states):
                            st["active"], st["phase"] = 0, "perform"
                        rerender()

                playhead = int((now % loop_secs) / sd0) % total
                state = StudioUIState(
                    stems=rack.states, active_stem=st["active"], phase=phase,
                    playhead=playhead, bpm=st["bpm"], pitch01=comh01,
                    flashes=list(st["flashes"]), prompts=st["prompts"],
                    score=st["score"], combo=st["combo"],
                    rating="PERFECT" if (now - st["last_fire"]) < 0.6 else "",
                    meters={"energy": min(1.0, f.energy_env), "pitch": comh01},
                    live_xyz=xyz, present=present)

            canvas = compose_studio(state, 1280, 720, camera_bgr=None, t=now)
            dt = now - last; last = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1 / dt if dt > 0 else 0)
            anote = ("audio off" if (player is None or not player.ok) else "audio ON")
            cv2.putText(canvas, f"{fps_ema:.0f}fps | {anote} | preset {args.preset}",
                        (16, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                st["paused"] = not st["paused"]
            if k == ord("n") and st["phase"] != "calib":
                st["active"] = (st["active"] + 1) % len(rack.states)
            if k == ord("m") and st["phase"] != "calib":
                s = rack.states[st["active"]]
                rack.set_muted(st["active"], not s.muted); rerender()
            if k == ord("r") and st["phase"] != "calib":
                rack.clear(st["active"]); st["phase"] = "record"; st["stem_start"] = now; rerender()
            if k == ord("c"):
                st.update(phase="calib", t0=now); cal = BuildCalibrator()
    except KeyboardInterrupt:
        pass
    finally:
        if player:
            player.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
