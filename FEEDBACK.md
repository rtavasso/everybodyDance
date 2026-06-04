# Sandbox feedback

Run `python -m tools.live` (or `scripts\sandbox_windows.ps1`), dance a full song,
and dump raw notes here. Don't polish — bullet points are perfect. I'll triage.

## Setup / does it even run
- OS + Python version:
- Did `pip install -r requirements-realtime.txt` succeed?
- Did the pose weights auto-download and the camera open?
- Any crash / traceback (paste it):

## Performance
- Camera FPS shown in the HUD:
- Inference ms / draw ms:
- Did audio say ON? Any glitches / latency / crackle?
- Did it feel real-time, or laggy?

## Pose tracking
- Was the skeleton stable? Where did it break (occlusion, fast moves, distance)?
- Calibration length right? (`--calibrate N`)

## The interaction (the important part)
- RHYTHM pass: could you feel your hits landing as onsets? Too many / too few?
- PITCH pass: did crouch→rise actually control pitch in a way you could aim?
- Auto-advance timing: too fast / too slow per instrument?
- Did "dance continuously, the song builds" feel good, or fight you?

## The result
- Did the song sound musical? Which stem worked best / worst?
- Would you want finer control over anything (which is currently automatic)?

## Anything else / wishlist
-
