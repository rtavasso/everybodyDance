#!/usr/bin/env bash
# Fetch the real data used for the iteration (kept out of git):
#   1) LAFAN1 mocap (Ubisoft) -> dance BVH for multiple subjects
#   2) MediaPipe pose-landmarker model
#   3) (optional) a few real-person videos to run pose estimation on
set -e
cd "$(dirname "$0")/.."
mkdir -p data/bvh data/videos data/poses models

echo ">> LAFAN1 mocap (dance subjects)"
if [ ! -f data/lafan1.zip ]; then
  curl -L --fail -o data/lafan1.zip \
    "https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip"
fi
python3 - <<'PY'
import zipfile
z = zipfile.ZipFile("data/lafan1.zip")
for n in z.namelist():
    if n.startswith("dance"):
        z.extract(n, "data/bvh")
        print("  extracted", n)
PY

echo ">> MediaPipe pose_landmarker model"
[ -f models/pose_landmarker.task ] || curl -L --fail -o models/pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

echo ">> Real-person videos (optional; for MediaPipe pose estimation)"
BASE="https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master"
for f in bolt-detection.mp4 face-demographics-walking.mp4 face-demographics-walking-and-pause.mp4; do
  [ -f "data/videos/$f" ] || curl -L --fail -o "data/videos/$f" "$BASE/$f"
done

echo ">> done. Next:"
echo "   python tools/extract_pose.py data/videos/bolt-detection.mp4 -o data/poses/bolt.npz --max-seconds 20"
echo "   python -m tools.run_dataset 'data/bvh/dance*.bvh' --max-seconds 50"
