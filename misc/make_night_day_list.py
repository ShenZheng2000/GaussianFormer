"""Build val night/day scene lists from nuScenes scene.json descriptions.

Usage: python misc/make_night_day_list.py
Output: misc/val_night_day.json
"""
import json
import os
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, "data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl")
SCENE_JSON = os.path.join(ROOT, "data/nuscenes/v1.0-trainval/scene.json")
OUT = os.path.join(ROOT, "misc/val_night_day.json")

scenes = {s["token"]: s for s in json.load(open(SCENE_JSON))}
data = pickle.load(open(PKL, "rb"))
metadata = data["metadata"]  # list of (scene_token, frame_idx), same as dataset.keyframes

out = {"night": {}, "day": {}}
for tok in data["infos"]:
    s = scenes[tok]
    tag = "night" if "night" in s["description"].lower() else "day"
    out[tag][tok] = {"name": s["name"], "description": s["description"]}

night = set(out["night"])
n_night = sum(t in night for t, _ in metadata)
out["counts"] = {
    "night_scenes": len(out["night"]),
    "day_scenes": len(out["day"]),
    "night_keyframes": n_night,
    "day_keyframes": len(metadata) - n_night,
}
json.dump(out, open(OUT, "w"), indent=1)
print(out["counts"])
print("saved", OUT)
