"""Watch replayed demos: render a video, or play live in the MuJoCo viewer.

Plays back the RECORDED states of data/replayed/<clip>.npz (arm, fingers, tube, coaster) kinematically,
so what you see is exactly what happened in the physics replay, at real speed.

Run:  python view_replay.py IMG_8903 IMG_8883            -> data/videos/<clip>.mp4 (side view + robot head view)
      python view_replay.py IMG_8903 --live              -> interactive viewer (mouse: rotate/zoom; close to stop)
"""
import argparse
import time
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).parent
PHASES = ["approach", "grasp", "transport", "place", "retreat"]


def setup():
    from build_g1_scene import make
    from ik import ArmIK
    from test_g1_grasp import FINGER_JOINTS
    m, d, stand = make()
    ik = ArmIK(m)
    fq = [m.joint(j).qposadr[0] for j in FINGER_JOINTS]
    cup = m.joint("cup_free").qposadr[0]
    return m, d, ik, fq, cup


def set_state(m, d, ik, fq, cup, e, k):
    d.qpos[ik.qadr] = e["arm_q"][k]
    d.qpos[fq] = e["finger_q"][k]
    d.qpos[cup:cup + 3] = e["cup_pos"][k]
    d.qpos[cup + 3:cup + 7] = e["cup_quat"][k]
    mujoco.mj_forward(m, d)


def load(clip):
    e = dict(np.load(ROOT / "data" / "replayed" / f"{clip}.npz"))
    return e


def place_coaster(m, e):
    c = e["coaster_pos"][0]
    m.geom_pos[m.geom("coaster").id][:2] = c[:2]
    m.site_pos[m.site("coaster_center").id][:2] = c[:2]


def video(clip, out_dir):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    m, d, ik, fq, cup = setup()
    e = load(clip)
    place_coaster(m, e)
    W, H = 640, 480
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.36, -0.08, 0.88]
    cam.distance, cam.azimuth, cam.elevation = 1.1, 135, -25
    ok = bool(e["success"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip}.mp4"
    w = imageio.get_writer(path, fps=25, macro_block_size=8)
    for k in range(0, len(e["action"]), 2):                    # 50 Hz ticks -> 25 fps, real speed
        set_state(m, d, ik, fq, cup, e, k)
        r.update_scene(d, cam); side = r.render()
        r.update_scene(d, "ego"); ego = r.render()
        img = Image.fromarray(np.hstack([side, ego]))
        dr = ImageDraw.Draw(img)
        ph = PHASES[int(e["phase_auto"][k])]
        dr.rectangle([0, 0, 2 * W, 34], fill=(0, 0, 0))
        dr.text((10, 8), f"{clip}   t={k*0.02:4.1f}s   phase: {ph}   grasped: {bool(e['grasped'][k])}",
                fill=(255, 255, 255))
        dr.text((2 * W - 250, 8), "RESULT: SUCCESS" if ok else "RESULT: FAILED",
                fill=(80, 255, 80) if ok else (255, 90, 90))
        dr.text((10, H - 20), "side view", fill=(255, 255, 255))
        dr.text((W + 10, H - 20), "robot head camera", fill=(255, 255, 255))
        w.append_data(np.array(img))
    for _ in range(25):                                        # hold the last frame 1 s
        w.append_data(np.array(img))
    w.close()
    return path


def live(clip):
    import mujoco.viewer
    m, d, ik, fq, cup = setup()
    e = load(clip)
    place_coaster(m, e)
    print(f"{clip}: {'SUCCESS' if e['success'] else 'FAILED'} - close the viewer window to stop")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.36, -0.08, 0.88]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.1, 135, -25
        while v.is_running():                                  # loop the replay until closed
            for k in range(len(e["action"])):
                t0 = time.time()
                set_state(m, d, ik, fq, cup, e, k)
                v.sync()
                if not v.is_running():
                    return
                time.sleep(max(0, 0.02 - (time.time() - t0)))
            time.sleep(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live:
        live(a.clips[0])
    else:
        for c in a.clips:
            print("wrote", video(c, ROOT / "data" / "videos"))
