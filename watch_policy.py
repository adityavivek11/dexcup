"""Watch a trained policy do the task, live or as a video. Uses the provided models; no training needed.

Layouts are the evaluation's 100 test layouts (seed 0), so --layout 7 is the same episode as row 7 of
results/<agent>_random.json.

Run:  python watch_policy.py router --layout 0 --live         (interactive viewer; close the window to stop)
      python watch_policy.py router --layout 0 --video r.mp4  (side view + robot head view)
Agents: teacher, router, random, student.
"""
import argparse
import time
import numpy as np
import torch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=["teacher", "router", "random", "student"])
    ap.add_argument("--layout", type=int, default=0, help="index into the 100 test layouts")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--video", default=None, help="output .mp4 path")
    a = ap.parse_args()
    if not a.live and not a.video:
        ap.error("choose --live or --video FILE")
    if __import__("sys").platform.startswith("linux") and a.video:
        __import__("os").environ.setdefault("MUJOCO_GL", "egl")    # headless rendering on Linux
    torch.set_num_threads(1)
    import mujoco
    import sim_env
    from sim_env import Env, T_MAX
    from eval_policy import layouts, make_agent
    threshold = getattr(sim_env, "GRIP_THRESHOLD", 0.5)            # per-hand on the inspire-hand branch
    cup, coaster = layouts("random", a.layout + 1)[a.layout]
    env = Env()
    agent = make_agent(a.agent, "random", seed=a.layout)
    env.reset(np.array(cup), np.array(coaster))
    agent.reset()
    m, d = env.m, env.d
    status = None

    def tick(t):
        o, palm = env.obs(), env.palm()
        target, grip = agent(o, palm, t)
        env.step(target, grip > threshold)
        return env.status()

    if a.live:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(m, d) as viewer:
            t = 0
            while viewer.is_running():
                t0 = time.time()
                if t < T_MAX and status is None:
                    status = tick(t)
                    t += 1
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.time() - t0)))      # real time: 50 Hz control
    else:
        import imageio
        renderer = mujoco.Renderer(m, 360, 480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.3, -0.1, 0.9]
        cam.distance, cam.azimuth, cam.elevation = 1.2, 150, -25
        with imageio.get_writer(a.video, fps=25) as w:
            for t in range(T_MAX):
                status = tick(t)
                if t % 2 == 0:
                    renderer.update_scene(d, cam)
                    side = renderer.render()
                    renderer.update_scene(d, "ego")
                    w.append_data(np.hstack([side, renderer.render()]))
                if status:
                    break
        print(f"wrote {a.video}")
    print(f"{a.agent} on layout {a.layout}: {status or 'timeout'}")
