import numpy as np
import matplotlib.pyplot as plt
from particle_filter import load_map
from data_io import load_bag
from test import run
from metrics import consistency, interp_pose, dead_reckon, moving_mean

START = (0.3, 0.25, -0.06)          # start pose from the first-frame grid search (used for odometry-only)


def check(odom, det_t, frames, colours, xy, other=None, N=5000, start=START):
    res = run(odom, det_t, frames, colours, xy, N=N)
    t, x, y, yaw = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
    keep = [i for i, tt in enumerate(det_t) if odom["t"][0] <= tt <= odom["t"][-1]]
    fr = [frames[i] for i in keep]                               # the frames the filter actually used
    t0 = odom["t"][0]

    c_filter = consistency(colours, xy, fr, x, y, yaw)
    dx, dy, dyaw = dead_reckon(odom, start, t)
    c_odom = consistency(colours, xy, fr, dx, dy, dyaw)

    print(f"frames used: {len(fr)}")
    print(f"map consistency, particle filter : {np.nanmean(c_filter):.2f}  "
          f"(frames above 0.5: {np.nanmean(c_filter > 0.5):.0%})")
    print(f"map consistency, odometry only   : {np.nanmean(c_odom):.2f}  "
          f"(frames above 0.5: {np.nanmean(c_odom > 0.5):.0%})")

    c_other = None
    if other is not None:
        t2, fr2 = other
        k2 = [i for i, tt in enumerate(t2) if t[0] <= tt <= t[-1]]
        fr2 = [fr2[i] for i in k2]
        ex, ey, eyaw = interp_pose(np.array(t2)[k2], t, x, y, yaw)
        c_other = consistency(colours, xy, fr2, ex, ey, eyaw)
        print(f"map consistency, filter pose checked with the OTHER topic's detections: {np.nanmean(c_other):.2f}")

    # --- plot ---
    plt.figure(figsize=(10, 4.5))
    plt.plot(t - t0, moving_mean(c_filter, 30), color="tab:green", label="particle filter")
    plt.plot(t - t0, moving_mean(c_odom, 30), "r--", label="odometry only")
    if c_other is not None:
        plt.plot(np.array(t2)[k2] - t0, moving_mean(c_other, 30), color="tab:blue", lw=1, label="filter, other topic")
    plt.ylim(0, 1.02); plt.xlabel("time (s)"); plt.ylabel("fraction of detections on a map cone")
    plt.title("Map consistency over time (average of the last ~5 s)")
    plt.grid(alpha=0.3); plt.legend(loc="lower left"); plt.tight_layout(); plt.show()
    return res, c_filter, c_odom


if __name__ == "__main__":
    colours, xy = load_map()
    odom, det_t, frames = load_bag("data")                                        # fused: used by the filter
    try:
        _, det_t2, frames2 = load_bag("data", det_topic="/fsae/perception/cone_detection")   # camera only
        other = (det_t2, frames2)
    except Exception as e:
        print("could not load the second topic, skipping that check:", e)
        other = None
    check(odom, det_t, frames, colours, xy, other)