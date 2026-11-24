import numpy as np
from particle_filter import load_map
from data_io import load_bag
from test import run
from metrics import consistency
import faults

START = (0.3, 0.25, -0.06)        

colours, xy = load_map()
odom, det_t, frames = load_bag("data")

margin = 0.5
keep = [i for i, t in enumerate(det_t) if odom["t"][0] + margin <= t <= odom["t"][-1] - margin]
det_t, frames = np.array(det_t)[keep], [frames[i] for i in keep]

known = dict(start=START, N=500)                  # the same known start for every fault test
new_rng = lambda: np.random.default_rng(7)        # a fresh generator per fault, so each one repeats

# The broken inputs
bad_odometry = faults.corrupt_odom(odom, scale=1.3)
frozen_odometry = faults.corrupt_odom(odom, frozen=True)
yaw_bias = faults.corrupt_odom(odom, yaw_bias=0.1)
late_stamps = dict(odom, t=odom["t"] - 0.25)                      # same as detections stamped 0.25 s late
dropped_80 = faults.drop_detections(frames, 0.8, new_rng())
dropped_95 = faults.drop_detections(frames, 0.95, new_rng())
no_orange = faults.remove_orange(frames)
flipped = faults.flip_colours(frames, 0.3, new_rng())
false_cones = faults.add_false_positives(frames, 3, new_rng())
blackout = faults.blackout(det_t, frames, odom["t"][0] + 60, odom["t"][0] + 90)
camera_off = [(c, p + [0.5, 0.0]) for c, p in frames]             # every cone looks 0.5 m farther ahead
noise_rng = new_rng()
noisy = [(c, p + noise_rng.normal(0, 0.3, p.shape)) for c, p in frames]

# (what we break, odometry, detections, options for run)
cases = [
    ("clean (reference)",                        odom,            frames,      known),
    ("clean, different random seed",             odom,            frames,      dict(known, seed=1)),
    ("odometry distance 30% off",                bad_odometry,    frames,      known),
    ("odometry frozen",                          frozen_odometry, frames,      known),
    ("yaw-rate bias 0.1 rad/s",                  yaw_bias,        frames,      known),
    ("detections stamped 0.25 s late",           late_stamps,     frames,      known),
    ("80% of detections dropped",                odom,            dropped_80,  known),
    ("95% of detections dropped",                odom,            dropped_95,  known),
    ("30% colour flips",                         odom,            flipped,     known),
    ("+3 false detections per frame",            odom,            false_cones, known),
    ("camera blackout 30 s",                     odom,            blackout,    known),
    ("camera 0.5 m off",                         odom,            camera_off,  known),
    ("detection noise +0.3 m",                   odom,            noisy,       known),
    ("start heading reversed",                   odom,            frames,      dict(known, start=(START[0], START[1], START[2] + np.pi))),
    ("start 4 m off",                            odom,            frames,      dict(known, start=(START[0] + 4, START[1], START[2]))),
    ("known start, 20 particles",                odom,            frames,      dict(known, N=20)),
    ("unknown start, only 100 particles",        odom,            frames,      dict(N=100)),
    ("unknown start, no orange, 100 particles",  odom,            no_orange,   dict(N=100)),
]

for name, case_odom, case_frames, options in cases:
    result = run(case_odom, det_t, case_frames, colours, xy, **options)
    t, x, y, yaw = result[:, :4].T
    settled = t - odom["t"][0] > 10                                   # ignore the first 10 s (start-up)
    fit = np.nanmean(consistency(colours, xy, frames, x, y, yaw)[settled])
    if name.startswith("clean ("):
        ref_x, ref_y = x, y
    off = np.hypot(x - ref_x, y - ref_y)[settled].mean()
    print(f"{name:42s} fits the map {fit:.2f}   {off:5.2f} m from the clean run")