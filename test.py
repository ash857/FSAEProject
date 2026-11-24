import numpy as np
import matplotlib.pyplot as plt
from particle_filter import (load_map, corridor_particles, known_particles, body_increment, predict, log_likelihood,
                weight, estimate, n_eff, systematic_resample, inject_random, wrap_angle)
from data_io import load_bag
from metrics import consistency
from health import (SEARCHING, TRACKING, LOST, FIT_EMA, MIN_DETECTIONS, RECOVER_FRAC, update_health)


def odom_pose_at(odom, t):
    """Odometry pose (x, y, yaw) at time t, interpolated between samples.

    Needed because the odometry and the camera don't tick at the same moments.
    """
    return tuple(np.interp(t, odom["t"], odom[key]) for key in ("x", "y", "yaw"))


def run(odom, det_t, frames, colours, xy, N=3000, seed=42, callback=None, start=None, recover=True):
    """Run the particle filter over every detection frame.

    Each frame: Predict -> Weight/Normalise -> Estimate -> Lost? -> Resample -> Recover.

    start    : known start pose (x, y, yaw), or None to begin with particles spread over the track
    callback : optional function called after the predict, weight and resample stages (used by watch)
    recover  : if False, never inject fresh particles when lost

    Returns an array with one row per frame: (t, x, y, yaw, spread, n_eff, status),
    where status is SEARCHING (0), TRACKING (1) or LOST (2).
    """
    rng = np.random.default_rng(seed)
    if start is None:
        particles = corridor_particles(colours, xy, N, rng)
        status = SEARCHING
    else:
        particles = known_particles(start, N, rng)
        status = TRACKING
    weights = np.full(N, 1.0 / N)
    good_frames = bad_frames = 0
    fit_smoothed = 1.0
    prev_pose, results = None, []

    def notify(stage, frame_idx, t, det_colours, det_xy):
        # Reads the current particles and weights each time it is called, not a snapshot.
        if callback is not None:
            callback(frame_idx, t, particles, weights, estimate(particles, weights),
                     n_eff(weights), det_colours, det_xy, stage)

    for t, (det_colours, det_xy) in zip(det_t, frames):
        if not odom["t"][0] <= t <= odom["t"][-1]:
            continue
        frame_idx = len(results)
        has_detections = len(det_colours) > 0

        # PREDICT
        pose = odom_pose_at(odom, t)
        if prev_pose is not None:
            forward, left, turn = body_increment(prev_pose, pose)
            particles = predict(particles, forward, left, turn, rng)
        prev_pose = pose
        notify("predict", frame_idx, t, det_colours, det_xy)

        # WEIGHT + NORMALISE
        if has_detections:
            weights = weight(weights, log_likelihood(particles, colours, xy, det_colours, det_xy))

        # ESTIMATE
        x, y, yaw, spread = estimate(particles, weights)
        n_effective = n_eff(weights)

        # FILTER LOST?
        if has_detections:
            fit = consistency(colours, xy, [(det_colours, det_xy)], [x], [y], [yaw])[0]
        else:
            fit = np.nan
        if len(det_colours) >= MIN_DETECTIONS and not np.isnan(fit):
            fit_smoothed += FIT_EMA * (fit - fit_smoothed)
        prev_status = status
        status, good_frames, bad_frames = update_health(
            status, good_frames, bad_frames, spread, fit, fit_smoothed, len(det_colours))
        if status == TRACKING and prev_status != TRACKING:
            fit_smoothed = max(fit_smoothed, 0.7)     # start tracking from a healthy smoothed score

        results.append((t, x, y, yaw, spread, n_effective, status))
        notify("weight", frame_idx, t, det_colours, det_xy)

        # RESAMPLE (only when the weights have become lopsided)
        if n_effective < 0.5 * N:
            particles, weights = systematic_resample(particles, weights, rng)
            notify("resample", frame_idx, t, det_colours, det_xy)

        # RECOVER: after resampling, so the fresh particles survive to be tested on the next frame
        if recover and status == LOST:
            particles, weights = inject_random(particles, weights, RECOVER_FRAC, colours, xy, rng)

    return np.array(results)