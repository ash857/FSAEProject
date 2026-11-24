import numpy as np
from particle_filter import body_increment

# Each function returns a NEW copy with one thing broken. The original data is never changed.


def corrupt_odom(odom, yaw_bias=0.0, scale=1.0, frozen=False):
    """Re-build the odometry from its own motion steps, with damage.

    yaw_bias : extra turn added every second, rad/s (the car "thinks" it turns a little extra)
    scale    : multiplies every step (the car "thinks" it travels a different distance)
    frozen   : if True, ignore the other two and report no motion at all
    """
    t, x, y, yaw = odom["t"], odom["x"], odom["y"], odom["yaw"]
    if frozen:
        return {"t": t, "x": np.full_like(x, x[0]), "y": np.full_like(y, y[0]), "yaw": np.full_like(yaw, yaw[0])}

    forward, left, turn = body_increment((x[:-1], y[:-1], yaw[:-1]), (x[1:], y[1:], yaw[1:]))
    turn = turn + yaw_bias * np.diff(t)
    forward, left = forward * scale, left * scale

    # Add the damaged steps back up into a path, turning each step into the odometry frame.
    new_yaw = np.concatenate([[yaw[0]], yaw[0] + np.cumsum(turn)])
    cos_yaw, sin_yaw = np.cos(new_yaw[:-1]), np.sin(new_yaw[:-1])
    new_x = np.concatenate([[x[0]], x[0] + np.cumsum(cos_yaw * forward - sin_yaw * left)])
    new_y = np.concatenate([[y[0]], y[0] + np.cumsum(sin_yaw * forward + cos_yaw * left)])
    return {"t": t, "x": new_x, "y": new_y, "yaw": new_yaw}


def drop_detections(frames, p, rng):
    """Each detection is removed with probability p (cones the detector missed)."""
    damaged = []
    for det_colours, det_xy in frames:
        keep = rng.random(len(det_colours)) >= p
        damaged.append((det_colours[keep], det_xy[keep]))
    return damaged


def add_false_positives(frames, rate, rng, max_range=10.0, fov=np.deg2rad(50)):
    """Add about `rate` random fake cones per frame (random blue or yellow, random place in view)."""
    colour_names = np.array(["blue", "yellow"])
    damaged = []
    for det_colours, det_xy in frames:
        n_fake = rng.poisson(rate)
        fake_range = rng.uniform(2.0, max_range, n_fake)
        fake_bearing = rng.uniform(-fov, fov, n_fake)
        fake_xy = np.column_stack([fake_range * np.cos(fake_bearing), fake_range * np.sin(fake_bearing)])
        fake_colours = colour_names[rng.integers(0, 2, n_fake)]
        damaged.append((np.concatenate([det_colours, fake_colours]), np.vstack([det_xy, fake_xy])))
    return damaged


def flip_colours(frames, p, rng):
    """Swap blue and yellow on each detection with probability p. Orange is never flipped."""
    damaged = []
    for det_colours, det_xy in frames:
        det_colours = det_colours.copy()
        flip = (rng.random(len(det_colours)) < p) & (det_colours != "orange")
        det_colours[flip] = np.where(det_colours[flip] == "blue", "yellow", "blue")
        damaged.append((det_colours, det_xy))
    return damaged


def blackout(det_t, frames, t_from, t_to):
    """Remove every detection between t_from and t_to (same clock as det_t): a camera outage."""
    return [(det_colours[:0], det_xy[:0]) if t_from <= t < t_to else (det_colours, det_xy)
            for t, (det_colours, det_xy) in zip(det_t, frames)]


def remove_orange(frames):
    """Remove the orange gate cones (unique landmarks) so only blue and yellow are left."""
    return [(det_colours[det_colours != "orange"], det_xy[det_colours != "orange"])
            for det_colours, det_xy in frames]