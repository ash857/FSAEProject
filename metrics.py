import numpy as np

def consistency(colours, xy, frames, x, y, yaw, gate=0.7, max_range=8.0):
    """For each detection frame: the fraction of detections that land within `gate` metres
    of a mapped cone of the SAME colour, when placed on the map using that frame's pose."""
    fit_per_frame = np.full(len(frames), np.nan)

    for i, (det_colours, det_xy) in enumerate(frames):
        within_range = np.hypot(det_xy[:, 0], det_xy[:, 1]) <= max_range
        det_colours, det_xy = det_colours[within_range], det_xy[within_range]
        if len(det_colours) == 0:
            continue

        # Car frame to map frame: rotate by the heading, then add the car's position.
        cos_yaw, sin_yaw = np.cos(yaw[i]), np.sin(yaw[i])
        map_x = x[i] + cos_yaw * det_xy[:, 0] - sin_yaw * det_xy[:, 1]
        map_y = y[i] + sin_yaw * det_xy[:, 0] + cos_yaw * det_xy[:, 1]

        n_matched = 0
        for colour, det_x, det_y in zip(det_colours, map_x, map_y):
            same_colour_cones = xy[colours == colour]
            if len(same_colour_cones) == 0:         # this colour isn't on the map at all
                continue
            distances = np.hypot(same_colour_cones[:, 0] - det_x, same_colour_cones[:, 1] - det_y)
            if distances.min() < gate:
                n_matched += 1

        fit_per_frame[i] = n_matched / len(det_colours)

    return fit_per_frame


def interp_pose(t_query, t, x, y, yaw):
    """Pose (x, y, yaw) at the times `t_query`, interpolated from poses recorded at times `t`.
    """
    x_at = np.interp(t_query, t, x)
    y_at = np.interp(t_query, t, y)
    yaw_at = np.arctan2(np.interp(t_query, t, np.sin(yaw)),
                        np.interp(t_query, t, np.cos(yaw)))
    return x_at, y_at, yaw_at


def dead_reckon(odom, start_pose, t_query):
    """Odometry alone, placed on the map, evaluated at the times t_query."""
    start_x, start_y, start_yaw = start_pose
    rotation = start_yaw - odom["yaw"][0]
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)

    odom_dx = odom["x"] - odom["x"][0]
    odom_dy = odom["y"] - odom["y"][0]
    map_x = start_x + cos_r * odom_dx - sin_r * odom_dy
    map_y = start_y + sin_r * odom_dx + cos_r * odom_dy
    map_yaw = odom["yaw"] - odom["yaw"][0] + start_yaw

    return interp_pose(t_query, odom["t"], map_x, map_y, map_yaw)


def moving_mean(values, window):
    """Average of the last `window` values, ignoring NaN (for a smooth time plot)."""
    smoothed = np.full(len(values), np.nan)
    for i in range(len(values)):
        recent = values[max(0, i - window + 1):i + 1]
        if not np.isnan(recent).all():      # nanmean of all-NaN warns, so skip those
            smoothed[i] = np.nanmean(recent)
    return smoothed