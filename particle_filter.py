import numpy as np


def load_map(path="data/cone_map.csv"):
    """Returns colours (n,) and cone positions xy (n, 2)."""
    colours, pts = [], []
    with open(path) as f:
        next(f)                                         
        for line in f:                                  
            _, colour, x, y = line.strip().split(",")   
            colours.append(colour)
            pts.append((float(x), float(y)))
    return np.array(colours), np.array(pts)


def corridor_particles(colours, xy, N, rng, pos_sigma=0.6, head_sigma=0.6):
    """Make N random particles [x, y, heading] scattered along the track.

    pos_sigma  : metres.  Std dev of the jitter added to x and y.
    head_sigma : radians. Std dev of the jitter added to the heading.
    """
    blue, yellow = xy[colours == "blue"], xy[colours == "yellow"]

    centreline = []
    for b in blue:
        dist_to_yellow = np.hypot(yellow[:, 0] - b[0], yellow[:, 1] - b[1])
        nearest_yellow = yellow[np.argmin(dist_to_yellow)]
        centreline.append((b + nearest_yellow) / 2)
    centreline = np.array(centreline)

    # Arrow from each centreline point to the next; roll wraps the last back to the first.
    segment_vec = np.roll(centreline, -1, axis=0) - centreline

    segment = rng.integers(0, len(centreline), N)
    fraction = rng.random(N)[:, None]          # [:, None] makes it a column so it scales both x and y
    pos = centreline[segment] + fraction * segment_vec[segment]
    pos += rng.normal(0, pos_sigma, pos.shape)

    heading = (np.arctan2(segment_vec[segment, 1], segment_vec[segment, 0])
               + rng.normal(0, head_sigma, N))

    return np.column_stack([pos, heading])      

def known_particles(pose, N, rng, sigma_xy=0.3, sigma_yaw=0.1):
    """Known start: a tight cloud around pose = (x, y, yaw)."""
    x, y, yaw = pose
    return np.column_stack([rng.normal(x, sigma_xy, N),
                            rng.normal(y, sigma_xy, N),
                            wrap_angle(rng.normal(yaw, sigma_yaw, N))])

def wrap_angle(a):
    """Keep an angle in [-pi, pi). 190 degrees becomes -170 degrees."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def body_increment(p0, p1):
    """Motion from odometry pose p0 to p1, measured in p0's own frame."""
    odom_dx, odom_dy = p1[0] - p0[0], p1[1] - p0[1]
    cos_yaw, sin_yaw = np.cos(p0[2]), np.sin(p0[2])

    # Rotate the odometry-frame shift by -yaw to express it in the car's frame.
    forward = cos_yaw * odom_dx + sin_yaw * odom_dy
    left = -sin_yaw * odom_dx + cos_yaw * odom_dy
    turn = wrap_angle(p1[2] - p0[2])
    return forward, left, turn


def predict(particles, forward, left, turn, rng):
    """PREDICT: move every particle by the odometry increment plus its OWN random noise.

    Noise grows with how far the car moved: 5% of the distance (plus 5 mm) for
    position, and 10% of the turn plus 0.01 rad per metre (plus 0.002 rad) for heading.
    """
    n = len(particles)
    dist = np.hypot(forward, left)
    trans_noise = 0.05 * dist + 0.005                       # metres
    rot_noise = 0.10 * abs(turn) + 0.01 * dist + 0.002      # radians

    noisy_forward = forward + rng.normal(0, trans_noise, n)
    noisy_left = left + rng.normal(0, trans_noise, n)
    noisy_turn = turn + rng.normal(0, rot_noise, n)

    heading = particles[:, 2]   
    particles[:, 0] += np.cos(heading) * noisy_forward - np.sin(heading) * noisy_left
    particles[:, 1] += np.sin(heading) * noisy_forward + np.cos(heading) * noisy_left
    particles[:, 2] = wrap_angle(heading + noisy_turn)
    return particles


def log_likelihood(particles, cone_colours, cone_xy, det_colours, det_xy,
                   sigma_r0=0.30, sigma_r1=0.05, sigma_phi=0.07, gate=9.21, floor=0.01):
    """Score each particle by how well it explains this frame's detections (log scale).

    For each particle and each detection, find the map cone of the same colour that
    matches best in range and bearing. A good match scores high. A detection with no
    good match gets the small constant `floor`, so one false detection can't zero out
    a particle. Scores multiply over detections, which means the logs add.

    sigma_r0, sigma_r1 : range noise in metres: sigma_r = sigma_r0 + sigma_r1 * range
    sigma_phi          : bearing noise in radians
    gate               : a mismatch above this counts as "no match" (9.21 is the 99% chi-squared value for 2 dof)
    floor              : likelihood given to a detection with no match
    """
    log_score = np.zeros(len(particles))
    px, py, heading = particles[:, 0], particles[:, 1], particles[:, 2]
    cos_h, sin_h = np.cos(heading)[:, None], np.sin(heading)[:, None]

    for colour in ("blue", "yellow", "orange"):
        map_cones = cone_xy[cone_colours == colour]
        detections = det_xy[det_colours == colour]
        if len(map_cones) == 0 or len(detections) == 0:
            continue

        # Expected: where each particle would see each map cone. Shape (n_particles, n_cones).
        dx = map_cones[:, 0] - px[:, None]
        dy = map_cones[:, 1] - py[:, None]
        ahead = cos_h * dx + sin_h * dy
        left = -sin_h * dx + cos_h * dy
        expected_range = np.hypot(ahead, left)
        expected_bearing = np.arctan2(left, ahead)

        # Observed: what the camera reported. Shape (n_detections,).
        seen_range = np.hypot(detections[:, 0], detections[:, 1])
        seen_bearing = np.arctan2(detections[:, 1], detections[:, 0])

        # Every (particle, detection, cone) combination. Shape (n_particles, n_detections, n_cones).
        range_err = seen_range[None, :, None] - expected_range[:, None, :]
        bearing_err = wrap_angle(seen_bearing[None, :, None] - expected_bearing[:, None, :])
        range_sigma = (sigma_r0 + sigma_r1 * seen_range)[None, :, None]
        mismatch = (range_err / range_sigma) ** 2 + (bearing_err / sigma_phi) ** 2

        best_mismatch = mismatch.min(axis=2)       # best cone for each (particle, detection)
        likelihood = np.where(best_mismatch < gate, np.exp(-0.5 * best_mismatch), floor)
        log_score += np.log(likelihood).sum(axis=1)

    return log_score


def weight(weights, log_score):
    """WEIGHT + NORMALISE: multiply old weights by the likelihood, then rescale to sum to 1.

    Works in log space: multiplying becomes adding, which avoids numbers too small to store.
    `log_score` is the output of log_likelihood.
    """
    log_weights = np.log(weights + 1e-300) + log_score      # 1e-300 stops log(0) giving -inf
    log_weights -= log_weights.max()                        # best particle becomes 0, so exp() can't underflow
    new_weights = np.exp(log_weights)
    return new_weights / new_weights.sum()

def estimate(particles, weights):
    """ESTIMATE: collapse the cloud into one pose, plus how spread out it is."""
    px, py, heading = particles[:, 0], particles[:, 1], particles[:, 2]

    x = np.sum(weights * px)
    y = np.sum(weights * py)

    # Heading is an angle, so average its sin and cos, never the raw numbers.
    yaw = np.arctan2(np.sum(weights * np.sin(heading)),
                     np.sum(weights * np.cos(heading)))

    dist_sq = (px - x) ** 2 + (py - y) ** 2
    spread = np.sqrt(np.sum(weights * dist_sq))
    return x, y, yaw, spread


def n_eff(weights):
    """Effective number of particles that really matter (N if equal, 1 if one has everything)."""
    return 1.0 / np.sum(weights ** 2)


def systematic_resample(particles, weights, rng):
    """RESAMPLE: copy heavy particles, drop light ones."""
    N = len(weights)
    pointers = (rng.random() + np.arange(N)) / N
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0                    # rounding can leave this at 0.9999999
    chosen = np.searchsorted(cumulative_weights, pointers)
    return particles[chosen].copy(), np.full(N, 1.0 / N)


def inject_random(particles, weights, frac, colours, xy, rng):
    """RECOVER: replace a fraction of the particles with fresh ones spread over the whole track."""
    N = len(particles)
    n_new = int(round(frac * N))
    replace = rng.choice(N, n_new, replace=False)       # which particles to overwrite
    particles[replace] = corridor_particles(colours, xy, n_new, rng)
    return particles, np.full(N, 1.0 / N)