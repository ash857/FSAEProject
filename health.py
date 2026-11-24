"""The 'Filter lost?' box: decide whether the filter is searching, tracking or lost."""
import numpy as np

SEARCHING, TRACKING, LOST = 0, 1, 2
NAMES = {SEARCHING: "searching", TRACKING: "tracking", LOST: "lost"}

CONVERGE_SPREAD = 1.0     # m   cloud must be tighter than this ...
CONVERGE_FIT = 0.6        #     ... and at least this fraction of detections must land on map cones ...
CONVERGE_FRAMES = 3       #     ... for this many frames in a row
LOST_FIT = 0.45           #     a frame is 'bad' if the smoothed fit is below this ...
LOST_SPREAD = 3.0         # m   ... or the cloud is wider than this
LOST_FRAMES = 5           #     lost after this many bad frames in a row
MIN_DETECTIONS = 3        #     frames with fewer cones are not judged either way
FIT_EMA = 0.1             #     smoothing of the fit score (about the last 10 frames)
RECOVER_FRAC = 0.2        #     while lost: fraction of particles replaced by fresh random ones each frame


def update_health(status, good_frames, bad_frames, spread, fit, fit_smoothed, n_detections):
    """Decide whether the filter is SEARCHING, TRACKING or LOST. Call once per detection frame"""
    judged = n_detections >= MIN_DETECTIONS and not np.isnan(fit)

    if status != TRACKING:
        looks_converged = judged and spread < CONVERGE_SPREAD and fit > CONVERGE_FIT
        good_frames = good_frames + 1 if looks_converged else 0
        if good_frames >= CONVERGE_FRAMES:
            status, bad_frames = TRACKING, 0
    else:
        is_bad_frame = (judged and fit_smoothed < LOST_FIT) or spread > LOST_SPREAD
        if is_bad_frame:
            bad_frames += 1
        elif judged:
            bad_frames = 0      # a frame with too few detections proves nothing, so it leaves the count alone
        if bad_frames >= LOST_FRAMES:
            status, good_frames = LOST, 0

    return status, good_frames, bad_frames