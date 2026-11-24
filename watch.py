from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from particle_filter import load_map
from data_io import load_bag
from test import run

DET_COLOURS = {"blue": "tab:blue", "yellow": "goldenrod", "orange": "tab:orange"}
STAGE_TEXT = {"predict": "PREDICT: every particle moved by the odometry",
              "weight": "WEIGHT: darker = fits the cones better",
              "resample": "RESAMPLE: heavy particles copied, light ones dropped"}
CONFIDENT_SPREAD = 1.0      # metres: below this the particle cloud counts as converged


def _smooth(values, window):
    """Moving average over `window` samples, centred, same length as the input."""
    return np.convolve(values, np.ones(window) / window, mode="same")


def _odometry_traces(odom, odom_start):
    """The odometry-only path placed on the map, plus speed and turn rate over time.
    """
    t = odom["t"] - odom["t"][0]

    start_x, start_y, start_yaw = odom_start
    rotation = start_yaw - odom["yaw"][0]
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    odom_dx, odom_dy = odom["x"] - odom["x"][0], odom["y"] - odom["y"][0]
    path_x = start_x + cos_r * odom_dx - sin_r * odom_dy
    path_y = start_y + sin_r * odom_dx + cos_r * odom_dy

    dt = np.diff(t)
    window = max(3, int(0.5 / np.median(dt)))               # about 0.5 s of samples
    speed = _smooth(np.hypot(np.diff(odom["x"]), np.diff(odom["y"])) / dt, window)
    turn_rate = _smooth(np.diff(odom["yaw"]) / dt, window)
    step_times = 0.5 * (t[1:] + t[:-1])
    return path_x, path_y, speed, turn_rate, step_times


def _build_figure(colours, xy, speed, turn_rate):
    """Create the window: the map on the left, three time graphs on the right, all empty.

    Returns (fig, ui), where ui holds the objects that draw() updates.
    """
    fig = plt.figure(figsize=(14, 7.5))
    grid = GridSpec(3, 2, width_ratios=[1.15, 1], hspace=0.45, wspace=0.18)
    map_ax = fig.add_subplot(grid[:, 0])
    spread_ax = fig.add_subplot(grid[0, 1])
    speed_ax = fig.add_subplot(grid[1, 1])
    turn_ax = fig.add_subplot(grid[2, 1])

    for name in ["blue", "yellow", "orange"]:
        is_colour = colours == name
        map_ax.scatter(xy[is_colour, 0], xy[is_colour, 1], c=DET_COLOURS[name], s=35, zorder=3)
    particles = map_ax.scatter([], [], c=[], s=10, cmap="Greens", vmin=0, vmax=1, zorder=2)
    rings = map_ax.scatter([], [], s=90, facecolors="none", linewidths=1.5, zorder=4)
    odom_line, = map_ax.plot([], [], "r--", lw=0.9, zorder=4, label="odometry only")
    trail_line, = map_ax.plot([], [], "k-", lw=1.2, zorder=5, label="particle filter")
    car, = map_ax.plot([], [], "r^", ms=12, zorder=6)
    map_ax.set_xlim(xy[:, 0].min() - 2, xy[:, 0].max() + 2)
    map_ax.set_ylim(xy[:, 1].min() - 2, xy[:, 1].max() + 2)
    map_ax.set_aspect("equal")
    map_ax.legend(loc="upper right", fontsize=8)

    spread_line, = spread_ax.plot([], [], color="tab:green", lw=1.2)
    spread_ax.axhline(CONFIDENT_SPREAD, color="gray", lw=0.8, ls="--")
    spread_ax.text(0.99, 1.15, "converged below 1 m", transform=spread_ax.get_yaxis_transform(),
                   ha="right", fontsize=8, color="gray")
    spread_ax.set_yscale("log"); spread_ax.set_ylim(0.01, 15); spread_ax.set_ylabel("spread (m)")
    spread_ax.set_title("How unsure is the filter?", fontsize=10, loc="left")

    speed_line, = speed_ax.plot([], [], color="tab:blue", lw=1.0)
    speed_ax.set_ylim(0, 1.15 * speed.max()); speed_ax.set_ylabel("speed (m/s)")
    speed_ax.set_title("Odometry: speed", fontsize=10, loc="left")

    turn_line, = turn_ax.plot([], [], color="tab:orange", lw=1.0)
    turn_limit = 1.15 * np.abs(turn_rate).max()
    turn_ax.set_ylim(-turn_limit, turn_limit)
    turn_ax.set_ylabel("turn rate (rad/s)"); turn_ax.set_xlabel("time (s)")
    turn_ax.set_title("Odometry: turning", fontsize=10, loc="left")

    for axis in (spread_ax, speed_ax, turn_ax):
        axis.set_xlim(0, 20); axis.grid(alpha=0.3)

    ui = SimpleNamespace(map_ax=map_ax, spread_ax=spread_ax, speed_ax=speed_ax, turn_ax=turn_ax,
                         particles=particles, rings=rings, odom_line=odom_line, trail_line=trail_line,
                         car=car, spread_line=spread_line, speed_line=speed_line, turn_line=turn_line)
    return fig, ui


def watch(odom, det_t, frames, colours, xy, N=3000, every=2, pause=0.001,
          slow_frames=6, slow_pause=1.2, zoom=None, save_dir=None, 
          odom_start=(0.3, 0.25, -0.06)):
    """Run the filter and redraw it as it goes.

    Left: the map, particles, the cones seen this frame, the filter's path and the odometry-only path.
    Right: the filter's spread, then the odometry's speed and turn rate, growing with time.

    slow_frames : for the first this-many frames, show every stage and wait slow_pause seconds on each.
    every       : after that, draw only every `every`-th frame (the weight stage).
    zoom        : if set (metres), the map view follows the estimate with this half-width.
    odom_start  : pose used to place the odometry-only path on the map (from the start-pose search).
    """

    t0 = odom["t"][0]
    odom_t = odom["t"] - t0
    path_x, path_y, speed, turn_rate, step_times = _odometry_traces(odom, odom_start)
    fig, ui = _build_figure(colours, xy, speed, turn_rate)

    trail_x, trail_y = [], []
    history_t, history_spread = [], []
    n_saved = 0

    def draw(frame_idx, t, particles, weights, est, n_effective, det_colours, det_xy, stage):
        nonlocal n_saved
        x, y, yaw, spread = est
        elapsed = t - t0
        confident = spread < CONFIDENT_SPREAD

        if stage == "weight":               # recorded for every frame, even those that aren't drawn
            history_t.append(elapsed); history_spread.append(spread)
            if confident:
                trail_x.append(x); trail_y.append(y)

        is_slow = frame_idx < slow_frames
        if not is_slow and (stage != "weight" or frame_idx % every):
            return
        if not plt.fignum_exists(fig.number):
            raise SystemExit

        ui.particles.set_offsets(particles[:, :2])
        ui.particles.set_array(weights / weights.max())

        if stage != "predict" and len(det_colours) and confident:
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            map_x = x + cos_yaw * det_xy[:, 0] - sin_yaw * det_xy[:, 1]
            map_y = y + sin_yaw * det_xy[:, 0] + cos_yaw * det_xy[:, 1]
            ui.rings.set_offsets(np.column_stack([map_x, map_y]))
            ui.rings.set_edgecolors([DET_COLOURS[colour] for colour in det_colours])
        else:
            ui.rings.set_offsets(np.empty((0, 2)))

        if stage == "weight":
            if confident:
                ui.trail_line.set_data(trail_x, trail_y)
                ui.car.set_data([x], [y])
            else:
                ui.car.set_data([], [])     # an average of several rival guesses isn't a real position

        n_odom = np.searchsorted(odom_t, elapsed)           # odometry samples up to now
        ui.odom_line.set_data(path_x[:n_odom], path_y[:n_odom])
        n_steps = np.searchsorted(step_times, elapsed)
        ui.speed_line.set_data(step_times[:n_steps], speed[:n_steps])
        ui.turn_line.set_data(step_times[:n_steps], turn_rate[:n_steps])
        ui.spread_line.set_data(history_t, history_spread)
        for axis in (ui.spread_ax, ui.speed_ax, ui.turn_ax):
            axis.set_xlim(0, max(20, 1.05 * elapsed))
        if zoom and confident:
            ui.map_ax.set_xlim(x - zoom, x + zoom); ui.map_ax.set_ylim(y - zoom, y + zoom)

        caption = ("START: first frame, cloud spread over the track"
                   if (frame_idx == 0 and stage == "predict") else STAGE_TEXT[stage])
        if not confident and stage != "predict":
            caption += "\nNOT CONVERGED: several rival guesses, so no single position is shown"
        ui.map_ax.set_title(f"frame {frame_idx}  t={elapsed:5.1f}s  spread={spread:5.2f} m  "
                            f"n_eff={n_effective:6.0f}  cones={len(det_colours)}\n{caption}", fontsize=10)

        if save_dir:
            fig.savefig(f"{save_dir}/frame_{n_saved:05d}.png", dpi=80)
            n_saved += 1
        plt.pause(slow_pause if is_slow else pause)

    plt.ion()
    results = run(odom, det_t, frames, colours, xy, N=N, callback=draw)
    plt.ioff(); plt.show()
    return results


if __name__ == "__main__":
    colours, xy = load_map()
    odom, det_t, frames = load_bag("data")
    watch(odom, det_t, frames, colours, xy, N=3000, every=10, slow_frames=5, slow_pause=0.75)