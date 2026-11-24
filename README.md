# Where Am I on the Track?

Cone-map localisation for the F:SAE:47 Autonomous 2026 newbie project.

Given a map of the track's cones, a stream of (drifting) odometry and a stream of noisy cone detections, a
particle filter works out where the car is on the map. It works with a **known start pose** and with an
**unknown start** (the car could be anywhere on the track).

## How it works

Odometry says how the car moved, but it drifts. The cone detections say what the car sees, relative to itself,
with noise, false detections and misses. The map says where the cones really are.

The filter keeps thousands of guesses of the car's pose (**particles**, each an x, y and heading) and repeats
this loop once per detection frame:

```
      ┌─────────────────────────────────────────────────────────────────────────┐
      ▼                                                                         │
   PREDICT ──► WEIGHT ──► ESTIMATE ──► HEALTH CHECK ──► RESAMPLE ──► RECOVER ───┘
                                       (sets the        only if      only if the
                                       status)          N_eff < N/2  status is LOST
```

The first four stages run on every frame. Resample and recover are conditional, and they are independent of each other: resampling happens whenever the weights become lopsided, and recovery happens only when the health check says the filter is lost.

| Stage | What happens |
|---|---|
| **Predict** | Every particle is moved by the odometry, plus its own random noise so the guesses spread out. |
| **Weight** | Each particle is scored on how well the detected cones match the map cones, as seen from that particle's pose. Wrong guesses score low. |
| **Estimate** | The weighted average of the particles gives the car's pose, and their spread says how sure the filter is. |
| **Lost?** | A health check (spread, plus how well the cones fit the map) decides whether the filter is searching, tracking or lost. |
| **Resample** | Particles with high weight are copied and low-weight ones dropped, when the weights become concentrated in few particles. |
| **Recover** | If the filter is lost, some fresh particles are spread over the track so it can find itself again. |

**Known vs unknown start:** the only difference is the first cloud of particles. For a known start it is a tight
cluster around the given pose. For an unknown start it is thousands of particles spread along the track.

### Key design choices

- Odometry is used only for the **motion between two moments**, applied in each particle's own heading, so its arbitrary origin doesn't matter.
- Detections are compared with the map in **range and bearing**, which is what the camera measures.
- Each detection is matched to its **best same-colour map cone**, with a gate and a small floor value so a false detection can't wipe out a good particle.
- Scores are combined as **logs**, to avoid numbers too small to store.
- Detection timestamps use the **header stamp** (when the sensor measured), and the odometry is interpolated to that time.

## Parameters

The values below control how the filter behaves. Most are **estimates**: the sensor and motion noise were not measured from the data, so they are the first things to tune if the filter misbehaves.

| Parameter | Value | What it controls | Basis |
|---|---|---|---|
| Particles (N) | 500 for a known start; 3000 by default and 5000 in `step9_check` for an unknown start | How many guesses the filter holds | Chosen by trying values; too few fail on an unknown start |
| Known-start spread | 0.3 m, 0.1 rad | How tight the starting cloud is | Estimate |
| Unknown-start jitter | 0.6 m, 0.6 rad | How wide the cloud is around the track centreline | Estimate |
| Motion noise | 0.05 × distance + 0.005 m; 0.10 × turn + 0.01 × distance + 0.002 rad | How much each particle's move is randomised | Estimate (odometry drift not measured) |
| Range noise | 0.30 + 0.05 × range (m) | How much a cone's measured distance is trusted | Estimate (sensor noise not measured) |
| Bearing noise | 0.07 rad | How much a cone's measured angle is trusted | Estimate |
| Match gate | 9.21 | Largest mismatch counted as a match | Derived: the 99% limit for 2 degrees of freedom |
| No-match floor | 0.01 | Score given to a detection with no match | Chosen, so one false detection cannot zero a particle |
| Resample threshold | N_eff < N/2 | When to copy and drop particles | Standard rule of thumb |
| Consistency gate | 0.7 m | How close a detection must land to a map cone to count | Derived: under half the closest same-colour cone spacing (about 1.57 m) |
| Consistency range | 8 m | Detections farther than this are ignored in the score | Estimate (far cones are noisier) |
| Converged | spread < 1 m and fit > 0.6, for 3 frames in a row | When "searching" becomes "tracking" | Tuned on simulated data |
| Lost | smoothed fit < 0.45 or spread > 3 m, for 5 frames in a row | When "tracking" becomes "lost" | Tuned on simulated data |
| Recovery | 20% of particles replaced | How many guesses are replaced when lost | Chosen |

The health thresholds were tuned on simulated data. On the real bag the filter was never lost, so the lost and recovery settings have not been exercised on real data.

### Assumptions

- Noise values and thresholds are estimates, not measured.
- The camera is assumed to be at the car's origin.
- The bag has no ground-truth pose, so accuracy in metres can't be measured on it. Map consistency, 
  agreement and deliberate fault injection are used instead.

## Architecture

The code is split into four layers. Data flows one way, from the files on disk down to the results.

```
  cone_map.csv                      rosbag (odometry + cone detections)
       │                                        │
  particle_filter.load_map               data_io.load_bag
       │                                        │
       ▼                                        ▼
  map: colours, xy             odom: t, x, y, yaw      det_t, frames
       │                                        │
       │        faults.py can corrupt these     │
       │        inputs on purpose, for testing  │
       └───────────────────┬────────────────────┘
                           ▼
             ┌───────────────────────────────┐
             │          test.run             │   one pass over the detection frames;
             │  holds the state: particles,  │   calls the functions below in order
             │  weights, health counters     │ ───► optional callback ──► watch.py (draws, never changes results)
             └──────────────┬────────────────┘
                            │ calls
        ┌───────────────────┼─────────────────────────────┐
        ▼                   ▼                             ▼
  particle_filter.py    metrics.consistency ──►     health.update_health
  body_increment        "do the cones fit the       "searching, tracking
  predict               map?" score                  or lost?"
  log_likelihood, weight
  estimate, n_eff           │
  systematic_resample       │
  inject_random             │
                            │
                            ▼
        results: one row per frame (t, x, y, yaw, spread, n_eff, status)
                            │
                            ▼
        check_map_consistency, break
```

| Layer | Files | Role |
|---|---|---|
| **Input** | `data_io.py`, `pf.load_map` | Turn the bag and the map file into plain numpy arrays. Nothing downstream knows about ROS. |
| **Filter** | `particle_filter.py`, `health.py` | The algorithm, as small functions. They take arrays in and give arrays out, and hold no state of their own. |
| **Orchestration** | `test.run` | Owns the state (particles, weights, health counters) and calls the filter functions in order, once per frame. |
| **Evaluation and display** | `metrics.py`, `faults.py`, `watch.py` | Check, stress-test and show the result. They read the results and don't change the filter. |

### What gets passed around

| Data | Shape and contents |
|---|---|
| `odom` | dict of arrays `t`, `x`, `y`, `yaw` (yaw unwrapped, so it never jumps by 2π) |
| `det_t`, `frames` | `det_t` is one time per frame. `frames` is a list of `(colours, xy)`, where `colours` are names and `xy` is (forward, left) in metres |
| map | `colours` (names) and `xy` (positions) of the 33 cones |
| `particles` | array of shape (N, 3): x, y, heading for each particle |
| `weights` | array of shape (N,), summing to 1 |
| results | array with one row per frame: `(t, x, y, yaw, spread, n_eff, status) |

## Files

### The filter

| File | What it is for |
|---|---|
| `particle_filter.py` | The particle filter itself: loading the map, creating particles, predict, scoring, weighting, estimate, resampling, recovery |
| `health.py` | Decides whether the filter is searching, tracking or lost |
| `test.py` | `run(...)`, which ties everything into the full loop. Running it directly processes the whole bag |

### Data and tools

| File | What it is for |
|---|---|
| `data_io.py` | Reads the rosbag into odometry arrays and detection frames, and caches the result |
| `metrics.py` | Map consistency, odometry-only path and pose interpolation (used to check the result) |
| `faults.py` | Corrupts the data on purpose (bias, dropped or false detections, flipped colours, blackout) for testing |
| `watch.py` | Live visualiser: particles, cones seen, filter path against odometry-only path, spread graph |


## Installation

Requires **Python 3** and these packages (listed in `requirements.txt`):

- `numpy`
- `matplotlib`
- `rosbags` (reads ROS 2 bags without a ROS install)

```bash
pip install -r requirements.txt
```

## Running it

```bash
python test.py        # run the filter (unknown start) on the whole bag, then plot the path and spread
python watch.py               # watch it live: particles, cones seen, odometry-only path, spread graph
python check_map_consistency.py         # check the result against the map
python break.py  # break the inputs on purpose
```

The first run reads the bag and saves a cache (`data/cache_*.npz`), so later runs start straight away.