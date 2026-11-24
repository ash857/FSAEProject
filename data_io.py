"""Read the rosbag once, keep only what we need, cache it so later runs are instant."""
import numpy as np
from pathlib import Path

COLOUR_NAMES = ["blue", "yellow", "orange"]          # code 0, 1, 2


def _stamp(msg):
    """Header stamp = when the sensor measured it (better than when the recorder received it)."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _read_bag(data_dir, bag_name, odom_topic, det_topic):
    """Read the raw messages. Returns (odom_rows, det_t, frames).

    odom_rows : (M, 4) array of (t, x, y, yaw), with yaw wrapped to -pi..pi
    det_t     : (F,) header time of each detection frame
    frames    : F arrays of shape (D, 3), each row (colour_code, x, y); small and big orange both count as 2
    """
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    custom_types = {}
    for msg_file in Path(data_dir, "msg").glob("*/msg/*.msg"):   # our own message types
        package = msg_file.parent.parent.name
        custom_types.update(get_types_from_msg(msg_file.read_text(), f"{package}/msg/{msg_file.stem}"))
    typestore.register(custom_types)

    odom_rows, det_t, frames = [], [], []
    with Reader(Path(data_dir, bag_name)) as reader:
        connections = [c for c in reader.connections if c.topic in (odom_topic, det_topic)]
        for connection, _, raw in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(raw, connection.msgtype)
            if connection.topic == odom_topic:
                q, p = msg.pose.pose.orientation, msg.pose.pose.position
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))   # quaternion -> yaw
                odom_rows.append((_stamp(msg), p.x, p.y, yaw))
            else:
                cones = []
                for code, points in ((0, msg.blue), (1, msg.yellow), (2, msg.small_orange), (2, msg.big_orange)):
                    cones += [(code, pt.x, pt.y) for pt in points]
                det_t.append(_stamp(msg))
                frames.append(np.array(cones, float).reshape(-1, 3))
    return np.array(odom_rows, float), np.array(det_t), frames


def load_bag(data_dir="data", bag_name="gokart_rosbag",
             odom_topic="/zed/zed_node/odom", det_topic="/cone_detection_fused"):
    """Returns (odom, det_t, frames).
    odom: dict of arrays t, x, y, yaw (yaw unwrapped so it never jumps by 2*pi).
    det_t: (F,) time of each detection frame.
    frames: list of (colours (D,) as names, xy (D, 2)); x forward, y left, metres.
    """
    cache = Path(data_dir, "cache_" + det_topic.strip("/").replace("/", "_") + ".npz")
    if not cache.exists():
        odom_rows, det_t, frames = _read_bag(data_dir, bag_name, odom_topic, det_topic)
        odom_rows = odom_rows[np.argsort(odom_rows[:, 0])]
        order = np.argsort(det_t)
        det_t, frames = det_t[order], [frames[i] for i in order]
        # An .npz can't hold a list of different-length arrays, so stack them and keep the cut points.
        offsets = np.concatenate([[0], np.cumsum([len(f) for f in frames])])
        np.savez(cache, odom=odom_rows, det_t=det_t, offsets=offsets,
                 pts=np.vstack(frames) if offsets[-1] else np.zeros((0, 3)))

    cached = np.load(cache)
    odom_rows = cached["odom"]
    odom = {"t": odom_rows[:, 0], "x": odom_rows[:, 1], "y": odom_rows[:, 2], "yaw": np.unwrap(odom_rows[:, 3])}
    offsets, all_cones = cached["offsets"], cached["pts"]
    colour_names = np.array(COLOUR_NAMES)
    frames = []
    for i in range(len(cached["det_t"])):
        cones = all_cones[offsets[i]:offsets[i + 1]]
        frames.append((colour_names[cones[:, 0].astype(int)], cones[:, 1:3]))
    return odom, cached["det_t"], frames