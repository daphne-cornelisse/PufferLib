#!/usr/bin/env python3
"""Sanity-check plots for I24 macro spawn + goal assignment.

Spawns a small number of agents on each highway direction (sparse enough
that they should reach a short/mid goal well before episode end) and
checks that goals lie ahead along heading and on a lane.

  uv run python ocean/drive/plot_macro_sanity.py
  uv run python ocean/drive/plot_macro_sanity.py --agents-per-direction 8 --out-dir logs/drive
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ocean" / "drive"))
from dataset import load_map_json  # noqa: E402

ROAD_LANE = "lane"
SIM_DT = 0.1
EPISODE_LENGTH = 512
CRUISE_SPEED = 20.0  # m/s, conservative highway cruise
GOAL_SHORT_MIN = 20.0
GOAL_SHORT_MAX = 40.0
GOAL_MID_MAX = 600.0
MIN_DISTANCE_TO_GOAL = 5.0
GRAPH_JOIN = 12.0


def _max_seg(pts: list[tuple[float, float]]) -> float:
    mx = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        mx = max(mx, math.hypot(x1 - x0, y1 - y0))
    return mx


def lane_polylines(map_data: dict) -> list[list[tuple[float, float]]]:
    lanes = []
    for road in map_data.get("roads", []):
        word = str(road.get("type", "")).lower()
        if word not in ("lane", "road_lane"):
            continue
        geom = road.get("geometry") or []
        if len(geom) < 2:
            continue
        pts = [(float(p["x"]), float(p["y"])) for p in geom]
        lanes.append(pts)
    return lanes


def edge_polylines(map_data: dict) -> list[list[tuple[float, float]]]:
    edges = []
    for road in map_data.get("roads", []):
        word = str(road.get("type", "")).lower()
        if word not in ("road_edge",):
            continue
        geom = road.get("geometry") or []
        if len(geom) < 2:
            continue
        edges.append([(float(p["x"]), float(p["y"])) for p in geom])
    return edges


def polyline_length(pts: list[tuple[float, float]]) -> float:
    L = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L += math.hypot(x1 - x0, y1 - y0)
    return L


def point_along(pts: list[tuple[float, float]], s: float):
    if s < 0:
        s = 0.0
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= s or math.hypot(x1 - pts[-1][0], y1 - pts[-1][1]) < 1e-9:
            t = 0.0 if seg < 1e-8 else (s - acc) / seg
            if t < 0:
                t = 0.0
            if t > 1:
                t = 1.0
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            heading = math.atan2(y1 - y0, x1 - x0)
            return x, y, heading
        acc += seg
    x, y = pts[-1]
    dx = pts[-1][0] - pts[-2][0]
    dy = pts[-1][1] - pts[-2][1]
    return x, y, math.atan2(dy, dx)


def highway_ends(lanes: list[list[tuple[float, float]]]):
    pts = []
    for lane in lanes:
        pts.append(lane[0])
        pts.append(lane[-1])
    best = -1.0
    a, b = pts[0], pts[1]
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d2 > best:
                best = d2
                a, b = p, q
    return a, b


def snap_to_lane(lanes, x, y):
    best = 1e30
    ox, oy = x, y
    for lane in lanes:
        for (ax, ay), (bx, by) in zip(lane, lane[1:]):
            dx, dy = bx - ax, by - ay
            len2 = dx * dx + dy * dy
            t = 0.0 if len2 < 1e-8 else ((x - ax) * dx + (y - ay) * dy) / len2
            t = min(1.0, max(0.0, t))
            px, py = ax + t * dx, ay + t * dy
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best:
                best = d2
                ox, oy = px, py
    return ox, oy, math.sqrt(best)


def sample_goal_dist(rng: random.Random, dist_to_end: float) -> float:
    u = rng.random()
    if u < 0.5:
        d = GOAL_SHORT_MIN + (u / 0.5) * (GOAL_SHORT_MAX - GOAL_SHORT_MIN)
    elif u < 0.8:
        d = GOAL_SHORT_MAX + ((u - 0.5) / 0.3) * (GOAL_MID_MAX - GOAL_SHORT_MAX)
    else:
        span = max(0.0, dist_to_end - GOAL_MID_MAX)
        d = GOAL_MID_MAX + ((u - 0.8) / 0.2) * span
    d = min(d, dist_to_end)
    floor_d = min(GOAL_SHORT_MIN, dist_to_end)
    d = max(d, floor_d, MIN_DISTANCE_TO_GOAL + 1.0)
    return d


def _heading(pts, at_end=False):
    if at_end:
        a, b = pts[-2], pts[-1]
    else:
        a, b = pts[0], pts[1]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def build_succ(lanes):
    succ = [-1] * len(lanes)
    for i, a in enumerate(lanes):
        ax, ay = a[-1]
        ah = _heading(a, at_end=True)
        best_d, best_dh, best = GRAPH_JOIN, 1e9, -1
        for j, b in enumerate(lanes):
            if i == j:
                continue
            d = math.hypot(b[0][0] - ax, b[0][1] - ay)
            if d > GRAPH_JOIN:
                continue
            dh = abs(((_heading(b) - ah + math.pi) % (2 * math.pi)) - math.pi)
            if dh > 1.2:
                continue
            if d < best_d - 0.1 or (abs(d - best_d) < 0.1 and dh < best_dh):
                best_d, best_dh, best = d, dh, j
        succ[i] = best
    return succ


def walk(lanes, succ, lane_i, s, dist):
    guard = 0
    while guard < 128:
        guard += 1
        L = polyline_length(lanes[lane_i])
        s = min(max(s, 0.0), L)
        rem = L - s
        if dist <= rem + 1e-3:
            x, y, h = point_along(lanes[lane_i], s + dist)
            return x, y, h, lane_i, s + dist
        dist -= rem
        nxt = succ[lane_i]
        if nxt < 0:
            return None
        lane_i, s = nxt, 0.0
    return None


def chain_remaining(lanes, succ, lane_i, s):
    dist = 0.0
    seen = set()
    for _ in range(256):
        L = polyline_length(lanes[lane_i])
        s = min(max(s, 0.0), L)
        dist += L - s
        nxt = succ[lane_i]
        if nxt < 0 or nxt in seen:
            break
        seen.add(nxt)
        lane_i, s = nxt, 0.0
    return dist


def assign_goal(lanes, succ, lane_i, along, x, y, hx, hy, rng):
    d = GOAL_SHORT_MIN + rng.random() * (GOAL_SHORT_MAX - GOAL_SHORT_MIN)
    walked = walk(lanes, succ, lane_i, along, d)
    if walked is None:
        rem = chain_remaining(lanes, succ, lane_i, along)
        walked = walk(lanes, succ, lane_i, along, max(0.0, rem - 0.5))
        if walked is None:
            return x, y, d, 0.0
    gx, gy, _h, _li, _s = walked
    return gx, gy, d, math.hypot(gx - x, gy - y)


def lane_heading(lane) -> float:
    dx = lane[-1][0] - lane[0][0]
    dy = lane[-1][1] - lane[0][1]
    return math.atan2(dy, dx)


def split_directions(lanes, ends):
    (ax, ay), (bx, by) = ends
    px, py = bx - ax, by - ay
    n = math.hypot(px, py) or 1.0
    px, py = px / n, py / n
    a, b = [], []
    for lane in lanes:
        hx = math.cos(lane_heading(lane))
        hy = math.sin(lane_heading(lane))
        if hx * px + hy * py >= 0:
            a.append(lane)
        else:
            b.append(lane)
    return a, b


def spawn_direction(lanes, succ, all_lanes, n_agents, rng_offset=0.0):
    lengths = [polyline_length(l) for l in lanes]
    need = GOAL_SHORT_MAX + 1.0
    spawn_len = []
    for lane, L in zip(lanes, lengths):
        try:
            lane_i = all_lanes.index(lane)
        except ValueError:
            lane_i = 0
        rem0 = chain_remaining(all_lanes, succ, lane_i, 0.0)
        valid = min(L, max(0.0, rem0 - need))
        spawn_len.append(valid)
    total = sum(spawn_len)
    if total <= 0:
        spawn_len = lengths
        total = sum(lengths)
    if total <= 0 or n_agents <= 0:
        return []
    spacing = total / n_agents
    agents = []
    for i in range(n_agents):
        target = (i + 0.5 + rng_offset) * spacing
        while target >= total:
            target -= total
        acc = 0.0
        chosen = lanes[0]
        along = 0.0
        for lane, L in zip(lanes, spawn_len):
            if acc + L >= target:
                chosen = lane
                along = target - acc
                break
            acc += L
            chosen = lane
            along = L
        x, y, heading = point_along(chosen, along)
        agents.append({
            "x": x, "y": y, "heading": heading,
            "lane": chosen, "along": along,
        })
    return agents


def spawn(map_data, n_per_dir: int, seed: int):
    lanes = lane_polylines(map_data)
    if len(lanes) < 1:
        raise SystemExit("no ROAD_LANE polylines in map")
    ends = highway_ends(lanes)
    dir_a, dir_b = split_directions(lanes, ends)
    rng = random.Random(seed)
    succ = build_succ(lanes)
    agents = []
    for di, group in enumerate((dir_a, dir_b)):
        if not group:
            continue
        spawned = spawn_direction(group, succ, lanes, n_per_dir)
        for e in spawned:
            hx, hy = math.cos(e["heading"]), math.sin(e["heading"])
            try:
                lane_i = lanes.index(e["lane"])
            except ValueError:
                lane_i = 0
            gx, gy, sampled_d, dist_end = assign_goal(
                lanes, succ, lane_i, e["along"], e["x"], e["y"], hx, hy, rng
            )
            ahead = (gx - e["x"]) * hx + (gy - e["y"]) * hy
            lateral = abs(-(gx - e["x"]) * hy + (gy - e["y"]) * hx)
            goal_dist = math.hypot(gx - e["x"], gy - e["y"])
            _, _, snap_d = snap_to_lane(lanes, gx, gy)
            agents.append({
                **e,
                "hx": hx,
                "hy": hy,
                "gx": gx,
                "gy": gy,
                "direction": di,
                "ahead": ahead,
                "lateral": lateral,
                "goal_dist": goal_dist,
                "sampled_d": sampled_d,
                "dist_to_end": dist_end,
                "on_road": snap_d,
            })
    return agents, lanes, edge_polylines(map_data), ends


def plot_sanity(agents, lanes, edges, ends, out_dir: Path, n_per_dir: int, episode_length: int):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    reachable = episode_length * SIM_DT * CRUISE_SPEED
    n_ahead = sum(1 for a in agents if a["ahead"] > 0)
    n_reach = sum(1 for a in agents if a["goal_dist"] <= reachable)
    n_on = sum(1 for a in agents if a["on_road"] < 1.0)
    max_off = max((a["on_road"] for a in agents), default=0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    if edges:
        segs = [[(x, y) for x, y in poly] for poly in edges]
        ax.add_collection(LineCollection(segs, colors="#444444", linewidths=0.6, alpha=0.8))
    if lanes:
        segs = [[(x, y) for x, y in poly] for poly in lanes]
        ax.add_collection(LineCollection(segs, colors="#888888", linewidths=0.4, alpha=0.5))
    colors = ["tab:blue", "tab:orange"]
    for a in agents:
        c = colors[a["direction"] % 2]
        ax.plot(a["x"], a["y"], "o", color=c, markersize=5, zorder=3)
        ax.plot(a["gx"], a["gy"], "*", color=c, markersize=9, zorder=3)
        ax.annotate(
            "",
            xy=(a["gx"], a["gy"]),
            xytext=(a["x"], a["y"]),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0, alpha=0.85),
        )
    (ax1, ay1), (ax2, ay2) = ends
    ax.scatter([ax1, ax2], [ay1, ay2], marker="s", c="red", s=30, zorder=4, label="highway ends")
    ax.set_aspect("equal")
    ax.set_title(
        f"I24 macro goals  {n_per_dir}/direction  "
        f"ahead={n_ahead}/{len(agents)}  "
        f"on-road={n_on}/{len(agents)} (max {max_off:.2f}m)  "
        f"reachable@{CRUISE_SPEED:.0f}m/s={n_reach}/{len(agents)}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.plot([], [], "o", color="tab:blue", label="start dir 0")
    ax.plot([], [], "*", color="tab:blue", label="goal dir 0")
    ax.plot([], [], "o", color="tab:orange", label="start dir 1")
    ax.plot([], [], "*", color="tab:orange", label="goal dir 1")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    overview = out_dir / "macro_sanity_goals.png"
    fig.savefig(overview, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dists = [a["goal_dist"] for a in agents]
    axes[0].hist(dists, bins=min(12, max(4, len(agents))), color="tab:blue", edgecolor="white")
    axes[0].axvline(GOAL_SHORT_MAX, color="tab:orange", ls="--", label="short max 200m")
    axes[0].axvline(GOAL_MID_MAX, color="tab:green", ls="--", label="mid max 600m")
    axes[0].axvline(reachable, color="tab:red", ls=":", label=f"reachable {reachable/1000:.1f}km")
    axes[0].set_xlabel("goal distance (m)")
    axes[0].set_ylabel("agents")
    axes[0].set_title("Goal distance")
    axes[0].legend(fontsize=7)

    axes[1].scatter([a["ahead"] for a in agents], [a["lateral"] for a in agents],
                    c=[a["direction"] for a in agents], cmap="coolwarm", s=28)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("ahead along heading (m)")
    axes[1].set_ylabel("lateral |offset| (m)")
    axes[1].set_title("Goal is ahead and on-lane")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(dists, [a["dist_to_end"] for a in agents],
                    c=[a["direction"] for a in agents], cmap="coolwarm", s=28)
    axes[2].plot([0, max(dists + [1])], [0, max(dists + [1])], "k--", lw=0.8)
    axes[2].set_xlabel("goal distance (m)")
    axes[2].set_ylabel("distance to corridor end (m)")
    axes[2].set_title("Goal before corridor end")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"episode={episode_length} steps ({episode_length * SIM_DT:.0f}s)  "
        f"cruise {CRUISE_SPEED:.0f} m/s → {reachable/1000:.1f} km budget  "
        f"max goal {max(dists):.0f} m"
    )
    fig.tight_layout()
    reach = out_dir / "macro_sanity_reach.png"
    fig.savefig(reach, dpi=140)
    plt.close(fig)
    return overview, reach, n_ahead, n_reach, n_on, max_off, len(agents), max(dists) if dists else 0.0, reachable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=Path("resources/drive/i24/I24_map.json"))
    parser.add_argument("--agents-per-direction", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/drive_macro"))
    args = parser.parse_args()

    map_data = load_map_json(args.map)
    agents, lanes, edges, ends = spawn(map_data, args.agents_per_direction, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overview, reach, n_ahead, n_reach, n_on, max_off, n, max_d, budget = plot_sanity(
        agents, lanes, edges, ends, args.out_dir, args.agents_per_direction, args.episode_length
    )
    print(
        f"wrote {overview} {reach}  agents={n}  "
        f"ahead={n_ahead}/{n}  on-road={n_on}/{n} (max {max_off:.2f}m)  "
        f"reachable={n_reach}/{n}  max_goal={max_d:.0f}m  budget={budget:.0f}m"
    )
    if n_ahead < n:
        print("WARNING: some goals are not ahead of the agent heading", file=sys.stderr)
        return 1
    if n_on < n:
        print("WARNING: some goals are off the lane centerline", file=sys.stderr)
        return 1
    if n_reach < n:
        print("WARNING: some goals exceed the conservative reach budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
