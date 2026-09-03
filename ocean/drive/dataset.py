"""Convert Waymo Open Motion Dataset (WOMD) JSON maps to binary format for PufferLib drive env.

Step 0: Download the preprocessed JSON scenarios from HuggingFace e.g.,
  https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train_1000

  uv pip install huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='daphne-cornelisse/pufferdrive_womd_train_1000', repo_type='dataset', local_dir='drive_data')"

Step 1: Unzip to get folder with .json files
    mkdir -p drive_data/training
    tar xzf drive_data/pufferdrive_womd_train_1000.tar.gz --strip-components=1 -C drive_data/training/

Step 2: Process to map binaries
  python ocean/drive/dataset.py --data_folder drive_data/training --output_dir drive_data/binaries

I24 (macro) single-map conversion:
  python ocean/drive/dataset.py --input resources/drive/I24_map.json --output resources/drive/I24_map.bin --preview resources/drive/I24_map.png
"""

import json
import struct
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

TRAJECTORY_LENGTH = 91


def calculate_area(p1, p2, p3):
    """Calculate the area of the triangle using the determinant method."""
    return 0.5 * abs(
        (p1["x"] - p3["x"]) * (p2["y"] - p1["y"])
        - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"])
    )


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if (
                area < polyline_reduction_threshold
                and dist(point1, point3) <= max_segment_length
            ):
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


I24_TYPE_TO_WORD = {
    "LANE": "lane",
    "ROAD_LINE": "road_line",
    "ROAD_EDGE": "road_edge",
    "STOP_SIGN": "stop_sign",
    "CROSSWALK": "crosswalk",
    "SPEED_BUMP": "speed_bump",
    "DRIVEWAY": "driveway",
}


def _polyline_to_geometry(polyline):
    geometry = []
    for point in polyline:
        if isinstance(point, dict):
            geometry.append({
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "z": float(point.get("z", 0.0)),
            })
        else:
            geometry.append({
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if len(point) > 2 else 0.0,
            })
    return geometry


def i24_scenario_to_map_data(scenario, include_objects=False):
    """Convert an I24 JSON scenario (map_features polylines) to WOMD-style map_data.

    Macro traffic sim spawns agents in C from NUM_AGENTS, so objects are omitted
    unless include_objects is set.
    """
    roads = []
    for feat in scenario.get("map_features", []):
        geometry = _polyline_to_geometry(feat.get("polyline", []))
        if len(geometry) < 2:
            continue
        feat_type = str(feat.get("type", "LANE")).upper()
        roads.append({
            "type": I24_TYPE_TO_WORD.get(feat_type, "lane"),
            "map_element_id": 2 if feat_type == "LANE" else 15 if feat_type == "ROAD_EDGE" else 0,
            "geometry": geometry,
            "width": 0.0,
            "length": 0.0,
            "height": 0.0,
            "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
            "mark_as_expert": 0,
        })

    objects = []
    if include_objects:
        for obj in scenario.get("objects", []):
            track = obj.get("track") or []
            state = obj.get("state") or {}
            if not track and state:
                track = [state]
            positions, velocities, headings, valids = [], [], [], []
            for step in track:
                pos = step.get("position", [0.0, 0.0, 0.0])
                vel = step.get("velocity", [0.0, 0.0, 0.0])
                if isinstance(pos, dict):
                    pos = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
                if isinstance(vel, dict):
                    vel = [vel.get("x", 0.0), vel.get("y", 0.0), vel.get("z", 0.0)]
                positions.append({"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]) if len(pos) > 2 else 0.0})
                velocities.append({"x": float(vel[0]), "y": float(vel[1]), "z": float(vel[2]) if len(vel) > 2 else 0.0})
                headings.append(float(step.get("heading", 0.0)))
                valids.append(int(bool(step.get("valid", True))))
            geom = obj.get("geometry") or {}
            objects.append({
                "type": obj.get("type", "vehicle"),
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": valids,
                "width": float(geom.get("width", 2.0)),
                "length": float(geom.get("length", 4.5)),
                "height": float(geom.get("height", 1.5)),
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            })

    return {"objects": objects, "roads": roads}


def load_map_json(path):
    """Load WOMD dict or I24 list-of-scenarios JSON into WOMD-style map_data."""
    with open(path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"{path} is an empty list")
        return i24_scenario_to_map_data(raw[0])
    if isinstance(raw, dict) and "map_features" in raw and "roads" not in raw:
        return i24_scenario_to_map_data(raw)
    return raw


def write_map_preview_png(map_data, output_file, size=2048, pad=0.04):
    """Write a top-down RGB PNG of road polylines (no extra deps)."""
    import zlib

    xs, ys = [], []
    polylines = []
    for road in map_data.get("roads", []):
        geom = road.get("geometry", [])
        if len(geom) < 2:
            continue
        pts = [(float(p["x"]), float(p["y"])) for p in geom]
        polylines.append((road.get("type", "lane"), pts))
        xs.extend(p[0] for p in pts)
        ys.extend(p[1] for p in pts)
    if not xs:
        raise ValueError("no road geometry to preview")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    span = max(span_x, span_y)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    scale = (1.0 - 2.0 * pad) * (size - 1) / span

    pixels = bytearray(size * size * 3)
    bg = (18, 22, 28)
    for i in range(0, len(pixels), 3):
        pixels[i:i + 3] = bg

    def put(px, py, rgb):
        if 0 <= px < size and 0 <= py < size:
            idx = ((size - 1 - py) * size + px) * 3
            pixels[idx:idx + 3] = rgb

    def draw_line(x0, y0, x1, y1, rgb):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            put(x0, y0, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def to_px(x, y):
        px = int(round((size - 1) * 0.5 + (x - cx) * scale))
        py = int(round((size - 1) * 0.5 + (y - cy) * scale))
        return px, py

    colors = {
        "lane": (255, 214, 64),
        "road_line": (230, 230, 230),
        "road_edge": (180, 180, 180),
    }
    for road_type, pts in polylines:
        rgb = colors.get(road_type, (255, 214, 64))
        for a, b in zip(pts, pts[1:]):
            x0, y0 = to_px(a[0], a[1])
            x1, y1 = to_px(b[0], b[1])
            draw_line(x0, y0, x1, y1, rgb)

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + bytes(pixels[y * size * 3:(y + 1) * size * 3]) for y in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(output_file, "wb") as f:
        f.write(png)


def convert_json_file(json_path, binary_path, preview_path=None):
    map_data = load_map_json(json_path)
    save_map_binary(map_data, str(binary_path), 0)
    n_obj = len(map_data.get("objects", []))
    n_road = len(map_data.get("roads", []))
    print(f"Wrote {binary_path} ({n_obj} objects, {n_road} roads)")
    if preview_path:
        write_map_preview_png(map_data, str(preview_path))
        print(f"Wrote {preview_path}")


def save_map_binary(map_data, output_file, unique_map_id):
    """Save map data in a binary format readable by C."""
    with open(output_file, "wb") as f:
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))

        # Write objects
        for obj in map_data.get("objects", []):
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))
            f.write(struct.pack("i", TRAJECTORY_LENGTH))

            positions = obj.get("position", [])
            for coord in ["x", "y", "z"]:
                for i in range(TRAJECTORY_LENGTH):
                    pos = (
                        positions[i]
                        if i < len(positions)
                        else {"x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    f.write(struct.pack("f", float(pos.get(coord, 0.0))))

            velocities = obj.get("velocity", [])
            for coord in ["x", "y", "z"]:
                for i in range(TRAJECTORY_LENGTH):
                    vel = (
                        velocities[i]
                        if i < len(velocities)
                        else {"x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    f.write(struct.pack("f", float(vel.get(coord, 0.0))))

            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{TRAJECTORY_LENGTH}f",
                    *[
                        float(headings[i]) if i < len(headings) else 0.0
                        for i in range(TRAJECTORY_LENGTH)
                    ],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{TRAJECTORY_LENGTH}i",
                    *[
                        int(valids[i]) if i < len(valids) else 0
                        for i in range(TRAJECTORY_LENGTH)
                    ],
                )
            )

            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for road in map_data.get("roads", []):
            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15

            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)

            if 0 <= road_type <= 3:
                road_type = 4
            elif 5 <= road_type <= 13:
                road_type = 5
            elif 14 <= road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10

            f.write(struct.pack("i", road_type))
            f.write(struct.pack("i", size))

            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))


def _process_single_map(args):
    """Worker function to process a single map file."""
    i, map_path, binary_path = args
    try:
        map_data = load_map_json(map_path)
        save_map_binary(map_data, str(binary_path), i)
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder,
    output_dir,
    max_maps=50_000,
    num_workers=None,
):
    """Process JSON map files into binary format using multiprocessing.

    Args:
        data_folder: Path to the folder containing JSON map files.
        output_dir: Path to the directory where binary files will be written.
        max_maps: Maximum number of maps to process.
        num_workers: Number of parallel workers (defaults to cpu_count()).
    """
    if num_workers is None:
        num_workers = cpu_count()

    data_dir = Path(data_folder)
    binary_dir = Path(output_dir)
    binary_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_path = binary_dir / f"map_{i:03d}.bin"
        tasks.append((i, map_path, binary_path))

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_single_map, tasks),
                total=len(tasks),
                desc="Processing maps",
                unit="map",
            )
        )

    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    print(f"\nProcessed {successful}/{len(results)} maps successfully.")
    if failed > 0:
        print(f"Failed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON map files to binary format.")
    parser.add_argument("--data_folder", type=str, default=None, help="Path to folder containing JSON map files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory for binary files.")
    parser.add_argument("--input", type=str, default=None, help="Single JSON map (WOMD or I24) to convert.")
    parser.add_argument("--output", type=str, default=None, help="Output .bin path for --input.")
    parser.add_argument("--preview", type=str, default=None, help="Optional top-down PNG preview path.")
    parser.add_argument("--max_maps", type=int, default=50_000, help="Maximum number of maps to process.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers.")
    args = parser.parse_args()

    if args.input:
        output = args.output
        if not output:
            output = str(Path(args.input).with_suffix(".bin"))
        convert_json_file(args.input, output, preview_path=args.preview)
    else:
        if not args.data_folder or not args.output_dir:
            parser.error("--data_folder and --output_dir are required unless --input is set")
        process_all_maps(
            data_folder=args.data_folder,
            output_dir=args.output_dir,
            max_maps=args.max_maps,
            num_workers=args.num_workers,
        )