#!/usr/bin/env python3
"""Rhizome structure-dump analyser.

Reads the `logs/structure_dump_mode<M>_<stamp>.txt` files written by
RhizomeLog's [EYE] structure dump (native/src/core/growth_controller.cpp,
write_structure_dump_controller) and turns them into morphology metrics.

Why this exists: the dump is the only artifact that carries *topology*
(parent links, branch order, per-root identity). The exported OBJ carries
geometry only, and the JSON stats carry counts only. Branch-order distribution,
depth, tip length and crowding cannot be recovered from the mesh, so tuning
without this file is guesswork.

Dump format (v1):
    # RHIZOME STRUCTURE DUMP v1
    # key=value            (header, see parse_header)
    # COLUMNS idx x y z radius parent order root_id
    0 0.30 0.30 18.30 1.400 -1 0 0

Stdlib only. No third-party dependencies.

Usage:
    python analyze_dump.py dump1.txt [dump2.txt ...] [--json out.json]
    python analyze_dump.py logs/                # every dump in a directory
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

@dataclass
class Section:
    idx: int
    x: float
    y: float
    z: float
    radius: float
    parent: int
    order: int
    root_id: int


@dataclass
class Dump:
    path: str
    header: Dict[str, str] = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)

    @property
    def mode(self) -> Optional[int]:
        try:
            return int(self.header.get("mode", ""))
        except ValueError:
            return None


def parse_header(key: str, raw: str) -> Tuple[str, str]:
    return key, raw


def parse_dump(path: str) -> Dump:
    dump = Dump(path=path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    key, value = body.split("=", 1)
                    dump.header[key.strip()] = value.strip()
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                dump.sections.append(
                    Section(
                        idx=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                        z=float(parts[3]),
                        radius=float(parts[4]),
                        parent=int(parts[5]),
                        order=int(parts[6]),
                        root_id=int(parts[7]),
                    )
                )
            except ValueError:
                # A malformed row should not sink an otherwise usable dump.
                continue
    return dump


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def depth_map(sections: Sequence[Section]) -> Dict[int, int]:
    """Depth in parent links from the root. Cycles are broken, not trusted."""
    by_idx = {s.idx: s for s in sections}
    depths: Dict[int, int] = {}

    for section in sections:
        if section.idx in depths:
            continue
        # Walk up to a known depth or the root, then unwind.
        chain: List[int] = []
        cur: Optional[Section] = section
        seen = set()
        while cur is not None and cur.idx not in depths:
            if cur.idx in seen:  # cycle: treat as root-level
                depths[cur.idx] = 0
                cur = None
                break
            seen.add(cur.idx)
            chain.append(cur.idx)
            parent = by_idx.get(cur.parent)
            cur = parent if parent is not None and cur.parent != cur.idx else None
        base = depths.get(cur.idx, 0) if cur is not None else 0
        if cur is not None and cur.idx in depths:
            base = depths[cur.idx]
        for offset, idx in enumerate(reversed(chain)):
            depths[idx] = base + offset + (1 if cur is not None else 0)
        if not chain and section.idx not in depths:
            depths[section.idx] = 0
    return depths


def crowding(sections: Sequence[Section]) -> Tuple[int, int]:
    """Count *cross-strand* capsule overlaps and the sections involved.

    Consecutive sections along one strand are spaced by growth_step and are
    wider than that spacing, so they always overlap — that is how a continuous
    tube is built, not a defect. Counting those would report ~97% of every
    structure as "crowded" and mean nothing.

    So parent/child and immediate index neighbours are excluded, and what
    remains is genuine cross-strand fusion: separate branches growing into the
    same space. That is the geometry that reads as a blob on the printer.
    """
    slack = 0.90
    pairs = 0
    involved = set()
    n = len(sections)
    for i in range(n):
        a = sections[i]
        for j in range(i + 1, n):
            b = sections[j]
            # Local chain neighbours are intended to overlap.
            if abs(a.idx - b.idx) <= 2:
                continue
            if a.parent == b.idx or b.parent == a.idx:
                continue
            dx = a.x - b.x
            if dx > 8.0 or dx < -8.0:
                continue
            dy = a.y - b.y
            if dy > 8.0 or dy < -8.0:
                continue
            dz = a.z - b.z
            limit = (a.radius + b.radius) * slack
            if dx * dx + dy * dy + dz * dz < limit * limit:
                pairs += 1
                involved.add(a.idx)
                involved.add(b.idx)
    return pairs, len(involved)


def occupancy_fraction(sections: Sequence[Section], bins: int = 16) -> float:
    """Fraction of a coarse grid (over the structure's own bbox) occupied."""
    if not sections:
        return 0.0
    xs = [s.x for s in sections]
    ys = [s.y for s in sections]
    zs = [s.z for s in sections]
    span = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if min(span) <= 0.0:
        return 0.0
    occupied = set()
    for s in sections:
        gx = min(bins - 1, int((s.x - min(xs)) / span[0] * bins))
        gy = min(bins - 1, int((s.y - min(ys)) / span[1] * bins))
        gz = min(bins - 1, int((s.z - min(zs)) / span[2] * bins))
        occupied.add((gx, gy, gz))
    return len(occupied) / float(bins ** 3)


def analyse(dump: Dump) -> dict:
    sections = dump.sections
    if not sections:
        return {"source": os.path.basename(dump.path), "error": "no sections"}

    by_idx = {s.idx: s for s in sections}
    children: Dict[int, List[int]] = {}
    for s in sections:
        if s.parent in by_idx and s.parent != s.idx:
            children.setdefault(s.parent, []).append(s.idx)

    depths = depth_map(sections)
    tips = [s for s in sections if not children.get(s.idx)]
    radii = [s.radius for s in sections]

    order_hist: Dict[int, int] = {}
    for s in sections:
        order_hist[s.order] = order_hist.get(s.order, 0) + 1

    root_hist: Dict[int, int] = {}
    for s in sections:
        root_hist[s.root_id] = root_hist.get(s.root_id, 0) + 1

    try:
        min_print_radius = float(dump.header.get("min_print_radius", "0"))
    except ValueError:
        min_print_radius = 0.0
    below = [r for r in radii if min_print_radius > 0 and r < min_print_radius]

    # Straightness per root: direct distance from the seed over path length.
    straightness: List[float] = []
    root_sections = [s for s in sections if s.parent not in by_idx]
    for tip in tips:
        # Walk the parent chain of this tip.
        path_len = 0.0
        cur = tip
        guard = 0
        while cur.parent in by_idx and guard < len(sections):
            parent = by_idx[cur.parent]
            path_len += math.dist((cur.x, cur.y, cur.z), (parent.x, parent.y, parent.z))
            cur = parent
            guard += 1
        direct = math.dist((tip.x, tip.y, tip.z), (cur.x, cur.y, cur.z))
        if path_len > 1e-6:
            straightness.append(direct / path_len)

    xs = [s.x for s in sections]
    ys = [s.y for s in sections]
    zs = [s.z for s in sections]
    z_min, z_max = min(zs), max(zs)
    z_span = max(z_max - z_min, 1e-6)
    z_bands = [0] * 5
    for z in zs:
        band = min(4, int((z - z_min) / z_span * 5))
        z_bands[band] += 1

    overlap_pairs, overlap_sections = crowding(sections)
    depth_values = [depths.get(s.idx, 0) for s in sections]

    return {
        "source": os.path.basename(dump.path),
        "mode": dump.mode,
        "header": dump.header,
        "counts": {
            "sections": len(sections),
            "tips": len(tips),
            "roots": len(root_sections),
            "by_order": {str(k): order_hist[k] for k in sorted(order_hist)},
            "by_root": {str(k): root_hist[k] for k in sorted(root_hist)},
        },
        "radius_mm": {
            "min": min(radii),
            "p05": percentile(radii, 0.05),
            "median": percentile(radii, 0.50),
            "p95": percentile(radii, 0.95),
            "max": max(radii),
            "min_print_radius": min_print_radius,
            "below_min_print_radius": len(below),
            "fraction_below_min_print_radius": (len(below) / len(radii)) if radii else 0.0,
        },
        "depth": {
            "max": max(depth_values) if depth_values else 0,
            "mean": (sum(depth_values) / len(depth_values)) if depth_values else 0.0,
            "histogram": {
                str(d): depth_values.count(d) for d in sorted(set(depth_values))
            },
        },
        "geometry": {
            "bbox_min": [min(xs), min(ys), min(zs)],
            "bbox_max": [max(xs), max(ys), max(zs)],
            "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
            "centroid": [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)],
            "z_bands_bottom_to_top": z_bands,
            "occupancy_fraction_16bin": occupancy_fraction(sections),
        },
        "crowding": {
            "overlap_pairs": overlap_pairs,
            "sections_in_overlap": overlap_sections,
            "fraction_in_overlap": (overlap_sections / len(sections)) if sections else 0.0,
        },
        "straightness": {
            "tip_count": len(straightness),
            "mean": (sum(straightness) / len(straightness)) if straightness else 0.0,
            "min": min(straightness) if straightness else 0.0,
            "max": max(straightness) if straightness else 0.0,
        },
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    if "error" in result:
        print(f"{result['source']}: {result['error']}")
        return
    c = result["counts"]
    r = result["radius_mm"]
    g = result["geometry"]
    d = result["depth"]
    cr = result["crowding"]
    st = result["straightness"]

    print(f"\n=== {result['source']} (mode {result['mode']}) ===")
    print(f"  sections {c['sections']}  tips {c['tips']}  roots {c['roots']}")
    print(f"  order: " + "  ".join(f"o{k}={v}" for k, v in c["by_order"].items()))
    print(f"  radius mm: min {r['min']:.3f}  median {r['median']:.3f}  "
          f"p95 {r['p95']:.3f}  max {r['max']:.3f}")
    print(f"  below print floor ({r['min_print_radius']} mm): "
          f"{r['below_min_print_radius']} ({r['fraction_below_min_print_radius']*100:.1f}%)")
    print(f"  depth: max {d['max']}  mean {d['mean']:.2f}")
    print(f"  size mm: {g['size'][0]:.2f} x {g['size'][1]:.2f} x {g['size'][2]:.2f}")
    print(f"  z bands (bottom->top): {g['z_bands_bottom_to_top']}")
    print(f"  occupancy: {g['occupancy_fraction_16bin']*100:.1f}% of own bbox")
    print(f"  crowding: {cr['overlap_pairs']} overlapping pairs, "
          f"{cr['fraction_in_overlap']*100:.1f}% of sections")
    print(f"  tip straightness: mean {st['mean']:.3f} (1.0 = dead straight)")


def collect_inputs(paths: Sequence[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.startswith("structure_dump") and name.endswith(".txt"):
                    files.append(os.path.join(path, name))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"warning: not found: {path}", file=sys.stderr)
    return files


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse Rhizome structure dumps.")
    parser.add_argument("paths", nargs="+", help="dump files or directories")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write all results to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="suppress per-dump summary")
    args = parser.parse_args(argv)

    files = collect_inputs(args.paths)
    if not files:
        print("error: no structure dumps found", file=sys.stderr)
        return 1

    results = []
    for path in files:
        dump = parse_dump(path)
        result = analyse(dump)
        results.append(result)
        if not args.quiet:
            print_summary(result)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json_out} ({len(results)} dumps)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
