#!/usr/bin/env python3
"""
Analyze corridor costmap ablation results.
Produces paper-ready summary table and detailed stats.
"""

import csv
import json
from pathlib import Path

import numpy as np


def analyze_config(csv_path: str, label: str):
    """Analyze a single costmap config's stats."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    occ_ratios = [float(r['occupied_ratio']) for r in rows]
    occupied = [int(r['occupied_cells']) for r in rows]
    free = [int(r['free_cells']) for r in rows]

    t0 = float(rows[0]['timestamp'])
    t_end = float(rows[-1]['timestamp'])
    duration = t_end - t0

    # Compute time-averaged metrics
    avg_occ_ratio = np.mean(occ_ratios)
    std_occ_ratio = np.std(occ_ratios)
    max_occ_ratio = np.max(occ_ratios)
    min_occ_ratio = np.min(occ_ratios)

    # Compute occupied cell growth rate
    n_obs = len(rows)
    first_quarter = occ_ratios[:n_obs // 4]
    last_quarter = occ_ratios[-(n_obs // 4):]
    avg_first = np.mean(first_quarter) if first_quarter else 0
    avg_last = np.mean(last_quarter) if last_quarter else 0

    # Spatial spread (centroid variance)
    cx = [float(r['obstacle_centroid_x']) for r in rows]
    cy = [float(r['obstacle_centroid_y']) for r in rows]
    centroid_spread = np.sqrt(np.var(cx) + np.var(cy))

    return {
        'config': label,
        'n_frames': len(rows),
        'duration_s': round(duration, 1),
        'avg_occupied_ratio': round(avg_occ_ratio * 100, 2),
        'std_occupied_ratio': round(std_occ_ratio * 100, 2),
        'max_occupied_ratio': round(max_occ_ratio * 100, 2),
        'min_occupied_ratio': round(min_occ_ratio * 100, 2),
        'avg_occupied_cells': round(np.mean(occupied), 1),
        'avg_free_cells': round(np.mean(free), 1),
        'first_quarter_occ': round(avg_first * 100, 2),
        'last_quarter_occ': round(avg_last * 100, 2),
        'growth_pct': round((avg_last - avg_first) * 100, 2),
        'centroid_spread': round(centroid_spread, 4),
    }


def main():
    base = Path('/home/nishant/maps/corridor_ablation_results')
    configs = ['L', 'L+S', 'L+D', 'D']

    results = []
    for cfg in configs:
        csv_path = base / cfg / 'costmap_stats.csv'
        if csv_path.exists():
            r = analyze_config(str(csv_path), cfg)
            if r:
                results.append(r)

    if not results:
        print("No results found!")
        return

    print("=" * 80)
    print("CORRIDOR COSTMAP ABLATION RESULTS (Bag 172822, 263s)")
    print("=" * 80)

    # Print detailed stats
    for r in results:
        print(f"\n--- {r['config']} ---")
        for k, v in r.items():
            if k != 'config':
                print(f"  {k:25s}: {v}")

    # Print paper-ready comparison table
    print("\n")
    print("=" * 80)
    print("PAPER TABLE: Costmap Obstacle Detection Comparison")
    print("=" * 80)
    header = f"{'Config':<8} {'Avg Occ%':>9} {'Std%':>7} {'Max%':>7} " \
             f"{'Cells':>8} {'Growth':>8} {'Spread':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['config']:<8} {r['avg_occupied_ratio']:>8.2f}% "
              f"{r['std_occupied_ratio']:>6.2f}% {r['max_occupied_ratio']:>6.2f}% "
              f"{r['avg_occupied_cells']:>7.0f} {r['growth_pct']:>+7.2f}% "
              f"{r['centroid_spread']:>7.4f}")

    # Key findings
    print("\n")
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    l_only = next((r for r in results if r['config'] == 'L'), None)
    ls = next((r for r in results if r['config'] == 'L+S'), None)
    ld = next((r for r in results if r['config'] == 'L+D'), None)
    d_only = next((r for r in results if r['config'] == 'D'), None)

    if l_only and ls:
        delta = ls['avg_occupied_ratio'] - l_only['avg_occupied_ratio']
        print(f"  L+S vs L (sensor depth added):  {delta:+.2f}% obstacle coverage")
    if l_only and ld:
        delta = ld['avg_occupied_ratio'] - l_only['avg_occupied_ratio']
        print(f"  L+D vs L (DA3 depth added):     {delta:+.2f}% obstacle coverage")
    if ls and ld:
        delta = ld['avg_occupied_ratio'] - ls['avg_occupied_ratio']
        print(f"  L+D vs L+S (DA3 vs sensor):     {delta:+.2f}% obstacle coverage")
    if d_only and l_only:
        delta = d_only['avg_occupied_ratio'] - l_only['avg_occupied_ratio']
        print(f"  D vs L (depth-only vs lidar):    {delta:+.2f}% obstacle coverage")
    if ld and d_only:
        delta = ld['avg_occupied_ratio'] - d_only['avg_occupied_ratio']
        print(f"  L+D vs D (lidar+da3 vs da3):     {delta:+.2f}% obstacle coverage")

    # Save JSON
    with open(base / 'ablation_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {base / 'ablation_summary.json'}")


if __name__ == '__main__':
    main()
