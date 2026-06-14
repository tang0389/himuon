#!/usr/bin/env python3
"""
Unified plot generator. Reads saved JSON data and regenerates figures.

Usage:
  uv run python microbench/plot.py --test shapes
  uv run python microbench/plot.py --test all
  uv run python microbench/plot.py --test shapes ns_convergence --out-dir /tmp/figs
"""

import argparse
import sys

from bench_utils import load_results

TESTS = {
    "shapes": ("real_shapes_bench", "real_shapes_bench"),
    "ns_convergence": ("ns_convergence", "ns_convergence"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified plot generator (reads from data/)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tests: {', '.join(TESTS.keys())}, all",
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="+",
        required=True,
        help="Plots to generate (shapes, ns_convergence, all)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Output directory for figures"
    )
    args = parser.parse_args()

    tests = list(TESTS.keys()) if "all" in args.test else args.test
    for t in tests:
        if t not in TESTS:
            print(f"Unknown test: {t}. Available: {', '.join(TESTS.keys())}")
            sys.exit(1)

    for t in tests:
        module_name, data_name = TESTS[t]
        print(f"Plotting: {t} (from data/{data_name}.json)")

        try:
            data = load_results(data_name)
        except FileNotFoundError:
            print(f"  No saved data found for {t}. Run the experiment first.")
            continue

        if t == "shapes":
            from experiments.exp_real_shapes_bench import plot as shapes_plot

            shapes_plot(data, out_dir=args.out_dir)

        elif t == "ns_convergence":
            from experiments.exp_ns_convergence import plot as nsc_plot

            nsc_plot(data, out_dir=args.out_dir)

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
