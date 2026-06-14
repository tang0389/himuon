#!/usr/bin/env python3
"""
Unified benchmark runner.

Usage:
  uv run python microbench/run.py --test shapes
  uv run python microbench/run.py --test shapes ns_convergence
  uv run python microbench/run.py --test all
  uv run python microbench/run.py --test shapes --tile 256

Common args are forwarded to each benchmark module's run() function.
Each module also accepts --plot-only when run standalone.
"""

import argparse
import sys

TESTS = {
    "shapes": "real_shapes_bench",
    "ns_convergence": "ns_convergence",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tests: {', '.join(TESTS.keys())}, all",
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="+",
        required=True,
        help="Tests to run (shapes, ns_convergence, all)",
    )
    # Common args
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    # shapes-specific
    parser.add_argument("--models", type=str, nargs="+", default=None)
    # no-plot
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting after run"
    )

    args = parser.parse_args()

    tests = list(TESTS.keys()) if "all" in args.test else args.test
    for t in tests:
        if t not in TESTS:
            print(f"Unknown test: {t}. Available: {', '.join(TESTS.keys())}")
            sys.exit(1)

    for t in tests:
        print(f"\n{'=' * 60}")
        print(f"  Running: {t} ({TESTS[t]})")
        print(f"{'=' * 60}\n")

        if t == "shapes":
            from experiments.exp_real_shapes_bench import run as shapes_run, plot as shapes_plot

            models = args.models or ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
            data = shapes_run(
                models=models,
                tile=args.tile,
                ns_steps=args.ns_steps,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            if not args.no_plot:
                shapes_plot(data)

        elif t == "ns_convergence":
            from experiments.exp_ns_convergence import run as nsc_run, plot as nsc_plot

            data = nsc_run(
                max_k=10,
                tiles=[128, 256, 512],
                D=(args.sizes or [4096])[0],
                n_seeds=3,
            )
            if not args.no_plot:
                nsc_plot(data)

    print("\nAll done.")


if __name__ == "__main__":
    main()
