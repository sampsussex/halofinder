from utils import ConfigReader
from halo_finder import RunHaloFinder
import itertools
import numpy as np
import os
import sys

def run_single(config_reader):
    halo_finder = RunHaloFinder(config_reader)
    halo_finder.run()
    return halo_finder


def should_tune_completeness(config_reader):
    """Tune completeness only when a completeness input column is configured."""
    return config_reader.get_column_names().get("completeness") is not None


import math


def round_sig(x, sig=2):
    if x == 0:
        return 0.0
    magnitude = int(math.floor(math.log10(abs(x))))
    return round(x, -magnitude + (sig - 1))


def golden_section_search(objective, low, high, tol=0.01):
    gr = (math.sqrt(5) + 1) / 2
    cache = {}

    def f(x):
        x_r = round_sig(x)
        if x_r not in cache:
            cache[x_r] = objective(x_r)
        return cache[x_r]

    a, b = low, high
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return round_sig((a + b) / 2)


def run_finder(config_reader, blue_a, red_a, completeness=None):
    halo_finder = RunHaloFinder(config_reader)
    halo_finder.blue_a_threshold = blue_a
    halo_finder.red_a_threshold  = red_a
    halo_finder.blue_b_threshold = 0.0
    halo_finder.red_b_threshold  = 0.0
    if completeness is not None:
        halo_finder.completeness_coefficient = completeness
    print(
        f"  blue_a={blue_a:.4f}, red_a={red_a:.4f}"
        + (f", completeness={completeness:.4f}" if completeness is not None else "")
    )
    halo_finder.run()
    return -halo_finder.s_tot


def optimize_on_mock(config_reader):
    tune_completeness = should_tune_completeness(config_reader)

    # ----------------------------------------------------------------
    # Initial values — edit these to change starting search ranges
    # ----------------------------------------------------------------
    BLUE_A_INITIAL = 2.  # used as the fixed default during Stage 1
    RED_A_INITIAL  = 1.  # used as the fixed default during Stage 1

    BLUE_A_BOUNDS       = (1, 3.0)
    RED_A_BOUNDS        = (0.1, 2.0)
    COMPLETENESS_BOUNDS = (0.0, 2.0)
    # ----------------------------------------------------------------

    # Stage 1 — optimise blue_a, red_a held at its initial value
    print("Stage 1: optimising blue_a_threshold...")
    blue_a = golden_section_search(
        lambda v: run_finder(config_reader, blue_a=v, red_a=RED_A_INITIAL),
        *BLUE_A_BOUNDS
    )
    print(f"  → blue_a_threshold = {blue_a}")

    # Stage 2 — optimise red_a, blue_a fixed at Stage 1 result
    print("Stage 2: optimising red_a_threshold...")
    red_a = golden_section_search(
        lambda v: run_finder(config_reader, blue_a=blue_a, red_a=v),
        *RED_A_BOUNDS
    )
    print(f"  → red_a_threshold = {red_a}")

    # Stage 3 — optimise completeness, both a_thresholds fixed
    completeness = None
    if tune_completeness:
        print("Stage 3: optimising completeness_coefficient...")
        completeness = golden_section_search(
            lambda v: run_finder(config_reader, blue_a=blue_a, red_a=red_a, completeness=v),
            *COMPLETENESS_BOUNDS
        )
        print(f"  → completeness_coefficient = {completeness}")

    print(f"\nFinal: blue_a={blue_a}, red_a={red_a}"
          + (f", completeness={completeness}" if completeness is not None else ""))

    return {
        "blue_a_threshold":       blue_a,
        "red_a_threshold":        red_a,
        "blue_b_threshold":       0.0,
        "red_b_threshold":        0.0,
        **({"completeness_coefficient": completeness} if completeness is not None else {})
    }


def grid_search_on_mock(config_reader, num_points=1000):
    tune_completeness = should_tune_completeness(config_reader)
    bounds = [
        (1, 4),  # red_a_threshold
        (-3, 0.1),  # red_b_threshold
        (1, 4),  # blue_a_threshold
        (-3, 0.1),  # blue_b_threshold
    ]
    param_names = [
        "red_a_threshold",
        "red_b_threshold",
        "blue_a_threshold",
        "blue_b_threshold",
    ]
    if tune_completeness:
        bounds.append((0.0, 4.0))  # completeness_coefficient
        param_names.append("completeness_coefficient")

    file_locations = config_reader.get_file_locations()
    results_path = f"{file_locations['s_tot_path']}_grid.csv"
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    grid_size = 1
    while grid_size ** len(bounds) < num_points:
        grid_size += 1

    grid_axes = [np.linspace(lower, upper, grid_size) for lower, upper in bounds]

    file_exists = os.path.exists(results_path)
    with open(results_path, "a", encoding="utf-8") as handle:
        if not file_exists:
            header = ",".join(["index", *param_names, "s_tot"])
            handle.write(f"{header}\n")

        for idx, params in enumerate(itertools.product(*grid_axes)):
            if idx >= num_points:
                break
            halo_finder = RunHaloFinder(config_reader)
            (
                halo_finder.red_a_threshold,
                halo_finder.red_b_threshold,
                halo_finder.blue_a_threshold,
                halo_finder.blue_b_threshold,
            ) = params[:4]
            if tune_completeness:
                halo_finder.completeness_coefficient = params[4]
            halo_finder.run()
            row = ",".join(
                [
                    str(idx),
                    *[f"{value:.6f}" for value in params],
                    f"{halo_finder.s_tot:.6f}",
                ]
            )
            handle.write(f"{row}\n")
            handle.flush()

    return results_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file_name>")
        sys.exit(1)

    config_file = sys.argv[1]

    config_reader = ConfigReader(config_file)
    config_reader.load_config()

    run_options = config_reader.get_run_options()

    if run_options.get("optimse_on_mock"):
        print("Optimising on mock comparison...")
        result = optimize_on_mock(config_reader)
        #best_score = -result.fun
        #print(f"Best score = {best_score:.3f}")
        #print(f"Best parameters = {result.x}")
    elif run_options.get("optimse_parameter_space"):
        print("Running parameter-space grid search on mock comparison...")
        results_path = grid_search_on_mock(config_reader, num_points=90)
        print(f"Grid search complete. Results saved to {results_path}")
    elif run_options.get("run_group_finder") or run_options.get("run_mock_comparison"):
        print("Running Halo Finder...")
        finder = run_single(config_reader)
        if run_options.get("run_mock_comparison"):
            print(f"S-score = {finder.s_tot:.6f}")
