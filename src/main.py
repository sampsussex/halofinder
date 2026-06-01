from utils import ConfigReader
from halo_finder import RunHaloFinder
from scipy.optimize import minimize
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


def optimize_on_mock(config_reader):
    threshold_options = config_reader.get_threshold_model_params()
    tune_completeness = should_tune_completeness(config_reader)
    initial_params = [
        threshold_options["red_a_threshold"],
        threshold_options["red_b_threshold"],
        threshold_options["blue_a_threshold"],
        threshold_options["blue_b_threshold"],
    ]

    bounds = [
        (0.1, 4),  # red_a_threshold
        (-4.0, 0.1),  # red_b_threshold
        (0.1, 4),  # blue_a_threshold
        (-4.0, 0.1),  # blue_b_threshold
    ]

    if tune_completeness:
        initial_params.append(threshold_options.get("completeness_coefficient", 0.0))
        bounds.append((0.0, 4.0))  # completeness_coefficient

    def objective_function(params):
        red_a, red_b, blue_a, blue_b = params[:4]
        halo_finder = RunHaloFinder(config_reader)
        halo_finder.red_a_threshold = red_a
        halo_finder.red_b_threshold = red_b
        halo_finder.blue_a_threshold = blue_a
        halo_finder.blue_b_threshold = blue_b
        if tune_completeness:
            halo_finder.completeness_coefficient = params[4]
        message = (
            "Optimising params: "
            f"red_a={red_a:.4f}, red_b={red_b:.4f}, "
            f"blue_a={blue_a:.4f}, blue_b={blue_b:.4f}"
        )
        if tune_completeness:
            message += f", completeness_coefficient={params[4]:.4f}"
        print(message)
        halo_finder.run()
        return -halo_finder.s_tot

    return minimize(
        objective_function, initial_params, bounds=bounds, method="Nelder-Mead"
    )


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
        best_score = -result.fun
        print(f"Best score = {best_score:.3f}")
        print(f"Best parameters = {result.x}")
    elif run_options.get("optimse_parameter_space"):
        print("Running parameter-space grid search on mock comparison...")
        results_path = grid_search_on_mock(config_reader, num_points=90)
        print(f"Grid search complete. Results saved to {results_path}")
    elif run_options.get("run_group_finder") or run_options.get("run_mock_comparison"):
        print("Running Halo Finder...")
        finder = run_single(config_reader)
        if run_options.get("run_mock_comparison"):
            print(f"S-score = {finder.s_tot:.6f}")
