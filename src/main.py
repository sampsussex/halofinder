from utils import ConfigReader
from halo_finder import TinkerFinder
from scipy.optimize import minimize
import itertools
import numpy as np
import os
import sys


def run_single(config_reader):
    tinker_finder = TinkerFinder(config_reader)
    tinker_finder.run()
    return tinker_finder


def optimize_on_mock(config_reader):
    setup_options = config_reader.get_setup_options()
    initial_params = [
        setup_options['red_a_threshold'],
        setup_options['red_b_threshold'],
        setup_options['red_c_threshold'],
        setup_options['blue_a_threshold'],
        setup_options['blue_b_threshold'],
        setup_options['blue_c_threshold'],
    ]

    bounds = [
        (5.0, 15.0),    # red_a_threshold
        (-2.0, 2.0),    # red_b_threshold
        (-2.0, 2.0),    # red_c_threshold
        (5.0, 15.0),    # blue_a_threshold
        (-2.0, 2.0),    # blue_b_threshold
        (-2.0, 2.0),    # blue_c_threshold
    ]

    def objective_function(params):
        (
            red_a,
            red_b,
            red_c,
            blue_a,
            blue_b,
            blue_c,
        ) = params
        tinker_finder = TinkerFinder(config_reader)
        tinker_finder.red_a_threshold = red_a
        tinker_finder.red_b_threshold = red_b
        tinker_finder.red_c_threshold = red_c
        tinker_finder.blue_a_threshold = blue_a
        tinker_finder.blue_b_threshold = blue_b
        tinker_finder.blue_c_threshold = blue_c
        print(
            "Optimising params: "
            f"red_a={red_a:.4f}, red_b={red_b:.4f}, red_c={red_c:.4f}, "
            f"blue_a={blue_a:.4f}, blue_b={blue_b:.4f}, blue_c={blue_c:.4f}"
        )
        tinker_finder.run()
        return -tinker_finder.s_tot

    return minimize(objective_function, initial_params, bounds=bounds, method="L-BFGS-B")


def grid_search_on_mock(config_reader, num_points=5):
    bounds = [
        (1, 7),    # red_a_threshold
        (0, 0),    # red_b_threshold
        (0, 0),    # red_c_threshold
        (1, 7),    # blue_a_threshold
        (0, 0),    # blue_b_threshold
        (0, 0),    # blue_c_threshold
    ]
    param_names = [
        "red_a_threshold",
        "red_b_threshold",
        "red_c_threshold",
        "blue_a_threshold",
        "blue_b_threshold",
        "blue_c_threshold",
    ]

    file_locations = config_reader.get_file_locations()
    results_path = f"{file_locations['s_tot_path']}_grid.csv"
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    grid_size = 1
    while grid_size ** len(bounds) < num_points:
        grid_size += 1

    grid_axes = [
        np.linspace(lower, upper, grid_size)
        for lower, upper in bounds
    ]

    file_exists = os.path.exists(results_path)
    with open(results_path, "a", encoding="utf-8") as handle:
        if not file_exists:
            header = ",".join(["index", *param_names, "s_tot"])
            handle.write(f"{header}\n")

        for idx, params in enumerate(itertools.product(*grid_axes)):
            if idx >= num_points:
                break
            tinker_finder = TinkerFinder(config_reader)
            (
                tinker_finder.red_a_threshold,
                tinker_finder.red_b_threshold,
                tinker_finder.red_c_threshold,
                tinker_finder.blue_a_threshold,
                tinker_finder.blue_b_threshold,
                tinker_finder.blue_c_threshold,
            ) = params
            tinker_finder.run()
            row = ",".join([str(idx), *[f"{value:.6f}" for value in params], f"{tinker_finder.s_tot:.6f}"])
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

    if run_options.get('optimse_on_mock'):
        print("Optimising on mock comparison...")
        result = optimize_on_mock(config_reader)
        best_score = -result.fun
        print(f"Best score = {best_score:.3f}")
        print(f"Best parameters = {result.x}")
    elif run_options.get('optimse_parameter_space'):
        print("Running parameter-space grid search on mock comparison...")
        results_path = grid_search_on_mock(config_reader, num_points=100)
        print(f"Grid search complete. Results saved to {results_path}")
    elif run_options.get('run_group_finder') or run_options.get('run_mock_comparison'):
        print("Running Halo Finder...")
        finder = run_single(config_reader)
        if run_options.get('run_mock_comparison'):
            print(f"S-score = {finder.s_tot:.6f}")
        
