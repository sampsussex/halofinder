from utils import ConfigReader
from halo_finder import TinkerFinder
from scipy.optimize import minimize
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
        setup_options['blue_a_threshold'],
        setup_options['blue_b_threshold'],
        setup_options['red_effective_luminosity_boost_a'],
        setup_options['red_effective_luminosity_boost_b'],
    ]

    bounds = [
        (-10.0, 0.0),   # red_a_threshold
        (10.0, 30.0),   # red_b_threshold
        (-10.0, 0.0),   # blue_a_threshold
        (10.0, 30.0),   # blue_b_threshold
        (-2.0, 2.0),   # red_effective_luminosity_boost_a
        (-2.0, 2.0),   # red_effective_luminosity_boost_b
    ]

    def objective_function(params):
        (
            red_a,
            red_b,
            blue_a,
            blue_b,
            red_boost_a,
            red_boost_b,
        ) = params
        tinker_finder = TinkerFinder(config_reader)
        tinker_finder.red_a_threshold = red_a
        tinker_finder.red_b_threshold = red_b
        tinker_finder.blue_a_threshold = blue_a
        tinker_finder.blue_b_threshold = blue_b
        tinker_finder.red_effective_luminosity_boost_a = red_boost_a
        tinker_finder.red_effective_luminosity_boost_b = red_boost_b
        print(
            "Optimising params: "
            f"red_a={red_a:.4f}, red_b={red_b:.4f}, "
            f"blue_a={blue_a:.4f}, blue_b={blue_b:.4f}, "
            f"red_boost_a={red_boost_a:.4f}, red_boost_b={red_boost_b:.4f}"
        )
        tinker_finder.run()
        return -tinker_finder.s_tot

    return minimize(objective_function, initial_params, bounds=bounds, method="L-BFGS-B")


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
    elif run_options.get('run_group_finder') or run_options.get('run_mock_comparison'):
        print("Running Halo Finder...")
        finder = run_single(config_reader)
        if run_options.get('run_mock_comparison'):
            print(f"S-score = {finder.s_tot:.6f}")
        
