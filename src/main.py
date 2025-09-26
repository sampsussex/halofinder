from utils import ConfigReader
from halo_finder import TinkerFinder
from scipy.optimize import minimize_scalar
import sys

bs_to_trial = [10]

if __name__ == "__main__":
    # Check if config file argument is provided
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file_name>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration
    config_reader = ConfigReader(config_file)
    config_reader.load_config()
    #config_reader.validate_config()
    #config_reader.print_config_summary()  # Added missing parentheses
    
    #for i in bs_to_trial:

    if config_reader.should_run_module('group_finder'):
        print("Running Halo Finder...")

        def objective_function(b_threshold):
            tinker_finder = TinkerFinder(config_reader)
            tinker_finder.b_threshold = b_threshold
            tinker_finder.run()
            return -tinker_finder.s_tot            

        #tinker_finder = TinkerFinder(config_reader)
        #tinker_finder.b_threshold = i

        res = minimize_scalar(objective_function, bounds=(1, 5), method="bounded")

        best_x_cont = res.x
        best_score_cont = -res.fun

        print(f"Best b_threshold (continuous) = {best_x_cont:.4f}, Score = {best_score_cont:.3f}")
        

