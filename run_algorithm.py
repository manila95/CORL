#!/usr/bin/env python3
"""
Unified script to run any algorithm in offline or finetune mode.

Usage:
    python run_algorithm.py --algorithm cql --mode offline
    python run_algorithm.py --algorithm iql --mode finetune
    python run_algorithm.py -a td3_bc -m offline --env mujoco/hopper/medium-v0
"""

import argparse
import importlib
import sys
from pathlib import Path

# Add the project root to Python path so we can import CORL modules
# Get the directory containing this script (CORL/)
script_dir = Path(__file__).parent.absolute()
# Get the parent directory (project root, e.g., O2O/)
project_root = script_dir.parent
# Add it to sys.path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Map of algorithm names to their module paths
ALGORITHM_MODULES = {
    # Offline algorithms
    "cql": {
        "offline": "CORL.algorithms.offline.cql",
        "finetune": "CORL.algorithms.finetune.cql",
    },
    "iql": {
        "offline": "CORL.algorithms.offline.iql",
        "finetune": "CORL.algorithms.finetune.iql",
    },
    "td3_bc": {
        "offline": "CORL.algorithms.offline.td3_bc",
        "finetune": None,  # TD3_BC doesn't have a finetune version
    },
    "cal_ql": {
        "offline": None,  # Cal-QL doesn't have an offline-only version
        "finetune": "CORL.algorithms.finetune.cal_ql",
    },
    "spot": {
        "offline": None,  # SPOT doesn't have an offline-only version
        "finetune": "CORL.algorithms.finetune.spot",
    },
    "awac": {
        "offline": None,  # AWAC doesn't have an offline-only version
        "finetune": "CORL.algorithms.finetune.awac",
    },
}


def get_available_algorithms():
    """Get list of available algorithms."""
    return list(ALGORITHM_MODULES.keys())


def get_available_modes(algorithm: str):
    """Get list of available modes for a given algorithm."""
    if algorithm not in ALGORITHM_MODULES:
        return []
    modes = []
    if ALGORITHM_MODULES[algorithm]["offline"] is not None:
        modes.append("offline")
    if ALGORITHM_MODULES[algorithm]["finetune"] is not None:
        modes.append("finetune")
    return modes


def run_algorithm(algorithm: str, mode: str, extra_args: list = None):
    """
    Run an algorithm in the specified mode.
    
    Args:
        algorithm: Name of the algorithm (e.g., 'cql', 'iql', 'td3_bc')
        mode: Mode to run in ('offline' or 'finetune')
        extra_args: Additional command-line arguments to pass to the algorithm
    """
    # Validate algorithm
    if algorithm not in ALGORITHM_MODULES:
        available = ", ".join(get_available_algorithms())
        print(f"Error: Unknown algorithm '{algorithm}'")
        print(f"Available algorithms: {available}")
        sys.exit(1)
    
    # Validate mode
    available_modes = get_available_modes(algorithm)
    if mode not in available_modes:
        modes_str = ", ".join(available_modes) if available_modes else "none"
        print(f"Error: Mode '{mode}' not available for algorithm '{algorithm}'")
        print(f"Available modes for {algorithm}: {modes_str}")
        sys.exit(1)
    
    # Get module path
    module_path = ALGORITHM_MODULES[algorithm][mode]
    if module_path is None:
        print(f"Error: {algorithm} does not have a {mode} mode")
        sys.exit(1)
    
    # Import the module
    try:
        print(f"Loading {algorithm} in {mode} mode from {module_path}...")
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error: Failed to import {module_path}")
        print(f"Import error: {e}")
        sys.exit(1)
    
    # Check if train function exists
    if not hasattr(module, "train"):
        print(f"Error: Module {module_path} does not have a 'train' function")
        sys.exit(1)
    
    # Get the train function
    train_func = module.train
    
    # Prepare arguments for pyrallis
    # pyrallis.wrap() expects sys.argv to contain the arguments
    original_argv = sys.argv.copy()
    try:
        # Set sys.argv to include the script name and any extra arguments
        # pyrallis expects the first argument to be the script name
        script_name = module_path.replace(".", "/") + ".py"
        sys.argv = [script_name] + (extra_args if extra_args else [])
        
        # Call the train function (pyrallis will handle argument parsing)
        print(f"\n{'='*70}")
        print(f"Running {algorithm.upper()} in {mode.upper()} mode")
        if extra_args:
            print(f"Additional arguments: {' '.join(extra_args)}")
        print(f"{'='*70}\n")
        train_func()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(
        description="Run offline RL algorithms in offline or finetune mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CQL in offline mode
  python run_algorithm.py --algorithm cql --mode offline

  # Run IQL in finetune mode with custom environment
  python run_algorithm.py -a iql -m finetune --env mujoco/hopper/medium-v0

  # Run TD3_BC in offline mode with custom seed
  python run_algorithm.py -a td3_bc -m offline --seed 42

  # Pass additional arguments (they will be forwarded to the algorithm)
  python run_algorithm.py -a cql -m offline --device cpu --batch_size 128
        """
    )
    
    parser.add_argument(
        "-a", "--algorithm",
        type=str,
        required=True,
        choices=get_available_algorithms(),
        help="Algorithm to run (choices: %(choices)s)"
    )
    
    parser.add_argument(
        "-m", "--mode",
        type=str,
        required=True,
        choices=["offline", "finetune"],
        help="Mode to run in: 'offline' or 'finetune'"
    )
    
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List all available algorithms and their supported modes"
    )
    
    # Parse known args first to handle --list-algorithms
    args, extra_args = parser.parse_known_args()
    
    # Handle list-algorithms flag
    if args.list_algorithms:
        print("Available algorithms and modes:")
        print("=" * 70)
        for alg in sorted(get_available_algorithms()):
            modes = get_available_modes(alg)
            modes_str = ", ".join(modes) if modes else "none"
            print(f"  {alg:12s} : {modes_str}")
        print("=" * 70)
        return
    
    # Validate algorithm and mode combination
    available_modes = get_available_modes(args.algorithm)
    if args.mode not in available_modes:
        modes_str = ", ".join(available_modes) if available_modes else "none"
        parser.error(
            f"Algorithm '{args.algorithm}' does not support mode '{args.mode}'. "
            f"Available modes: {modes_str}"
        )
    
    # Run the algorithm
    # Pass extra_args to the algorithm (pyrallis will handle them)
    run_algorithm(args.algorithm, args.mode, extra_args)


if __name__ == "__main__":
    main()
