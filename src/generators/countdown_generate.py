"""
Script for generating data for the countdown task.
"""
import json
import argparse
import random
import os
import sys
import time
from tqdm import tqdm

from src.generators.countdown import CountDown

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="../../data/countdown", help="Directory to store data")

    # countdown specific
    parser.add_argument("--list_length", type=int, default=8, help="Length of the starting number list")
    parser.add_argument("--max_target", type=int, default=100, help="Maximum target number")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of samples to generate")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                       help="Ratio of validation samples (default: 0.2)")
    
    # duplicate handling
    parser.add_argument("--check_duplicates", action="store_true", 
                       help="Whether to check for duplicates within the dataset")
    parser.add_argument("--max_attempts", type=int, default=100,
                       help="Maximum number of attempts to generate a non-duplicate sample")
    
    return parser.parse_args()

def calculate_solution_complexity(operations):
    """
    Calculate the complexity of a solution based on the number of operations.
    
    Args:
        operations: List of operations in the solution
        
    Returns:
        float: Complexity score (0.0 to 1.0)
    """
    # Simple complexity measure based on number of operations
    return min(1.0, len(operations) / 10.0) 

def generate_target_numbers(args):
    """
    Generate a list of target numbers based on the specified strategy.
    
    Args:
        args: Command line arguments
        
    Returns:
        list: Generated target numbers
    """
    min_target = 10  # Minimum target number
    max_target = args.max_target

    targets = list(range(min_target, max_target + 1))
    random.shuffle(targets)
    return targets

def is_duplicate(sample, existing_samples):
    """
    Check if a sample is a duplicate of any existing sample.
    
    A sample is considered a duplicate if:
    1. It has the same starting numbers and target number, or
    2. It has the same solution
    
    Args:
        sample: The new sample to check
        existing_samples: List of existing samples to check against
        
    Returns:
        bool: True if duplicate, False otherwise
    """
    if not existing_samples:
        return False
    
    # Create sorted version of nums for comparison
    sorted_nums = sorted(sample["nums"])
    
    for existing in existing_samples:
        # Check if same starting numbers and target
        if sorted(existing["nums"]) == sorted_nums and existing["target"] == sample["target"]:
            return True
        
        # Check if same solution
        if existing["solution"] == sample["solution"]:
            return True
    
    return False

def generate_sample(cd, list_length, existing_samples, target_nums, max_attempts=10):
    """
    Generate a single data sample that is not a duplicate.
    
    Args:
        cd: CountDown instance
        list_length: Length of the starting number list
        existing_samples: List of samples already generated
        target_nums: List of available target numbers
        max_attempts: Maximum number of attempts to generate a non-duplicate
        
    Returns:
        dict: Generated sample data
    """
    attempts = 0
    while attempts < max_attempts:
        # Select a target randomly from the available targets
        target = random.choice(target_nums)
        
        # Generate a puzzle with a solution
        nums, solution = cd.generate(target)
        no_backtrack_trace = cd.convert_to_path(target, nums, solution)
        
        # Create the sample
        sample = {
            "nums": nums,
            "target": target,
            "solution": solution,
            "complexity": calculate_solution_complexity(solution),
            "optimal_path": no_backtrack_trace,
            "start_size": list_length
        }
        
        # Check for duplicates if needed
        if existing_samples and is_duplicate(sample, existing_samples):
            attempts += 1
            continue
        
        # If we got here, the sample is not a duplicate
        return sample
    
    # If we exhausted attempts, just return the last sample generated
    print(f"Warning: Could not generate a non-duplicate sample after {max_attempts} attempts")
    return sample

def generate_dataset(args):
    """
    Generate a full dataset of unique samples.
    
    Args:
        args: Command line arguments
        
    Returns:
        list: Generated samples
    """
    # Set random seed
    random.seed(args.seed)
    
    # Generate target numbers based on specified strategy
    target_nums = generate_target_numbers(args)
    
    print(f"Generated {len(target_nums)} target numbers")
    
    # Statistics tracking
    samples_count = 0
    complexity_values = []
    duplicate_attempts = 0
    
    # Generate all samples in one batch
    all_samples = []
    
    print(f"Generating {args.num_samples} total samples with list length {args.list_length}...")
    
    for t in tqdm(range(args.num_samples), desc="Generating samples"):
        try:
            cd = CountDown(args.max_target, args.list_length)
            
            # Generate sample, checking for duplicates if enabled
            if args.check_duplicates:
                sample = generate_sample(
                    cd, 
                    args.list_length, 
                    all_samples, 
                    target_nums,
                    args.max_attempts
                )
            else:
                sample = generate_sample(cd, args.list_length, [], target_nums)
            
            all_samples.append(sample)
            
            # Track statistics
            samples_count += 1
            complexity_values.append(sample["complexity"])
            
        except Exception as e:
            print(f"Error generating sample {t}: {e}")
            # Continue with next sample
            continue
    
    # Print statistics for the generated dataset
    print("\nGenerated dataset statistics:")
    if samples_count > 0:
        avg_complexity = sum(complexity_values) / samples_count
        print(f"  List length {args.list_length}: {samples_count} samples, avg complexity: {avg_complexity:.4f}")
    
    return all_samples

def split_dataset(samples, val_ratio=0.2, seed=None):
    """
    Split a dataset into training and validation sets.
    
    Args:
        samples: List of all samples
        val_ratio: Ratio of validation samples (default: 0.2)
        seed: Random seed for shuffling
        
    Returns:
        tuple: (train_samples, val_samples)
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle the samples
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    val_size = int(len(shuffled) * val_ratio)
    
    # Split the dataset
    train_samples = shuffled[val_size:]
    val_samples = shuffled[:val_size]
    
    return train_samples, val_samples

def generate_data(args):
    """
    Generate the full dataset based on arguments.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    print(f"Starting data generation")
    
    # Generate a single dataset with all samples
    all_samples = generate_dataset(args)
    
    # Split the dataset into train and validation sets
    train_samples, val_samples = split_dataset(all_samples, args.val_ratio, args.seed)
    
    print(f"\nSplit dataset into:")
    print(f"  Training: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")
    
    # Save the data
    data_samples = {
        "train": train_samples,
        "val": val_samples
    }
    
    for split, samples in data_samples.items():
        save_data(args, split, samples)

    end_time = time.time()
    print(f"Completed data generation in {end_time - start_time:.2f} seconds")
    
    # Print additional statistics
    if args.check_duplicates:
        print(f"Duplicate checking was enabled")
        print(f"Generated {len(all_samples)} unique samples")

def save_data(args, split, data):
    """
    Save data to a JSON file.
    
    Args:
        args: Command line arguments
        split: Data split (train, val)
        data: Data to save
    """
    os.makedirs(args.data_dir, exist_ok=True)
    filename = f"{args.data_dir}/{split}_{args.list_length}.json"
    
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(data)} samples to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")
        # Try to save to a backup location
        backup_filename = f"backup_{split}.json"
        try:
            with open(backup_filename, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved backup to {backup_filename}")
        except:
            print("Failed to save backup. Data may be lost.")

if __name__ == "__main__":
    try:
        args = parse_arguments()
        generate_data(args)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
