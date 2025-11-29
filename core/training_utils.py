"""
Training utilities for MyPT.

Includes functions for calculating dataset coverage, training statistics, etc.
"""

def calculate_dataset_coverage(max_iters, batch_size, block_size, total_tokens):
    """
    Calculate how many times the dataset will be viewed during training.
    
    Args:
        max_iters: Number of training iterations
        batch_size: Batch size
        block_size: Context length
        total_tokens: Total tokens in dataset
    
    Returns:
        dict with coverage statistics
    """
    # Tokens processed per iteration
    tokens_per_iter = batch_size * block_size
    
    # Total tokens that will be viewed
    total_tokens_viewed = max_iters * tokens_per_iter
    
    # Coverage ratio (how many times dataset is seen)
    coverage_ratio = total_tokens_viewed / total_tokens if total_tokens > 0 else 0
    
    # Calculate recommended iterations for 2-5x coverage
    recommended_min = int((2.0 * total_tokens) / tokens_per_iter)
    recommended_max = int((5.0 * total_tokens) / tokens_per_iter)
    recommended_optimal = int((3.5 * total_tokens) / tokens_per_iter)
    
    return {
        'tokens_per_iter': tokens_per_iter,
        'total_tokens_viewed': total_tokens_viewed,
        'dataset_tokens': total_tokens,
        'coverage_ratio': coverage_ratio,
        'coverage_percentage': coverage_ratio * 100,
        'recommended_min_iters': recommended_min,
        'recommended_max_iters': recommended_max,
        'recommended_optimal_iters': recommended_optimal,
    }


def print_coverage_analysis(coverage, current_iters):
    """
    Print dataset coverage analysis with recommendations.
    
    Args:
        coverage: dict from calculate_dataset_coverage()
        current_iters: Current max_iters setting
    """
    print()
    print("=" * 70)
    print("Dataset Coverage Analysis")
    print("=" * 70)
    
    # Basic stats
    print(f"Dataset size:           {coverage['dataset_tokens']:,} tokens")
    print(f"Tokens per iteration:   {coverage['tokens_per_iter']:,} tokens")
    print(f"Total iterations:       {current_iters:,}")
    print(f"Total tokens to view:   {coverage['total_tokens_viewed']:,} tokens")
    print()
    
    # Coverage ratio
    ratio = coverage['coverage_ratio']
    percentage = coverage['coverage_percentage']
    
    print(f"Dataset coverage:       {ratio:.2f}x ({percentage:.1f}%)")
    
    # Visual indicator
    if ratio < 1.0:
        bar_length = int(ratio * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"Progress:               [{bar}]")
    else:
        full_passes = int(ratio)
        partial = ratio - full_passes
        print(f"Full passes:            {full_passes}x + {partial:.1%} of dataset")
    
    print()
    
    # Assessment and recommendations
    if ratio < 0.5:
        print("⚠️  WARNING: Very low coverage!")
        print("   Your model will only see a small fraction of the data.")
        print("   Risk: Underfitting, poor generalization")
        print()
    elif ratio < 1.0:
        print("⚠️  WARNING: Low coverage")
        print("   Your model won't see the entire dataset even once.")
        print("   Risk: Underfitting")
        print()
    elif ratio < 2.0:
        print("ℹ️  Note: Below recommended coverage")
        print("   Model will see dataset ~1x. Consider training longer.")
        print()
    elif ratio >= 2.0 and ratio <= 5.0:
        print("✅ GOOD: Coverage is in optimal range (2-5x)")
        print("   Model will see dataset multiple times for good learning.")
        print()
    elif ratio > 5.0 and ratio <= 10.0:
        print("ℹ️  Note: High coverage")
        print("   Model will see dataset many times. May overfit on small datasets.")
        print()
    else:  # ratio > 10.0
        print("⚠️  WARNING: Very high coverage")
        print("   Model will see dataset many times. Risk of overfitting.")
        print("   Consider: Reducing max_iters, or use a larger dataset.")
        print()
    
    # Recommendations
    print("Recommendations:")
    print(f"  Minimum (2x):         --max_iters {coverage['recommended_min_iters']:,}")
    print(f"  Optimal (3.5x):       --max_iters {coverage['recommended_optimal_iters']:,}")
    print(f"  Maximum (5x):         --max_iters {coverage['recommended_max_iters']:,}")
    
    # Special recommendations based on ratio
    if ratio < 1.0:
        print()
        print("💡 Suggestion:")
        print(f"   Increase max_iters to at least {coverage['recommended_min_iters']:,}")
        print(f"   for 2x coverage (current: {current_iters:,})")
    elif ratio > 10.0:
        print()
        print("💡 Suggestion:")
        print(f"   Reduce max_iters to {coverage['recommended_max_iters']:,} for 5x coverage")
        print(f"   OR use a larger dataset to avoid overfitting")
    
    print("=" * 70)
    print()


def estimate_training_time(max_iters, eval_interval, time_per_iter_seconds=None):
    """
    Estimate total training time.
    
    Args:
        max_iters: Number of training iterations
        eval_interval: Evaluation frequency
        time_per_iter_seconds: Average time per iteration (optional)
    
    Returns:
        dict with time estimates
    """
    num_evals = max_iters // eval_interval
    
    # Rough estimates if time_per_iter not provided
    if time_per_iter_seconds is None:
        # These are very rough estimates and depend on GPU, model size, etc.
        time_per_iter_seconds = 0.5  # Assume ~0.5 seconds per iteration
    
    training_time_seconds = max_iters * time_per_iter_seconds
    eval_time_seconds = num_evals * 10  # Assume ~10 seconds per evaluation
    total_seconds = training_time_seconds + eval_time_seconds
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return {
        'training_time_seconds': training_time_seconds,
        'eval_time_seconds': eval_time_seconds,
        'total_seconds': total_seconds,
        'hours': hours,
        'minutes': minutes,
        'num_evaluations': num_evals,
    }


def print_training_estimates(max_iters, eval_interval, batch_size, block_size):
    """
    Print estimated training statistics.
    
    Args:
        max_iters: Number of training iterations
        eval_interval: Evaluation frequency
        batch_size: Batch size
        block_size: Context length
    """
    time_est = estimate_training_time(max_iters, eval_interval)
    
    print("Training Estimates:")
    print(f"  Total iterations:     {max_iters:,}")
    print(f"  Evaluations:          {time_est['num_evaluations']}")
    print(f"  Estimated time:       ~{time_est['hours']}h {time_est['minutes']}m")
    print(f"                        (rough estimate, depends on GPU)")

