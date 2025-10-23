#!/usr/bin/env python3
"""
Direct Calibration Script
Directly measures processing time without OpenFaaS overhead
"""

import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

def process_thumbnail_generation(image_size):
    """Complete thumbnail generation workflow"""
    start_time = time.perf_counter()

    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # Create PIL image
    img = Image.fromarray(image_data)

    # Resize to thumbnail
    thumbnail = img.resize((128, 128), Image.Resampling.LANCZOS)

    # Convert back to array
    result = np.array(thumbnail)

    end_time = time.perf_counter()
    duration = end_time - start_time

    return duration, result.shape

def process_image_compression(image_size):
    """Image compression workflow"""
    start_time = time.perf_counter()

    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # Quantization (reduce color depth)
    quantized = (image_data // 32) * 32

    # Normalization per channel
    for channel in range(3):
        channel_data = quantized[:, :, channel].astype(np.float32)
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        normalized = (channel_data - mean) / (std + 1e-8)
        quantized[:, :, channel] = np.clip(normalized * std + mean, 0, 255).astype(np.uint8)

    end_time = time.perf_counter()
    duration = end_time - start_time

    return duration, quantized.shape

def process_metadata_extraction(image_size):
    """Metadata extraction workflow"""
    start_time = time.perf_counter()

    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    metadata = {
        "width": image_data.shape[1],
        "height": image_data.shape[0],
        "channels": image_data.shape[2],
    }

    # Per-channel statistics
    for c, color in enumerate(['R', 'G', 'B']):
        channel = image_data[:,:,c]
        metadata[f"{color}_mean"] = float(np.mean(channel))
        metadata[f"{color}_std"] = float(np.std(channel))
        metadata[f"{color}_min"] = int(np.min(channel))
        metadata[f"{color}_max"] = int(np.max(channel))

        # Histogram
        hist, _ = np.histogram(channel, bins=16, range=(0, 256))
        metadata[f"{color}_histogram"] = hist.tolist()

    end_time = time.perf_counter()
    duration = end_time - start_time

    return duration, metadata

def process_format_conversion(image_size):
    """Format conversion workflow"""
    start_time = time.perf_counter()

    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # RGB to Grayscale
    grayscale = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])

    # Convert back to RGB
    rgb_again = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)

    end_time = time.perf_counter()
    duration = end_time - start_time

    return duration, rgb_again.shape

def process_complete_pipeline(image_size):
    """
    Process image through ALL stages sequentially
    This represents the complete workflow for one image
    """
    start_time = time.perf_counter()

    # Stage 1: Thumbnail generation
    duration1, _ = process_thumbnail_generation(image_size)

    # Stage 2: Image compression
    duration2, _ = process_image_compression(image_size)

    # Stage 3: Metadata extraction
    duration3, _ = process_metadata_extraction(image_size)

    # Stage 4: Format conversion
    duration4, _ = process_format_conversion(image_size)

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    return total_duration, {
        "thumbnail": duration1,
        "compression": duration2,
        "metadata": duration3,
        "conversion": duration4
    }

def calibrate_image_size(image_size, num_trials=10):
    """Calibrate processing time for complete image pipeline"""
    print(f"\n{'='*70}")
    print(f"Calibrating {image_size}x{image_size} images ({num_trials} trials)")
    print(f"Pipeline: Thumbnail → Compression → Metadata → Conversion")
    print(f"{'='*70}")

    times = []
    stage_times = {
        "thumbnail": [],
        "compression": [],
        "metadata": [],
        "conversion": []
    }

    for trial in range(num_trials):
        # Reset seed for each trial to ensure reproducibility
        np.random.seed(SEED + trial)

        total_duration, stages = process_complete_pipeline(image_size)
        times.append(total_duration)

        # Track individual stage times
        for stage, duration in stages.items():
            stage_times[stage].append(duration)

        print(f"  Trial {trial+1}/{num_trials}: {total_duration*1000:.2f}ms total")
        print(f"    (Thumb: {stages['thumbnail']*1000:.1f}ms, "
              f"Comp: {stages['compression']*1000:.1f}ms, "
              f"Meta: {stages['metadata']*1000:.1f}ms, "
              f"Conv: {stages['conversion']*1000:.1f}ms)")

    # Calculate statistics
    stats = {
        "image_size": image_size,
        "num_trials": num_trials,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "raw_times": times,
        "stage_stats": {
            stage: {
                "mean": np.mean(durations),
                "std": np.std(durations)
            }
            for stage, durations in stage_times.items()
        }
    }

    print(f"\n  📊 Pipeline Statistics:")
    print(f"    Total Mean:  {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")
    print(f"    Min:   {stats['min']*1000:.2f} ms")
    print(f"    Max:   {stats['max']*1000:.2f} ms")
    print(f"    P50:   {stats['p50']*1000:.2f} ms")
    print(f"    P95:   {stats['p95']*1000:.2f} ms")

    print(f"\n  📊 Average Stage Breakdown:")
    for stage, stage_stat in stats['stage_stats'].items():
        print(f"    {stage.capitalize():12s}: {stage_stat['mean']*1000:.2f}ms ± {stage_stat['std']*1000:.2f}ms")

    return stats

def fit_timing_model(calibration_results):
    """Fit timing model to calibration data"""
    print(f"\n{'='*70}")
    print("Fitting Timing Model")
    print(f"{'='*70}")

    # Extract data
    sizes = np.array([r["image_size"] for r in calibration_results])
    times = np.array([r["mean"] for r in calibration_results])

    # Define model: time = a * size^2 + b
    def quadratic_model(size, a, b):
        return a * size**2 + b

    # Fit
    try:
        params, covariance = curve_fit(quadratic_model, sizes, times)
        a, b = params

        # Calculate R^2
        predicted = quadratic_model(sizes, a, b)
        ss_res = np.sum((times - predicted) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\n  Model: time = {a:.2e} * size² + {b:.6f}")
        print(f"  R² = {r_squared:.4f}")

        # Validate predictions
        print(f"\n  Validation:")
        for size, actual in zip(sizes, times):
            predicted = quadratic_model(size, a, b)
            error = abs(predicted - actual) / actual * 100
            print(f"    {size}x{size}: actual={actual*1000:.2f}ms, "
                  f"predicted={predicted*1000:.2f}ms, error={error:.1f}%")

        return {
            "model": "quadratic",
            "a": float(a),
            "b": float(b),
            "r_squared": float(r_squared),
            "formula": f"time = {a:.2e} * size² + {b:.6f}"
        }

    except Exception as e:
        print(f"  ❌ Model fitting failed: {e}")
        return None

def plot_calibration_results(results, model, output_file="calibration_plot.png"):
    """Plot calibration results"""
    try:
        sizes = [r["image_size"] for r in results]
        means = [r["mean"] * 1000 for r in results]
        stds = [r["std"] * 1000 for r in results]

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot data points with error bars
        plt.errorbar(sizes, means, yerr=stds, fmt='o', capsize=5, 
                    label='Measured', markersize=8, color='blue')

        # Plot fitted model
        if model:
            size_range = np.linspace(min(sizes), max(sizes), 100)
            predicted = (model["a"] * size_range**2 + model["b"]) * 1000
            plt.plot(size_range, predicted, 'r--', linewidth=2,
                    label=f'Model: {model["a"]:.2e}*size² + {model["b"]:.4f}')

        plt.xlabel('Image Size (pixels)', fontsize=12)
        plt.ylabel('Processing Time (ms)', fontsize=12)
        plt.title('Thumbnail Generation Processing Time Calibration', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to {output_file}")

    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")

def save_calibration_results(results, model, output_file="calibration_results.json"):
    """Save calibration results to file"""
    # Remove raw_times for cleaner JSON
    results_clean = []
    for r in results:
        r_clean = {k: v for k, v in r.items() if k != 'raw_times'}
        results_clean.append(r_clean)

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "model": model,
        "results": results_clean
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Results saved to {output_file}")

def update_configuration_yml(model, config_path="../orchestrator/configuration.yml"):
    """Update configuration.yml with calibration results"""
    import yaml

    try:
        # Read existing configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update calibrated model section
        config['calibrated_model'] = {
            'a': model['a'],
            'b': model['b'],
            'seed': SEED,
            'r_squared': model['r_squared']
        }

        # Write back to file
        with open(config_path, 'w') as f:
            # Write with comments preserved
            f.write("# Configuration file for OpenFaaS Farm Skeleton\n")
            f.write("# Auto-generated sections will be marked\n\n")

            # Write all config except calibrated_model
            for key, value in config.items():
                if key != 'calibrated_model':
                    yaml.dump({key: value}, f, default_flow_style=False, sort_keys=False)

            # Write calibrated model with comments
            f.write("\n\n")
            f.write("# Calibrated timing model\n")
            f.write("# Generated by: python3 calibrate_direct.py\n")
            f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# To update: Run calibration and execute this script again\n")
            yaml.dump({'calibrated_model': config['calibrated_model']}, f, 
                     default_flow_style=False, sort_keys=False)

        print(f"✅ Configuration updated: {config_path}")
        print(f"   Model: time = {model['a']:.2e} * size² + {model['b']:.6f}")
        print(f"   R² = {model['r_squared']:.4f}")

    except Exception as e:
        print(f"⚠️  Could not update configuration.yml: {e}")
        print(f"   Please manually update the calibrated_model section")

def main():
    """Main calibration routine"""
    print(f"\n{'='*70}")
    print("COMPLETE PIPELINE CALIBRATION")
    print(f"{'='*70}")

    # Configuration
    image_sizes = [512, 1024, 2048, 4096]
    num_trials = 10

    print(f"\nConfiguration:")
    print(f"  Image sizes: {image_sizes}")
    print(f"  Pipeline stages: Thumbnail → Compression → Metadata → Conversion")
    print(f"  Trials per size: {num_trials}")
    print(f"  Method: Complete pipeline (all 4 stages sequentially)")
    print(f"  Random seed: {SEED} (for reproducibility)")

    # Run calibration for each image size
    calibration_results = []

    for image_size in image_sizes:
        stats = calibrate_image_size(image_size, num_trials)
        calibration_results.append(stats)

    # Fit timing model
    model = fit_timing_model(calibration_results)

    # Save results
    save_calibration_results(calibration_results, model)

    # Update configuration.yml
    if model:
        update_configuration_yml(model)

    # Plot results
    plot_calibration_results(calibration_results, model)

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Calibrated {len(calibration_results)} image sizes")
    if model:
        print(f"✅ Model R² = {model['r_squared']:.4f}")
        print(f"\n📝 Model Formula:")
        print(f"   {model['formula']}")
    print(f"\n📝 Next steps:")
    print(f"  1. Review calibration_results.json")
    print(f"  2. Check calibration_plot.png")
    print(f"  3. Configuration.yml has been updated automatically")
    print(f"  4. Use model in simulated processing")
    print()

if __name__ == "__main__":
    main()
