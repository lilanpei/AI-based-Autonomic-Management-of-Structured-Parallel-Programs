#!/usr/bin/env python3
"""
Test Task Generator and Plot Arrival Rates

This script simulates the four-phase task generator and plots:
- Task arrival rate over time
- Phase boundaries
- Expected vs actual rates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys


def get_phase_rate(phase, base_rate, time_in_phase, phase_duration):
    """
    Calculate task arrival rate for current phase and time
    (Same logic as task_generator.py)
    """
    if phase == 1:
        # Phase 1: Steady Low Load (30% of base rate)
        return base_rate * 0.3

    elif phase == 2:
        # Phase 2: Steady High Load (150% of base rate)
        return base_rate * 1.5

    elif phase == 3:
        # Phase 3: Slow Oscillation (period = phase_duration)
        # Oscillates between 50% and 150% of base rate
        progress = time_in_phase / phase_duration  # 0 to 1
        oscillation = 0.5 * np.sin(2 * np.pi * progress) + 1.0  # 0.5 to 1.5
        return base_rate * oscillation

    elif phase == 4:
        # Phase 4: Fast Oscillation (4 cycles per phase)
        # Oscillates between 30% and 170% of base rate
        progress = time_in_phase / phase_duration  # 0 to 1
        oscillation = 0.7 * np.sin(8 * np.pi * progress) + 1.0  # 0.3 to 1.7
        return base_rate * oscillation

    else:
        return base_rate


def simulate_arrival_rates(base_rate, phase_duration, window_duration):
    """
    Simulate arrival rates for all four phases

    Returns:
        times: Array of time points
        rates: Array of arrival rates
        phases: Array of phase numbers
        phase_boundaries: List of phase boundary times
    """
    total_duration = phase_duration * 4
    num_windows = int(total_duration / window_duration)

    times = []
    rates = []
    phases = []

    for window in range(num_windows):
        current_time = window * window_duration

        # Determine current phase
        if current_time < phase_duration:
            phase = 1
            time_in_phase = current_time
        elif current_time < phase_duration * 2:
            phase = 2
            time_in_phase = current_time - phase_duration
        elif current_time < phase_duration * 3:
            phase = 3
            time_in_phase = current_time - phase_duration * 2
        else:
            phase = 4
            time_in_phase = current_time - phase_duration * 3

        # Calculate rate
        rate = get_phase_rate(phase, base_rate, time_in_phase, phase_duration)
        
        times.append(current_time)
        rates.append(rate)
        phases.append(phase)

    phase_boundaries = [0, phase_duration, phase_duration*2, phase_duration*3, phase_duration*4]

    return np.array(times), np.array(rates), np.array(phases), phase_boundaries


def plot_arrival_rates(base_rate, phase_duration, window_duration, output_file='task_arrival_rates.png'):
    """
    Plot arrival rates for all four phases
    """
    # Simulate rates
    times, rates, phases, phase_boundaries = simulate_arrival_rates(
        base_rate, phase_duration, window_duration
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Arrival Rate Over Time

    # Plot rate line
    ax1.plot(times / 60, rates, 'b-', linewidth=2, label='Arrival Rate')

    # Add phase backgrounds
    colors = ['#e8f4f8', '#fff4e6', '#f0f8e8', '#fce8f0']
    phase_names = ['Phase 1: Steady Low', 'Phase 2: Steady High', 
                   'Phase 3: Slow Oscillation', 'Phase 4: Fast Oscillation']

    for i in range(4):
        start = phase_boundaries[i] / 60
        end = phase_boundaries[i+1] / 60
        ax1.axvspan(start, end, alpha=0.2, color=colors[i], label=phase_names[i])

    # Add phase boundary lines
    for boundary in phase_boundaries[1:-1]:
        ax1.axvline(boundary / 60, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add reference lines
    ax1.axhline(base_rate, color='green', linestyle=':', linewidth=1, alpha=0.7, label=f'Base Rate ({base_rate} tasks/min)')
    ax1.axhline(base_rate * 0.3, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(base_rate * 1.5, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Arrival Rate (tasks/min)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Four-Phase Task Arrival Pattern (Base Rate: {base_rate} tasks/min)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_xlim(0, phase_boundaries[-1] / 60)
    ax1.set_ylim(0, base_rate * 1.8)

    # Plot 2: Rate Distribution by Phase

    phase_data = []
    phase_labels = []

    for phase_num in range(1, 5):
        phase_mask = phases == phase_num
        phase_rates = rates[phase_mask]
        phase_data.append(phase_rates)

        mean_rate = np.mean(phase_rates)
        min_rate = np.min(phase_rates)
        max_rate = np.max(phase_rates)

        phase_labels.append(f'Phase {phase_num}\n({min_rate:.1f}-{max_rate:.1f})\nμ={mean_rate:.1f}')

    # Box plot
    bp = ax2.boxplot(phase_data, labels=phase_labels, patch_artist=True,
                     widths=0.6, showmeans=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add base rate reference
    ax2.axhline(base_rate, color='green', linestyle=':', linewidth=2, alpha=0.7, 
                label=f'Base Rate ({base_rate} tasks/min)')
    
    ax2.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Arrival Rate (tasks/min)', fontsize=12, fontweight='bold')
    ax2.set_title('Rate Distribution by Phase', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', fontsize=10)

    # Add statistics text

    stats_text = f"""Configuration:
Base Rate: {base_rate} tasks/min
Phase Duration: {phase_duration}s ({phase_duration/60:.1f} min)
Window Duration: {window_duration}s
Total Duration: {phase_boundaries[-1]}s ({phase_boundaries[-1]/60:.1f} min)

Phase Statistics:
Phase 1: {np.mean(rates[phases==1]):.1f} tasks/min (steady)
Phase 2: {np.mean(rates[phases==2]):.1f} tasks/min (steady)
Phase 3: {np.mean(rates[phases==3]):.1f} tasks/min (slow osc)
Phase 4: {np.mean(rates[phases==4]):.1f} tasks/min (fast osc)"""

    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")

    return fig


def print_phase_summary(base_rate, phase_duration, window_duration):
    """
    Print summary statistics for each phase
    """
    times, rates, phases, phase_boundaries = simulate_arrival_rates(
        base_rate, phase_duration, window_duration
    )

    print(f"\n{'='*70}")
    print(f"FOUR-PHASE ARRIVAL PATTERN SUMMARY")
    print(f"{'='*70}")
    print(f"Base Rate: {base_rate} tasks/min")
    print(f"Phase Duration: {phase_duration}s ({phase_duration/60:.1f} min)")
    print(f"Window Duration: {window_duration}s")
    print(f"Total Duration: {phase_boundaries[-1]}s ({phase_boundaries[-1]/60:.1f} min)")
    print(f"{'='*70}\n")

    phase_names = [
        "Steady Low Load",
        "Steady High Load",
        "Slow Oscillation",
        "Fast Oscillation"
    ]

    for phase_num in range(1, 5):
        phase_mask = phases == phase_num
        phase_rates = rates[phase_mask]

        print(f"Phase {phase_num}: {phase_names[phase_num-1]}")
        print(f"  Time Range: {phase_boundaries[phase_num-1]:.0f}s - {phase_boundaries[phase_num]:.0f}s")
        print(f"  Duration: {phase_duration}s ({phase_duration/60:.1f} min)")
        print(f"  Rate Range: {np.min(phase_rates):.2f} - {np.max(phase_rates):.2f} tasks/min")
        print(f"  Mean Rate: {np.mean(phase_rates):.2f} tasks/min")
        print(f"  Std Dev: {np.std(phase_rates):.2f} tasks/min")

        # Estimate total tasks
        total_tasks = np.sum(phase_rates) * (window_duration / 60)
        print(f"  Estimated Tasks: ~{int(total_tasks)} tasks")
        print()

    # Overall statistics
    total_tasks_all = np.sum(rates) * (window_duration / 60)
    avg_rate = np.mean(rates)

    print(f"{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total Estimated Tasks: ~{int(total_tasks_all)} tasks")
    print(f"Average Rate: {avg_rate:.2f} tasks/min")
    print(f"Min Rate: {np.min(rates):.2f} tasks/min")
    print(f"Max Rate: {np.max(rates):.2f} tasks/min")
    print(f"{'='*70}\n")


def main():
    """Main test function"""

    # Parse arguments
    if len(sys.argv) == 4:
        base_rate = int(sys.argv[1])
        phase_duration = int(sys.argv[2])
        window_duration = int(sys.argv[3])
    else:
        # Default values
        print("Usage: python test_task_generator.py <base_rate> <phase_duration> <window_duration>")
        print("Using default values: 300 60 1\n")
        base_rate = 300
        phase_duration = 60
        window_duration = 1

    # Print summary
    print_phase_summary(base_rate, phase_duration, window_duration)

    # Generate plot
    print("Generating plot...")
    plot_arrival_rates(base_rate, phase_duration, window_duration)

    print("\n✅ Test complete!")
    print("\nTo run with custom parameters:")
    print("  python test_task_generator.py 300 60 1")
    print("\nTo view the plot:")
    print("  open task_arrival_rates.png")


if __name__ == "__main__":
    main()
