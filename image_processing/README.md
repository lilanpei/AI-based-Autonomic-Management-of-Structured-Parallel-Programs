# Calibrating Image Processing Duration: Script Overview

## 📋 Overview

`calibrate_direct.py` is a direct calibration script that measures the actual processing time of image processing tasks **without OpenFaaS overhead**. It provides accurate timing models for simulating realistic workloads in the autonomic management system.

**Purpose**: Establish a baseline timing model for image processing tasks that can be used to:
1. Generate realistic task deadlines
2. Simulate processing times in workers
3. Predict system behavior under different loads
4. Validate autonomic scaling decisions

---

## 🎯 Key Features

### **1. Complete Pipeline Calibration**
- Measures **4 sequential stages** of image processing:
  - **Thumbnail Generation**: Resize image to 128x128
  - **Image Compression**: Quantization and normalization
  - **Metadata Extraction**: Statistics and histograms
  - **Format Conversion**: RGB ↔ Grayscale transformations

### **2. Quadratic Timing Model**
- Fits processing time to image size using: **`time = a × size² + b`**
- Provides high accuracy (R² > 0.999)
- Accounts for computational complexity of image operations

### **3. Automatic Configuration Update**
- Saves calibration results to `calibration_results.json`
- **Automatically updates** `orchestrator/configuration.yml`
- Generates visualization in `calibration_plot.png`

### **4. Statistical Rigor**
- Multiple trials per image size (default: 10)
- Calculates mean, std, min, max, percentiles
- Per-stage timing breakdown
- Reproducible with fixed random seed

---

## 🔬 Image Processing Pipeline

### **Stage 1: Thumbnail Generation**
```python
def process_thumbnail_generation(image_size):
    """Complete thumbnail generation workflow"""
    # Generate random image data
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    # Create PIL image
    img = Image.fromarray(image_data)
    
    # Resize to thumbnail (128x128)
    thumbnail = img.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Convert back to array
    result = np.array(thumbnail)
    
    return duration, result.shape
```

**Complexity**: O(size²) - dominated by LANCZOS resampling

---

### **Stage 2: Image Compression**
```python
def process_image_compression(image_size):
    """Image compression workflow"""
    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    # Quantization (reduce color depth: 256 → 8 levels)
    quantized = (image_data // 32) * 32
    
    # Normalization per channel (RGB)
    for channel in range(3):
        channel_data = quantized[:, :, channel].astype(np.float32)
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        normalized = (channel_data - mean) / (std + 1e-8)
        quantized[:, :, channel] = np.clip(normalized * std + mean, 0, 255).astype(np.uint8)
    
    return duration, quantized.shape
```

**Complexity**: O(size²) - per-pixel operations

---

### **Stage 3: Metadata Extraction**
```python
def process_metadata_extraction(image_size):
    """Metadata extraction workflow"""
    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    metadata = {
        "width": image_data.shape[1],
        "height": image_data.shape[0],
        "channels": image_data.shape[2],
    }
    
    # Per-channel statistics (R, G, B)
    for c, color in enumerate(['R', 'G', 'B']):
        channel = image_data[:,:,c]
        metadata[f"{color}_mean"] = float(np.mean(channel))
        metadata[f"{color}_std"] = float(np.std(channel))
        metadata[f"{color}_min"] = int(np.min(channel))
        metadata[f"{color}_max"] = int(np.max(channel))
        
        # Histogram (16 bins)
        hist, _ = np.histogram(channel, bins=16, range=(0, 256))
        metadata[f"{color}_histogram"] = hist.tolist()
    
    return duration, metadata
```

**Complexity**: O(size²) - statistical computations

---

### **Stage 4: Format Conversion**
```python
def process_format_conversion(image_size):
    """Format conversion workflow"""
    # Generate image
    image_data = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    # RGB to Grayscale (weighted conversion)
    grayscale = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Convert back to RGB (3 channels)
    rgb_again = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)
    
    return duration, rgb_again.shape
```

**Complexity**: O(size²) - matrix operations

---

## 📊 Calibration Process

### **Complete Pipeline Processing**
```python
def process_complete_pipeline(image_size):
    """
    Process image through ALL stages sequentially
    This represents the complete workflow for one image
    """
    # Stage 1: Thumbnail generation
    duration1, _ = process_thumbnail_generation(image_size)
    
    # Stage 2: Image compression
    duration2, _ = process_image_compression(image_size)
    
    # Stage 3: Metadata extraction
    duration3, _ = process_metadata_extraction(image_size)
    
    # Stage 4: Format conversion
    duration4, _ = process_format_conversion(image_size)
    
    total_duration = duration1 + duration2 + duration3 + duration4
    
    return total_duration, {
        "thumbnail": duration1,
        "compression": duration2,
        "metadata": duration3,
        "conversion": duration4
    }
```

---

### **Calibration for Each Image Size**
```python
def calibrate_image_size(image_size, num_trials=10):
    """Calibrate processing time for complete image pipeline"""
    times = []
    stage_times = {"thumbnail": [], "compression": [], "metadata": [], "conversion": []}
    
    for trial in range(num_trials):
        # Reset seed for reproducibility
        np.random.seed(SEED + trial)
        
        total_duration, stages = process_complete_pipeline(image_size)
        times.append(total_duration)
        
        # Track individual stage times
        for stage, duration in stages.items():
            stage_times[stage].append(duration)
    
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
        "stage_stats": {
            stage: {"mean": np.mean(durations), "std": np.std(durations)}
            for stage, durations in stage_times.items()
        }
    }
    
    return stats
```

---

## 📐 Timing Model Fitting

### **Quadratic Model**
```python
def fit_timing_model(calibration_results):
    """Fit timing model to calibration data"""
    # Extract data
    sizes = np.array([r["image_size"] for r in calibration_results])
    times = np.array([r["mean"] for r in calibration_results])
    
    # Define model: time = a * size^2 + b
    def quadratic_model(size, a, b):
        return a * size**2 + b
    
    # Fit using curve_fit
    params, covariance = curve_fit(quadratic_model, sizes, times)
    a, b = params
    
    # Calculate R²
    predicted = quadratic_model(sizes, a, b)
    ss_res = np.sum((times - predicted) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        "model": "quadratic",
        "a": float(a),
        "b": float(b),
        "r_squared": float(r_squared),
        "formula": f"time = {a:.2e} * size² + {b:.6f}"
    }
```

**Why Quadratic?**
- Image processing complexity is O(size²)
- Each pixel requires processing
- Fits empirical data with R² > 0.999

---

## 🔄 Automatic Configuration Update

### **Update configuration.yml**
```python
def update_configuration_yml(model, config_path="orchestrator/configuration.yml"):
    """Update configuration.yml with calibration results"""
    import yaml
    
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
    
    # Write back with comments
    with open(config_path, 'w') as f:
        # Write header
        f.write("# Configuration file for OpenFaaS Farm Skeleton\n\n")
        
        # Write all config except calibrated_model
        for key, value in config.items():
            if key != 'calibrated_model':
                yaml.dump({key: value}, f, default_flow_style=False)
        
        # Write calibrated model with timestamp
        f.write("\n# Calibrated timing model\n")
        f.write("# Generated by: python3 calibrate_direct.py\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        yaml.dump({'calibrated_model': config['calibrated_model']}, f)
```

---

## 📈 Task Generation Using Calibrated Model

### **Generate Task with Realistic Deadline**
```python
def generate_task(task_id, task_gen_timestamp, model):
    """Generate task with calibrated timing"""
    # Select image size (weighted distribution)
    image_sizes = [512, 1024, 2048, 4096]
    size_weights = [0.1, 0.3, 0.5, 0.1]  # Favor mid-range sizes
    image_size = np.random.choice(image_sizes, p=size_weights)
    
    # Calculate expected duration using calibrated model
    a = model['a']  # e.g., 1.66e-07
    b = model['b']  # e.g., 0.00376
    expected_duration = a * image_size**2 + b
    
    # Set deadline (2x expected for normal priority)
    deadline_coefficient = 2.0
    task_deadline = deadline_coefficient * expected_duration
    
    task = {
        "task_id": f"task-{task_id:05d}",
        "task_gen_timestamp": task_gen_timestamp,
        "task_application": "image_processing",
        "task_priority": "normal",
        "task_data": {
            "image_size": image_size
        },
        "task_data_size": image_size,
        "task_deadline": task_deadline
    }
    
    return task
```

**Example Deadlines** (with coefficient=2.0):
- 512×512: ~45ms → deadline ~90ms
- 1024×1024: ~175ms → deadline ~350ms
- 2048×2048: ~708ms → deadline ~1.4s
- 4096×4096: ~2.79s → deadline ~5.6s

---

## 🎨 Task Arrival Rate Patterns

### **Multi-Phase Workload Design**

The calibration enables realistic workload generation with varying arrival rates:

```python
def get_phase_arrival_rate(phase, time_in_phase, phase_duration):
    """Get arrival rate for current phase"""
    if phase == 1:
        # Phase 1: Steady Low (0-60s)
        return 0.3  # tasks/second
    
    elif phase == 2:
        # Phase 2: Gradual Ramp-Up (60-120s)
        # Linear increase from 0.3 to 3.0
        progress = time_in_phase / phase_duration
        return 0.3 + (3.0 - 0.3) * progress
    
    elif phase == 3:
        # Phase 3: High Load (120-180s)
        return 3.0  # tasks/second
    
    elif phase == 4:
        # Phase 4: Spike (180-200s)
        return 5.0  # tasks/second (burst)
    
    elif phase == 5:
        # Phase 5: Gradual Decrease (200-240s)
        progress = time_in_phase / phase_duration
        return 5.0 - (5.0 - 0.5) * progress
    
    else:
        return 0.5  # Default low rate
```

### **Workload Characteristics**

| Phase | Duration | Arrival Rate | Total Tasks | Characteristics |
|-------|----------|--------------|-------------|-----------------|
| 1. Steady Low | 60s | 0.3/s | ~18 | Baseline load |
| 2. Ramp-Up | 60s | 0.3→3.0/s | ~99 | Gradual increase |
| 3. High Load | 60s | 3.0/s | ~180 | Sustained high load |
| 4. Spike | 20s | 5.0/s | ~100 | Burst traffic |
| 5. Decrease | 40s | 5.0→0.5/s | ~110 | Graceful decline |
| **Total** | **240s** | **Variable** | **~507** | **Multi-phase** |

---

## 🚀 Usage

### **Run Calibration**
```bash
# Navigate to project root
cd /home/lanpei/AI-based-Autonomic-Management-of-Structured-Parallel-Programs

# Run calibration
python3 calibrate_direct.py
```

### **Expected Output**
```
======================================================================
COMPLETE PIPELINE CALIBRATION
======================================================================

Configuration:
  Image sizes: [512, 1024, 2048, 4096]
  Pipeline stages: Thumbnail → Compression → Metadata → Conversion
  Trials per size: 10
  Method: Complete pipeline (all 4 stages sequentially)
  Random seed: 42 (for reproducibility)

======================================================================
Calibrating 512x512 images (10 trials)
Pipeline: Thumbnail → Compression → Metadata → Conversion
======================================================================
  Trial 1/10: 45.03ms total
    (Thumb: 6.3ms, Comp: 8.3ms, Meta: 18.8ms, Conv: 11.6ms)
  Trial 2/10: 44.16ms total
    (Thumb: 6.0ms, Comp: 7.9ms, Meta: 18.5ms, Conv: 11.8ms)
  ...

  📊 Pipeline Statistics:
    Total Mean:  45.03 ± 2.71 ms
    Min:   43.07 ms
    Max:   52.89 ms
    P50:   44.16 ms
    P95:   49.41 ms

  📊 Average Stage Breakdown:
    Thumbnail   : 6.26ms ± 0.29ms
    Compression : 8.28ms ± 0.90ms
    Metadata    : 18.79ms ± 0.86ms
    Conversion  : 11.61ms ± 0.75ms

... (similar for 1024, 2048, 4096)

======================================================================
Fitting Timing Model
======================================================================

  Model: time = 1.66e-07 * size² + 0.003759
  R² = 0.9999

  Validation:
    512x512: actual=45.03ms, predicted=47.17ms, error=4.8%
    1024x1024: actual=174.76ms, predicted=177.94ms, error=1.8%
    2048x2048: actual=707.56ms, predicted=700.32ms, error=1.0%
    4096x4096: actual=2789.48ms, predicted=2792.06ms, error=0.1%

✅ Results saved to calibration_results.json
✅ Configuration updated: orchestrator/configuration.yml
   Model: time = 1.66e-07 * size² + 0.003759
   R² = 0.9999
✅ Plot saved to calibration_plot.png

======================================================================
CALIBRATION COMPLETE
======================================================================

✅ Calibrated 4 image sizes
✅ Model R² = 0.9999

📝 Model Formula:
   time = 1.66e-07 * size² + 0.003759

📝 Next steps:
  1. Review calibration_results.json
  2. Check calibration_plot.png
  3. Configuration.yml has been updated automatically
  4. Use model in simulated processing
```

---

## 📁 Output Files

### **1. calibration_results.json**
```json
{
  "timestamp": "2025-10-21 17:25:14",
  "seed": 42,
  "model": {
    "model": "quadratic",
    "a": 1.6613189113279248e-07,
    "b": 0.003758800690149,
    "r_squared": 0.9999862963584111,
    "formula": "time = 1.66e-07 * size² + 0.003759"
  },
  "results": [
    {
      "image_size": 512,
      "mean": 0.0450280032120645,
      "std": 0.0027053884975961695,
      "stage_stats": { ... }
    },
    ...
  ]
}
```

### **2. orchestrator/configuration.yml** (Updated Section)
```yaml
# Calibrated timing model
# Generated by: python3 calibrate_direct.py
# Timestamp: 2025-10-21 17:25:14
# To update: Run calibration and execute this script again
calibrated_model:
  a: 1.6613189113279248e-07
  b: 0.003758800690149
  seed: 42
  r_squared: 0.9999862963584111
```

### **3. calibration_plot.png**
Visual plot showing:
- Measured processing times (with error bars)
- Fitted quadratic model curve
- R² value and formula
- All 4 image sizes

---

## 🔧 Configuration

### **Image Sizes**
```python
image_sizes = [512, 1024, 2048, 4096]
```
Covers typical image processing scenarios from thumbnails to high-resolution.

### **Number of Trials**
```python
num_trials = 10
```
Balances statistical accuracy with execution time (~2-3 minutes total).

### **Random Seed**
```python
SEED = 42
```
Ensures reproducibility across runs.

---

## 📊 Calibration Results (Example)

### **Timing Breakdown by Image Size**

| Image Size | Total Time | Thumbnail | Compression | Metadata | Conversion |
|------------|------------|-----------|-------------|----------|------------|
| 512×512 | 45.0ms | 6.3ms (14%) | 8.3ms (18%) | 18.8ms (42%) | 11.6ms (26%) |
| 1024×1024 | 174.8ms | 21.5ms (12%) | 34.1ms (19%) | 73.8ms (42%) | 45.3ms (26%) |
| 2048×2048 | 707.6ms | 80.9ms (11%) | 147.7ms (21%) | 289.6ms (41%) | 187.7ms (27%) |
| 4096×4096 | 2789.5ms | 323.6ms (12%) | 561.6ms (20%) | 1145.9ms (41%) | 751.9ms (27%) |

**Observations**:
- Metadata extraction dominates (~40-42% of total time)
- Consistent stage proportions across sizes
- Quadratic scaling confirmed

---

## 🎯 Integration with Baseline Runner

### **Load Model in Baseline**
```python
# In baseline_runner_openfaas.py
import json

def load_calibration_model():
    """Load calibrated timing model"""
    with open("calibration_results.json", 'r') as f:
        data = json.load(f)
    return data["model"]

# Use in task generation
model = load_calibration_model()
a, b = model['a'], model['b']

# Calculate expected duration
expected_duration = a * image_size**2 + b
task_deadline = 2.0 * expected_duration  # 2x for normal priority
```

---

## ✅ Benefits

1. **Realistic Timing**: Based on actual processing, not estimates
2. **Accurate Deadlines**: Proportional to actual workload
3. **Reproducible**: Fixed seed ensures consistent results
4. **Automated**: Updates configuration automatically
5. **Validated**: High R² (>0.999) confirms model accuracy
6. **Comprehensive**: Measures complete 4-stage pipeline

---

## 🔄 Re-calibration

Run calibration again if:
- Hardware changes (CPU, memory)
- Algorithm changes (different image processing)
- Python/library versions update
- Need different image sizes

Simply run:
```bash
python3 calibrate_direct.py
```

Configuration will be updated automatically!

---

## 📝 Summary

`calibrate_direct.py` provides:
- ✅ **Accurate timing model** for image processing (R² > 0.999)
- ✅ **4-stage pipeline** calibration (Thumbnail → Compression → Metadata → Conversion)
- ✅ **Automatic configuration update** to `orchestrator/configuration.yml`
- ✅ **Statistical rigor** with multiple trials and percentiles
- ✅ **Visualization** of calibration results
- ✅ **Reproducibility** with fixed random seed
- ✅ **Integration-ready** for baseline and RL experiments

**Use this calibration to generate realistic workloads and validate autonomic scaling decisions!** 🚀
