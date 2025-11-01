# Calibrating Image Processing Duration: Script Overview

## 📋 Overview

`calibrate_direct.py` is a direct calibration script that measures the actual processing time of image processing tasks **without OpenFaaS overhead**. It provides accurate timing models for simulating realistic workloads in the autonomic management system.

**Purpose**: Establish a baseline timing model for image processing tasks that can be used to:
1. Generate realistic task deadlines
2. Simulate processing times in workers (using `time.sleep()`)
3. Predict system behavior under different loads
4. Validate autonomic scaling decisions
5. Configure HPA (Horizontal Pod Autoscaler) thresholds

**Latest Calibration**: October 26, 2025 at 23:06:25
- **Model**: `time = 1.71e-07 × size² + 0.00167` seconds
- **R² = 0.9999** (excellent fit)
- **Configuration**: Automatically updated in `utilities/configuration.yml`
- **Sampling Defaults**: Mean 1.5 s, gamma shape 4.0 (configurable)

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
- Provides high accuracy (R² = 0.9999)
- Accounts for computational complexity of image operations
- **Current coefficients**:
  - `a = 1.7099013205346125e-07`
  - `b = 0.0016654777540409116` seconds
  - Model outputs time in **seconds** (not milliseconds)

### **3. Automatic Configuration Update**
- Saves calibration results to `calibration_results.json`
- **Automatically updates** `utilities/configuration.yml` (not orchestrator/)
- Generates visualization in `calibration_plot.png`
- Updates calibrated coefficients, processing-time sampling defaults, and phase definitions used by the emitter/worker stack

### **4. Statistical Rigor**
- Multiple trials per image size (default: 10)
- Calculates mean, std, min, max, percentiles
- Per-stage timing breakdown
- Reproducible with fixed random seed

---

## 🔬 Image Processing Pipeline

The calibration measures a **4-stage sequential pipeline**:

1. **Thumbnail Generation** - Resize to 128×128 using LANCZOS resampling
2. **Image Compression** - Quantization (256→8 levels) + per-channel normalization
3. **Metadata Extraction** - RGB statistics (mean/std/min/max) + 16-bin histograms
4. **Format Conversion** - RGB→Grayscale→RGB using weighted conversion

**Complexity**: O(size²) for all stages (per-pixel operations)

---

## 📊 Calibration Process

**Method**:
1. Process complete 4-stage pipeline for each image size
2. Run 10 trials per size with fixed random seed (42)
3. Calculate statistics: mean, std, min, max, p50, p95
4. Track per-stage timing breakdown
5. Fit quadratic model: `time = a × size² + b`
6. Auto-update `utilities/configuration.yml`

---

## 📐 Timing Model

**Quadratic Model**: `time = a × size² + b`
- **Why quadratic?** Image processing is O(size²) - every pixel requires processing
- **Fitting**: Uses `scipy.optimize.curve_fit` on measured data
- **Validation**: R² = 0.9999 confirms excellent fit
- **Auto-update**: Writes coefficients to `utilities/configuration.yml`

---

## 📈 Calibrated Processing Times

**Current Model** (October 2025): `time = 1.71e-07 × size² + 0.00167` seconds

| Image Size | Processing Time | Deadline (2×) |
|------------|----------------|---------------|
| 512×512 | 0.046s (46ms) | 0.09s |
| 1024×1024 | 0.181s (181ms) | 0.35s |
| 2048×2048 | 0.719s (719ms) | 1.42s |
| 4096×4096 | 2.870s (2870ms) | 5.58s |

**Processing-Time Sampling** (emitter `utils.py`):
- Draws a processing time from a gamma distribution (`shape`, `target_mean_processing_time`) defined in `utilities/configuration.yml`
- Clips the draw to calibrated min/max, then inverts the quadratic model to compute the corresponding image size
- Recomputes the final simulated duration from calibrated coefficients to stay consistent with the model
- Defaults target mean to **1.5 seconds** with shape **4.0**, but both are configurable

---

## 🎨 Phase-Based Workload Generation

**Phase Definitions** (default `phase_definitions` in `utilities/configuration.yml`):

| Phase | Rate Multiplier | Duration | Pattern | Notes |
|-------|-----------------|----------|---------|-------|
| 1. Steady Low Load | 30% | 60 s | steady | Baseline warm-up |
| 2. Steady High Load | 150% | 60 s | steady | Forces rapid scale-up |
| 3. Slow Oscillation | 100% avg | 60 s | oscillation_slow | 0.5–1.5×, 1 cycle |
| 4. Fast Oscillation | 100% avg | 60 s | oscillation_fast | 0.3–1.7×, 4 cycles |
| **Total** | — | **240 s** | — | ≈1140 tasks at base_rate 300 |

Customize the number of phases, multipliers, durations, and oscillation bounds directly in `utilities/configuration.yml`; emitter picks them up through `phase_definitions` at runtime.

**Autoscaling Requirements**:
- Single worker capacity: 0.67 tasks/s (1 / 1.5s)
- Phase 2 peak: 7.5 tasks/s → **needs ~11 workers**

---

## 🚀 Usage

### **Run Calibration**
```bash
# Navigate to image_processing directory
cd image_processing

# Run calibration
python3 calibrate_direct.py

# Output files:
# - calibration_results.json (detailed results)
# - calibration_plot.png (visualization)
# - ../utilities/configuration.yml (auto-updated)
```

### **Expected Output**
```
COMPLETE PIPELINE CALIBRATION
Image sizes: [512, 1024, 2048, 4096]
Trials per size: 10

Calibrating 512x512... Mean: 42.6ms ± 4.5ms
Calibrating 1024x1024... Mean: 170.2ms ± 5.3ms
Calibrating 2048x2048... Mean: 737.5ms ± 60.9ms
Calibrating 4096x4096... Mean: 2866.5ms ± 234ms

Model: time = 1.71e-07 × size² + 0.00167
R² = 0.9999

✅ Results saved to calibration_results.json
✅ Configuration updated: utilities/configuration.yml
✅ Plot saved to calibration_plot.png
```

---

## 📁 Output Files

### **1. calibration_results.json**
- Timestamp, seed, model coefficients (a, b, R²)
- Per-size statistics: mean, std, min, max, p50, p95
- Per-stage breakdown: thumbnail, compression, metadata, conversion

### **2. utilities/configuration.yml**
- Auto-updated with calibrated model coefficients
- Used by emitter and worker for task generation

### **3. calibration_plot.png**
- Measured times with error bars
- Fitted quadratic curve
- R² value and formula

---

## 🔧 Configuration

- **Trials per size**: 10 - balances accuracy with speed (~2-3 min total)
- **Random seed**: 42 - ensures reproducibility
- **Target mean processing time**: 1.5 s (gamma-based sampling default)
- **Gamma shape**: 4.0 (controls spread; tweak in configuration)

---

## 📊 Calibration Results (Example)

### **Timing Breakdown by Image Size**

| Image Size | Total Time | Thumbnail | Compression | Metadata | Conversion |
|------------|------------|-----------|-------------|----------|------------|
| 512×512 | 42.6ms | 5.9ms (14%) | 7.9ms (18%) | 17.6ms (41%) | 11.1ms (26%) |
| 1024×1024 | 170.2ms | 20.6ms (12%) | 33.8ms (20%) | 70.7ms (42%) | 44.9ms (26%) |
| 2048×2048 | 737.5ms | 78.0ms (11%) | 150.4ms (20%) | 291.8ms (40%) | 214.1ms (29%) |
| 4096×4096 | 2866.5ms | 310.7ms (11%) | 600.3ms (21%) | 1136.8ms (40%) | 809.7ms (28%) |

**Observations**:
- Metadata extraction dominates (~40-42% of total time)
- Consistent stage proportions across sizes
- Quadratic scaling confirmed

---

## 🎯 Integration

**Emitter** (`utils.py`):
- Loads model from `utilities/configuration.yml`
- Calculates: `processing_time = a × size² + b`
- Sets task deadline: `2.0 × expected_duration`

**Worker** (`handler.py`):
- Reads `task_processing_time_simulated` from task
- Simulates processing: `time.sleep(processing_time)`

---

## ✅ Benefits

1. **Realistic Timing**: Based on actual processing, not estimates
2. **Accurate Deadlines**: Proportional to actual workload (2× expected time)
3. **Reproducible**: Fixed seed ensures consistent results
4. **Automated**: Updates `utilities/configuration.yml` automatically
5. **Validated**: High R² = 0.9999 confirms model accuracy
6. **Comprehensive**: Measures complete 4-stage pipeline
7. **Autoscaling-ready**: Processing times trigger HPA scaling
8. **QoS-enabled**: Realistic deadlines for quality-of-service evaluation

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
- ✅ **Accurate timing model** for image processing (R² = 0.9999)
- ✅ **4-stage pipeline** calibration (Thumbnail → Compression → Metadata → Conversion)
- ✅ **Automatic configuration update** to `utilities/configuration.yml`
- ✅ **Statistical rigor** with 10 trials per size, mean/std/percentiles
- ✅ **Visualization** of calibration results (`calibration_plot.png`)
- ✅ **Reproducibility** with fixed random seed (42)
- ✅ **Integration-ready** for emitter, worker, and autoscaling
- ✅ **Phase-based workload** generation with varying arrival rates
- ✅ **HPA-compatible** processing times for autoscaling validation

**Latest Calibration** (Oct 26, 2025):
- Model: `time = 1.71e-07 × size² + 0.00167` seconds
- Average processing: 1.5s
- Worker capacity: 0.67 tasks/sec
- Phase 2 peak load: 7.5 tasks/sec → **requires 11 workers**

**Use this calibration to generate realistic workloads and validate autonomic scaling decisions!** 🚀
