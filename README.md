# Hybrid Sensor Fault Diagnosis and Evaluation Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, reproducible experimental framework for simulating, detecting, and evaluating sensor faults in IMU-like time-series data using a combination of **Classical State Estimation (Kalman Filtering)** and **Deep Learning (Autoencoders)**.

## 🚀 Overview

In critical systems (robotics, aerospace, autonomous vehicles), sensor reliability is paramount. This project provides a "digital twin" environment to:
- **Simulate** realistic IMU acceleration signals.
- **Inject** 5 types of common sensor faults (Drift, Spike, Noise, Dropout, Stuck).
- **Detect** anomalies using 5 different strategies, including a **Hybrid Detector** that fuses physics-based residuals with ML anomaly scores.
- **Evaluate** performance using precision, recall, latency, and ROC/AUC metrics.

## 🛠️ Architecture

The project is designed with modularity at its core:

- `data_generation.py`: Synthesizes 1D acceleration signals with realistic motion dynamics.
- `fault_injection.py`: Parameterized fault models for bias drift, sudden spikes, and more.
- `state_estimation.py`: Kalman Filter implementation using `filterpy` for real-time tracking.
- `detectors.py`: Individual detection logic (Threshold, Z-Score, NIS, Autoencoder).
- `hybrid_logic.py`: Fusion logic that merges Kalman innovation and ML reconstruction error.
- `evaluation.py`: Performance measurement engine (F1, FPR, Latency).
- `experiments.py`: Headless CLI tool for running batch parameter sweeps.
- `main_demo.py`: Visual demonstration script with full plotting capabilities.

## 📊 Sample Results

![Detection Process](demo_results.png)

The framework evaluates:
1. **Raw Measurement vs. Estimated State**
2. **Innovation Analysis (NIS)**: Monitoring the consistency of the Kalman filter.
3. **Hybrid Confidence**: A 0-100 score indicating the probability of a fault.
4. **Detection Timelines**: Comparing when different detectors "trigger" relative to the fault onset.

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/HybridFaultDiagnosis.git
   cd HybridFaultDiagnosis
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Running the Visual Demo
To run the end-to-end simulation with a visual dashboard of the results:
```bash
python main_demo.py
```

### Running Batch Experiments
To run headless experiments that sweep through noise levels and fault severities:
```bash
python experiments.py
```
This will generate `experiment_results.csv` containing raw performance data for all detectors.

## 🧪 Fault Models Supported

| Fault | Description |
| :--- | :--- |
| **Bias Drift** | Linear increase in sensor offset over time. |
| **Spike** | Sudden, extreme deviation in a single or few samples. |
| **Increased Noise** | Jump in the variance of measurement noise. |
| **Stuck-at** | Sensor output freezes at a constant value. |
| **Dropout** | Intermittent loss of data (signal goes to zero). |

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
