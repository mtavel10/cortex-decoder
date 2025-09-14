# Cortex Decoder

**Author:** Maddy Tavel  
**Lab:** Maclean Lab @ University of Chicago  
**Project:** Measuring Neuro-Behavior Relationships During Motor Learning Using
Decoder Performance

## Overview
This pipeline combines calcium imaging (neural spike probabilities) with kinematic tracking to predict mouse paw-positions from motor cortex activity. The system supports multiple analysis modes including behavior-specific decoding, cell-type comparisons, and cross-day analysis.


## Features
- **Data cleaning, pre-processing, and Integration**: Interpolates paw-kinematic data to calcium timestamps. 
- **Behavioral context analysis**: Decodes neural activity during different behaviors
associated with learning a motor task (reach, grasp, carry, etc.)
- **Cell type comparison**: Compares decoding performance between inhibitory and excitatory neurons
- **Cross-day analysis**: Analyzes neural consistency across multiple recording sessions
- **Temporal lag analysis**: Accounts for calcium indicator delay in neural responses
- **Comprehensive Validation**: Multiple cross-validation approaches for robust performance assessment


## Project Structure
```
cortex-decoder/
├── decode.py                    # Main decoder classes and analysis functions
├── mouse.py                     # MouseDay data class for session management
├── interpolate.py              # Kinematic data interpolation utilities
├── src/
│   ├── IO.py                   # Data loading/saving functions
│   ├── fextract.py             # Data extraction utilities
│   └── curr_data_paths.txt     # Current data paths reference
├── requirements.txt            # Python package dependencies
├── environment.yml             # Conda environment specification
├── plots.py
├──plots                        # Data Visuals + results
├──model_eval                   # Debugging output/evals
└── [mouseID]/                  # Data directories (e.g., mouse25/)
    └── [YYYYMMDD]/            # Session dates
        ├── calcium/           # Neural data files
        │   ├── cascade_spks.npy
        │   ├── calcium_event_times.npy
        │   ├── event_labels.npy
        │   └── ...
        └── kinematics/        # Behavioral tracking files
            └── *.h5           # DLC output files
```


## Installation

### Option 1: Conda Environment (Recommended)
```bash
# Create conda environment
conda env create -f environment.yml
conda activate cortex_decoder_env

# Install additional dependencies
pip install -r requirements.txt
```

### Option 2: Manual Installation
```bash
# Core dependencies
pip install numpy pandas matplotlib scikit-learn scipy
pip install blosc2 h5py mat73 msgpack munkres ndindex
```

## Data Requirements

### File Structure
Data must be organized in the following hierarchy:
```
[project_root]/
├── [mouseID]/           # e.g., "mouse25"
│   └── [YYYYMMDD]/     # e.g., "20240425"
│       ├── calcium/
│       │   ├── cascade_spks.npy          # Neural spike probabilities
│       │   ├── calcium_event_times.npy   # Event timing in calcium frames
│       │   ├── calcium_timestamps.pkl    # Calcium frame timestamps
│       │   ├── cam_event_times.pkl       # Event timing in camera frames
│       │   ├── cam_timestamps.pkl        # Camera frame timestamps
│       │   ├── event_labels.npy          # Behavioral event labels
│       │   └── red_labels.npy            # Cell type labels (inhibitory/excitatory)
│       └── kinematics/
│           └── *.h5                      # DeepLabCut pose estimation files
```

### Behavioral Categories
The pipeline recognizes these behavioral categories:
- **Learned behaviors**: reach (0), grasp (1), carry (2)
- **Natural behaviors**: non_movement (3), fidget (4), eating (5)
- **Other**: grooming (6), non_behavior_event (-1)

## Usage

### Basic Analysis

```python
from mouse import MouseDay
from decode import CortexDecoder, run_comprehensive_analysis

# Initialize decoder
decoder = CortexDecoder(random_state=42)

# Load session data
mouse_day = MouseDay('mouse25', '20240425')

# Run complete analysis pipeline
results = run_comprehensive_analysis(mouse_day, decoder, save_results=True)
```

### Individual Analysis Components

#### 1. General Population Decoding
```python
# Decode using entire neural population
scores, predictions = decoder.decode_general_population(
    mouse_day, 
    model_type="ridge",
    lag=None,  # Optional calcium lag correction
    n_trials=10,
    save_results=True
)
print(f"Average R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

#### 2. Behavior-Specific Analysis
```python
# Train separate decoders for each behavior
behavior_scores, behavior_preds = decoder.decode_by_behavior(
    mouse_day,
    save_results=True
)
```

#### 3. Cross-Behavioral Generalization
```python
# Test generalization across behaviors
cross_scores = decoder.decode_cross_behaviors(
    mouse_day,
    save_results=True
)
```

#### 4. Cell Type Comparison
```python
# Compare inhibitory vs excitatory neurons
inh_scores, exc_scores, inh_preds, exc_preds = decoder.decode_by_cell_type(
    mouse_day,
    save_results=True
)
```

#### 5. Cross-Day Analysis
```python
from decode import CrossDayDecoder

cross_decoder = CrossDayDecoder()
day1 = MouseDay('mouse25', '20240425')
day2 = MouseDay('mouse25', '20240426')

# Decode across days using registered neurons
scores, preds = cross_decoder.decode_two_days(
    train_day=day1,
    test_day=day2,
    cross_test=True,
    save_results=True
)
```

### Kinematic Data Interpolation

```python
from interpolate import multiday_interpolate_and_save

# Interpolate kinematic data to calcium timestamps
mouseID = 'mouse25'
days = ['20240420', '20240421', '20240422']
multiday_interpolate_and_save(mouseID, days)
```

### Visualization and Analysis
The pipeline includes comprehensive plotting functions for data visualization and results analysis, functions can be found in the plot.py file. 

## Key Classes

### `MouseDay`
Core data container for a single experimental session.

**Key Methods:**
- `get_neural_data(trim=True, registered_key=None)`: Returns neural spike data
- `get_locations(trimmed=True)`: Returns interpolated paw positions
- `get_behavior_labels(trim=True)`: Returns behavioral event labels

### `CortexDecoder`
Main decoder class with multiple analysis modes.

**Key Methods:**
- `decode_general_population()`: Standard population-level decoding
- `decode_by_behavior()`: Behavior-specific model training
- `decode_cross_behaviors()`: Cross-behavioral generalization testing
- `decode_by_cell_type()`: Inhibitory vs excitatory comparison

### `CrossDayDecoder`
Extension for cross-session analysis using registered neurons.

## Output Files

Results are automatically saved to:
```
decoded_data/
└── [mouseID]/
    └── [day]/
        ├── general_scores.npy       # Cross-validation scores
        ├── general_preds.npy        # Position predictions
        ├── [behavior]_scores.npy    # Behavior-specific scores
        ├── [behavior]_preds.npy     # Behavior-specific predictions
        └── [model_type]_model.pkl   # Trained model objects
```

## Configuration

### Model Parameters
- **Regularization**: Ridge/Lasso regression with cross-validated alpha selection
- **Cross-validation**: 10-fold stratified cross-validation by default
- **Test size**: 30% for train/test splits
- **Random state**: 42 for reproducibility

### Customization
```python
# Custom decoder configuration
decoder = CortexDecoder(
    alphas=[1e-3, 1e-2, 1e-1, 1.0, 10.0],  # Regularization parameters
    random_state=42,                         # Reproducibility seed
    log_level=logging.INFO                   # Logging verbosity
)
```

## Analysis Pipeline

The complete analysis pipeline includes:

1. **General Population Decoding**: Baseline performance using all neurons
2. **Behavior-Specific Models**: Individual decoders per behavior type
3. **Cross-Behavioral Testing**: Generalization across behavioral contexts
4. **Cell Type Analysis**: Inhibitory vs excitatory neuron comparison
5. **Behavioral Class Analysis**: Learned vs natural behavior comparison
6. **Temporal Lag Analysis**: Calcium indicator delay correction
7. **Cross-Day Analysis**: Testing performance accross learning sessions

## Data Quality Diagnostics

```python
from decode import diagnose_data_quality

# Check for potential numerical issues
quality_report = diagnose_data_quality(mouse_day)
```

## Troubleshooting

### Common Issues

1. **File Structure**: Ensure data follows the required directory structure
2. **Missing Files**: Check that all required .npy and .pkl files are present
3. **Dimension Mismatches**: Verify timestamp and data frame alignment
4. **Memory Issues**: Large datasets may require chunked processing

### Data Validation
```python
# Check data alignment
mouse_day.check_bin_tstamp_alignment()

# Verify dimensions
from decode import MouseDay
MouseDay.dimensions_check(mouse_day)
```