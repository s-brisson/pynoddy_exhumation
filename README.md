# pynoddy_exhumation

Probabilistic kinematic modeling with thermal resetting using Noddy and thermochronological data.

## Repository Structure

```
pynoddy_exhumation/
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── core/                         # Core functions (used by all analyses)
│   ├── __init__.py
│   ├── exh_functions.py          # Core geological modeling functions
│   ├── exh_scoring.py            # Scoring and likelihood computation
│   ├── exh_simulation.py         # Monte Carlo simulation runner
│   ├── exh_processing.py         # Data processing utilities
│   └── exh_plotting.py           # Visualization functions
│
├── scripts/                      # Analysis and simulation scripts
│   ├── sm/                       # Subalpine Molasse (SM) scripts
│   │   ├── __init__.py
│   │   ├── analyze_sm_bayes_factor.py    # Bayes factor analysis for SM models
│   │   └── count_sm_parameters.py         # Parameter counting utility
│   │
│   ├── transalp/                 # TRANSALP section scripts
│   │   ├── __init__.py
│   │   └── transalp_importance_weighting.py  # Importance weighting implementation
│   │
│   └── shared/                   # Shared utilities (to be added)
│       └── __init__.py
│
├── examples/                     # Example scripts and test files
│   ├── test_Cioff.py             # Example pynoddy usage
│   └── twofaults_translation.his  # Example Noddy history file
│
└── docs/                         # Documentation
    └── analysis/                 # Analysis documentation (local only, not versioned)
        ├── Code_review_guide.md
        ├── TRANSALP_*.md
        ├── SM_*.md
        └── ... (other analysis docs)
```

---

## Overview

This repository contains code for probabilistic kinematic modeling using Noddy geological models combined with thermochronological data. The code supports two main analysis scenarios:

1. **Subalpine Molasse (SM)** - Single model analysis
2. **TRANSALP Section** - Two-model comparison (subduction polarity reversal vs. no reversal)

---

## Key Features

### Statistical Analysis

- **Likelihood Calculation**: Gaussian likelihoods for thermochronological data
- **Model Comparison**: BIC-based Bayes factors and likelihood ratios
- **Posterior Distributions**: Importance weighting for proper posterior computation
- **Convergence Analysis**: MLE evolution tracking

### Simulation

- **Monte Carlo Sampling**: Parameter space exploration
- **Exhumation Tracking**: Computes exhumation paths for indicator layers
- **Scoring**: Binary and probabilistic scoring systems

---

## Main Scripts

### Core Functions (`core/exh_*.py`)

- **`core/exh_functions.py`**: Core geological modeling functions
  - `disturb()`: Parameter perturbation for Monte Carlo
  - `exhumationComplex()`: Exhumation computation
  - `ExtractCoords()`: Coordinate extraction

- **`core/exh_scoring.py`**: Scoring and likelihood computation
  - `interp_and_score()`: Interpolation and scoring
  - Likelihood calculation functions

- **`core/exh_simulation.py`**: Monte Carlo simulation runner
  - Main simulation loop
  - Parameter variation
  - Output saving

- **`core/exh_processing.py`**: Data processing utilities
  - Data loading and manipulation
  - Format conversion

- **`core/exh_plotting.py`**: Visualization functions
  - Plotting utilities
  - Figure generation

### SM Analysis Scripts (`scripts/sm/`)

- **`analyze_sm_bayes_factor.py`**: 
  - Computes proper Bayes factors using BIC approximation
  - Compares SM Model A vs. Model B
  - Includes harmonic mean estimator option
  - Usage: `python scripts/sm/analyze_sm_bayes_factor.py`

- **`count_sm_parameters.py`**: 
  - Utility to count free parameters from CSV files
  - Helps verify parameter counts for BIC calculation
  - Usage: `python scripts/sm/count_sm_parameters.py`

### TRANSALP Analysis Scripts (`scripts/transalp/`)

- **`transalp_importance_weighting.py`**: 
  - Implements importance weighting for posterior distributions
  - Works with existing simulation results (no re-run needed)
  - Computes weighted statistics for all parameters
  - Usage: `python scripts/transalp/transalp_importance_weighting.py`

### External Analysis Scripts

**Note**: The main TRANSALP analysis notebooks are located externally in:
```
/Users/flow/Documents/01_work/01_own_docs/02_paper_drafts/2025/Sofia_Paper_3/Thermokinematic/
```

These include:
- `transalp_clean.ipynb` - Main TRANSALP analysis
- `transalp_calibration.ipynb` - Calibration analysis
- `transalp.ipynb` - Detailed analysis with posterior distributions
- `transalp_funcs.py` - TRANSALP-specific functions

---

## Statistical Methods

### Model Comparison

**Likelihood Ratio:**
- Ratio of maximum likelihoods
- No complexity penalty
- Use for initial comparison

**Bayes Factor (BIC Approximation):**
- Includes complexity penalty
- Proper Bayesian model comparison
- Use for final model selection
- Formula: `BF ≈ exp(-0.5 × ΔBIC)`

### Posterior Distributions

**Importance Weighting:**
- Converts prior samples to posterior samples
- Uses likelihoods as importance weights
- Proper posterior distributions
- No information loss (unlike filtering)

**Effective Sample Size (ESS):**
- Diagnostic for importance sampling quality
- Low ESS (< 10%) indicates poor sampling
- High ESS (> 30%) indicates good sampling

---

## Usage Examples

### SM Model Comparison

```python
from scripts.sm.analyze_sm_bayes_factor import analyze_sm_models_bayes_factor

results = analyze_sm_models_bayes_factor(
    log_mle_A=-40,
    log_mle_B=-38,
    n_data_points=4,
    n_params_A=10,
    n_params_B=12
)
```

### TRANSALP Importance Weighting

```python
from scripts.transalp.transalp_importance_weighting import compute_posterior_statistics_importance_weighting

results, weights, likelihoods = compute_posterior_statistics_importance_weighting(
    params_dir='path/to/params/',
    exh_dir='path/to/exhumation/',
    geo_gradient=25
)
```

---

## Data Structure

### Simulation Outputs (Not in Repository - Too Large)

Simulation results are stored externally in:
```
cluster_outputs/
├── SM_20240406/              # SM simulation results
│   ├── model_params/         # Parameter CSV files
│   ├── model_scores/         # Score CSV files
│   ├── model_exhumation/     # Exhumation .npy files
│   └── model_samples/        # Sample CSV files
│
└── 03_20240312/              # TRANSALP simulation results
    ├── model1/
    │   ├── model1_params/
    │   ├── model1_scores/
    │   ├── model1_exhumation/
    │   └── model1_samples/
    └── model2/
        └── ... (same structure)
```

### Parameter Files

**Format**: CSV files with columns:
- `Event`: Event number
- `X`, `Z`: Spatial coordinates
- `Amplitude`, `Slip`: Varied parameters
- `n_draw`: Simulation iteration number
- `event_name`: Event identifier

### Exhumation Files

**Format**: `.npy` numpy arrays
- Shape: `(n_simulations, n_samples)`
- Contains predicted exhumation for each sample in each simulation

---

## Dependencies

### Required

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `pynoddy` - Noddy geological modeling (only for running NEW simulations)

### Optional

- `scipy` - Additional statistical functions
- `jupyter` - For running notebooks

**Note**: For analysis of existing results, `pynoddy` is NOT required. It's only needed when running new simulations.

---

## Recent Changes (Branch: `statistical-fixes`)

### Statistical Fixes

1. **Terminology Corrections**:
   - Fixed "Bayes factor" misnomer → Proper likelihood ratio + BIC-based Bayes factor
   - Updated TRANSALP notebooks with correct terminology

2. **Proper Bayes Factors**:
   - Implemented BIC-based Bayes factor calculation
   - Added to TRANSALP analysis notebooks
   - Available for SM analysis

3. **Importance Weighting**:
   - Implemented proper posterior computation
   - Replaces filtering approach
   - Works with existing simulation results

4. **Code Organization**:
   - Separated SM and TRANSALP scripts
   - Created analysis documentation folder
   - Updated repository structure

---

## Model Parameters

### SM Models

- **Model A**: 20 parameters (5 faults × 4 parameters: X, Z, Amplitude, Slip)
- **Model B**: 16 parameters (4 faults × 4 parameters: X, Z, Amplitude, Slip)
- **Data points**: 4 samples (B40, B50, B45, B55)

### TRANSALP Models

- **Model 1** (Subduction polarity reversal): 16 parameters (8 events × 2)
- **Model 2** (No subduction polarity reversal): 18 parameters (9 events × 2)
- **Data points**: 12 TRANSALP samples

---

## Documentation

### Analysis Documentation

Analysis documentation files are stored in `docs/analysis/` (local only, not versioned):
- `Code_review_guide.md` - Original review guide
- `TRANSALP_*.md` - TRANSALP analysis documentation
- `SM_*.md` - SM analysis documentation
- `IMPORTANCE_WEIGHTING_*.md` - Importance weighting guides
- `BAYES_FACTOR_GUIDE.md` - Bayes factor calculation guide

### Key Documents

- **`Code_review_guide.md`**: Original issues identified and fix options
- **`TRANSALP_IMPLEMENTATION_SUMMARY.md`**: Complete TRANSALP fixes summary
- **`IMPORTANCE_WEIGHTING_IMPLEMENTATION.md`**: Importance weighting guide
- **`SCENARIO_MAPPING.md`**: Maps scripts to SM vs. TRANSALP scenarios

---

## Contributing

When adding new code:

1. **SM scripts** → `scripts/sm/`
2. **TRANSALP scripts** → `scripts/transalp/`
3. **Shared utilities** → `scripts/shared/`
4. **Analysis docs** → `docs/analysis/` (local only, not versioned)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

[Add contact information here]
