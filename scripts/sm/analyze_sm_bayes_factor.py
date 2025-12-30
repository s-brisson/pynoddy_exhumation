"""
Proper Bayes Factor Analysis for Subalpine Molasse Models A and B

This script computes proper Bayes factors using:
1. BIC approximation (recommended)
2. Harmonic mean estimator (for verification)

Requires:
- Maximum log-likelihoods for each model
- Parameter counts for each model
- Number of data points
- (Optional) All likelihoods from Monte Carlo samples for harmonic mean
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


def compute_bic(log_likelihood_max, n_parameters, n_data_points):
    """
    Compute Bayesian Information Criterion (BIC).
    
    Lower BIC indicates better model (penalizes complexity).
    
    Parameters
    ----------
    log_likelihood_max : float
        Maximum log-likelihood achieved
    n_parameters : int
        Number of free parameters in the model
    n_data_points : int
        Number of data points (samples)
        
    Returns
    -------
    bic : float
        BIC value
    """
    return -2 * log_likelihood_max + n_parameters * np.log(n_data_points)


def compute_bic_bayes_factor(bic_A, bic_B):
    """
    Approximate Bayes factor from BIC values.
    
    BF ≈ exp(-0.5 × ΔBIC)
    
    This approximation is valid for large sample sizes and when
    priors are not too informative.
    
    Parameters
    ----------
    bic_A : float
        BIC for Model A
    bic_B : float
        BIC for Model B
        
    Returns
    -------
    bayes_factor : float
        Approximate Bayes factor P(D|M_A) / P(D|M_B)
        BF > 1 favors Model A
        BF < 1 favors Model B
    """
    return np.exp(-0.5 * (bic_A - bic_B))


def interpret_bayes_factor(bf):
    """
    Interpret Bayes factor using Wasserman (2000) scale.
    
    Parameters
    ----------
    bf : float
        Bayes factor
        
    Returns
    -------
    interpretation : str
        Textual interpretation
    """
    if bf < 1/150:
        return "Very strong evidence for Model B"
    elif bf < 1/20:
        return "Strong evidence for Model B"
    elif bf < 1/3:
        return "Positive evidence for Model B"
    elif bf < 1:
        return "Weak evidence for Model B"
    elif bf < 3:
        return "Weak evidence for Model A"
    elif bf < 20:
        return "Positive evidence for Model A"
    elif bf < 150:
        return "Strong evidence for Model A"
    else:
        return "Very strong evidence for Model A"


def estimate_marginal_likelihood_log(log_likelihoods):
    """
    Estimate marginal likelihood from log-likelihoods using log-sum-exp trick.
    
    More numerically stable than harmonic mean in raw space.
    
    Parameters
    ----------
    log_likelihoods : array-like
        Log-likelihood values from prior samples
        
    Returns
    -------
    log_marginal_likelihood : float
        Log of marginal likelihood
    """
    log_likelihoods = np.asarray(log_likelihoods)
    
    # Filter out -inf
    finite = log_likelihoods[np.isfinite(log_likelihoods)]
    
    if len(finite) == 0:
        return -np.inf
    
    # Log-sum-exp trick for numerical stability
    max_ll = np.max(finite)
    log_marginal = max_ll + np.log(np.mean(np.exp(finite - max_ll)))
    
    return log_marginal


def estimate_marginal_likelihood_harmonic(likelihoods):
    """
    Estimate marginal likelihood using harmonic mean estimator.
    
    WARNING: This estimator can be unstable and biased. It tends to
    overestimate the marginal likelihood. Use with caution.
    
    Parameters
    ----------
    likelihoods : array-like
        Likelihood values from prior samples
        
    Returns
    -------
    marginal_likelihood : float
        Marginal likelihood estimate
    """
    likelihoods = np.asarray(likelihoods)
    
    # Filter out zeros to avoid division errors
    nonzero = likelihoods[likelihoods > 0]
    
    if len(nonzero) == 0:
        return 0.0
    
    # Harmonic mean estimator
    marginal_likelihood = len(nonzero) / np.sum(1.0 / nonzero)
    
    return marginal_likelihood


def compute_bayes_factor_harmonic(likelihoods_A, likelihoods_B, use_log=True):
    """
    Compute Bayes factor using harmonic mean estimator.
    
    Parameters
    ----------
    likelihoods_A : array-like
        Likelihood values from Model A prior samples
    likelihoods_B : array-like
        Likelihood values from Model B prior samples
    use_log : bool
        If True, work in log space (more stable)
        
    Returns
    -------
    bayes_factor : float
        Ratio of marginal likelihoods P(D|M_A) / P(D|M_B)
    """
    if use_log:
        # Convert to log if needed
        if np.all(likelihoods_A > 0):
            log_likelihoods_A = np.log(likelihoods_A)
        else:
            log_likelihoods_A = likelihoods_A
            
        if np.all(likelihoods_B > 0):
            log_likelihoods_B = np.log(likelihoods_B)
        else:
            log_likelihoods_B = likelihoods_B
        
        log_ml_A = estimate_marginal_likelihood_log(log_likelihoods_A)
        log_ml_B = estimate_marginal_likelihood_log(log_likelihoods_B)
        
        return np.exp(log_ml_A - log_ml_B)
    else:
        ml_A = estimate_marginal_likelihood_harmonic(likelihoods_A)
        ml_B = estimate_marginal_likelihood_harmonic(likelihoods_B)
        
        if ml_B == 0:
            return np.inf
        
        return ml_A / ml_B


def count_parameters(params_dir):
    """
    Count number of free parameters from parameter CSV files.
    
    Parameters
    ----------
    params_dir : str or Path
        Directory containing parameter CSV files
        
    Returns
    -------
    n_params : int
        Number of free parameters
    param_names : list
        List of parameter column names
    """
    params_dir = Path(params_dir)
    csv_files = list(params_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {params_dir}")
    
    # Read first file to get column names
    first_file = csv_files[0]
    df = pd.read_csv(first_file)
    
    # Exclude non-parameter columns
    exclude_cols = ['Event', 'iteration', 'n_draw', 'event_name']
    param_cols = [col for col in df.columns if col not in exclude_cols]
    
    return len(param_cols), param_cols


def analyze_sm_models_bayes_factor(
    log_mle_A=-40,
    log_mle_B=-38,
    n_data_points=4,
    n_params_A=None,
    n_params_B=None,
    params_dir_A=None,
    params_dir_B=None,
    likelihoods_A=None,
    likelihoods_B=None
):
    """
    Analyze SM Models A and B using proper Bayes factor.
    
    Parameters
    ----------
    log_mle_A : float
        Maximum log-likelihood for Model A
    log_mle_B : float
        Maximum log-likelihood for Model B
    n_data_points : int
        Number of data points (samples)
    n_params_A : int, optional
        Number of parameters in Model A (will count from CSV if not provided)
    n_params_B : int, optional
        Number of parameters in Model B (will count from CSV if not provided)
    params_dir_A : str, optional
        Directory with Model A parameter files (for counting parameters)
    params_dir_B : str, optional
        Directory with Model B parameter files (for counting parameters)
    likelihoods_A : array-like, optional
        All likelihoods from Model A (for harmonic mean)
    likelihoods_B : array-like, optional
        All likelihoods from Model B (for harmonic mean)
        
    Returns
    -------
    results : dict
        Dictionary with all computed statistics
    """
    results = {}
    
    # Count parameters if not provided
    if n_params_A is None:
        if params_dir_A is None:
            raise ValueError("Must provide either n_params_A or params_dir_A")
        n_params_A, param_names_A = count_parameters(params_dir_A)
        print(f"Model A has {n_params_A} parameters: {param_names_A}")
    else:
        param_names_A = None
    
    if n_params_B is None:
        if params_dir_B is None:
            raise ValueError("Must provide either n_params_B or params_dir_B")
        n_params_B, param_names_B = count_parameters(params_dir_B)
        print(f"Model B has {n_params_B} parameters: {param_names_B}")
    else:
        param_names_B = None
    
    results['n_params_A'] = n_params_A
    results['n_params_B'] = n_params_B
    results['n_data_points'] = n_data_points
    
    # Compute BIC
    bic_A = compute_bic(log_mle_A, n_params_A, n_data_points)
    bic_B = compute_bic(log_mle_B, n_params_B, n_data_points)
    
    results['bic_A'] = bic_A
    results['bic_B'] = bic_B
    results['bic_difference'] = bic_B - bic_A  # Positive means B is worse
    
    # Compute Bayes factor from BIC
    bayes_factor_bic = compute_bic_bayes_factor(bic_A, bic_B)
    results['bayes_factor_bic'] = bayes_factor_bic
    results['interpretation_bic'] = interpret_bayes_factor(bayes_factor_bic)
    
    # Compute likelihood ratio (for comparison)
    max_likelihood_A = np.exp(log_mle_A)
    max_likelihood_B = np.exp(log_mle_B)
    likelihood_ratio = max_likelihood_B / max_likelihood_A
    results['likelihood_ratio'] = likelihood_ratio
    
    # Compute harmonic mean Bayes factor if likelihoods provided
    if likelihoods_A is not None and likelihoods_B is not None:
        bayes_factor_harmonic = compute_bayes_factor_harmonic(
            likelihoods_A, likelihoods_B, use_log=True
        )
        results['bayes_factor_harmonic'] = bayes_factor_harmonic
        results['interpretation_harmonic'] = interpret_bayes_factor(bayes_factor_harmonic)
        
        # Compute marginal likelihoods
        if np.all(np.asarray(likelihoods_A) > 0):
            log_likelihoods_A = np.log(likelihoods_A)
        else:
            log_likelihoods_A = likelihoods_A
            
        if np.all(np.asarray(likelihoods_B) > 0):
            log_likelihoods_B = np.log(likelihoods_B)
        else:
            log_likelihoods_B = likelihoods_B
        
        log_ml_A = estimate_marginal_likelihood_log(log_likelihoods_A)
        log_ml_B = estimate_marginal_likelihood_log(log_likelihoods_B)
        
        results['log_marginal_likelihood_A'] = log_ml_A
        results['log_marginal_likelihood_B'] = log_ml_B
    
    return results


def print_results(results):
    """Print analysis results in a formatted way."""
    print("=" * 70)
    print("BAYES FACTOR ANALYSIS: Subalpine Molasse Models A and B")
    print("=" * 70)
    
    print(f"\nModel Complexity:")
    print(f"  Model A: {results['n_params_A']} parameters")
    print(f"  Model B: {results['n_params_B']} parameters")
    print(f"  Data points: {results['n_data_points']}")
    
    print(f"\nMaximum Log-Likelihoods:")
    print(f"  Model A: {results.get('log_mle_A', 'N/A')}")
    print(f"  Model B: {results.get('log_mle_B', 'N/A')}")
    
    print(f"\nBayesian Information Criterion (BIC):")
    print(f"  Model A: {results['bic_A']:.2f}")
    print(f"  Model B: {results['bic_B']:.2f}")
    print(f"  Difference (B - A): {results['bic_difference']:.2f}")
    if results['bic_difference'] > 0:
        print(f"    → Model B is penalized more (has more parameters)")
    elif results['bic_difference'] < 0:
        print(f"    → Model A is penalized more (has more parameters)")
    else:
        print(f"    → Models have same complexity")
    
    print(f"\nLikelihood Ratio (for comparison):")
    print(f"  Ratio (B/A): {results['likelihood_ratio']:.2f}")
    print(f"  ⚠️  This does NOT penalize complexity")
    
    print(f"\nBayes Factor (BIC approximation):")
    print(f"  BF (A/B): {results['bayes_factor_bic']:.2f}")
    print(f"  Interpretation: {results['interpretation_bic']}")
    
    if 'bayes_factor_harmonic' in results:
        print(f"\nBayes Factor (Harmonic Mean):")
        print(f"  BF (A/B): {results['bayes_factor_harmonic']:.2f}")
        print(f"  Interpretation: {results['interpretation_harmonic']}")
        print(f"\nMarginal Likelihoods (log):")
        print(f"  Model A: {results['log_marginal_likelihood_A']:.2f}")
        print(f"  Model B: {results['log_marginal_likelihood_B']:.2f}")
    
    print("\n" + "=" * 70)
    print("Wasserman (2000) Interpretation Scale:")
    print("  BF < 1/150: Very strong evidence for Model B")
    print("  1/150 < BF < 1/20: Strong evidence for Model B")
    print("  1/20 < BF < 1/3: Positive evidence for Model B")
    print("  1/3 < BF < 1: Weak evidence for Model B")
    print("  1 < BF < 3: Weak evidence for Model A")
    print("  3 < BF < 20: Positive evidence for Model A")
    print("  20 < BF < 150: Strong evidence for Model A")
    print("  BF > 150: Very strong evidence for Model A")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Option 1: Using BIC only (requires parameter counts)
    print("=" * 70)
    print("EXAMPLE 1: BIC-Based Bayes Factor")
    print("=" * 70)
    
    # Parameter counts confirmed by colleague:
    # SM Model A: 5 faults × 4 parameters (X, Z, Amplitude, Slip) = 20 parameters
    # SM Model B: 4 faults × 4 parameters (X, Z, Amplitude, Slip) = 16 parameters
    
    results = analyze_sm_models_bayes_factor(
        log_mle_A=-40,
        log_mle_B=-38,
        n_data_points=4,
        n_params_A=20,  # Confirmed: 5 faults × 4 parameters (X, Z, Amplitude, Slip)
        n_params_B=16,  # Confirmed: 4 faults × 4 parameters (X, Z, Amplitude, Slip)
    )
    
    print_results(results)
    
    # Note: Model B actually has FEWER parameters (16) than Model A (20)
    # This means Model B is simpler AND fits better - strong evidence!
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("Model B has FEWER parameters (16) than Model A (20), yet fits better.")
    print("This provides strong evidence for Model B because:")
    print("  1. Model B fits the data better (higher likelihood)")
    print("  2. Model B is simpler (fewer parameters)")
    print("  3. Both factors favor Model B in the Bayes factor")
    print("=" * 70)
    
    # Option 2: Using parameter directories (auto-counts parameters)
    # Uncomment and adjust paths:
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Auto-counting parameters from CSV files")
    print("=" * 70)
    
    results = analyze_sm_models_bayes_factor(
        log_mle_A=-40,
        log_mle_B=-38,
        n_data_points=4,
        params_dir_A="cluster_outputs/SM_20240406/modelA_params/",
        params_dir_B="cluster_outputs/SM_20240406/modelB_params/",
    )
    
    print_results(results)
    """
    
    # Option 3: Using harmonic mean (requires all likelihoods)
    # Uncomment and provide likelihood arrays:
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Harmonic Mean Bayes Factor")
    print("=" * 70)
    
    # Load likelihoods from mle() function
    # likelihoods_A = ...  # Array of likelihoods from Model A
    # likelihoods_B = ...  # Array of likelihoods from Model B
    
    results = analyze_sm_models_bayes_factor(
        log_mle_A=-40,
        log_mle_B=-38,
        n_data_points=4,
        n_params_A=20,
        n_params_B=16,
        likelihoods_A=likelihoods_A,
        likelihoods_B=likelihoods_B,
    )
    
    print_results(results)
    """

