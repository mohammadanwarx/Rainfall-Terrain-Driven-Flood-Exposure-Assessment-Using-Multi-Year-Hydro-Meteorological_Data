"""
Flood Propensity Model Module

This module implements a scientifically defensible flood propensity index
that combines multi-year rainfall variability with DEM-derived hydrological
indicators.

Reference: Rainfall-Terrain Driven Flood Exposure Assessment Using Multi-Year
Hydro-Meteorological Data

Author: Geospatial Flood Analysis Pipeline
"""

import numpy as np
import xarray as xr
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: RAINFALL INDICATORS
# ============================================================================

def compute_rainfall_indicators(rainfall_cube: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract rainfall indicators from a 7-year (or multi-year) rainfall data cube.
    
    Indicators computed:
    - R1: Mean annual rainfall (baseline moisture)
    - R2: 95th percentile rainfall (extreme event propensity)
    - R3: Coefficient of variation (rainfall variability / saturation dynamics)
    
    Parameters
    ----------
    rainfall_cube : np.ndarray
        Shape (time, lat, lon) in months. Time axis must have ≥12 months
        for meaningful annual statistics.
        
    Returns
    -------
    dict
        Keys: 'R1_mean', 'R2_p95', 'R3_cv'
        Values: Arrays of shape (lat, lon)
        
    Notes
    -----
    R1 aggregates all months in cube (interpreted as total available period).
    For true annual climatology, pre-aggregate by calendar year first.
    """
    
    # Ensure rainfall is non-negative
    rainfall_cube = np.maximum(rainfall_cube, 0)
    
    # R1: Mean rainfall (baseline moisture availability)
    R1_mean = np.mean(rainfall_cube, axis=0)
    
    # R2: 95th percentile (extreme rainfall propensity)
    R2_p95 = np.percentile(rainfall_cube, 95, axis=0)
    
    # R3: Coefficient of variation (saturation dynamics)
    R3_std = np.std(rainfall_cube, axis=0)
    # Avoid division by zero
    R3_cv = np.divide(R3_std, R1_mean, 
                      where=(R1_mean > 0.1), 
                      out=np.zeros_like(R1_mean))
    
    return {
        'R1_mean': R1_mean,
        'R2_p95': R2_p95,
        'R3_cv': R3_cv
    }


# ============================================================================
# STEP 2: TERRAIN INDICATORS
# ============================================================================

def compute_terrain_indicators(dem: np.ndarray, 
                               slope: np.ndarray,
                               flow_accumulation: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute terrain-based hydrological indicators from DEM.
    
    Indicators computed:
    - T1: Flow accumulation (upslope contributing area, normalized log-scale)
    - T2: Inverted slope (low slope = high propensity)
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model, shape (lat, lon)
    slope : np.ndarray
        Slope in degrees, shape (lat, lon)
    flow_accumulation : np.ndarray, optional
        Pre-computed flow accumulation. If None, computed from DEM.
        
    Returns
    -------
    dict
        Keys: 'T1_flow_acc', 'T2_slope_inv'
        Values: Arrays of shape (lat, lon)
        
    Notes
    -----
    T1 uses log-scale normalization to handle power-law distribution of flow paths.
    T2 inverts slope because gentle slopes have higher flood propensity.
    """
    
    # T1: Flow accumulation (if not provided, compute D8)
    if flow_accumulation is None:
        # Import hydrology module
        from hydrology import calculate_flow_direction, calculate_flow_accumulation
        flow_dir = calculate_flow_direction(dem)
        flow_accumulation = calculate_flow_accumulation(flow_dir)
    
    T1_flow_acc = flow_accumulation.copy()
    
    # T2: Inverted slope (convert degrees to radians, then invert)
    slope_rad = np.radians(slope)
    slope_tan = np.tan(slope_rad)
    
    # Avoid division by zero and very small slopes
    slope_tan = np.maximum(slope_tan, 0.001)
    
    # Invert: high slope → low propensity
    T2_slope_inv = 1.0 / slope_tan
    
    # Normalize T2 to [0, 1]
    T2_min = np.percentile(T2_slope_inv[~np.isinf(T2_slope_inv)], 1)
    T2_max = np.percentile(T2_slope_inv[~np.isinf(T2_slope_inv)], 99)
    T2_slope_inv = (T2_slope_inv - T2_min) / (T2_max - T2_min + 1e-8)
    T2_slope_inv = np.clip(T2_slope_inv, 0, 1)
    
    return {
        'T1_flow_acc': T1_flow_acc,
        'T2_slope_inv': T2_slope_inv
    }


# ============================================================================
# STEP 3: NORMALIZE INDICATORS TO [0, 1]
# ============================================================================

def normalize_indicators(rainfall_indicators: Dict[str, np.ndarray],
                        terrain_indicators: Dict[str, np.ndarray],
                        method: str = 'minmax') -> Tuple[Dict, Dict]:
    """
    Normalize all indicators to [0, 1] using min-max scaling.
    
    Parameters
    ----------
    rainfall_indicators : dict
        Output from compute_rainfall_indicators()
    terrain_indicators : dict
        Output from compute_terrain_indicators()
    method : str, default 'minmax'
        'minmax': (x - min) / (max - min)
        'zscore': (x - mean) / std (then clip to [-3, 3] and rescale)
        
    Returns
    -------
    tuple of (R_norm, T_norm)
        R_norm: dict with keys 'R1_norm', 'R2_norm', 'R3_norm'
        T_norm: dict with keys 'T1_norm', 'T2_norm'
    """
    
    R_norm = {}
    T_norm = {}
    
    if method == 'minmax':
        
        # Normalize rainfall indicators
        for key in ['R1_mean', 'R2_p95', 'R3_cv']:
            data = rainfall_indicators[key]
            # Use 1st and 99th percentiles to avoid outliers
            data_min = np.percentile(data[~np.isnan(data)], 1)
            data_max = np.percentile(data[~np.isnan(data)], 99)
            data_norm = (data - data_min) / (data_max - data_min + 1e-8)
            data_norm = np.clip(data_norm, 0, 1)
            R_norm[key.replace('_mean', '_norm').replace('_p95', '_norm').replace('_cv', '_norm')] = data_norm
        
        # Normalize terrain indicators
        # T1: Log-scale normalization (handles power-law distribution)
        T1_data = terrain_indicators['T1_flow_acc']
        T1_log = np.log(T1_data + 1)
        T1_log_max = np.percentile(T1_log[~np.isnan(T1_log)], 99)
        T1_norm_data = T1_log / (T1_log_max + 1e-8)
        T1_norm_data = np.clip(T1_norm_data, 0, 1)
        T_norm['T1_norm'] = T1_norm_data
        
        # T2: Already normalized in compute_terrain_indicators
        T_norm['T2_norm'] = terrain_indicators['T2_slope_inv']
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return R_norm, T_norm


# ============================================================================
# STEP 4: COMPUTE FLOOD PROPENSITY INDEX
# ============================================================================

def compute_flood_propensity_index(R_norm: Dict[str, np.ndarray],
                                   T_norm: Dict[str, np.ndarray],
                                   weights: Optional[np.ndarray] = None,
                                   mode: str = 'linear') -> np.ndarray:
    """
    Compute Flood Propensity Index (FPI) from normalized indicators.
    
    FPI combines rainfall and terrain in a weighted linear combination:
    
    FPI = w1·R1 + w2·R2 + w3·R3 + w4·T1 + w5·T2
    
    Parameters
    ----------
    R_norm : dict
        Normalized rainfall indicators {'R1_norm', 'R2_norm', 'R3_norm'}
    T_norm : dict
        Normalized terrain indicators {'T1_norm', 'T2_norm'}
    weights : np.ndarray, optional
        Weight vector [w1, w2, w3, w4, w5]. If None, uses defaults:
        [0.20, 0.25, 0.15, 0.25, 0.15]
        Interpretation:
        - w1=0.20: Mean rainfall (baseline moisture)
        - w2=0.25: Extreme rainfall (event forcing)
        - w3=0.15: Rainfall variability (saturation dynamics)
        - w4=0.25: Flow accumulation (terrain concentration)
        - w5=0.15: Inverted slope (drainage resistance)
    mode : str, default 'linear'
        'linear': Weighted sum (transparent, explainable)
        'multiplicative': Geometric mean (emphasizes conjunction)
        
    Returns
    -------
    np.ndarray
        Flood Propensity Index, shape (lat, lon), range [0, 1]
    """
    
    if weights is None:
        weights = np.array([0.20, 0.25, 0.15, 0.25, 0.15])
    
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum()}")
    
    # Stack components in order: R1, R2, R3, T1, T2
    R1 = R_norm['R1_norm']
    R2 = R_norm['R2_norm']
    R3 = R_norm['R3_norm']
    T1 = T_norm['T1_norm']
    T2 = T_norm['T2_norm']
    
    if mode == 'linear':
        fpi = (weights[0] * R1 + 
               weights[1] * R2 + 
               weights[2] * R3 + 
               weights[3] * T1 + 
               weights[4] * T2)
        
    elif mode == 'multiplicative':
        # Geometric mean with equal weights
        fpi = np.power(R1 * R2 * R3 * T1 * T2, 1/5)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Final clip to [0, 1]
    fpi = np.clip(fpi, 0, 1)
    
    return fpi


# ============================================================================
# STEP 5: POPULATION EXPOSURE
# ============================================================================

def compute_exposure(fpi: np.ndarray,
                     population_density: np.ndarray,
                     vulnerability: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute flood exposure as FPI × population.
    
    Optionally include vulnerability weighting.
    
    Parameters
    ----------
    fpi : np.ndarray
        Flood Propensity Index, shape (lat, lon), range [0, 1]
    population_density : np.ndarray
        Population density (people/km²) or count, same shape as FPI
    vulnerability : np.ndarray, optional
        Vulnerability multiplier (e.g., poverty index, fragility).
        If None, set to 1.0 everywhere.
        
    Returns
    -------
    np.ndarray
        Exposure map (expected people at risk), shape (lat, lon)
        
    Formula
    -------
    Exposure = FPI × Population × Vulnerability
    """
    
    if population_density.shape != fpi.shape:
        raise ValueError("population_density must have same shape as fpi")
    
    if vulnerability is None:
        vulnerability = np.ones_like(fpi)
    
    exposure = fpi * population_density * vulnerability
    
    return exposure


# ============================================================================
# UTILITIES
# ============================================================================

def generate_flood_propensity_report(fpi: np.ndarray,
                                     exposure: Optional[np.ndarray] = None) -> Dict:
    """
    Generate summary statistics for FPI and exposure.
    
    Parameters
    ----------
    fpi : np.ndarray
        Flood Propensity Index
    exposure : np.ndarray, optional
        Exposure map
        
    Returns
    -------
    dict
        Summary statistics
    """
    
    report = {
        'FPI_statistics': {
            'mean': float(np.nanmean(fpi)),
            'std': float(np.nanstd(fpi)),
            'min': float(np.nanmin(fpi)),
            'max': float(np.nanmax(fpi)),
            'median': float(np.nanmedian(fpi)),
            'p25': float(np.nanpercentile(fpi, 25)),
            'p75': float(np.nanpercentile(fpi, 75)),
        }
    }
    
    if exposure is not None:
        report['Exposure_statistics'] = {
            'total_exposed': float(np.nansum(exposure)),
            'mean_exposure': float(np.nanmean(exposure)),
            'std_exposure': float(np.nanstd(exposure)),
            'high_risk_pixels': int(np.sum(fpi >= 0.66)),
            'medium_risk_pixels': int(np.sum((fpi >= 0.33) & (fpi < 0.66))),
            'low_risk_pixels': int(np.sum(fpi < 0.33)),
        }
    
    return report


if __name__ == '__main__':
    print("Flood Propensity Module loaded successfully.")
    print("Use compute_rainfall_indicators() and compute_terrain_indicators() to get started.")
