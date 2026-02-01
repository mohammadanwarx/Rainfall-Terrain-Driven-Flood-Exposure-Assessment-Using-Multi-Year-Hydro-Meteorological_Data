"""
Flood Exposure Analysis Module

This module handles flood exposure calculations, vulnerability assessment,
and risk mapping for various features of interest, including buildings.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from shapely.geometry import Point, Polygon, box
from rasterio.features import rasterize
from rasterio.transform import Affine
import logging

logger = logging.getLogger(__name__)


def calculate_flood_depth(dem: np.ndarray, water_level: float) -> np.ndarray:
    """
    Calculate flood depth based on DEM and water level.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model
    water_level : float
        Water surface elevation
        
    Returns
    -------
    np.ndarray
        Flood depth array (positive values indicate flooding)
    """
    flood_depth = water_level - dem
    flood_depth = np.where(flood_depth > 0, flood_depth, 0)
    return flood_depth


def identify_flooded_areas(flood_depth: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Identify flooded areas based on depth threshold.
    
    Parameters
    ----------
    flood_depth : np.ndarray
        Flood depth array
    threshold : float, optional
        Minimum depth to consider as flooded (default: 0.0)
        
    Returns
    -------
    np.ndarray
        Binary flood mask (1 = flooded, 0 = not flooded)
    """
    flood_mask = (flood_depth > threshold).astype(np.int32)
    return flood_mask


def _get_raster_bounds_geometry(flood_mask: np.ndarray, transform: Affine) -> Polygon:
    """
    Get the bounding geometry of a raster.
    
    Parameters
    ----------
    flood_mask : np.ndarray
        Raster array
    transform : Affine
        Raster affine transform
        
    Returns
    -------
    Polygon
        Bounding polygon of the raster
    """
    rows, cols = flood_mask.shape
    
    # Get corner coordinates
    ul_x = transform.c
    ul_y = transform.f
    lr_x = transform.c + (cols * transform.a)
    lr_y = transform.f + (rows * transform.e)
    
    # Create bounding box polygon
    bounds_poly = box(
        min(ul_x, lr_x),
        min(ul_y, lr_y),
        max(ul_x, lr_x),
        max(ul_y, lr_y)
    )
    
    return bounds_poly


def calculate_exposure_statistics(buildings_gdf: gpd.GeoDataFrame, 
                                  flood_mask: np.ndarray,
                                  transform: Affine,
                                  flood_depth: Optional[np.ndarray] = None,
                                  auto_clip: bool = True) -> gpd.GeoDataFrame:
    """
    Calculate exposure statistics for buildings.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings vector data with geometries
    flood_mask : np.ndarray
        Binary flood mask (1 = flooded, 0 = not flooded)
    transform : Affine
        Raster transform (geospatial affine transformation)
    flood_depth : np.ndarray, optional
        Flood depth raster for more detailed analysis
    auto_clip : bool, optional
        If True, automatically clip buildings to study area bounds (default: True)
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings with exposure flags and statistics (clipped to study area)
    """
    buildings = buildings_gdf.copy()
    
    # Ensure consistent CRS
    if buildings.crs is None:
        logger.warning("Buildings GeoDataFrame has no CRS. Assuming EPSG:4326")
        buildings.set_crs('EPSG:4326', inplace=True)
    
    # Automatically clip buildings to study area bounds
    if auto_clip:
        study_area = _get_raster_bounds_geometry(flood_mask, transform)
        buildings = buildings.clip(study_area)
        
        if len(buildings) == 0:
            logger.warning("No buildings found within study area after clipping")
            # Return empty GeoDataFrame with required columns
            buildings = buildings_gdf.copy()
            buildings = buildings.iloc[0:0]  # Empty with same structure
        
        logger.info(f"Clipped to {len(buildings)} buildings within study area")
    
    # Check exposure for each building
    buildings['exposed'] = False
    buildings['flood_depth'] = 0.0
    buildings['exposure_type'] = 'none'
    
    for idx, building in buildings.iterrows():
        try:
            # Get building geometry bounds
            bounds = building.geometry.bounds
            
            # Convert bounds to raster indices
            row_min, col_min = _world_to_pixel(bounds[1], bounds[0], transform)
            row_max, col_max = _world_to_pixel(bounds[3], bounds[2], transform)
            
            # Ensure indices are within bounds
            row_min = max(0, int(row_min))
            row_max = min(flood_mask.shape[0], int(row_max) + 1)
            col_min = max(0, int(col_min))
            col_max = min(flood_mask.shape[1], int(col_max) + 1)
            
            if row_min < row_max and col_min < col_max:
                # Check if building overlaps with flooded areas
                building_flood_mask = flood_mask[row_min:row_max, col_min:col_max]
                
                if np.any(building_flood_mask):
                    buildings.at[idx, 'exposed'] = True
                    buildings.at[idx, 'exposure_type'] = 'partial' if np.any(~building_flood_mask) else 'full'
                    
                    # Calculate flood depth if available
                    if flood_depth is not None:
                        building_flood_depth = flood_depth[row_min:row_max, col_min:col_max]
                        buildings.at[idx, 'flood_depth'] = float(np.max(building_flood_depth))
        
        except Exception as e:
            logger.debug(f"Error processing building {idx}: {str(e)}")
            continue
    
    return buildings


def _world_to_pixel(x: float, y: float, transform: Affine) -> Tuple[float, float]:
    """
    Convert world coordinates to pixel indices using raster transform.
    
    Parameters
    ----------
    x : float
        X coordinate
    y : float
        Y coordinate
    transform : Affine
        Raster affine transform
        
    Returns
    -------
    Tuple[float, float]
        (row, col) in pixel coordinates
    """
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    return row, col


def assess_population_exposure(population_raster: np.ndarray, 
                               flood_mask: np.ndarray) -> Dict[str, float]:
    """
    Assess population exposure to flooding.
    
    Parameters
    ----------
    population_raster : np.ndarray
        Population density raster
    flood_mask : np.ndarray
        Binary flood mask
        
    Returns
    -------
    Dict[str, float]
        Dictionary with exposure statistics
    """
    exposed_population = np.sum(population_raster * flood_mask)
    total_population = np.sum(population_raster)
    exposure_rate = (exposed_population / total_population * 100) if total_population > 0 else 0
    
    return {
        'exposed_population': float(exposed_population),
        'total_population': float(total_population),
        'exposure_rate_percent': float(exposure_rate)
    }


def calculate_vulnerability_index(features_gdf: gpd.GeoDataFrame, 
                                  weights: Dict[str, float]) -> gpd.GeoDataFrame:
    """
    Calculate vulnerability index for features.
    
    Parameters
    ----------
    features_gdf : gpd.GeoDataFrame
        Features with vulnerability attributes
    weights : Dict[str, float]
        Weights for different vulnerability factors
        
    Returns
    -------
    gpd.GeoDataFrame
        Features with vulnerability index
    """
    # TODO: Implement vulnerability index calculation
    features_gdf['vulnerability_index'] = 0.0
    return features_gdf


def generate_risk_map(flood_depth: np.ndarray, 
                     vulnerability: np.ndarray,
                     depth_weights: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Generate risk map combining flood depth and vulnerability.
    
    Risk = Hazard × Vulnerability
    
    Parameters
    ----------
    flood_depth : np.ndarray
        Flood depth array
    vulnerability : np.ndarray
        Vulnerability array
    depth_weights : List[Tuple[float, float]], optional
        Depth thresholds and corresponding weights
        
    Returns
    -------
    np.ndarray
        Risk map
    """
    if depth_weights is None:
        # Default depth-damage relationship
        depth_weights = [(0.0, 0.0), (0.5, 0.2), (1.0, 0.4), (2.0, 0.7), (3.0, 1.0)]
    
    # Apply depth-damage curve
    hazard = np.zeros_like(flood_depth)
    for i, (depth_thresh, weight) in enumerate(depth_weights[:-1]):
        next_depth, next_weight = depth_weights[i + 1]
        mask = (flood_depth >= depth_thresh) & (flood_depth < next_depth)
        interpolated_weight = weight + (next_weight - weight) * \
                            (flood_depth - depth_thresh) / (next_depth - depth_thresh)
        hazard = np.where(mask, interpolated_weight, hazard)
    
    # For depths above maximum threshold
    hazard = np.where(flood_depth >= depth_weights[-1][0], depth_weights[-1][1], hazard)
    
    # Calculate risk
    risk = hazard * vulnerability
    
    return risk


def assess_building_vulnerability(buildings_gdf: gpd.GeoDataFrame,
                                  use_physical_attributes: bool = True) -> gpd.GeoDataFrame:
    """
    Assess vulnerability of buildings to flooding.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings GeoDataFrame with attributes
    use_physical_attributes : bool, optional
        If True, use building height/levels to estimate vulnerability (default: True)
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings with vulnerability index (0-1)
    """
    buildings = buildings_gdf.copy()
    
    # Initialize vulnerability score
    if 'vulnerability_index' not in buildings.columns:
        buildings['vulnerability_index'] = 0.5  # Default mid-range value
    
    # Base vulnerability by building class
    class_vulnerability = {
        'residential': 0.7,
        'commercial': 0.5,
        'industrial': 0.4,
        'public': 0.8,  # Higher vulnerability for public buildings
        'other': 0.5
    }
    
    if 'building_class' in buildings.columns:
        buildings['vulnerability_index'] = buildings['building_class'].map(
            lambda x: class_vulnerability.get(x, 0.5)
        )
    
    # Adjust based on physical attributes
    if use_physical_attributes:
        # Ground floor height reduces vulnerability
        if 'height' in buildings.columns:
            height_factor = 1.0 - (buildings['height'] / 20.0).clip(0, 0.5)
            buildings['vulnerability_index'] *= height_factor
        
        # Building age (if available) - older buildings more vulnerable
        if 'construction_year' in buildings.columns:
            age_factor = (pd.Timestamp.now().year - buildings['construction_year']) / 100
            age_factor = age_factor.clip(0.5, 1.5)  # Bound between 0.5 and 1.5
            buildings['vulnerability_index'] *= age_factor
    
    # Ensure values are between 0 and 1
    buildings['vulnerability_index'] = buildings['vulnerability_index'].clip(0, 1)
    
    return buildings


def calculate_building_flood_risk(buildings_gdf: gpd.GeoDataFrame,
                                 depth_damage_curve: Optional[Dict[str, float]] = None) -> gpd.GeoDataFrame:
    """
    Calculate flood risk for buildings combining exposure and vulnerability.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings with 'exposed', 'flood_depth', and 'vulnerability_index' columns
    depth_damage_curve : Dict[str, float], optional
        Depth-damage relationship. Keys are depth thresholds, values are damage fractions
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings with risk metrics
    """
    buildings = buildings_gdf.copy()
    
    # Default depth-damage curve for buildings (relative damage)
    if depth_damage_curve is None:
        depth_damage_curve = {
            0.0: 0.0,
            0.3: 0.1,
            0.6: 0.25,
            1.0: 0.4,
            1.5: 0.6,
            2.0: 0.8,
            3.0: 1.0
        }
    
    # Calculate hazard from flood depth
    buildings['hazard'] = 0.0
    
    if 'flood_depth' in buildings.columns:
        depths = sorted(depth_damage_curve.keys())
        
        for i in range(len(depths) - 1):
            d1, d2 = depths[i], depths[i + 1]
            h1, h2 = depth_damage_curve[d1], depth_damage_curve[d2]
            
            mask = (buildings['flood_depth'] >= d1) & (buildings['flood_depth'] < d2)
            # Linear interpolation
            slope = (h2 - h1) / (d2 - d1)
            interp_hazard = h1 + slope * (buildings.loc[mask, 'flood_depth'] - d1)
            buildings.loc[mask, 'hazard'] = interp_hazard
        
        # For depths above max
        buildings.loc[buildings['flood_depth'] >= depths[-1], 'hazard'] = depth_damage_curve[depths[-1]]
    
    # Calculate overall risk: Risk = Hazard × Vulnerability × Exposure
    buildings['risk'] = 0.0
    
    if 'vulnerability_index' in buildings.columns:
        buildings['risk'] = (
            buildings['hazard'] * 
            buildings['vulnerability_index'] * 
            buildings['exposed'].astype(int)
        )
    
    # Classify risk level
    buildings['risk_level'] = 'none'
    buildings.loc[buildings['risk'] > 0, 'risk_level'] = 'low'
    buildings.loc[buildings['risk'] > 0.2, 'risk_level'] = 'medium'
    buildings.loc[buildings['risk'] > 0.5, 'risk_level'] = 'high'
    buildings.loc[buildings['risk'] > 0.75, 'risk_level'] = 'very_high'
    
    return buildings


def generate_building_exposure_summary(buildings_gdf: gpd.GeoDataFrame) -> Dict[str, Union[int, float, str]]:
    """
    Generate summary statistics of building exposure and risk.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings with exposure and risk metrics
        
    Returns
    -------
    Dict[str, Union[int, float, str]]
        Summary statistics
    """
    summary = {
        'total_buildings': len(buildings_gdf),
        'exposed_buildings': int(buildings_gdf['exposed'].sum()) if 'exposed' in buildings_gdf.columns else 0,
        'exposure_rate_percent': 0.0,
        'high_risk_buildings': 0,
        'affected_area_sqm': 0.0,
        'by_risk_level': {}
    }
    
    # Calculate exposure rate
    if 'exposed' in buildings_gdf.columns and summary['total_buildings'] > 0:
        summary['exposure_rate_percent'] = (
            summary['exposed_buildings'] / summary['total_buildings'] * 100
        )
    
    # Count by risk level
    if 'risk_level' in buildings_gdf.columns:
        summary['high_risk_buildings'] = int(
            buildings_gdf[buildings_gdf['risk_level'].isin(['high', 'very_high'])].shape[0]
        )
        summary['by_risk_level'] = buildings_gdf['risk_level'].value_counts().to_dict()
    
    # Calculate affected area
    if 'area_sqm' in buildings_gdf.columns and 'exposed' in buildings_gdf.columns:
        summary['affected_area_sqm'] = float(
            buildings_gdf[buildings_gdf['exposed']]['area_sqm'].sum()
        )
    
    return summary


# ===============================================
# Raster-Vector Flood Exposure Assessment
# ===============================================

def assess_building_exposure_to_raster(buildings_gdf: gpd.GeoDataFrame,
                                      flood_raster: np.ndarray,
                                      raster_bounds,
                                      raster_transform) -> gpd.GeoDataFrame:
    """
    Assess building exposure to flood raster using 5-level classification.
    
    METHOD:
    -------
    For each building centroid:
    1. Identify corresponding raster pixel (using raster bounds & transform)
    2. Read flood raster value at that pixel
    3. Classify into 5 exposure levels based on FPI value thresholds
    
    EXPOSURE LEVELS (5-tier classification):
    - No Exposure: FPI = 0
    - Very Low: 0 < FPI < 0.15
    - Medium: 0.15 ≤ FPI < 0.35
    - High: 0.35 ≤ FPI < 0.65
    - Very High: FPI ≥ 0.65
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings with point or polygon geometries
    flood_raster : np.ndarray
        2D flood raster (e.g., FPI values [0-1])
    raster_bounds : rasterio.coords.BoundingBox
        Raster geographic extent (left, bottom, right, top)
    raster_transform : rasterio.transform.Affine
        Raster georeference transform
        
    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with added columns:
        - 'flood_raster_value': float, FPI value at building centroid
        - 'exposure_class': str, one of ['No Exposure', 'Low', 'Medium', 'High', 'Very High']
        - 'exposure_level': int, 0-4 (0=No Exposure, 4=Very High)
        
    Examples
    --------
    >>> buildings_exp = assess_building_exposure_to_raster(
    ...     buildings_gdf, fpi_array, fpi_bounds, fpi_transform
    ... )
    >>> print(buildings_exp['exposure_class'].value_counts())
    """
    gdf = buildings_gdf.copy()
    
    # Convert to centroids if necessary
    if gdf.geometry.type.isin(['Polygon', 'MultiPolygon']).any():
        gdf['geometry'] = gdf.geometry.centroid
        logger.info("Converted polygon geometries to centroids for raster sampling")
    
    # Initialize exposure columns - 5-level classification
    gdf['flood_raster_value'] = np.nan
    gdf['exposure_class'] = 'unknown'
    gdf['exposure_level'] = -1
    
    # Sample raster at building locations
    for idx, row in gdf.iterrows():
        try:
            # Get building centroid coordinates
            x, y = row.geometry.x, row.geometry.y
            
            # Convert geographic coordinates to raster indices
            px = int((x - raster_bounds.left) / raster_transform.a)
            py = int((raster_bounds.top - y) / (-raster_transform.e))
            
            # Check bounds
            if 0 <= py < flood_raster.shape[0] and 0 <= px < flood_raster.shape[1]:
                value = float(flood_raster[py, px])
                gdf.at[idx, 'flood_raster_value'] = value
                
                # 5-level exposure classification
                if not np.isnan(value):
                    if value == 0:
                        gdf.at[idx, 'exposure_class'] = 'No Exposure'
                        gdf.at[idx, 'exposure_level'] = 0
                    elif value < 0.15:
                        gdf.at[idx, 'exposure_class'] = 'Very Low'
                        gdf.at[idx, 'exposure_level'] = 1
                    elif value < 0.35:
                        gdf.at[idx, 'exposure_class'] = 'Medium'
                        gdf.at[idx, 'exposure_level'] = 2
                    elif value < 0.65:
                        gdf.at[idx, 'exposure_class'] = 'High'
                        gdf.at[idx, 'exposure_level'] = 3
                    else:
                        gdf.at[idx, 'exposure_class'] = 'Very High'
                        gdf.at[idx, 'exposure_level'] = 4
            else:
                gdf.at[idx, 'exposure_class'] = 'outside_raster'
                gdf.at[idx, 'exposure_level'] = -1
                
        except Exception as e:
            logger.warning(f"Error sampling raster at building {idx}: {e}")
            gdf.at[idx, 'exposure_class'] = 'error'
            gdf.at[idx, 'exposure_level'] = -2
    
    # Summarize exposure distribution
    exposure_dist = gdf['exposure_class'].value_counts()
    logger.info(f"Building exposure assessment complete. 5-level distribution:")
    for exp_class, count in exposure_dist.items():
        if exp_class not in ['outside_raster', 'error', 'unknown']:
            pct = 100 * count / len(gdf[gdf['exposure_level'] >= 0])
            logger.info(f"  {exp_class}: {count} ({pct:.1f}%)")
    
    return gdf


def calculate_population_exposure(buildings_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Calculate population exposure metrics from buildings with population attribute.
    
    REQUIREMENTS:
    - buildings_gdf must have 'population' column (derived from estimate_population_from_buildings)
    - buildings_gdf must have 'exposed' column (from assess_building_exposure_to_raster)
    
    METRICS COMPUTED:
    1. Total exposed population
    2. Total population
    3. Exposure rate (%)
    4. Average population per exposed building
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings with 'population' and 'exposed' columns
        
    Returns
    -------
    dict
        Exposure metrics including:
        - 'total_population': int
        - 'exposed_population': int
        - 'exposure_rate_percent': float
        - 'unexposed_population': int
        - 'avg_pop_per_exposed_building': float
        
    Raises
    ------
    KeyError
        If required columns ('population', 'exposed') are missing
        
    Examples
    --------
    >>> pop_metrics = calculate_population_exposure(buildings_exp)
    >>> print(f"Exposed population: {pop_metrics['exposed_population']}")
    """
    if 'population' not in buildings_gdf.columns:
        raise KeyError("'population' column not found. Run estimate_population_from_buildings first.")
    if 'exposed' not in buildings_gdf.columns:
        raise KeyError("'exposed' column not found. Run assess_building_exposure_to_raster first.")
    
    # Calculate metrics
    total_pop = int(buildings_gdf['population'].sum())
    exposed_pop = int(buildings_gdf[buildings_gdf['exposed']]['population'].sum())
    unexposed_pop = total_pop - exposed_pop
    exposure_rate = (exposed_pop / total_pop * 100) if total_pop > 0 else 0
    
    exposed_buildings = buildings_gdf['exposed'].sum()
    avg_pop_per_exposed = (exposed_pop / exposed_buildings) if exposed_buildings > 0 else 0
    
    metrics = {
        'total_population': total_pop,
        'exposed_population': exposed_pop,
        'unexposed_population': unexposed_pop,
        'exposure_rate_percent': exposure_rate,
        'exposed_buildings': int(exposed_buildings),
        'total_buildings': len(buildings_gdf),
        'building_exposure_rate_percent': (exposed_buildings / len(buildings_gdf) * 100) if len(buildings_gdf) > 0 else 0,
        'avg_pop_per_exposed_building': avg_pop_per_exposed
    }
    
    logger.info(
        f"Population exposure: {exposed_pop}/{total_pop} persons exposed "
        f"({exposure_rate:.1f}%)"
    )
    
    return metrics


def generate_exposure_summary(buildings_gdf: gpd.GeoDataFrame,
                              geometry_column: Optional[str] = None) -> pd.DataFrame:
    """
    Generate detailed exposure summary table.
    
    Produces a summary of exposure metrics with optional grouping by
    administrative boundaries or analysis zones.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Buildings with exposure attributes
    geometry_column : str, optional
        Column name for spatial grouping (e.g., 'admin_zone', 'district')
        
    Returns
    -------
    pd.DataFrame
        Summary table with exposure metrics per zone or overall
        
    Examples
    --------
    >>> summary = generate_exposure_summary(buildings_exp, 'district')
    >>> print(summary)
    """
    summary_data = []
    
    if geometry_column and geometry_column in buildings_gdf.columns:
        # Group by geometry column
        for zone, group in buildings_gdf.groupby(geometry_column):
            row_dict = {
                'zone': zone,
                'total_buildings': len(group),
                'exposed_buildings': int(group['exposed'].sum()),
                'total_population': int(group['population'].sum()) if 'population' in group.columns else 0,
                'exposed_population': int(group[group['exposed']]['population'].sum()) if 'population' in group.columns else 0,
            }
            
            if row_dict['total_buildings'] > 0:
                row_dict['building_exposure_pct'] = 100 * row_dict['exposed_buildings'] / row_dict['total_buildings']
            else:
                row_dict['building_exposure_pct'] = 0
                
            if row_dict['total_population'] > 0:
                row_dict['population_exposure_pct'] = 100 * row_dict['exposed_population'] / row_dict['total_population']
            else:
                row_dict['population_exposure_pct'] = 0
            
            summary_data.append(row_dict)
    else:
        # Overall summary
        row_dict = {
            'zone': 'OVERALL',
            'total_buildings': len(buildings_gdf),
            'exposed_buildings': int(buildings_gdf['exposed'].sum()),
            'total_population': int(buildings_gdf['population'].sum()) if 'population' in buildings_gdf.columns else 0,
            'exposed_population': int(buildings_gdf[buildings_gdf['exposed']]['population'].sum()) if 'population' in buildings_gdf.columns else 0,
        }
        
        if row_dict['total_buildings'] > 0:
            row_dict['building_exposure_pct'] = 100 * row_dict['exposed_buildings'] / row_dict['total_buildings']
        else:
            row_dict['building_exposure_pct'] = 0
            
        if row_dict['total_population'] > 0:
            row_dict['population_exposure_pct'] = 100 * row_dict['exposed_population'] / row_dict['total_population']
        else:
            row_dict['population_exposure_pct'] = 0
        
        summary_data.append(row_dict)
    
    return pd.DataFrame(summary_data)
