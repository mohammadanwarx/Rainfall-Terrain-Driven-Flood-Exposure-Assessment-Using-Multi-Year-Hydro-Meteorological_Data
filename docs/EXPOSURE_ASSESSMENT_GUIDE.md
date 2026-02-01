# Flood Exposure Assessment - Technical Guide

## Overview

The flood exposure assessment module implements a raster-vector integrated approach to evaluate building and population exposure to flood hazards using the Flood Propensity Index (FPI) raster dataset.

## Methodology

### Spatial Integration Approach: Point-in-Raster Sampling

The assessment uses building **centroid coordinates** sampled against the FPI raster:

```
Geographic Coordinates (Building Centroid)
    ↓
Affine Coordinate Transform
    ↓
Pixel Coordinates (x, y) in FPI grid
    ↓
Sample FPI value at [pixel_x, pixel_y]
    ↓
Classify Exposure: Low/Medium/High based on thresholds
```

### Mathematical Foundation

#### Coordinate Transformation (Geographic → Pixel)

Given:
- Raster bounds: `[left, bottom, right, top]`
- Raster dimensions: `(rows, cols)` = shape of FPI array
- Building centroid: `(geo_x, geo_y)` in geographic coordinates

Pixel coordinates calculated as:
```
pixel_x = (geo_x - bounds.left) / pixel_width
pixel_y = (bounds.top - geo_y) / pixel_height
```

Where:
- `pixel_width = (bounds.right - bounds.left) / cols`
- `pixel_height = (bounds.top - bounds.bottom) / rows`

#### Exposure Classification

Buildings are classified based on FPI value at centroid:

| Exposure Class | FPI Range | Risk Level |
|---|---|---|
| **Low** | FPI < 0.33 | Low risk |
| **Medium** | 0.33 ≤ FPI < 0.66 | Medium risk |
| **High** | FPI ≥ 0.66 | High risk |

**Exposure Threshold (Exposed):** Building is considered **exposed** if FPI ≥ 0.33 (medium-to-high risk)

#### Population Estimation

Population is estimated uniformly across buildings:
```
Population per building = 1 household × 6 persons
Total population = Number of buildings × 6
Exposed population = Number of exposed buildings × 6
```

**Assumption Rationale:** No external population dataset is available. The 6-person household represents an average African household size suitable for the Nile Basin region context.

### Metrics Computed

#### Building-Level Metrics
- `total_buildings`: Total number of buildings in study area
- `exposed_buildings`: Number of buildings with FPI ≥ 0.33
- `building_exposure_rate`: (exposed_buildings / total_buildings) × 100%
- FPI value statistics: min, max, mean, median at building centroids

#### Population-Level Metrics
- `total_population`: Total population (buildings × 6)
- `exposed_population`: Population in exposed buildings
- `unexposed_population`: Population in non-exposed buildings
- `exposure_rate_percent`: (exposed_population / total_population) × 100%
- `avg_pop_per_exposed_building`: Average household size (always 6 in this implementation)

#### Summary Statistics
- Exposure class distribution (Low/Medium/High count and percentage)
- Spatial clustering of exposure (zone-based if available)
- Per-zone exposure metrics

## Function Reference

### `assess_building_exposure_to_raster()`

**Purpose:** Classify buildings by flood exposure using raster sampling

**Signature:**
```python
def assess_building_exposure_to_raster(
    buildings_gdf,
    flood_raster,
    raster_bounds,
    raster_transform,
    exposure_threshold=0.33
)
```

**Parameters:**
- `buildings_gdf` (GeoDataFrame): Buildings with geometry column (polygons or points)
- `flood_raster` (ndarray): FPI raster [0-1] with shape (rows, cols)
- `raster_bounds` (Bounds): Raster extent with attributes `left, bottom, right, top`
- `raster_transform` (Affine): Coordinate transformation (currently simplified to pixel calculation)
- `exposure_threshold` (float): FPI threshold for exposure classification (default 0.33)

**Output:** Modified GeoDataFrame with new columns:
- `flood_raster_value` (float): FPI value at building centroid
- `exposed` (bool): True if FPI ≥ threshold
- `exposure_class` (str): 'Low' / 'Medium' / 'High'

**Algorithm:**
1. Extract building centroids from polygon geometries
2. Convert geographic coordinates to pixel indices using Affine transform
3. Sample FPI values at pixel locations
4. Classify each building as Low/Medium/High risk
5. Return GeoDataFrame with exposure columns

### `calculate_population_exposure()`

**Purpose:** Compute population-level exposure metrics

**Signature:**
```python
def calculate_population_exposure(buildings_gdf, household_size=6)
```

**Parameters:**
- `buildings_gdf` (GeoDataFrame): Buildings with `population` and `exposed` columns
- `household_size` (int): Persons per household (used for population scaling)

**Output:** Dictionary with keys:
```python
{
    'total_population': int,
    'exposed_population': int,
    'unexposed_population': int,
    'exposure_rate_percent': float,  # 0-100
    'exposed_buildings': int,
    'total_buildings': int,
    'building_exposure_rate_percent': float,
    'avg_pop_per_exposed_building': float
}
```

**Computation:**
- Counts buildings where `exposed == True`
- Sums population in exposed vs unexposed buildings
- Calculates exposure rates as percentages

### `generate_exposure_summary()`

**Purpose:** Generate summary statistics table

**Signature:**
```python
def generate_exposure_summary(buildings_gdf, geometry_column=None)
```

**Parameters:**
- `buildings_gdf` (GeoDataFrame): Buildings with exposure columns
- `geometry_column` (str, optional): Column name for grouping (e.g., 'zone', 'district')

**Output:** DataFrame with summary metrics
- If `geometry_column` provided: Per-zone statistics with rows for each zone
- If None: Overall summary with single row

**Columns:**
- `zone` / `geometry`: Zone identifier (if grouped)
- `total_buildings`: Count of buildings
- `exposed_buildings`: Count of exposed buildings
- `total_population`: Total population
- `exposed_population`: Population in exposed buildings
- `building_exposure_pct`: (exposed_buildings / total_buildings) × 100%
- `population_exposure_pct`: (exposed_population / total_population) × 100%

## Data Requirements

### Input Data

**1. FPI Raster Dataset**
- Format: NetCDF or GeoTIFF
- Data type: Float32 (values 0-1)
- Spatial extent: Covers Area of Interest (AOI)
- Resolution: ~5km (50×50 pixels for Nile Basin case study)
- Source: `data/processed/fpi_raster_[year].nc`

**2. Building Footprints**
- Format: GeoJSON or Shapefile
- Geometry type: Polygon (representing building outlines)
- Attributes: At minimum, should have valid geometry
- CRS: Must match raster CRS (typically EPSG:4326 or local UTM)
- Source: `data/raw/buildings/`

**3. Area of Interest (AOI)**
- Format: Shapefile or GeoJSON
- Geometry type: Polygon
- Purpose: Define study area boundary
- Source: `data/raw/vector/AOI.shp`

### Output Data

**Primary Outputs:**
- Modified buildings GeoDataFrame with exposure columns (saved to notebook variables)
- Exposure metrics dictionary
- Summary statistics DataFrame

**Visualization Outputs:**
- `outputs/exposure_map_buildings_on_fpi.png`: Map showing buildings overlaid on FPI raster
- `outputs/exposure_summary_charts.png`: 3-panel summary chart (classes, status, population)

## Validation and Error Handling

### Built-in Checks

1. **Empty GeoDataFrame:** Validates non-empty input
2. **Missing Columns:** Checks for required columns (`exposed`, `population`)
3. **Out-of-Bounds Buildings:** Flags buildings with centroids outside raster extent
4. **CRS Mismatch:** Warns if buildings and raster have different coordinate systems
5. **Invalid FPI Values:** Handles NaN/inf values in raster sampling

### Error Messages

```python
# Example: Missing population column
ValueError: 'population' column not found in buildings_gdf. 
Run add_population_attribute() first.

# Example: Out of bounds
Warning: 42 buildings have centroids outside raster extent. 
These will be assigned NaN and marked as unexposed.
```

## Assumptions and Limitations

### Key Assumptions

1. **One Building = One Household** (exactly one household per building footprint)
2. **Uniform Household Size** = 6 persons (across entire study area)
3. **Centroid-Based Exposure** (FPI value at centroid represents building exposure)
4. **Binary Exposure Classification** (building either exposed or not, no gradients)
5. **Static Population Distribution** (population not changing temporally)

### Limitations

1. **No Building Height Data**: Exposure does not account for vertical structure or floor levels
2. **No Temporal Dynamics**: Uses static FPI; doesn't model dynamic flood progression
3. **Centroid Limitation**: Large buildings may have parts in low-risk areas but centroid in high-risk area (and vice versa)
4. **No Vulnerability Assessment**: Exposure ≠ Risk (no socioeconomic/adaptive capacity factors)
5. **Uniform Population Assumption**: Does not account for actual population distributions, age structures, or density variation

### Recommended Improvements

1. **Polygon-Based Sampling**: Use zonal statistics instead of centroid sampling for more robust exposure
2. **Multi-Level Classification**: Sample at multiple building locations (centroid + corners)
3. **Vulnerability Integration**: Combine exposure with socioeconomic indicators for risk assessment
4. **Temporal Analysis**: Time-series exposure assessment across multiple years/scenarios
5. **Uncertainty Quantification**: Propagate uncertainty from FPI model through exposure metrics

## Reproducibility

All functions include:
- ✅ Comprehensive docstrings with assumptions documented
- ✅ Parameter validation with informative error messages
- ✅ Logging of processing steps and results
- ✅ Deterministic algorithms (no randomness, same input → same output)
- ✅ Version control of source data and thresholds

To reproduce results:
1. Use same FPI raster dataset (version/date)
2. Use same building footprint dataset
3. Set household_size=6 and exposure_threshold=0.33 (defaults)
4. Run `flood_exposure_assessment.ipynb` cells sequentially

## References

- Raster-Vector Integration: GDAL/Rasterio documentation
- Flood Propensity Index: [See `_03_analysis/flood_propensity.py`]
- Point-in-polygon / Point-in-raster: Standard geospatial algorithms
- Household size estimate: African Development Bank (average household size ~6 in Sub-Saharan Africa)

## Contact

For questions or improvements, see project README.md or examine source code comments in:
- `src/_01_data_loading/buildings.py`
- `src/_03_analysis/exposure.py`
- `notebooks/flood_exposure_assessment.ipynb`
