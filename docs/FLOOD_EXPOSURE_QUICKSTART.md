# Flood Exposure Assessment - Quick Start Guide

## Overview

The **flood_exposure_assessment.ipynb** notebook implements a raster-vector integrated approach to assess building and population exposure to flood hazards using the Flood Propensity Index (FPI) raster.

## Key Components

### 1. Data Loading (Section 1)
- Loads FPI raster from NetCDF
- Loads buildings from shapefile/GeoJSON
- Loads Area of Interest (AOI) boundary
- Clips all data to study area
- Validates data integrity and prints summaries

### 2. Population Estimation (Section 2)
- Adds population attribute to buildings
- **Assumption**: 6 persons per household per building
- Computes total and average population statistics
- Suitable for regions without detailed population data

### 3. Building Exposure Assessment (Section 3)
- **Method**: Point-in-raster sampling (centroid coordinates → FPI raster)
- Classifies buildings into 3 exposure classes:
  - **Low**: FPI < 0.33
  - **Medium**: 0.33 ≤ FPI < 0.66
  - **High**: FPI ≥ 0.66
- Marks buildings as "exposed" if FPI ≥ 0.33 (medium-to-high risk)
- Outputs exposure statistics and class distribution

### 4. Population Exposure Metrics (Section 4)
- Calculates 8 key metrics:
  1. Total population
  2. Exposed population
  3. Unexposed population
  4. Exposure rate (%)
  5. Exposed buildings count
  6. Total buildings count
  7. Building exposure rate (%)
  8. Average population per exposed building
- Displays metrics in formatted table

### 5. Summary Tables (Section 5)
- Generates exposure summary by zone (if zone column available)
- Computes per-zone exposure percentages
- Produces overall summary DataFrame
- Suitable for administrative reporting

### 6. Visualizations (Section 6)

#### Map Visualization (6.1)
- FPI raster displayed as background with colorbar
- Building points overlaid and color-coded:
  - **Green**: Unexposed buildings
  - **Red**: Exposed buildings
- Shows AOI boundary as black outline
- Includes legend, scale, and title

#### Summary Charts (6.2)
- **Panel 1**: Buildings by exposure class (histogram)
- **Panel 2**: Building exposure status (exposed vs unexposed)
- **Panel 3**: Population exposure status (by population count)

## Running the Notebook

### Prerequisites
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

### Execution Steps

1. **Open the notebook**:
   ```bash
   jupyter notebook notebooks/flood_exposure_assessment.ipynb
   ```

2. **Set parameters** (in first code cell):
   ```python
   HOUSEHOLD_SIZE = 6  # Persons per building
   EXPOSURE_THRESHOLD = 0.33  # FPI threshold for exposure classification
   ```

3. **Run all cells** (Kernel → Restart & Run All or Ctrl+Shift+Enter)

4. **Check outputs**:
   - Console prints: Data summaries, metrics tables
   - Maps and charts: Display inline in notebook
   - Files saved to `outputs/`:
     - `exposure_map_buildings_on_fpi.png`
     - `exposure_summary_charts.png`

## Expected Results

### Sample Output (Example Case Study)

For a hypothetical study area with 150 buildings:

```
DATA LOADING RESULTS:
- FPI Raster Shape: (50, 50)
- Buildings Loaded: 150
- Buildings in AOI: 150
- AOI Area: ~640 km²

POPULATION ESTIMATION:
- Total Population: 900 persons (150 buildings × 6 persons/building)
- Population Distribution: Uniform across all buildings

BUILDING EXPOSURE:
- Total Buildings: 150
- Exposed Buildings: 45 (30%)
- Low Risk: 50 (33%)
- Medium Risk: 70 (47%)
- High Risk: 30 (20%)

POPULATION EXPOSURE:
- Total Population: 900
- Exposed Population: 270 (30%)
- Unexposed Population: 630 (70%)
- Building Exposure Rate: 30%
- Population Exposure Rate: 30%
```

### Visualization Outputs

**exposure_map_buildings_on_fpi.png:**
- 16×12" map showing FPI raster with building overlay
- Color scale: 0 (green/low risk) → 1 (red/high risk)
- Green points: Safe buildings, Red points: Exposed buildings

**exposure_summary_charts.png:**
- 3-panel figure showing exposure statistics
- Bar charts for class distribution and status comparison

## Spatial Logic Explanation

### Point-in-Raster Sampling Algorithm

The notebook uses building **centroid coordinates** sampled against the FPI raster:

```
1. Extract building centroid from polygon geometry
2. Convert geographic coordinates (lat, lon) to pixel indices (x, y)
3. Sample FPI value at pixel location
4. Classify based on FPI threshold
5. Assign exposure status (exposed/unexposed)
```

### Coordinate Transformation Formula

```
pixel_x = (geo_x - raster_left) / pixel_width
pixel_y = (raster_top - geo_y) / pixel_height
```

## Key Assumptions

| Assumption | Value | Rationale |
|---|---|---|
| Household size | 6 persons | Average African household (Sub-Saharan Africa context) |
| Exposure threshold | FPI ≥ 0.33 | Medium-to-high risk classification |
| One building = one household | Uniform | No detailed building use data available |
| Population distribution | Uniform | No external population dataset available |
| Centroid-based classification | Single point | Simplification; assumes exposure uniform across building |

## Interpreting Results

### Building Exposure Rate
- **Definition**: Percentage of buildings with FPI ≥ 0.33
- **Interpretation**: Higher % → More buildings in flood-prone areas
- **Use Case**: Infrastructure planning, building-level interventions

### Population Exposure Rate
- **Definition**: Percentage of people living in exposed buildings
- **Interpretation**: Higher % → More people vulnerable to floods
- **Use Case**: Humanitarian response, population-based risk management

### Exposure Classes
- **High Risk (FPI ≥ 0.66)**: Immediate hazard mitigation recommended
- **Medium Risk (0.33-0.66)**: Monitoring and preparedness recommended
- **Low Risk (FPI < 0.33)**: Standard development guidelines sufficient

## Customization Options

### Change Household Size
```python
# In Section 2 or at top of notebook:
HOUSEHOLD_SIZE = 4  # For smaller households
HOUSEHOLD_SIZE = 8  # For larger households
```

### Change Exposure Threshold
```python
# In Section 3:
EXPOSURE_THRESHOLD = 0.5  # More conservative (only high-risk)
EXPOSURE_THRESHOLD = 0.25  # More inclusive (include low-risk)
```

### Add Zoning Analysis
```python
# In Section 5, add zone column to buildings_gdf before calling:
summary_df = generate_exposure_summary(
    buildings_exposed, 
    geometry_column='district'  # Group by district
)
```

## Troubleshooting

### Issue: "FileNotFoundError: data/raw/raster/..."
**Solution**: Ensure data files are in correct location. Check `data/raw/` structure.

### Issue: "ValueError: 'population' column not found"
**Solution**: Make sure Section 2 has been run before Section 4. Population must be estimated first.

### Issue: "All values are NaN" in exposure sampling
**Solution**: Check CRS alignment between buildings and raster. Both must use same coordinate system.

### Issue: Maps not displaying
**Solution**: Ensure matplotlib backend is configured. Try `%matplotlib inline` in Jupyter.

## Function Reference

### From buildings.py
- `add_population_attribute(buildings_gdf, household_size=6)` → GeoDataFrame with population column

### From exposure.py
- `assess_building_exposure_to_raster(buildings_gdf, flood_raster, raster_bounds, raster_transform, exposure_threshold=0.33)` → GeoDataFrame with exposure columns
- `calculate_population_exposure(buildings_gdf)` → Dict with 8 exposure metrics
- `generate_exposure_summary(buildings_gdf, geometry_column=None)` → DataFrame with summary statistics

## Next Steps

1. **Validation**: Compare results with observed flood extents or historical data
2. **Refinement**: Add building type classification for refined population estimates
3. **Integration**: Incorporate vulnerability factors (building age, construction quality, etc.)
4. **Scenarios**: Model exposure under different climate/rainfall scenarios
5. **Mapping**: Create interactive maps with Folium or Leaflet
6. **Reporting**: Generate stakeholder reports with risk maps and statistics

## References

- **Technical Details**: See `docs/EXPOSURE_ASSESSMENT_GUIDE.md`
- **Source Code**: `src/_01_data_loading/buildings.py`, `src/_03_analysis/exposure.py`
- **Related Notebooks**: `Visulization.ipynb` (6-panel overview), `hydrology_analysis.ipynb` (TWI calculation)

---

**Last Updated**: February 2025
**Notebook Version**: 1.0
**Contact**: See project README.md
