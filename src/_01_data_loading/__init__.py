"""
Data Loading Module

Functions for loading all types of geospatial data:
- Buildings (CSV, GeoJSON, Shapefile)
- Rainfall (CHIRPS netCDF files)
- Vector data (AOI, boundaries)
"""

from .buildings import (
    filter_buildings_by_extent,
)

from .rainfall_processing import (
    CHIRPSProcessor,
    create_rainfall_datacube,
)

__all__ = [
    # Buildings
    "filter_buildings_by_extent", 
    # Rainfall
    "CHIRPSProcessor",
    "create_rainfall_datacube",
]
