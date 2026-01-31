"""
Digital Elevation Model (DEM) Processing Module

This module handles DEM data loading, preprocessing, and manipulation
for flood exposure analysis.
"""

import numpy as np
from typing import Tuple, Optional

# Optional rasterio import
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def load_dem(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load a DEM raster file.
    
    Parameters
    ----------
    filepath : str
        Path to the DEM raster file
    
    Returns
    -------
    Tuple[np.ndarray, dict]
        DEM array and metadata dictionary
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio required. Install: pip install rasterio")
    with rasterio.open(filepath) as src:
        dem = src.read(1)
        metadata = src.meta.copy()
    return dem, metadata


def fill_depressions(dem: np.ndarray) -> np.ndarray:
    """
    Fill depressions in DEM for hydrological analysis.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM array
        
    Returns
    -------
    np.ndarray
        Depression-filled DEM
    """
    # Simple iterative pit filling: raise any cell lower than all 8 neighbors
    # to the minimum of its neighbors. This is a basic approach and may be
    # slow for large rasters compared to specialized hydrology libraries.
    filled = dem.astype(float, copy=True)
    valid = np.isfinite(filled)

    max_iter = 1000
    for _ in range(max_iter):
        padded = np.pad(filled, 1, mode="edge")
        neighbors = [
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2],                     padded[1:-1, 2:],
            padded[2:, :-2],  padded[2:, 1:-1],  padded[2:, 2:]
        ]
        neighbor_min = np.minimum.reduce(neighbors)

        pits = valid & (filled < neighbor_min)
        if not np.any(pits):
            break

        filled[pits] = neighbor_min[pits]

    return filled


def calculate_slope(dem: np.ndarray, cellsize: float) -> np.ndarray:
    """
    Calculate slope from DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM array
    cellsize : float
        Cell size in meters
        
    Returns
    -------
    np.ndarray
        Slope in degrees
    """
    # Calculate gradients
    dy, dx = np.gradient(dem, cellsize)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope


def save_raster(array: np.ndarray, metadata: dict, output_path: str) -> None:
    """
    Save a NumPy array as a GeoTIFF raster.

    Parameters
    ----------
    array : np.ndarray
        Array to save
    metadata : dict
        Raster metadata from original file
    output_path : str
        Path to save the output file
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio required. Install: pip install rasterio")

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    meta = metadata.copy()
    meta.update({'dtype': array.dtype, 'count': 1})

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(array, 1)
    print(f"✓ Saved: {output_path}")


def process_dem(input_path: str, output_dir: str, cellsize: float = 30.0) -> dict:
    """
    Process DEM: fill depressions, calculate slope, and save results.

    Parameters
    ----------
    input_path : str
        Path to input DEM file
    output_dir : str
        Directory to save processed outputs
    cellsize : float
        Cell size in meters (default: 30m)

    Returns
    -------
    dict
        Dictionary with paths to output files
    """
    from pathlib import Path

    print("=" * 50)
    print("DEM PROCESSING")
    print("=" * 50)

    # Load DEM
    print(f"\nLoading DEM: {input_path}")
    dem, metadata = load_dem(input_path)
    print(f"  Shape: {dem.shape}")

    # Fill depressions
    print("\nFilling depressions...")
    filled = fill_depressions(dem)

    # Calculate slope
    print("Calculating slope...")
    slope = calculate_slope(filled, cellsize)

    # Save outputs
    output_dir = Path(output_dir)
    filled_path = str(output_dir / "dem_filled.tif")
    slope_path = str(output_dir / "slope.tif")

    print("\nSaving outputs...")
    save_raster(filled.astype(np.float32), metadata, filled_path)
    save_raster(slope.astype(np.float32), metadata, slope_path)

    print("\n✓ Processing complete!")

    return {
        'filled_dem': filled_path,
        'slope': slope_path
    }


#==================
# Run DEM processing when executed as script
'''
this is a usage for above functions, Dem.tif been used and callculate the fill and  slope
and it saves the both raster files

'''
if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Determine base directory
    try:
        base_dir = Path(__file__).parent.parent.parent
    except NameError:
        base_dir = Path.cwd()
        if not (base_dir / "data").exists():
            for parent in [base_dir.parent, base_dir.parent.parent]:
                if (parent / "data").exists():
                    base_dir = parent
                    break

    # Define paths
    input_path = base_dir / "data" / "raw" / "raster" / "DEM2.tif"
    output_dir = base_dir / "data" / "processed"

    print(f"Base Directory: {base_dir}")
    print(f"Input DEM: {input_path}")
    print(f"Output Directory: {output_dir}")

    # Run processing
    results = process_dem(str(input_path), str(output_dir), cellsize=30)

    print(f"\nOutputs:")
    for name, path in results.items():
        print(f"  {name}: {path}")
