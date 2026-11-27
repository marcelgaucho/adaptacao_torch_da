# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:55:58 2026

@author: Marcel
"""

# %% Imports

# From standard modules
from pathlib import Path

# From other modules
from osgeo import gdal, gdal_array
import numpy as np

# %%

class RasterReader:
    @staticmethod
    def read(path):
        """Loads a geographic file via pathlib and returns a (C, H, W) array."""
        # GDAL requires strings, so we cast the Path object
        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(f"Could not open {path}") # Error while opening
            
        data = ds.ReadAsArray()
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0) # GDAL (H, W) -> (1, H, W) for masks
        
        # GDAL (C, H, W) -> (H, W, C)
        return np.transpose(data, (1, 2, 0))
    
    @staticmethod
    def export_to_geotiff(array, base_geotiff, out_path):
        ''' Export to geotiff an array (H, W, C) '''
        # Open base geotiff
        ds_base = gdal.Open(base_geotiff)
        if ds_base is None:
            raise FileNotFoundError(f"Could not open {base_geotiff}")
            
        # Array dimensions
        y_pixels, x_pixels, nbands = array.shape
         
        # Map numpy dtype to GDAL type 
        code_type = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
         
        # Create output dataset
        driver = gdal.GetDriverByName('GTiff') 
        out_ds = driver.Create(out_path, xsize=x_pixels, ysize=y_pixels,
                               bands=nbands, eType=code_type)
        
        # Write arrays
        for b in range(nbands):
            out_ds.GetRasterBand(b+1).WriteArray(array[..., b])
            
        # Sync Georeferencing
        out_ds.SetProjection(ds_base.GetProjection())
        out_ds.SetGeoTransform(ds_base.GetGeoTransform())
        
        # Write on disk and close files
        out_ds.FlushCache()
        out_ds = None 
        ds_base = None
            
# %%

if __name__ == "__main__":
    # Use forward slashes ("/vsimem/...") directly instead of Path(...) on Windows,
    # because Path converts "/" into "\\", which GDAL does not recognize as /vsimem/.
    base_path = "/vsimem/base_test.tif"
    out_path = "/vsimem/output_test.tif"

    # Create a simple 4x4 single-band GeoTIFF in GDAL's virtual RAM filesystem
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(base_path, xsize=4, ysize=4, bands=1, eType=gdal.GDT_Byte)
    if ds is None:
        raise RuntimeError("GDAL failed to create virtual GeoTIFF in /vsimem/")

    ds.SetGeoTransform((10.0, 0.5, 0.0, 20.0, 0.0, -0.5))
    ds.GetRasterBand(1).WriteArray(np.zeros((4, 4), dtype=np.uint8))
    ds.FlushCache()
    ds = None

    # Test reading -> returns (C, H, W)
    read_base = RasterReader.read(base_path)
    print("read_base shape (C, H, W):", read_base.shape)

    # Test export -> expects (H, W, C) array
    array_hwc = np.arange(4 * 4 * 2, dtype=np.uint8).reshape(4, 4, 2)
    RasterReader.export_to_geotiff(array_hwc, base_path, out_path)

    read_out = RasterReader.read(out_path)
    print("read_out shape (C, H, W):", read_out.shape)
    print("Band 0 matches:", np.array_equal(read_out[0], array_hwc[..., 0]))
    print("Band 1 matches:", np.array_equal(read_out[1], array_hwc[..., 1]))

    # Free the virtual in-memory files from GDAL
    gdal.Unlink(base_path)
    gdal.Unlink(out_path)