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
            
        return data
    
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

if __name__ == '__main__':
    pass
    # tile_path =  