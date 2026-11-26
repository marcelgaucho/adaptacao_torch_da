# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:57:01 2026

@author: Marcel
"""

# %% Imports

# From standard modules
import csv
import json
from pathlib import Path
from dataclasses import dataclass
import tempfile

# From our modules
from .extractor import PatchExtractor

# From other modules
import numpy as np

# %%

class PatchStore:
    """Stores all patches of a (domain, split) into two numpy arrays:

        images.npy  (N, C, S, S)   fused T1+T2 image, in CHW
        labels.npy  (N, S, S)      target (0, 1 or ignore_index)

    Row i of index.csv describes patch i of the two arrays.
    """

    COLUMNS = ["tile", "idx", "ymin", "xmin", "ymax", "xmax", "pos_frac"]

    def __init__(self, out_dir, image_dtype=np.float32):
        self.out = Path(out_dir)
        self.image_dtype = image_dtype

    def write(self, pipeline, samples, domain, split):
        images, labels, rows, shapes = [], [], [], {}
        for sample in samples: # Loop through tile samples
            tp = pipeline.extract(sample) # Patch extraction
            shapes[sample.name] = list(tp.tile_hw) # Respective tile shape
            images.append(tp.x.transpose(0, 3, 1, 2).astype(self.image_dtype)) # NHWC -> NCHW
            labels.append(tp.y.astype(np.uint8))
            for i, (ymin, xmin, ymax, xmax) in enumerate(tp.coords): # Loop through patch coordinates
                rows.append([sample.name, i, int(ymin), int(xmin), int(ymax), int(xmax),
                             float((tp.y[i] == 1).mean())]) # Append patch info

        if not rows: # empty patch list
            raise ValueError(f"{domain}/{split}: no patch extracted")

        d = self.out / domain / split # save directory
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "images.npy", np.concatenate(images))
        np.save(d / "labels.npy", np.concatenate(labels))
        with open(d / "index.csv", "w", newline="") as f: # Write info to csv
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            writer.writerows(rows)
        
        (d / "tile_shapes.json").write_text(json.dumps(shapes)) # Write tiles shapes to json 

        return len(rows) # Number of patches is returned
    
# %%

if __name__ == "__main__":
    @dataclass
    class DummySample:
        name: str
        image: np.ndarray  # (H, W, C)
        label: np.ndarray  # (H, W)


    @dataclass
    class ExtractedTiles:
        tile_hw: tuple[int, int]
        x: np.ndarray       # (N, S, S, C)
        y: np.ndarray       # (N, S, S)
        coords: np.ndarray  # (N, 4)


    class DummyPipeline:
        def __init__(self, extractor):
            self.extractor = extractor

        def extract(self, sample):
            h, w, _ = sample.image.shape
            x, coords = self.extractor(sample.image, get_coords=True)
            # Extract matching label patches using the same extractor
            y, _ = self.extractor(sample.label[..., None], get_coords=True)
            return ExtractedTiles(tile_hw=(h, w), x=x, y=y[..., 0], coords=coords)

    extractor = PatchExtractor(patch_size=4, overlap_percent=0.5, padding=True)
    pipeline = DummyPipeline(extractor)

    # Create two synthetic 8x8 samples with 2 channels (e.g., T1+T2) and binary labels
    s1_img = np.arange(8 * 8 * 2, dtype=np.float32).reshape(8, 8, 2)
    s1_lbl = np.zeros((8, 8), dtype=np.uint8)
    s1_lbl[2:6, 2:6] = 1

    s2_img = np.ones((8, 8, 2), dtype=np.float32) * 100
    s2_lbl = np.ones((8, 8), dtype=np.uint8)

    samples = [
        DummySample(name="tile_01", image=s1_img, label=s1_lbl),
        DummySample(name="tile_02", image=s2_img, label=s2_lbl),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = PatchStore(out_dir=tmp_dir)
        total = store.write(pipeline, samples, domain="area_a", split="train")

        out_path = Path(tmp_dir) / "area_a" / "train"
        images = np.load(out_path / "images.npy")
        labels = np.load(out_path / "labels.npy")
        shapes = json.loads((out_path / "tile_shapes.json").read_text())
        index_lines = (out_path / "index.csv").read_text().splitlines()

        print(f"Total patches written: {total}")
        print("images.npy shape:", images.shape, "| dtype:", images.dtype)  # (32, 2, 4, 4), float32
        print("labels.npy shape:", labels.shape, "| dtype:", labels.dtype)  # (32, 4, 4), uint8
        print("tile_shapes.json:", shapes)
        print("CSV header:", index_lines[0])
        print("First row: ", index_lines[1])
        print("Last row:  ", index_lines[-1])