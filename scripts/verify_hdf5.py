import h5py
import sys
import numpy as np

path = "data/sift-128-euclidean.hdf5"

with h5py.File(path, "r") as f:
    def print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  [{name}]  shape={obj.shape}  dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  [{name}/]  (group)")

    print(f"File: {path}")
    print("Keys:")
    f.visititems(print_item)

    print("\nSample values:")
    for key in f.keys():
        ds = f[key]
        if isinstance(ds, h5py.Dataset) and len(ds.shape) >= 1:
            print(f"  {key}[0]: {ds[0]}")

    neighbors = f["neighbors"][:]
    train_len = f["train"].shape[0]
    print(f"\nmax neighbor index: {neighbors.max()},  train length: {train_len}")
    print(f"any out of range: {(neighbors >= train_len).any()}")
