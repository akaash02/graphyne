import h5py
import numpy as np
import struct
import os

SRC  = "data/sift-128-euclidean.hdf5"
DEST = "bin"

# Each file: [int32 rows][int32 cols][data row-major]
def write_bin(path, array):
    rows, cols = array.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", rows, cols))
        f.write(array.tobytes())
    print(f"  wrote {path}  ({rows}x{cols}  {array.dtype}  {os.path.getsize(path) / 1e6:.1f} MB)")

os.makedirs(DEST, exist_ok=True)

with h5py.File(SRC, "r") as f:
    datasets = {
        "train":     (f["train"][:],     "train.bin"),
        "test":      (f["test"][:],      "test.bin"),
        "neighbors": (f["neighbors"][:], "neighbors.bin"),
        "distances": (f["distances"][:], "distances.bin"),
    }

for name, (array, filename) in datasets.items():
    write_bin(os.path.join(DEST, filename), array)

print("Done.")
