import h5py
import numpy as np
import struct

SRC  = "data/sift-128-euclidean.hdf5"
BINS = {
    "train":     ("bin/train.bin",     np.float32),
    "test":      ("bin/test.bin",      np.float32),
    "neighbors": ("bin/neighbors.bin", np.int32),
    "distances": ("bin/distances.bin", np.float32),
}

def read_bin(path, dtype):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=dtype).reshape(rows, cols)
    return data

with h5py.File(SRC, "r") as f:
    for name, (path, dtype) in BINS.items():
        original = f[name][:]
        recovered = read_bin(path, dtype)

        shape_ok  = original.shape == recovered.shape
        values_ok = np.array_equal(original, recovered)

        status = "OK" if (shape_ok and values_ok) else "FAIL"
        print(f"[{status}] {name}: shape={recovered.shape}  match={values_ok}")
