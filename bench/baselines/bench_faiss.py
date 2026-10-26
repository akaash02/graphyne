# faiss IndexHNSWFlat baseline matching Graphyne: M=16, efC=200, k=10, one thread.
# usage: python bench_faiss.py <dataDir> <indexPath> build|query
import sys, time
import numpy as np
import faiss

faiss.omp_set_num_threads(1)
d_dir, idx_path, mode = sys.argv[1:4]

def load(p, dt):
    with open(p, "rb") as f:
        r, c = np.frombuffer(f.read(8), dtype=np.int32)
        return np.frombuffer(f.read(), dtype=dt).reshape(r, c)

Q = np.ascontiguousarray(load(f"{d_dir}/test.bin", np.float32))
G = load(f"{d_dir}/neighbors.bin", np.int32)[:, :10]
k = 10

if mode == "build":
    X = np.ascontiguousarray(load(f"{d_dir}/train.bin", np.float32))
    index = faiss.IndexHNSWFlat(X.shape[1], 16)
    index.hnsw.efConstruction = 200
    t0 = time.perf_counter()
    step = 100_000
    for s in range(0, len(X), step):
        index.add(X[s:s + step])
        print(f"  {s + step}  {time.perf_counter() - t0:.0f}s", flush=True)
    bt = time.perf_counter() - t0
    print(f"BUILD faiss n={len(X)} {bt:.1f}s {len(X) / bt:.0f} vec/s", flush=True)
    faiss.write_index(index, idx_path)
else:
    index = faiss.read_index(idx_path)

for ef in [10, 20, 30, 50, 80, 100, 150, 200, 400]:
    index.hnsw.efSearch = ef
    # one query per call, same as Graphyne's loop, so per-call overhead is counted
    t0 = time.perf_counter()
    I = np.empty((len(Q), k), dtype=np.int64)
    for i in range(len(Q)):
        _, I[i] = index.search(Q[i:i + 1], k)
    qt = time.perf_counter() - t0
    hits = sum(len(np.intersect1d(I[i], G[i])) for i in range(len(Q)))
    print(f"RESULT faiss {mode} ef={ef} recall={hits / (len(Q) * k):.4f} qps={len(Q) / qt:.0f}", flush=True)
