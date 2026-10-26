# Baseline drivers — faiss & hnswlib

Reproducible benchmarks of [faiss](https://github.com/facebookresearch/faiss) and
[hnswlib](https://github.com/nmslib/hnswlib) on the same data, parameters, and recall scorer
that Graphyne's harness uses. Both read the flat `.bin` dataset format
(`[int32 rows][int32 cols][row-major payload]`) produced by `scripts/convert_hdf5.py`, build
an HNSW index with `M=16`, `efConstruction=200`, one thread, score recall@10 against
`neighbors.bin`, and sweep `ef`. One query per `search` call, so per-call overhead is counted
the same way Graphyne's loop counts it.

## faiss (`IndexHNSWFlat`)

```sh
python -m venv venv && venv/bin/pip install faiss-cpu numpy
python bench_faiss.py <dataDir> <indexPath> build   # build, save, then sweep (hot)
python bench_faiss.py <dataDir> <indexPath> query   # reload saved index, sweep (cold)
```

## hnswlib (`HierarchicalNSW`)

Needs the hnswlib headers (`git clone https://github.com/nmslib/hnswlib`) on the include path.
Build with the same flags as Graphyne, and use `lld` if GNU `ld` gives linker trouble:

```sh
clang++ -std=c++20 -O3 -mavx2 -mfma -mtune=native -I /path/to/hnswlib \
        bench_hnswlib.cpp -o bench_hnswlib
./bench_hnswlib <dataDir> <indexPath> build   # build, save, then sweep (hot)
./bench_hnswlib <dataDir> <indexPath> query   # reload saved index, sweep (cold)
```

The `query` mode exists because these libraries can serialize an index, so they can be
measured on a cold CPU in a fresh process — the apples-to-apples state that Graphyne cannot
yet match (no serialization). The cold numbers are the ones quoted in
[../../BENCHMARKS.md](../../BENCHMARKS.md).
