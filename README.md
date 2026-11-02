# Graphyne

A vector similarity index built from scratch in C++20 — an implementation of
**HNSW** (Hierarchical Navigable Small World) graphs for approximate nearest-neighbour
search, with AVX2/FMA distance kernels and a flat, cache-friendly memory layout.

On **SIFT1M** (1,000,000 × 128 base vectors), single-threaded at `ef=50`, it reaches
**6,449 QPS at recall@10 = 0.956** — **~26% higher throughput *and* higher recall than
[faiss](https://github.com/facebookresearch/faiss)** (5,101 QPS / 0.951), and higher recall
than [hnswlib](https://github.com/nmslib/hnswlib) at equal search depth — all benchmarked on
the same machine.

> This is a systems/learning project: the goal was to implement HNSW correctly from the
> paper, make it fast with full memory control, and — just as importantly — **benchmark it
> honestly**. See [BENCHMARKS.md](BENCHMARKS.md) for the full numbers, the library
> comparison, and a write-up of the measurement pitfalls (thermal throttling, PGO
> miscompiles, run-to-run graph variance) found along the way.

---

## Benchmark summary

SIFT1M, 1M base vectors, 10,000 queries, single thread, `M=16`, `efConstruction=200`,
`k=10`, on an Intel i7-13700H. Query at `ef=50`:

| Engine                         | QPS    | Recall@10 | Notes                          |
| ------------------------------ | ------ | --------- | ------------------------------ |
| hnswlib 0.8                    | 7,401 | 0.9466    | cold query from saved index     |
| **Graphyne**                   | 6,449 | **0.9556**| hot query (no serialization yet) |
| faiss 1.15.1 (`IndexHNSWFlat`) | 5,101 | 0.9506    | cold query from saved index     |

**vs faiss — a clean win:** at the same `ef`, Graphyne is both faster (6,449 vs 5,101 QPS,
+26%) *and* higher recall (0.9556 vs 0.9506).

**vs hnswlib — a split:** hnswlib has higher raw throughput (7,401 QPS) but at lower recall
(0.9466). Graphyne reaches higher recall at the same search depth; to match Graphyne's
0.9556 recall hnswlib must raise `ef` (do more work). So Graphyne wins on recall-per-`ef`,
not on peak QPS.

One caveat on state: Graphyne's number is measured *hot* (query runs right after a ~9-minute
build) because it cannot yet serialize an index, whereas the libraries query cold from a
saved index. The full `ef` sweep, methodology, and caveats are in
[BENCHMARKS.md](BENCHMARKS.md).

---

## Design

- **HNSW graph.** Multi-layer navigable small-world graph. Upper layers are sparse with
  long-range edges; layer 0 holds every node. Insertion runs greedy descent to the node's
  assigned layer, then beam search + a diversity neighbour-selection heuristic with
  bidirectional, pruned wiring. Search descends to layer 1, then one beam search at layer 0.
- **Separated storage (`NodeBuffer`).** Vector data lives in a single 32-byte-aligned flat
  `float` buffer, sized once at construction (`n_nodes × dim`), addressed by node id. This
  keeps the graph structs small and the distance path AVX2-friendly (aligned loads).
- **Flat neighbour slab (`NeighborStore`).** Graph topology is a contiguous `int` slab with
  a parallel degree array, replacing per-node `vector<vector<int>>` — one allocation
  instead of ~1M, and a prefetchable neighbour walk on the hot loop.
- **AVX2 + FMA distance.** Hand-written 8-wide L2 kernel (`dist_avx2`) with `__restrict__`
  pointers; scalar reference kept as fallback and for correctness checks.
- **Integer-index discipline.** All graph references are integer indices into flat arrays,
  never raw pointers — so a buffer resize can never dangle a reference.

See [Dev.md](Dev.md) for the development log and the step-by-step optimisation history
(scalar → `-O3 -mavx2 -mfma` → AVX2 intrinsics → visited-table + heap reserve).

---

## Build

Requires **CMake ≥ 3.20** and **Clang** (developed with clang 22 from MSYS2 UCRT64 on
Windows; any recent clang with AVX2 support works).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
cmake --build build
```

### Data

The benchmark uses the ANN-Benchmarks `sift-128-euclidean.hdf5` dataset. Convert it to the
flat binary format the loaders expect (`[int32 rows][int32 cols][row-major payload]`):

```sh
python scripts/convert_hdf5.py      # writes bin/{train,test,neighbors,distances}.bin
```

### Run the benchmark

```sh
./build/start/graphyne            # full SIFT1M (expects ./bin)
./build/start/graphyne bin50k     # 50K subset for quick iteration
```

The harness builds the index with a progress/ETA readout, queries all 10K vectors, scores
recall@10 against the ground truth, and prints build time and QPS.

---

## Comparing against faiss / hnswlib

Reproducible baseline drivers (matched parameters, same data, same recall scorer, one query
per call) live in [bench/baselines/](bench/baselines/). See the README there.

---

## Status

| Component                              | Status                                   |
| -------------------------------------- | ---------------------------------------- |
| HNSW insertion / search                | ✅ implemented, validated on SIFT1M       |
| Diversity neighbour selection          | ✅ implemented                            |
| Flat storage (`NodeBuffer`)            | ✅ implemented                            |
| Flat neighbour slab (`NeighborStore`)  | ✅ implemented                            |
| Soft deletion (tombstone + ep promote) | ✅ implemented                            |
| AVX2 + FMA distance                    | ✅ implemented                            |
| IO loaders                             | ✅ implemented                            |
| Benchmark harness + library comparison | ✅ implemented                            |
| Index serialization (save / load)      | ⬜ planned (unblocks cold benchmark + API)|
| Fixed RNG seed                         | ⬜ planned (run-to-run reproducibility)   |
| Multithreaded queries                  | ⬜ planned                                |
| Product Quantization                   | ⬜ planned                                |
| REST API / deployment                  | ⬜ planned                                |

### Known issue

A profile-guided-optimization (PGO) build miscompiles at 1M scale and segfaults (fine at
50K). This points at a latent memory bug in the hot path, likely surfaced by aggressive
codegen — to be run under AddressSanitizer. Details in [BENCHMARKS.md](BENCHMARKS.md).

---

## Repository layout

```
core/hnsw   HNSW index: hnsw.{hpp,cpp}, storage.hpp (NodeBuffer), neighbors.hpp (NeighborStore)
core/io     loaders for the flat .bin dataset format
core/pq     Product Quantization (planned)
start       benchmark entry point (main.cpp)
tests       recall/QPS test harness
bench       baseline drivers for faiss / hnswlib comparison
scripts     dataset conversion / verification (Python)
```

---

## License

MIT — see [LICENSE](LICENSE).
