# Graphyne HNSW — Development Log

## Context

Dataset: SIFT-128 (`sift-128-euclidean.hdf5`)

- `train.bin` — 1,000,000 × 128 float32 vectors (~512 MB)
- `test.bin` — 10,000 × 128 float32 query vectors
- `neighbors.bin` — 10,000 × 100 int32 ground-truth neighbor IDs
- `distances.bin` — 10,000 × 100 float32 ground-truth distances

Hardware: AMD Ryzen 7 PRO 6850U @ 2.70 GHz, single core, ~15 GB RAM.

Index parameters: `M=16`, `efConstruction=200`, `ef=50` at query time, `k=10`.

---

## Benchmark Summary

| Version                       | Build time | vec/s (avg) | QPS  | Recall@10 |
| ----------------------------- | ---------- | ----------- | ---- | --------- |
| Baseline (scalar `dist`)      | 3851s      | 260         | 962  | 0.9554    |
| + `-O3 -mavx2 -mfma`          | ~3800s     | ~266        | ~980 | 0.9554    |
| + AVX2 `dist_avx2` intrinsics | 2204s      | 454         | 1474 | 0.9556    |
| + visited set + heap reserve  | 1825s      | 548         | 2098 | 0.9554    |

Recall is stable across all versions — optimisations are all correctness-preserving.

---

## Session Work

### 1. Benchmark Harness (`tests/test_hnsw.cpp`)

**Problem:** The test file only loaded `train.bin` into a standalone `NodeBuffer` and printed the shape. There was no index construction, no querying, and no recall evaluation.

**Problem with wiring:** `HNSW` owns its own internal `NodeBuffer` allocated at construction. `load_matrix()` returned a separate `NodeBuffer`. There was no way to feed binary data directly into the HNSW's internal buffer without an intermediate allocation or a redesign.

**Solution — Option 1 (load_into):** Add a `load_into(path, float* dest, rows, cols)` function to `io.cpp` that streams the binary payload directly into a caller-supplied pointer. Expose a `node_data()` accessor on `HNSW` returning `nodeBuffer.data()`. The test then:

1. Calls `read_header("bin/train.bin")` to get dimensions without loading data
2. Constructs `HNSW(M, efConstruction, n_nodes, dim)` — preallocates everything (~1.4 GB total)
3. Calls `load_into("bin/train.bin", hnsw.node_data(), rows, cols)` — streams 512 MB directly in
4. Loops `addNode(i)` for all 1M vectors with progress + ETA printed every 50K
5. Loads `test.bin` via `load_matrix()` and `neighbors.bin` via `load_matrix_int()`
6. Runs `searchANN(query, k=10, ef=50)` for all 10K queries
7. Computes recall@10 by intersecting results with ground-truth top-10
8. Prints recall@10 and QPS

**New functions added to `io.hpp` / `io.cpp`:**

```cpp
// Peek dimensions without loading data
std::pair<int32_t, int32_t> read_header(const std::string& path);

// Stream float payload into existing buffer; validates dimensions match
void load_into(const std::string& path, float* dest,
               int32_t expected_rows, int32_t expected_cols);

// Load int32 matrix (neighbors.bin); returns flat row-major vector
std::vector<int32_t> load_matrix_int(const std::string& path,
                                      int32_t& out_rows, int32_t& out_cols);
```

A private `open_and_read_header` helper was factored out to avoid header-reading duplication across all three loaders.

---

### 2. Compiler Flags (`CMakeLists.txt`)

Changed `-O2` → `-O3` and added `-mavx2 -mfma`.

```cmake
add_compile_options(
  -Wall
  -Wextra
  -O3
  -mavx2
  -mfma
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-O0>
)
```

**Effect:** `-O3` gave a small improvement (~2%) from more aggressive inlining and loop optimisation. `-mavx2 -mfma` alone did not trigger auto-vectorisation of `dist()` — the compiler could not prove the two `float*` arguments didn't alias, so it stayed scalar. These flags are still required for the explicit AVX2 intrinsics added in the next step.

---

### 3. AVX2 Distance Function (`core/hnsw/hnsw.cpp`)

**Problem:** `dist()` is the innermost hot function — called thousands of times per `addNode` and for every query. The scalar implementation processes 1 float per iteration over 128 dimensions.

```cpp
// original — kept as reference
float HNSW::dist(const float* a, const float* b)
{
    float sum = 0;
    for (size_t i = 0; i < nodeBuffer.dim(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}
```

**Solution:** Added `dist_avx2` alongside the original (not replacing it). Uses AVX2 to process 8 floats per iteration and FMA to fold the multiply-accumulate into one instruction.

```cpp
float HNSW::dist_avx2(const float* __restrict__ a, const float* __restrict__ b)
{
    __m256 acc = _mm256_setzero_ps();

    // 128-dim / 8 floats per register = 16 iterations, no remainder
    for (size_t i = 0; i < 128; i += 8)
    {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc         = _mm256_fmadd_ps(diff, diff, acc);
    }

    // Horizontal sum: reduce 8 lanes to scalar
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum        = _mm_hadd_ps(sum, sum);
    sum        = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
```

**Design notes:**

- `__restrict__` on both pointers eliminates aliasing ambiguity — the compiler can assume `a` and `b` never overlap
- `_mm256_loadu_ps` (unaligned load) used even though `NodeBuffer` rows are 32-byte aligned — safe for both the aligned node buffer rows and the unaligned query `std::vector<float>` in `searchANN`
- Loop count hardcoded to 128 — SIFT-specific, avoids a branch/load on every call. `dist()` retains the generic `nodeBuffer.dim()` loop for non-SIFT use
- Horizontal sum uses the standard `hadd` idiom: fold high/low 128-bit halves together, two `hadd` passes reduce 4 lanes to 1

All 6 call sites in `hnsw.cpp` were switched to `dist_avx2`: `selectNeighbors`, `wireNeighbors`, `greedyDescend` (×2), `beamSearch` (×2).

**Result:** Build time 3851s → 2204s (1.75×), QPS 962 → 1474 (1.53×).

The improvement is bounded by Amdahl's law — `dist()` is hot but `beamSearch` also spends significant time in heap operations and visited-set lookups, which were unchanged at this point.

---

### 4. Visited Set + Priority Queue Optimisation (`core/hnsw/hnsw.cpp`, `hnsw.hpp`)

#### Visited Set

**Problem:** `beamSearch` constructed a fresh `unordered_set<int>` on every call. This means:

- Heap allocation of hash table buckets from scratch each call
- Hash computation + potential collision chaining on every `insert`/`find`
- Full bucket-walk and deallocation at the end of each call

During build with `efConstruction=200`, `beamSearch` is called once per layer per node insertion — millions of times total.

**Solution:** Flat array with a generation counter, allocated once at HNSW construction and reused forever.

Added to `hnsw.hpp`:

```cpp
std::vector<uint32_t> visited_table_; // size = n_nodes
uint32_t              visited_gen_ = 0;
```

Initialised in constructor:

```cpp
visited_table_(n_nodes, 0),
visited_gen_(0)
```

Used in `beamSearch`:

```cpp
++visited_gen_;
// wraparound guard: after 4 billion calls, reset table
if (visited_gen_ == 0) {
    ++visited_gen_;
    fill(visited_table_.begin(), visited_table_.end(), 0);
}

auto mark_visited = [&](int id) { visited_table_[id] = visited_gen_; };
auto is_visited   = [&](int id) { return visited_table_[id] == visited_gen_; };
```

"Clearing" between calls is a single integer increment — O(1), zero memory writes. A node is considered unvisited if its table slot holds any generation older than the current one.

#### Priority Queue

**Problem:** Both `priority_queue` instances in `beamSearch` started empty and grew dynamically. The backing `std::vector` reallocated as elements were pushed. `std::priority_queue` does not expose `reserve()`.

**Solution:** Replace both `priority_queue` instances with `std::vector` + `std::push_heap`/`pop_heap`/`sort_heap` from `<algorithm>`. This exposes the backing vector so it can be `reserve()`d upfront.

```cpp
vector<pair<float, int>> candidates, results;
candidates.reserve(ef * 2 * M);  // e.g. 200 * 32 = 6400 during build
results.reserve(ef + 1);          // one extra slot: push before pop
```

- `candidates` is a min-heap: `greater<pair<float,int>>{}` comparator passed to all heap operations
- `results` is a max-heap: default `less<>` comparator (largest distance at `front()`)
- `sort_heap` at the end replaces the old drain-into-output loop — sorts in-place ascending, returning results closest-first as callers expect
- `results.front()` gives the worst (largest distance) element currently in the bounded result set — used correctly in the pruning condition

**Result:** Build time 2204s → 1825s (1.21×), QPS 1474 → 2098 (1.42×). Build rate in the second half stabilised rather than decaying — the visited set fix disproportionately helped later insertions when the graph is denser and `beamSearch` explores more nodes per call.

---

## What's Left

Remaining single-core improvements (diminishing returns):

- **`selectNeighbors` sort** — still a full `std::sort` on candidates each call; a partial sort would suffice
- **`searchANN` query vector copy** — harness currently copies each query row into a `std::vector<float>`; passing a raw pointer would avoid 10K small allocations

Bigger levers:

- **Multithreaded queries** — 10K queries are independent reads on a built index. `std::for_each` with `std::execution::par_unseq` or a manual thread pool would give near-linear scaling. Expected: 8× on 8 cores → ~16K QPS
- **Multithreaded build** — harder; `addNode` mutates `neighborStore`. Requires per-node mutex array (hnswlib approach) or lock-free edge insertion
- **Product Quantisation (PQ)** — compress vectors to a small code; approximate `dist()` in compressed space. Massive reduction in distance computation cost at the expense of some recall. Larger implementation effort

---

## Build & Run

From the MSYS2 ucrt64 shell, project root:

```bash
# First time or after CMakeLists changes
cmake -S . -B build -G Ninja

# Every subsequent build + run
cmake --build build && ./build/tests/test_hnsw
```

Binary must be run from the project root so relative paths bin/train.bin etc. resolve correctly.
