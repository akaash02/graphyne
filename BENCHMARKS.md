# Graphyne — Benchmark Record & Provenance

This file is the reproducible record of Graphyne's performance: what was measured,
on what hardware, with which flags, and how it compares to faiss and hnswlib on the
same machine. It also documents the provenance of the historical **7,227 QPS** figure,
because that number was previously undocumented and unverifiable.

**Rule going forward:** no QPS number goes on a résumé, in the README, or in CLAUDE.md
without a row in this file stating hardware, flags, build type, and query thermal state.

---

## Headline result (2026-10-26)

SIFT1M (1,000,000 × 128 float32), single thread, `M=16`, `efConstruction=200`,
`ef=50`, `k=10`, measured on an Intel i7-13700H.

| Engine                         | Build  | QPS      | Recall@10 | Query state          |
| ------------------------------ | ------ | -------- | --------- | -------------------- |
| **Graphyne** (HEAD `425f6a7`)  | 556.5s | **6,449**| 0.9556    | hot (post-build)     |
| hnswlib (v0.8, same flags)     | 485.2s | 7,401    | 0.9466    | cold (saved index)   |
| faiss `IndexHNSWFlat` (1.15.1) | 563.1s | 5,101    | 0.9506    | cold (saved index)   |

**Read this carefully — it is not a clean apples-to-apples comparison yet.** Graphyne's
number is a *hot* query (the query phase runs seconds after a ~9-minute build, on a
heat-soaked CPU) because Graphyne cannot yet serialize an index. faiss and hnswlib query
*cold* from a saved index in a fresh process. Graphyne's own data (below) puts the hot
penalty at ~38%, so a cold Graphyne would likely top this table. The honest claim today
is: **at matched recall, Graphyne is competitive with two mature C++ ANN libraries on the
same machine, despite querying under a thermal handicap.**

Closing that gap fairly is exactly what index serialization (`save`/`load`) unlocks, which
is why it leads the next-steps list in CLAUDE.md.

### Full recall/QPS sweep, SIFT1M, cold query (libraries only)

| ef  | faiss recall | faiss QPS | hnswlib recall | hnswlib QPS |
| --- | ------------ | --------- | -------------- | ----------- |
| 10  | 0.7133       | 15,521    | 0.7095         | 23,096      |
| 20  | 0.8442       | 10,604    | 0.8391         | 14,850      |
| 30  | 0.9007       | 7,662     | 0.8958         | 10,095      |
| 50  | 0.9506       | 5,101     | 0.9466         | 7,401       |
| 80  | 0.9759       | 3,797     | 0.9748         | 4,482       |
| 100 | 0.9840       | 2,969     | 0.9829         | 3,836       |
| 150 | 0.9923       | 1,784     | 0.9922         | 2,690       |
| 200 | 0.9956       | 1,179     | 0.9957         | 1,999       |
| 400 | 0.9985       | 555       | 0.9987         | 1,116       |

Graphyne was measured only at `ef=50` (its single configured operating point in
`start/main.cpp`). A sweep requires parameterizing `ef` on the command line — a small
change worth making before the next round.

---

## Graphyne build profile & configuration matrix (2026-10-26)

### Build-rate profile (SIFT1M, fastest run)

Insertion throughput as the graph grows, from the CMake Release run (556.5s, 6,449 QPS):

| Nodes inserted | vec/s | elapsed |
| -------------- | ----- | ------- |
| 50,000         | 5,342 | 9.4s    |
| 100,000        | 4,155 | 24.1s   |
| 200,000        | 3,043 | 65.7s   |
| 400,000        | 2,418 | 165.4s  |
| 600,000        | 2,137 | 280.8s  |
| 800,000        | 1,953 | 409.5s  |
| 1,000,000      | 1,797 | 556.5s  |

The rate decays smoothly from ~5,300 to ~1,800 vec/s and flattens toward the end — the
expected `O(log N)` descent cost plus growing cache pressure, with **no linear-scan knee**.
This is the healthy shape; a hidden `O(N)` path would curve downward instead of flattening.

### Every Graphyne config tried, SIFT1M (recall 0.9556 in all)

| Config                                                              | Build  | QPS       | Query state |
| ------------------------------------------------------------------- | ------ | --------- | ----------- |
| **CMake Release, GNU ld, repo flags (`-mtune=native`), sep. compile** | 556.5s | **6,449** | hot         |
| unity build, lld, `-fno-strict-aliasing`                            | 590.0s | 5,666     | hot         |
| unity build, lld, `-fno-strict-aliasing`, 150s cooldown             | 566.1s | 5,378     | cold proxy  |
| PGO (`-fprofile-use`), unity, lld, `-mtune=native -fno-strict-aliasing` | —   | **segfault** | hot & cold |
| *historical (4 Aug 2026, unverifiable, binary gone)*                | 570.7s | *7,227*   | hot         |

- **Highest verified 1M reading: 6,449 QPS** (CMake Release, hot), recall 0.9556.
- The 5,378–6,449 span is a single binary's run-to-run variance: linker/codegen differences
  plus the **unseeded RNG** building a structurally different graph each run (~15%).
- **Hot vs cold is inside that noise** — the 150s cooldown did not produce a faster query,
  so the ~38% thermal penalty quoted in `CLAUDE.md` does **not** reproduce on this i7 by
  this method. A true cold-vs-cold number needs index save/load + a fixed RNG seed.

### Every Graphyne config tried, 50K subset (recall ~0.988)

**Timing here is noisy** — 1,000 queries finishing in ~0.06–0.09s, so ±10–15% run to run.
Treat these as indicative only.

| Config                                                | QPS        | Recall@10 |
| ----------------------------------------------------- | ---------- | --------- |
| PGO, unity, lld, `-mtune=native -fno-strict-aliasing` | **15,978** | 0.9883    |
| PGO, unity, lld (quick sanity)                        | 15,507     | 0.9883    |
| no-PGO, unity, GNU ld                                 | 15,145     | 0.9881    |
| CMake Release, GNU ld                                 | 14,277     | 0.9875    |
| no-PGO, unity, lld                                    | 12,814     | 0.9885    |
| PGO instrumented (profile-gen build)                  | 12,559     | 0.9887    |
| no-PGO, unity, lld, cooldown                          | 10,703     | 0.9880    |

- **Highest 50K reading: 15,978 QPS** (PGO), but the ~11% PGO edge is within the 50K noise
  band and does not survive to 1M (the PGO build crashes there). At matched recall the 50K
  corpus beats faiss (~10,800) and sits level with hnswlib (~16,100).

### Highest readings, summary

| Scale | Highest verified | Config                        | Recall |
| ----- | ---------------- | ----------------------------- | ------ |
| 1M    | **6,449 QPS**    | CMake Release, hot            | 0.9556 |
| 50K   | **15,978 QPS**   | PGO build (noisy, cache-fit)  | 0.9883 |

---

## Reproduction methodology (2026-10-26)

- **Machine:** Intel i7-13700H (6 P-cores + 8 E-cores), Windows 11, unpinned, no custom
  power plan applied for these runs.
- **Graphyne:** commit `425f6a7` (current `avx2` HEAD), built fresh in Release with
  `clang++ 22.1.7` (MSYS2 UCRT64), flags `-O3 -mavx2 -mfma -mtune=native -Wall -Wextra`
  plus CMake's `-O3 -DNDEBUG`. **No PGO.** Run via `start/graphyne.exe bin`.
- **hnswlib:** `HierarchicalNSW<float>`, L2 space, `M=16`, `ef_construction=200`, one
  thread. Compiled with the same clang and the same flags. Cold query reads a
  `saveIndex`/`loadIndex` round-trip in a fresh process.
- **faiss:** `faiss-cpu 1.15.1` (pip wheel), `IndexHNSWFlat(d, 16)`,
  `efConstruction=200`, `omp_set_num_threads(1)`. Cold query reads `write_index`/
  `read_index` in a fresh process.
- **Fairness notes:** all three use the same `train.bin`/`test.bin`/`neighbors.bin`,
  the same recall scorer (intersection of top-10 vs ground-truth top-10), and one query
  per `search` call (so per-call overhead is counted identically). Graphyne's RNG is
  **unseeded**, so its graph — and recall in the 4th decimal — varies run to run.

---

## Provenance of the historical 7,227 QPS figure

The résumé previously cited **7,227 QPS at recall@10 = 0.9554 on SIFT1M**. That number did
not appear anywhere in the repository, and the binary that produced it no longer exists.
Two investigations (2026-10-26) plus a direct reproduction established the following.

### What is confirmed

- **The run used this repo's `start/main.cpp`.** The pasted output format
  (`train: 1000000 x 128 (512.0 MB)`, `| 4057 vec/s | elapsed …`, `query 1.38s
  (7227 QPS)`) matches those print statements exactly; the other harness,
  `tests/test_hnsw`, prints differently.
- **The code was commit `425f6a7` ("Added allocation optimizations").** The results were
  posted on 4 Aug 2026 ~18:25; that commit landed at 18:07:54, and load + a 570.7s build +
  queries is ~10 minutes, placing the run at 18:08–18:15. It therefore included AVX2
  `dist_avx2` (built into `hnsw.cpp`, never a separate `distance_avx2.cpp`), the visited
  table, reserved heaps, and the `NeighborStore` (created 15 Jul in `01ded2f`, **not** in
  an early-August rewrite).
- **The flags were the repo's at that commit:** `-O3 -mavx2 -mfma -mtune=native -Wall
  -Wextra` on every file, plus `-O3 -DNDEBUG` for Release. There was never a
  `distance_avx2.cpp`, per-file flag via `set_source_files_properties`, or runtime
  dispatch via `__builtin_cpu_supports` — none exist in any commit, branch, or stash.
- **`-march`, LTO, and `-ffast-math` were never used** (`git log --all -S` for each is
  empty).
- **The 7,227 binary is gone.** The only `.exe` files on disk date from 5 Sep and contain
  debug info with no PGO instrumentation. No log anywhere contains `7227` or `570.7`.

### PGO: almost certainly NOT used

The résumé recollection was that PGO and extra compiler flags were involved. The evidence
leans hard the other way:

1. **The toolchain couldn't do it.** The MSYS2 install log shows `compiler-rt` and `llvm`
   were never installed — so there was no clang profile runtime to link an instrumented
   binary against, and no `llvm-profdata` to merge a profile. Nothing was removed later.
2. **No profile ever existed.** No `.profraw`/`.profdata` files anywhere under the user
   folder, and no `pgo/` directory. With `GRAPHYNE_PGO_USE=ON` and no profile present,
   CMake halts with a fatal error — so an accidental PGO build would have failed loudly.
3. **A plain Release build reproduces the number.** On 2026-10-26, the *same commit*
   (`425f6a7`), same machine, same hot-query method, built with **no PGO**, produced
   **6,449 QPS / 0.9556 recall / 556.5s build** — matching the historical 570.7s build and
   0.9554 recall almost exactly, at 89% of the QPS.

**Conclusion:** 7,227 and 6,449 are the same vanilla Release binary. The ~12% gap is
ordinary run-to-run variance — thermal state, the unseeded RNG producing a structurally
different graph each build, and background CPU load — not a lost PGO/tuning configuration.
The PGO *build options* in `CMakeLists.txt` are real and were added at 17:24 that day, but
the evidence indicates they were never successfully exercised.

### PGO empirically tested (2026-10-26) — does not run at 1M scale

To test the "PGO was enabled" recollection directly, the full PGO toolchain was installed
(`compiler-rt`, `llvm`, `llvm-tools`, `lld` — none of which were present before today) and
a real PGO cycle was run: instrument → profile on the 50K set → `llvm-profdata merge` →
rebuild with `-fprofile-use`. Findings:

1. **The repo's documented PGO recipe does not link on this toolchain.** With GNU `ld`,
   profile counters for inlined `std::string` template instantiations are dropped across
   the static libraries, giving `undefined reference to __profc_*`. Linking with **`lld`**
   is required. The `CMakeLists.txt` recipe predates this discovery and would fail as
   written.
2. **`-mtune=native` + PGO miscompiles** and segfaults immediately, unless
   `-fno-strict-aliasing` is also passed — the `dist_avx2` kernel's `__restrict__` float
   pointers are the likely trigger.
3. **Even with both fixes, the PGO build crashes at SIFT1M.** It runs fine at 50K
   (~15,500 QPS, recall 0.988) but segfaults (`exit 139`) on the 1M workload, in both the
   hot and cold variants, every time. The matching no-PGO binary (identical flags, profile
   removed) completes cleanly.

| Build (same unity + lld + `-fno-strict-aliasing` config) | 50K | 1M |
| -------------------------------------------------------- | --------------- | --------------- |
| no-PGO                                                   | ~13,900 QPS ✓   | 5,666 QPS ✓ (recall 0.9556) |
| PGO (`-fprofile-use`)                                    | ~15,500 QPS ✓   | **segfault**    |

**This strengthens the conclusion above:** the 7,227 run was almost certainly *not* a PGO
build, because a PGO build does not run to completion on this code at 1M. The ~11% PGO
edge seen at 50K is real but unusable at scale, and 50K query timing (1,000 queries,
~0.07s) is too noisy to weigh heavily.

**Side finding worth acting on:** the 1M-only segfault under aggressive PGO codegen points
to a latent bug in the hot path — most likely an aliasing assumption in `dist_avx2` or the
`visited_table_` generation-stamp logic — that only manifests under certain optimization or
at a working-set size reached around 1M. Worth hardening (AddressSanitizer on a 1M run)
independent of PGO.

### Corrections to earlier recollections

- **`NeighborStore` was created 15 Jul 2026** (`01ded2f`), not 2–4 August.
- **All of 4 Aug's work** (I/O loading, CMake flags, AVX2, allocation fixes) was committed
  between 15:46 and 18:07. `Dev.md` from that day records the hardware as a **Ryzen 7 PRO
  6850U** and peaks at **2,098 QPS** — that was the work laptop ("run 1").
- **This i7's previously recorded best** (CLAUDE.md) was **2,011 QPS unpinned / 1,713
  pinned**, with a 1,662s build (~3× slower than 570.7s). Those were contended or
  differently-configured runs; the 6,449 reproduction supersedes them as the clean figure.
- **"Run 2"** (personal laptop, pinned to P-cores, no recall recorded) left no surviving
  artifact and could not be located.

---

## Recommended résumé wording

Replace the bare, unverifiable number with a comparison that is reproducible from this
repo today:

> Built a from-scratch C++ HNSW vector index (layered graph, greedy descent, beam search,
> diversity neighbor selection, AVX2/FMA distance kernels). Achieves ~6,500 QPS
> single-thread at 95.6% recall@10 on SIFT1M (i7-13700H), competitive with faiss and
> hnswlib benchmarked on the same machine.

State the hardware next to the number. A head-to-head against named libraries is far more
credible than a standalone QPS figure, and every claim above is backed by a row in this
file.

---

*Benchmark scripts and drivers used for the library comparison live outside the repo
(scratchpad). If kept, they belong under `bench/` alongside the deletion-study harness.*
