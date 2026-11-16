# METIS (bundled)

METIS 5.2.1 (George Karypis and the University of Minnesota, release tag v5.2.1, commit f5ae915a84d3bbf1508b529af90292dd7085b9ec of
https://github.com/KarypisLab/METIS) with GKlib (commit 8bd6bad750b2b0d90800c632cf18e8ee93ad72d7 of https://github.com/KarypisLab/GKlib,
the one packagers pair with METIS 5.2.1), both under the Apache License 2.0 (`LICENSE`, `GKlib/LICENSE.txt`). Only the library's sources
are kept: `libmetis/`, `include/metis.h` and GKlib's top-level sources; no programs, tests or build files.

BRISK uses METIS's nested dissection (`METIS_NodeND`) as a fill-reducing order of its sparse Cholesky factorizations, next to AMD (`amd/`)
and its own dissection (`nd.c`). It is built with 32-bit indices and reals (`-DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=32`) by the Makefile
(`METIS ?= 1`; `make METIS=0` builds without it). The sources are unchanged; any change made for BRISK is listed here.

Changes for BRISK (13 November 2026, the SOCP developer):
- `GKlib/random.c`: the random generator's state (`mt`, `mti`) per thread (`__thread`).
- `libmetis/ometis.c`: `BRISK_METIS_SetThreads(nt)`. With nt >= 1, `METIS_NodeND` orders the two halves of each separator in parallel
  (OpenMP tasks, when both have more than 5,000 vertices; each task with its own copy of the control structure and its workspace),
  reseeds the random numbers at each step of the recursion from its own subgraph (its last position in the order, its vertices and
  edges), so that the order is the same at any number of threads, and does not set up GKlib's memory tracking (a graph may be freed by
  another thread than the one that allocated it). Without the call (nt = 0) METIS behaves as released.
- `Makefile`: `-DUSE_GKRAND`, so that every random number comes from GKlib's generator (by default `gk_randint32` calls the C
  library's `rand()`, shared by all threads and not seeded by METIS); and each task's copy of the control structure has its own
  arrays that the ordering writes (`maxvwgt`, `ubfactors`, `tpwgts`, `pijbm`).
- Measured (one thread against four, the same order bit for bit): db-joint-soerensen's normal equations (SOCP, 791,903 vertices)
  7.34e8 flops of the factor, 7.4 s -> 2.5 s; rail02's (LP) 1.17 -> 0.53 s; small graphs unchanged (the halves below 5,000 vertices
  run as one task).
