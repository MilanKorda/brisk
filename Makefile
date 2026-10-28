# BRISK build.
#   make                      -> system BLAS/LAPACK (Linux: -llapack -lblas; macOS: Accelerate)
#   make BLAS=scipy           -> SciPy's bundled OpenBLAS (symbols prefixed scipy_)
#   make BLASLIB="-lopenblas" -> custom library
# Threads (OpenMP):
#   make OMP=bundled          -> BRISK's own OpenMP runtime (omp/brisk_omp.c): needs clang, nothing
#                                to install (default on macOS: Apple's clang has no OpenMP library)
#   make OMP=1                -> the compiler's runtime (gcc: libgomp, default on Linux; clang: libomp)
#   make OMP=0                -> no threads of BRISK's own (BLAS threads only)
# Shared library (Julia, ctypes):
#   make libbrisk             -> libbrisk.so / libbrisk.dylib (capi.c; same options as above)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
CC      ?= cc
ARCHFLAG := -march=native
ifeq ($(UNAME_S),Darwin)
  OMP ?= bundled
  ifneq ($(filter arm64,$(UNAME_M)),)
    ARCHFLAG := -mcpu=native
  endif
  # 5.15: the first release of Apple's new linker (Xcode 15.0, `ld -v`: dyld-1015) changed an
  # instruction that carries no relocation (sedumi_solve: `fmov d3, x15` linked as an undefined
  # word; SIGILL in the Python tests on an M3); the classic linker links the same objects right.
  # Every link also goes through tools/link_check.sh, which looks for such words in the result
  # that the input objects do not have (compilers put data in code: jump tables on x86-64), with
  # any linker, and links again with the classic linker if it finds one.
  ifneq ($(findstring dyld-1015,$(shell ld -v 2>&1)),)
    LINKFIX := -Wl,-ld_classic
  endif
else
  OMP ?= 1
endif
OMPSRC  :=
OMPLD   :=
ifeq ($(OMP),1)
  OMPFLAG := -fopenmp
else ifeq ($(OMP),bundled)
  # the pragmas are compiled (the runtime interface is clang's, __kmpc_*), the runtime is ours;
  # -Xpreprocessor keeps the driver from linking a runtime library
  OMPFLAG := -Xpreprocessor -fopenmp -Iomp
  OMPSRC  := omp/brisk_omp.c
  OMPLD   := -lpthread
else
  # without OpenMP (OMP=0; ISSUES 713): the runtime's queries declared for every file and defined for one thread
  OMPFLAG := -Wno-unknown-pragmas -include omp/ompstub.h
  OMPSRC  := omp/ompstub.c
endif
# -ffp-contract=off (5.18): each expression is rounded as written; with gcc's default (fast) the fusion of a*b + c into
# an FMA follows the inlining, so an edit of one function moved the last bits of others (the ill-posed files of the
# regression sets moved with every switch of a change off); the time per iteration is the same within the noise
CFLAGS  ?= -O3 $(ARCHFLAG) -funroll-loops -ffp-contract=off $(OMPFLAG) -std=gnu11 -Wall -Wextra -Wno-unused-parameter
SRC      = problem.c weylpd.c presolve.c postsolve.c dictroute.c chordal.c freeelim.c dualize.c momaff.c blkscale.c fom.c mfipm.c lralm.c bm.c bmip.c symred.c symalg.c sparsechol.c dualscale.c ddend.c crossover.c solver.c bound.c boundcert.c main.c $(OMPSRC) hpmp.c hpsolve.c socp.c sedumi.c lpio.c lpsolve.c nd.c simplex.c simplex_lu.c simplex_pre.c qp.c qpio.c qpipm.c qpas.c qcqp.c qpdbl.c socpmodel.c
BLAS    ?= system

ifeq ($(BLAS),scipy)
  SCIPY_LIBDIR := $(shell python3 -c "import scipy,os;print(os.path.join(os.path.dirname(scipy.__file__)+'.libs'))")
  SCIPY_OBLAS  := $(firstword $(wildcard $(SCIPY_LIBDIR)/libscipy_openblas*.so))
  ifeq ($(SCIPY_OBLAS),)
    $(error BLAS=scipy: no SciPy OpenBLAS found (python3 = $(shell command -v python3)); install scipy, or with pip use --no-build-isolation)
  endif
  DEFS     = -DBLAS_PREFIX=scipy_
  BLASLIB ?= $(SCIPY_OBLAS) -Wl,-rpath,$(SCIPY_LIBDIR)
else
  DEFS     =
  ifeq ($(UNAME_S),Darwin)
    BLASLIB ?= -framework Accelerate
  else
    BLASLIB ?= -llapack -lblas
  endif
endif

OBJ      = $(SRC:.c=.o)

# AMD (the fill-reducing ordering) is bundled: amd/ (AMD 2.4.6, Timothy A. Davis, Patrick R.
# Amestoy and Iain S. Duff, used under the BSD 3-clause license, amd/LICENSE_AMD.txt);
# no SuiteSparse installation is needed
AMDSRC   = $(wildcard amd/*.c)
AMDOBJ   = $(AMDSRC:.c=.o)
DEFS    += -DHAVE_AMD -DNDEBUG
# optional: CHOLMOD's nested dissection ordering for large Schur patterns (static; make CHOLMOD=0 to build without)
CHOLMOD ?= 0
ifeq ($(CHOLMOD),1)
  DEFS += -DHAVE_CHOLMOD
  CHOLMODLIB = -l:libcholmod.a -l:libcamd.a -l:libcolamd.a -l:libccolamd.a
endif
# optional: METIS nested dissection for large Schur patterns (make METIS=1; off: AMD won on every AC-OPF pattern)
METIS ?= 0
ifeq ($(METIS),1)
  DEFS += -DHAVE_METIS
  METISLIB = -lmetis
endif
brisk: $(OBJ) $(AMDOBJ)
	sh tools/link_check.sh $@ -- $(CC) $(CFLAGS) $(DEFS) $(LINKFIX) -o $@ $(OBJ) $(AMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm

# the QP solver's command line (the SOCP developer, 22 October 2026): qp.h, qp.c, qpio.c, qpipm.c, qpas.c, qpmain.c
QPOBJ    = qp.o qpio.o qpipm.o qpas.o qcqp.o qpdbl.o
brisk_qp: qpmain.o $(QPOBJ) sparsechol.o nd.o $(OMPSRC:.c=.o) $(AMDOBJ)
	$(CC) $(CFLAGS) $(DEFS) $(LINKFIX) -o $@ qpmain.o $(QPOBJ) sparsechol.o nd.o $(OMPSRC:.c=.o) $(AMDOBJ) $(BLASLIB) $(OMPLD) -lm
qp.o qpio.o qpipm.o qpas.o qpmain.o: qp.h
lpio.o qpio.o: fastnum.h
tests/qp_test: tests/qp_test.c $(QPOBJ) sparsechol.o nd.o $(OMPSRC:.c=.o) $(AMDOBJ) qp.h
	$(CC) $(CFLAGS) $(DEFS) -o $@ tests/qp_test.c $(QPOBJ) sparsechol.o nd.o $(OMPSRC:.c=.o) $(AMDOBJ) $(BLASLIB) $(OMPLD) -lm
qptest: tests/qp_test
	./tests/qp_test

amd/%.o: amd/%.c
	$(CC) $(CFLAGS) -DNDEBUG -Wno-all -Wno-extra -c -o $@ $<

# (23 October 2026) every object on every header of the tree: a changed structure in qp.h left the library's capi.o
# with the old size of QpRes, and brisk_qp_resolve wrote past it (stack smashing in the QP model's test)
HDRS := $(wildcard *.h)
%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $(DEFS) -c -o $@ $<

# 4.39: the rigorous certifier changes the rounding mode
boundcert.o libobj/boundcert.o: CFLAGS += -frounding-math -fno-fast-math
hpsolve.o libobj/hpsolve.o: hp.h hpipm.inc hpfom.inc hplr.inc hppre.inc hpcert.inc hpred.inc
hpmp.o libobj/hpmp.o: hp.h
socp.o libobj/socp.o: socp.h blas_one.h
sedumi.o libobj/sedumi.o main.o libobj/main.o: socp.h sedumi.h
# (22 October 2026) the LP method's options struct is shared by these: a field added to it in lpsolve.h left main.o with
# the old layout (memset of the smaller struct, the new field from the stack)
lpsolve.o libobj/lpsolve.o main.o libobj/main.o sedumi.o libobj/sedumi.o simplex.o libobj/simplex.o: lpsolve.h
# (22 October 2026) the simplex method's internal structs (simplex_int.h: the factorization's SxLU inside Sx) and its
# headers: a field added to SxLU left simplex.o with the old layout ("corrupted size" at the first solve)
simplex.o libobj/simplex.o simplex_lu.o libobj/simplex_lu.o: simplex_int.h simplex.h
simplex.o libobj/simplex.o simplex_pre.o libobj/simplex_pre.o: simplex_pre.h
simplex.o libobj/simplex.o: sxmodel.h socp.h

# the bundled OpenMP runtime against serial results, 1..8 threads (needs clang)
omptest: omp/test_omp.c omp/brisk_omp.c omp/omp.h
	clang -O2 -std=gnu11 -Xpreprocessor -fopenmp -Iomp -o omp/test_omp omp/test_omp.c omp/brisk_omp.c -lpthread
	./omp/test_omp 20

# 4.37: the shared library for foreign-function callers (Julia: julia/Brisk.jl; capi.c has
# the interface). Separate position-independent objects in libobj/ (the command line's
# objects are not -fPIC); only the brisk_* symbols are exported, so nothing of BRISK's (AMD,
# the bundled OpenMP runtime, helpers) can clash with the host's libraries.
ifeq ($(UNAME_S),Darwin)
  LIBBRISK   := libbrisk.dylib
  LIBLDFLAGS := -dynamiclib -install_name @rpath/libbrisk.dylib -Wl,-exported_symbols_list,libbrisk.exp
else
  LIBBRISK   := libbrisk.so
  LIBLDFLAGS := -shared -Wl,--version-script=libbrisk.map -Wl,-Bsymbolic -Wl,--no-undefined
endif
LIBOBJ    = $(patsubst %.c,libobj/%.o,$(SRC) capi.c)
LIBAMDOBJ = $(patsubst %.c,libobj/%.o,$(AMDSRC))

libbrisk: $(LIBBRISK)

$(LIBBRISK): $(LIBOBJ) $(LIBAMDOBJ) libbrisk.map libbrisk.exp
	sh tools/link_check.sh $@ -- $(CC) $(CFLAGS) -fPIC $(DEFS) -DBRISK_LIBRARY $(LIBLDFLAGS) $(LINKFIX) -o $@ $(LIBOBJ) $(LIBAMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm

libobj/amd/%.o: amd/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC -DNDEBUG -Wno-all -Wno-extra -c -o $@ $<

libobj/%.o: %.c $(HDRS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC $(DEFS) -DBRISK_LIBRARY -c -o $@ $<

# the C interface on an SDPA file and in memory against the command line (needs libbrisk)
capitest: $(LIBBRISK) tools/capi_test.c
	$(CC) -O1 -std=gnu11 -Wall -I. -o tools/capi_test tools/capi_test.c -L. -lbrisk -Wl,-rpath,$(CURDIR) -lm
	./tools/capi_test

# the library under AddressSanitizer: the C API test and the bound-mode handle test (memory errors and leaks)
asantest:
	tools/asan_test.sh

# reproducibility (CLAUDE.md, ISSUES 708): the same bits at 1, 2 and all threads, with thread changes, and from the C
# library with problems of the same dimensions alternated; reprotest-mkl: the same under MKL (tools/setup_env.sh --mkl)
reprotest: brisk $(LIBBRISK)
	tools/repro_suite.sh
reprotest-mkl: brisk $(LIBBRISK)
	tools/repro_suite.sh --mkl

# the MATLAB build's integer-widening BLAS layer (blas64.c) against an ILP64 OpenBLAS (libopenblas64-dev)
blas64test: brisk
	tools/blas64_test.sh

# bound OpenMP threads on the command line (Linux): one processor per thread
bindtest: brisk
	tools/bind_test.sh ./brisk

# the reductions of the high-precision solver on a problem with a known value (theta of an odd cycle)
hpredtest: brisk
	tools/hpred_test.sh ./brisk

# the restoration of the primal feasibility at the end of a solve, and the options of the first-method rule
restoretest: brisk
	tools/restore_test.sh ./brisk

# the advice at the end of the log (more accuracy, a guaranteed bound) on the command line
advicetest: brisk
	tools/advice_test.sh ./brisk

# every method keeps the time limit (a generated problem, -timelimit 3 on each method; ISSUES 274)
timetest: brisk
	tools/timelimit_test.sh ./brisk

# doubly nonnegative programs: the bounds on X's entries as a cone in the first-order engine, the hand-over, the options (ISSUES 288)
dnntest: brisk
	tools/dnn_test.sh ./brisk

# the dual after a facial reduction whose removed duals cannot be rebuilt: the default (no re-solve, the message), -nofr, -frfallback 4, BRISK_FRFIX=1
frtest: brisk
	tools/fr_test.sh ./brisk

pytest: $(LIBBRISK)
	cd python && python3 -m pytest -q tests

clean:
	rm -f brisk $(OBJ) $(AMDOBJ) omp/*.o omp/test_omp libbrisk.so libbrisk.dylib tools/capi_test
	rm -rf libobj python/build python/*.egg-info python/brisk/__pycache__ python/tests/__pycache__ python/.pytest_cache

# the simplex method's presolve: an optimal point of the problem as read restricted to the presolved problem (feasible and
# optimal there?), tools/sxpre_restrict_check.c
sxprecheck: $(LIBOBJ) $(LIBAMDOBJ) tools/sxpre_restrict_check.c
	$(CC) $(CFLAGS) $(DEFS) -I. -o tools/sxpre_restrict_check tools/sxpre_restrict_check.c $(LIBOBJ) $(LIBAMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lpthread -lm

# the re-solve interface for a mixed-integer solver (lpmodel.h, internal): dives against solves from scratch, cuts from
# the tableau, the basis, trials, the cutoff, threads (tools/lpmodel_test.c), and the example branch and bound
# (tools/lpmodel_bb.c) on the small mixed-integer problems of benchmarks/mip_small; linked with the library's objects
LPMODEL_FILES = $(addprefix benchmarks/mip_small/,flugpl.mps gr4x6.mps gt2.mps khb05250.mps p0201.mps dcmulti.mps fiber.mps)
lpmodeltest: $(LIBOBJ) $(LIBAMDOBJ) lpmodel.c lpmodel.h sxmodel.h tools/lpmodel_test.c tools/lpmodel_bb.c
	$(CC) $(CFLAGS) $(DEFS) -I. -o tools/lpmodel_test tools/lpmodel_test.c lpmodel.c $(LIBOBJ) $(LIBAMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lpthread -lm
	$(CC) $(CFLAGS) $(DEFS) -I. -o tools/lpmodel_bb tools/lpmodel_bb.c lpmodel.c $(LIBOBJ) $(LIBAMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lpthread -lm
	./tools/lpmodel_test $(LPMODEL_FILES)
	./tools/lpmodel_bb $(addprefix benchmarks/mip_small/,flugpl.mps gr4x6.mps khb05250.mps)
	./tools/lpmodel_bb -weights 1 $(addprefix benchmarks/mip_small/,gr4x6.mps khb05250.mps)
	./tools/lpmodel_bb -weights 2 $(addprefix benchmarks/mip_small/,gr4x6.mps)

# the same under AddressSanitizer (memory errors and leaks)
lpmodelasan:
	tools/lpmodel_asan.sh

# (22 October 2026, the SOCP developer) warm starts of the SOCP solver in a branch and bound over CBLIB's mixed-integer SOCPs
tests/main_nomain.o: main.c brisk.h socp.h sedumi.h lpio.h lpsolve.h qp.h
	$(CC) $(CFLAGS) $(DEFS) -DBRISK_NO_MAIN -c -o $@ main.c
tests/misocp_model: tests/misocp_model.c tests/main_nomain.o $(filter-out main.o,$(OBJ)) $(AMDOBJ) socp.h socpmodel.h sedumi.h
	$(CC) $(CFLAGS) $(DEFS) -o $@ tests/misocp_model.c tests/main_nomain.o $(filter-out main.o,$(OBJ)) $(AMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm
tests/misocp_bb: tests/misocp_bb.c tests/main_nomain.o $(filter-out main.o,$(OBJ)) $(AMDOBJ) socp.h sedumi.h lpio.h
	$(CC) $(CFLAGS) $(DEFS) -o $@ tests/misocp_bb.c tests/main_nomain.o $(filter-out main.o,$(OBJ)) $(AMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm

tests/reader_test: tests/reader_test.c tests/main_nomain.o $(filter-out main.o,$(OBJ)) $(AMDOBJ) lpio.h sedumi.h qp.h
	$(CC) $(CFLAGS) $(DEFS) -o $@ tests/reader_test.c tests/main_nomain.o $(filter-out main.o,$(OBJ)) $(AMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm
readertest: tests/reader_test
	./tests/reader_test

bench/qp/small/qp_rep: bench/qp/small/qp_rep.c $(QPOBJ) sparsechol.o nd.o $(OMPSRC:.c=.o) $(AMDOBJ) qp.h
	$(CC) $(CFLAGS) $(DEFS) -I. -o $@ bench/qp/small/qp_rep.c $(QPOBJ) sparsechol.o nd.o $(OMPSRC:.c=.o) $(AMDOBJ) $(BLASLIB) $(OMPLD) -lm
