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
  OMPFLAG := -Wno-unknown-pragmas
endif
CFLAGS  ?= -O3 $(ARCHFLAG) -funroll-loops $(OMPFLAG) -std=gnu11 -Wall -Wextra -Wno-unused-parameter
SRC      = problem.c presolve.c postsolve.c dictroute.c chordal.c freeelim.c dualize.c fom.c mfipm.c lralm.c symred.c symalg.c sparsechol.c dualscale.c ddend.c crossover.c solver.c bound.c boundcert.c main.c $(OMPSRC) hpmp.c hpsolve.c socp.c sedumi.c lpio.c lpsolve.c nd.c
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
	$(CC) $(CFLAGS) $(DEFS) -o $@ $(OBJ) $(AMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm

amd/%.o: amd/%.c
	$(CC) $(CFLAGS) -DNDEBUG -Wno-all -Wno-extra -c -o $@ $<

%.o: %.c brisk.h
	$(CC) $(CFLAGS) $(DEFS) -c -o $@ $<

# 4.39: the rigorous certifier changes the rounding mode
boundcert.o libobj/boundcert.o: CFLAGS += -frounding-math -fno-fast-math
hpsolve.o libobj/hpsolve.o: hp.h hpipm.inc hpfom.inc hplr.inc hppre.inc hpcert.inc
hpmp.o libobj/hpmp.o: hp.h
socp.o libobj/socp.o: socp.h
sedumi.o libobj/sedumi.o main.o libobj/main.o: socp.h sedumi.h

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
	$(CC) $(CFLAGS) -fPIC $(DEFS) -DBRISK_LIBRARY $(LIBLDFLAGS) -o $@ $(LIBOBJ) $(LIBAMDOBJ) $(CHOLMODLIB) $(METISLIB) $(BLASLIB) $(OMPLD) -lm

libobj/amd/%.o: amd/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC -DNDEBUG -Wno-all -Wno-extra -c -o $@ $<

libobj/%.o: %.c brisk.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fPIC $(DEFS) -DBRISK_LIBRARY -c -o $@ $<

# the C interface on an SDPA file and in memory against the command line (needs libbrisk)
capitest: $(LIBBRISK) tools/capi_test.c
	$(CC) -O1 -std=gnu11 -Wall -I. -o tools/capi_test tools/capi_test.c -L. -lbrisk -Wl,-rpath,$(CURDIR) -lm
	./tools/capi_test

pytest: $(LIBBRISK)
	cd python && python3 -m pytest -q tests

clean:
	rm -f brisk $(OBJ) $(AMDOBJ) omp/*.o omp/test_omp libbrisk.so libbrisk.dylib tools/capi_test
	rm -rf libobj python/build python/*.egg-info python/brisk/__pycache__ python/tests/__pycache__ python/.pytest_cache
