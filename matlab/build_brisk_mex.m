function build_brisk_mex(varargin)
%BUILD_BRISK_MEX  Compile the BRISK MEX file (brisk_mex) for MATLAB or GNU Octave.
%
%   build_brisk_mex                      % defaults for this platform
%   build_brisk_mex('cpu', 'apple-m1')   % -mcpu on Apple Silicon (default apple-m1: runs on M1-M4)
%   build_brisk_mex('openmp', 'bundled') % threads: BRISK's own OpenMP runtime (../omp), nothing to
%                                        %   install - the default on macOS (Apple's clang has no
%                                        %   OpenMP library)
%   build_brisk_mex('openmp', 'system')  % the compiler's runtime: Linux gcc -fopenmp (libgomp, the
%                                        %   default on Linux); macOS Homebrew's libomp (brew install libomp)
%   build_brisk_mex('openmp', false)     % no threads of BRISK's own (BLAS threads only)
%   build_brisk_mex('verbose', true)     % show the compiler commands
%   build_brisk_mex('blas', 'matlab')    % Linux MATLAB (the default there): MATLAB's own BLAS/LAPACK
%                                        %   (libmwblas, libmwlapack: MKL with 64-bit integers)
%                                        %   through the integer-widening layer blas64.c
%   build_brisk_mex('blas', 'system')    % the system BLAS/LAPACK with 32-bit integers (-lblas -llapack;
%                                        %   the default in Octave). On Linux MATLAB this crashes: the
%                                        %   symbols resolve to MATLAB's 64-bit MKL whatever is linked
%   build_brisk_mex('blas', 'system64')  % a system BLAS/LAPACK with 64-bit integers (an ILP64 build of
%                                        %   OpenBLAS; an Octave built with --enable-64 and such a BLAS)
%
% macOS (Apple Silicon or Intel): Xcode command-line tools (xcode-select --install) and a
% MATLAB-supported compiler (mex -setup C). BLAS/LAPACK come from Apple's Accelerate
% framework, the AMD ordering (../amd) and the OpenMP runtime (../omp) are bundled: no other
% library is needed. The build takes about a minute.
%
% Linux MATLAB: gcc (gcc brings libgomp); BLAS/LAPACK are MATLAB's own (no library to install).
% Linux Octave: gcc and a system BLAS/LAPACK (-lblas -llapack, e.g. OpenBLAS).
%
% The source files are the ones of the command-line solver, one directory up.

p = inputParser;
p.addParameter('cpu', 'apple-m1');
p.addParameter('openmp', 'auto');   % 'auto' | 'bundled' | 'system' | false
p.addParameter('verbose', false);
p.addParameter('blas', 'auto');     % 'auto' | 'matlab' | 'system' | 'system64'
p.parse(varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
src = fullfile(here, '..');
% (25 October 2026) the solver's C files as the Makefile's SRC line lists them (the OpenMP runtime's sources, $(OMPSRC),
% added below when needed): a second list kept here went stale when the QP and QCQP solvers were added, and the MEX file
% then failed to load (an undefined symbol, qcqp_socp)
mk = fileread(fullfile(src, 'Makefile'));
tok = regexp(mk, '(?m)^SRC\s*=\s*([^\n]*)', 'tokens', 'once');
if isempty(tok), error('brisk:build', 'no SRC line in %s', fullfile(src, 'Makefile')); end
names = regexp(tok{1}, '\S+', 'match');
names = names(cellfun(@(f) numel(f) > 2 && strcmp(f(end-1:end), '.c'), names));
files = [cellfun(@(f) fullfile(src, f), names, 'UniformOutput', false), {fullfile(here, 'brisk_mex.c')}];
for k = 1:numel(files)
    if ~exist(files{k}, 'file'), error('brisk:build', 'missing source file %s', files{k}); end
end

isoct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
cflags = '-std=gnu11 -O3 -funroll-loops -ffp-contract=off -Wno-unknown-pragmas -DBRISK_LIBRARY -DNDEBUG -frounding-math';   % -ffp-contract=off: as the Makefile   % -frounding-math: boundcert.c changes the rounding mode
defs = {};
incs = {['-I' src]};
libs = {};
ldextra = '';

omp = o.openmp;
if islogical(omp) || isnumeric(omp)
    if omp, omp = 'auto'; else, omp = 'none'; end
end
if strcmp(omp, 'auto')
    if ismac, omp = 'bundled'; else, omp = 'system'; end
end
if ~any(strcmp(omp, {'bundled', 'system', 'none'}))
    error('brisk:build', 'openmp must be ''auto'', ''bundled'', ''system'' or false');
end

if ismac
    arch = computer('arch');
    if any(strcmp(arch, {'maca64'})) || (isoct && ~isempty(strfind(computer(), 'aarch64'))) %#ok<STREMP>
        cflags = [cflags ' -mcpu=' o.cpu];
    end
    ldextra = '-framework Accelerate';
else
    blas = o.blas;
    if strcmp(blas, 'auto'), blas = ternary(isoct, 'system', 'matlab'); end
    switch blas
        case 'matlab'
            if isoct, error('brisk:build', 'blas ''matlab'' is for MATLAB; Octave uses ''system'' or ''system64'''); end
            % MATLAB's MKL takes 64-bit integers: the solver calls bk_* (blas64.c), which widen
            % the integers and call MATLAB's dgemm_ etc. (ptrdiff_t, as in MATLAB's blas.h)
            defs = [defs, {'-DBLAS_PREFIX=bk_', '-DBLAS64_INT=ptrdiff_t'}];
            files{end + 1} = fullfile(src, 'blas64.c');
            libs = {'-lmwlapack', '-lmwblas'};
            fprintf('BLAS/LAPACK: MATLAB''s (64-bit integers, through blas64.c)\n');
        case 'system64'
            defs = [defs, {'-DBLAS_PREFIX=bk_', '-DBLAS64_INT=ptrdiff_t'}];
            files{end + 1} = fullfile(src, 'blas64.c');
            libs = {'-llapack', '-lblas'};
            fprintf('BLAS/LAPACK: the system libraries with 64-bit integers (through blas64.c)\n');
        case 'system'
            libs = {'-llapack', '-lblas'};
        otherwise
            error('brisk:build', 'blas must be ''auto'', ''matlab'', ''system'' or ''system64''');
    end
end
if ~ismac, libs{end + 1} = '-ldl'; end   % dlsym (ioFlush in brisk_mex.c) is in libdl before glibc 2.34
switch omp
    case 'bundled'
        % the pragmas are compiled by clang (which lowers them to the LLVM runtime interface,
        % __kmpc_*), the runtime is ../omp/brisk_omp.c, compiled into the MEX file: no library
        % to install, no dylib to find at run time, no clash with an OpenMP runtime MATLAB
        % may have loaded (the symbols are internal to the MEX file). Needs clang: gcc lowers
        % the pragmas to libgomp's interface instead.
        if ~ismac
            warning('brisk:build', ['openmp ''bundled'' needs clang as the MEX compiler ' ...
                    '(mex -setup C); with gcc use ''system''']);
        end
        cflags = [cflags ' -Xpreprocessor -fopenmp -I' fullfile(src, 'omp')];
        files{end + 1} = fullfile(src, 'omp', 'brisk_omp.c');
        ldextra = strtrim([ldextra ' -lpthread']);
        fprintf('OpenMP: BRISK''s bundled runtime (threads: OMP_NUM_THREADS or opts.threads)\n');
    case 'system'
        if ismac
            % Homebrew's libomp with Apple's clang (-Xpreprocessor -fopenmp, as the Makefile)
            omproots = {'/opt/homebrew/opt/libomp', '/usr/local/opt/libomp'};
            found = '';
            for r = omproots
                if exist(fullfile(r{1}, 'include', 'omp.h'), 'file'), found = r{1}; break; end
            end
            if isempty(found)
                error('brisk:build', 'openmp ''system'': libomp not found (brew install libomp), or use ''bundled''');
            end
            cflags = [cflags ' -Xpreprocessor -fopenmp -I' fullfile(found, 'include')];
            ldextra = [ldextra ' -L' fullfile(found, 'lib') ' -lomp'];
            fprintf('OpenMP: Homebrew libomp (%s)\n', found);
        else
            cflags = [cflags ' -fopenmp'];
            ldextra = strtrim([ldextra ' -fopenmp']);
            fprintf('OpenMP: the compiler''s runtime (gcc: libgomp)\n');
        end
    otherwise
        fprintf('OpenMP: off (BLAS threads only)\n');
end

% AMD (the fill-reducing ordering of sparse Schur complements) is bundled in ../amd
% (AMD 2.4.6, BSD 3-clause); nothing to install
amdfiles = dir(fullfile(src, 'amd', '*.c'));
files = [files, cellfun(@(f) fullfile(src, 'amd', f), {amdfiles.name}, 'UniformOutput', false)];
defs{end + 1} = '-DHAVE_AMD';
fprintf('BRISK MEX build: %s (AMD bundled)\n', ternary(isoct, 'Octave', 'MATLAB'));

olddir = pwd;
c = onCleanup(@() cd(olddir));
cd(here);
mexfile = fullfile(here, ['brisk_mex.' mexext]);
% 5.15: on macOS the linked file goes through the post-link check (../tools/link_check.sh): the
% linker of Xcode 15.0 changed an instruction of the solver into an undefined word (SIGILL); a
% damaged file is built again with the classic linker, or removed if there is none
for attempt = 1:2
    if isoct
        setenv('CFLAGS', cflags);
        if ~isempty(ldextra), setenv('LDFLAGS', ldextra); end
        args = [{'--mex', '-o', 'brisk_mex'}, defs, incs, files, libs];
        if o.verbose, args = [{'-v'}, args]; end
        [out, status] = mkoctfile(args{:});
        if status ~= 0, error('brisk:build', 'mkoctfile failed:\n%s', out); end
    else
        args = [{'-output', 'brisk_mex', ['CFLAGS=$CFLAGS ' cflags], 'COPTIMFLAGS=-O3 -DNDEBUG'}, defs, incs, files, libs];
        if ~isempty(ldextra), args{end + 1} = ['LDFLAGS=$LDFLAGS ' ldextra]; end
        if o.verbose, args = [{'-v'}, args]; end
        mex(args{:});
    end
    if ~ismac, break; end
    [st, out] = system(sprintf('sh "%s" --check "%s"', fullfile(src, 'tools', 'link_check.sh'), mexfile));
    if st == 0, break; end
    fprintf('%s', out);
    if attempt == 2 || ~isempty(strfind(ldextra, '-ld_classic')) || system('ld -ld_classic -v >/dev/null 2>&1') ~= 0 %#ok<STREMP>
        delete(mexfile);
        error('brisk:build', ['%s was linked damaged and is removed: update the command-line tools ' ...
              '(System Settings > Software Update)'], mexfile);
    end
    fprintf('building again with the classic linker (-Wl,-ld_classic)\n');
    ldextra = strtrim([ldextra ' -Wl,-ld_classic']);
end
fprintf('built %s\n', mexfile);
end

function r = ternary(c, a, b)
if c, r = a; else, r = b; end
end
