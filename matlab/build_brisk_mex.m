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
%
% macOS (Apple Silicon or Intel): Xcode command-line tools (xcode-select --install) and a
% MATLAB-supported compiler (mex -setup C). BLAS/LAPACK come from Apple's Accelerate
% framework, the AMD ordering (../amd) and the OpenMP runtime (../omp) are bundled: no other
% library is needed. The build takes about a minute.
%
% Linux: gcc and a system BLAS/LAPACK (-lblas -llapack, e.g. OpenBLAS); gcc brings libgomp.
%
% The source files are the ones of the command-line solver, one directory up.

p = inputParser;
p.addParameter('cpu', 'apple-m1');
p.addParameter('openmp', 'auto');   % 'auto' | 'bundled' | 'system' | false
p.addParameter('verbose', false);
p.parse(varargin{:});
o = p.Results;

here = fileparts(mfilename('fullpath'));
src = fullfile(here, '..');
names = {'problem.c', 'presolve.c', 'postsolve.c', 'dictroute.c', 'chordal.c', 'freeelim.c', ...
         'dualize.c', 'fom.c', 'mfipm.c', 'lralm.c', 'symred.c', 'symalg.c', 'sparsechol.c', 'dualscale.c', 'ddend.c', 'crossover.c', 'solver.c', 'bound.c', 'boundcert.c', 'hpmp.c', 'hpsolve.c', 'main.c'};
files = [cellfun(@(f) fullfile(src, f), names, 'UniformOutput', false), {fullfile(here, 'brisk_mex.c')}];
for k = 1:numel(files)
    if ~exist(files{k}, 'file'), error('brisk:build', 'missing source file %s', files{k}); end
end

isoct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
cflags = '-std=gnu11 -O3 -funroll-loops -Wno-unknown-pragmas -DBRISK_LIBRARY -DNDEBUG -frounding-math';   % -frounding-math: boundcert.c changes the rounding mode
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
    libs = {'-llapack', '-lblas'};
end
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
fprintf('built %s\n', fullfile(here, ['brisk_mex.' mexext]));
end

function r = ternary(c, a, b)
if c, r = a; else, r = b; end
end
