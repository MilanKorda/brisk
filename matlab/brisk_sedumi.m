function [x, y, info, z] = brisk_sedumi(A, b, c, K, opts)
%BRISK_SEDUMI  Solve a cone program given in SeDuMi format with BRISK.
%
%   [x, y, info] = brisk_sedumi(A, b, c, K)
%   [x, y, info] = brisk_sedumi(A, b, c, K, opts)
%   [x, y, info, z] = brisk_sedumi(...)            also z = c - A'*y
%
% The SeDuMi problem pair:
%   (P)  min  c'x   s.t.  A x = b,  x in K
%   (D)  max  b'y   s.t.  z = c - A'y in K*   (K is self-dual here)
% with the cone K described as in SeDuMi:
%   K.f   number of free variables (the first K.f entries of x)
%   K.l   number of nonnegative variables (next K.l entries)
%   K.q   dimensions of the second-order cones: x(1) >= norm(x(2:end))
%   K.r   dimensions of the rotated cones: 2 x(1) x(2) >= norm(x(3:end))^2, x(1), x(2) >= 0
%   K.s   sizes of the PSD blocks; block j takes K.s(j)^2 entries of x, vec(X_j)
%         column-major (the symmetric part of the data is used, as in SeDuMi)
% in this order. Not supported: K.scomplex / K.xcomplex / K.ycomplex.
%
% A may be m x N or N x m (N = K.f + K.l + sum(K.q) + sum(K.r) + sum(K.s.^2)), sparse or full.
% A problem without PSD blocks (a linear or second-order cone program) is solved by BRISK's
% second-order cone solver; opts.conesolver = 0 sends it to the semidefinite solver, as do
% opts.prec (high precision) and opts.bound. With PSD blocks the semidefinite solver is used
% (second-order cones as arrow blocks, free variables as split pairs x = x+ - x-, which
% BRISK detects and eliminates or handles natively).
%
% opts: see brisk_opts2args (verbose, acc, tol, maxit, timelimit, args).
%
% info fields (SeDuMi-like, plus BRISK's own):
%   pinf, dinf   1 when BRISK reports (P) resp. (D) infeasible (y resp. x is then an
%                infeasibility certificate); 0 otherwise
%   numerr       0 solved to the tolerance, 1 reduced accuracy (max DIMACS error <= 1e-6),
%                2 failure (iteration or time limit, numerical difficulties)
%   iter, cpusec, pobj = c'x, dobj = b'y, dimacs (1x6, DIMACS errors on the data as given),
%   status (BRISK's status string), statuscode, brisk (the raw BRISK info)
%
% See also BRISK_SDPA, BRISK_OPTS2ARGS.

if nargin < 4, error('brisk:input', 'usage: brisk_sedumi(A, b, c, K [, opts])'); end
if nargin < 5, opts = struct(); end
if ~isstruct(K), error('brisk:input', 'K must be a struct'); end
for fn = {'scomplex', 'xcomplex', 'ycomplex'}
    if isfield(K, fn{1}) && ~isempty(K.(fn{1}))
        error('brisk:input', 'complex data (K.%s) is not supported by BRISK', fn{1});
    end
end
nf = 0; nl = 0; ns = [];
if isfield(K, 'f') && ~isempty(K.f), nf = double(K.f); end
if isfield(K, 'l') && ~isempty(K.l), nl = double(K.l); end
if isfield(K, 's') && ~isempty(K.s), ns = double(K.s(:)'); ns = ns(ns > 0); end
nq = []; nr = [];
if isfield(K, 'q') && ~isempty(K.q), nq = double(K.q(:)'); nq = nq(nq > 0); end
if isfield(K, 'r') && ~isempty(K.r), nr = double(K.r(:)'); nr = nr(nr > 0); end
N = nf + nl + sum(nq) + sum(nr) + sum(ns .^ 2);
b = full(double(b(:)));
m = numel(b);
c = double(c(:));
if numel(c) ~= N, error('brisk:input', 'numel(c) = %d, but K describes %d variables', numel(c), N); end
if size(A, 1) ~= m && size(A, 2) == m, A = A'; end
if ~isequal(size(A), [m, N]), error('brisk:input', 'A must be %d x %d (or its transpose)', m, N); end
if ~issparse(A), A = sparse(double(A)); end
if m == 0, error('brisk:input', 'no constraints (m = 0)'); end

if ~isempty(nq) || ~isempty(nr) || isempty(ns)
    % second-order cones, or no PSD block: the problem goes to BRISK in SeDuMi form
    if N == 0, error('brisk:input', 'empty cone'); end
    [x, y, zz, bi] = brisk_mex('sedumi', A, b, full(c), nf, nl, nq, nr, ns, brisk_opts2args(opts));
    if ~bi.have_x
        warning('brisk:nox', 'BRISK returned no primal solution; x is zero. Set opts.returnx = 1 to force it.');
    end
    info = sedumi_info(bi, c, x, b, y);
    if nargout > 3, z = zz; end
    return
end

% ---- column map: SeDuMi column -> (BRISK block, i, j), plus the '-' slot of free columns
s1 = ns(ns == 1); sb = ns(ns > 1);          % 1x1 PSD blocks go to the LP block
nlp = 2 * nf + nl + numel(s1);
blk = zeros(N, 1); ii = zeros(N, 1); jj = zeros(N, 1); wt = ones(N, 1);
bsizes = [];
if nlp > 0, bsizes = -nlp; end
lpb = 1;                                     % the LP block, when there is one
col = 0;
% free: x+ in slots 1..nf, x- in nf+1..2nf
blk(1:nf) = lpb; ii(1:nf) = 1:nf; jj(1:nf) = 1:nf;
col = col + nf;
% nonnegative
blk(col + (1:nl)) = lpb; ii(col + (1:nl)) = 2 * nf + (1:nl); jj(col + (1:nl)) = 2 * nf + (1:nl);
col = col + nl;
% PSD blocks, in the order of K.s
kb = numel(bsizes);                          % BRISK block counter
lpslot = 2 * nf + nl;
sdpcols = cell(1, numel(ns));
for t = 1:numel(ns)
    n = ns(t);
    idx = col + (1:n ^ 2)';
    if n == 1
        lpslot = lpslot + 1;
        blk(idx) = lpb; ii(idx) = lpslot; jj(idx) = lpslot;
    else
        kb = kb + 1;
        bsizes(end + 1) = n; %#ok<AGROW>
        [p, q] = ndgrid(1:n, 1:n);
        blk(idx) = kb; ii(idx) = min(p(:), q(:)); jj(idx) = max(p(:), q(:));
        wt(idx) = 1 - 0.5 * (p(:) ~= q(:));  % SDPA entry (i<j) = (a_ij + a_ji)/2
    end
    sdpcols{t} = struct('n', n, 'cols', idx, 'blk', blk(idx(1)), 'slot', ii(idx(1)));
    col = col + n ^ 2;
end
if isempty(bsizes), error('brisk:input', 'empty cone'); end

% ---- triplets [mat blk i j value]: F_0 = -C, F_i = A_i
[r, cc, v] = find(A);
[r0, ~, v0] = find(c);
c0 = find(c);
rows = [r; zeros(numel(c0), 1)];
cols = [cc; c0];
vals = [v; -v0];
T = [rows, blk(cols), ii(cols), jj(cols), vals .* wt(cols)];
if nf > 0
    fr = cols <= nf;                         % the '-' slots of the free columns
    Tm = T(fr, :);
    Tm(:, 3) = Tm(:, 3) + nf; Tm(:, 4) = Tm(:, 4) + nf;
    Tm(:, 5) = -Tm(:, 5);
    T = [T; Tm];
end
% merge duplicates (the two triangles of a symmetric entry)
[u, ~, ic] = unique(T(:, 1:4), 'rows');
val = accumarray(ic, T(:, 5));
keep = val ~= 0;
T = [u(keep, :), val(keep)];
clear r cc v r0 v0

[yb, Xb, ~, bi] = brisk_mex(m, bsizes, b, T, brisk_opts2args(opts));

% ---- back to SeDuMi's x
x = zeros(N, 1);
if bi.have_x
    if nlp > 0
        xl = Xb{lpb};
        x(1:nf) = xl(1:nf) - xl(nf + (1:nf));
        x(nf + (1:nl)) = xl(2 * nf + (1:nl));
    end
    for t = 1:numel(ns)
        S = sdpcols{t};
        if S.n == 1
            x(S.cols) = Xb{lpb}(S.slot);
        else
            x(S.cols) = Xb{S.blk}(:);
        end
    end
else
    warning('brisk:nox', ['BRISK returned no primal solution (the point was verified on the cliques of the chordal decomposition); ' ...
        'x is zero. Set opts.returnx = 1 to force it, or opts.chordal = 0.']);
end
y = yb;
info = sedumi_info(bi, c, x, b, y);
if nargout > 3, z = c - A' * y; end
end

function info = sedumi_info(bi, c, x, b, y)
code = bi.statuscode;
info = struct();
info.pinf = double(code == 1);
info.dinf = double(code == 2);
if any(code == [0 1 2])
    info.numerr = 0;
elseif code == 5
    info.numerr = 1;
else
    info.numerr = 2;
end
info.iter = bi.iter;
info.cpusec = bi.time;
info.pobj = full(c' * x);
info.dobj = b' * y;
info.dimacs = bi.dimacs;
info.status = bi.status;
info.statuscode = code;
info.brisk = bi;
end
