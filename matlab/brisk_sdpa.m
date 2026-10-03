function [objVal, x, X, Y, INFO] = brisk_sdpa(varargin)
%BRISK_SDPA  Solve an SDP given in SDPA format (a file, SDPA-M data, or SDPA triplets).
%
%   [objVal, x, X, Y, INFO] = brisk_sdpa('problem.dat-s')
%   [objVal, x, X, Y, INFO] = brisk_sdpa('problem.dat-s', opts)
%   [objVal, x, X, Y, INFO] = brisk_sdpa(mDIM, nBLOCK, bLOCKsTRUCT, c, F)
%   [objVal, x, X, Y, INFO] = brisk_sdpa(mDIM, nBLOCK, bLOCKsTRUCT, c, F, opts)
%   [objVal, x, X, Y, INFO] = brisk_sdpa(mDIM, nBLOCK, bLOCKsTRUCT, c, T, opts)
%
% The SDPA problem pair (the SDPA and SDPA-M convention):
%   (SDPA primal)  min  c'x          s.t.  X = sum_{i=1..m} F_i x_i - F_0,  X psd
%   (SDPA dual)    max  F_0 . Y      s.t.  F_i . Y = c_i (i = 1..m),         Y psd
% with block-diagonal F_i; LP blocks (negative entries of bLOCKsTRUCT) are diagonal, i.e.
% nonnegative vectors.
%
% Inputs
%   mDIM         m, the number of variables x
%   nBLOCK       number of blocks
%   bLOCKsTRUCT  block sizes (negative: LP block)
%   c            m-vector
%   F            nBLOCK x (mDIM+1) cell array, F{k,i+1} = block k of F_i (SDPA-M layout):
%                SDP block: n x n symmetric (sparse or full); LP block: a vector of length n
%                or an n x n diagonal matrix; empty = zero
%   T            alternatively, an nnz x 5 array [i k r s value] as in the body of a
%                .dat-s file: entry (r,s) (r <= s) of block k of F_i, i = 0..m
%   opts         options struct, see brisk_opts2args (verbose, acc, tol, maxit,
%                timelimit, args)
%
% Outputs (SDPA-M conventions)
%   objVal   [c'x, F_0 . Y]  (primal and dual objective values)
%   x        m x 1
%   X, Y     nBLOCK x 1 cell arrays; SDP blocks n x n, LP blocks n x 1 vectors
%   INFO     struct: status ('OPTIMAL', 'SOLVED TO REDUCED ACCURACY', 'PRIMAL INFEASIBLE',
%            ... - in BRISK's naming, where primal is the SDPA dual (the Y problem)),
%            statuscode, iter, dimacs (1x6, the DIMACS errors on the data as given),
%            time (s), and the BRISK-convention objectives pobj = <C,Y> = -F_0.Y, dobj = -c'x;
%            have_x: 0 when Y was not formed (a large chordal-decomposed block, or
%            opts.returnx = 0): the blocks of Y are then empty
%
% In BRISK's own convention the same problem is (P) min <C,X> s.t. <A_i,X> = b_i,
% X psd with C = -F_0, A_i = F_i, b = c; so BRISK's X is SDPA's Y, BRISK's y is -x, and
% BRISK's Z = C - sum y_i A_i is SDPA's X. SDPA's primal value c'x is -b'y (BRISK's dual
% objective, negated), SDPA's dual value F_0.Y is -<C,X>.
%
% See also BRISK_SEDUMI, BRISK_OPTS2ARGS.

opts = struct();
if nargin >= 1 && ischar(varargin{1})
    fname = varargin{1};
    if nargin >= 2, opts = varargin{2}; end
    if nargin > 2, error('brisk:input', 'usage: brisk_sdpa(filename, opts)'); end
    if ~exist(fname, 'file'), error('brisk:input', 'file not found: %s', fname); end
    [y, Xb, Zb, info] = brisk_mex(fname, brisk_opts2args(opts));
else
    if nargin < 5 || nargin > 6
        error('brisk:input', 'usage: brisk_sdpa(mDIM, nBLOCK, bLOCKsTRUCT, c, F [, opts])');
    end
    [m, nb, bs, c, F] = varargin{1:5};
    if nargin == 6, opts = varargin{6}; end
    bs = double(bs(:)');
    if numel(bs) ~= nb, error('brisk:input', 'numel(bLOCKsTRUCT) must equal nBLOCK'); end
    c = full(double(c(:)));
    if numel(c) ~= m, error('brisk:input', 'numel(c) must equal mDIM'); end
    if iscell(F)
        T = sdpam_triplets(m, nb, bs, F);
    elseif isnumeric(F) && (isempty(F) || size(F, 2) == 5)
        T = full(double(F));
    else
        error('brisk:input', 'F must be an nBLOCK x (mDIM+1) cell array or an nnz x 5 array');
    end
    [y, Xb, Zb, info] = brisk_mex(m, bs, c, T, brisk_opts2args(opts));
end
% BRISK convention -> SDPA-M convention
x = -y;
X = Zb;
Y = Xb;
if isfield(info, 'have_x') && ~info.have_x && nargout >= 4
    % Y (BRISK's X) of a chordal-decomposed block is not formed when its dense matrix
    % would not fit, or with opts.returnx = 0: the blocks of Y are empty
    warning('brisk:nox', ['BRISK did not return Y (its primal X): the solution was verified on the cliques of the chordal ' ...
        'decomposition. Set opts.returnx = 1 to force the dense matrix, or opts.chordal = 0.']);
end
objVal = [-info.dobj, -info.pobj];
INFO = info;
end

function T = sdpam_triplets(m, nb, bs, F)
if ~isequal(size(F), [nb, m + 1])
    error('brisk:input', 'F must be nBLOCK x (mDIM+1), here %d x %d', nb, m + 1);
end
parts = cell(nb, m + 1);
for k = 1:nb
    n = abs(bs(k));
    for i = 0:m
        A = F{k, i + 1};
        if isempty(A), continue; end
        if bs(k) < 0
            if isvector(A) && numel(A) == n
                v = full(double(A(:)));
            elseif isequal(size(A), [n n])
                if nnz(A - diag(diag(A))) > 0
                    error('brisk:input', 'F{%d,%d}: LP block must be diagonal', k, i + 1);
                end
                v = full(double(diag(A)));
            else
                error('brisk:input', 'F{%d,%d}: LP block of size %d must be a vector or a diagonal matrix', k, i + 1, n);
            end
            r = find(v);
            parts{k, i + 1} = [i * ones(numel(r), 1), k * ones(numel(r), 1), r, r, v(r)];
        else
            if ~isequal(size(A), [n n])
                error('brisk:input', 'F{%d,%d} must be %d x %d', k, i + 1, n, n);
            end
            A = sparse(double(A));
            if nnz(A - A') > 0
                A = (A + A') / 2;       % the symmetric part
            end
            [r, s, v] = find(triu(A));
            parts{k, i + 1} = [i * ones(numel(r), 1), k * ones(numel(r), 1), r, s, v];
        end
    end
end
T = vertcat(parts{:});
if isempty(T), T = zeros(0, 5); end
end
