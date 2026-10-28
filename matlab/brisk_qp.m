function [x, info] = brisk_qp(H, f, A, l, u, lb, ub, opts)
%BRISK_QP  Solve a convex quadratic program with BRISK.
%
%   [x, info] = brisk_qp(H, f, A, l, u, lb, ub)
%   [x, info] = brisk_qp(H, f, A, l, u, lb, ub, opts)
%   [x, info] = brisk_qp('problem.qps')               a QPS file (a convex QP or QCQP), as ./brisk problem.qps
%   [x, info] = brisk_qp('problem.qps', opts)
%
%   min  1/2 x'Hx + f'x   s.t.  l <= A x <= u,  lb <= x <= ub
%
% H (n x n) symmetric, or one triangle of it; A (m x n); full or sparse. An absent bound is -Inf or
% Inf; [] for l, u, lb or ub: no bounds of that kind; an equality row has l(i) = u(i). By default a
% small QP with H positive definite is solved by the dual active-set method, any other by the
% interior-point method for QPs; opts.qpas = true or opts.qpipm = true chooses (other options:
% verbose, tol, maxit, timelimit, threads; see brisk_opts2args).
%
% info: status (string), statuscode (0 optimal, 1 primal infeasible, 2 dual infeasible, 3 iteration
% limit, 4 numerical difficulties, 5 reduced accuracy, 6 time limit), obj, y (the rows' multipliers),
% z (the bounds' multipliers; H x + f - A'y - z = 0), iter, time, errors (relative primal, dual and gap
% errors). A QP that is not convex is refused with an error.
%
% For a QP solved again after changes of its bounds or cost, see BriskQpModel (each solve from the
% previous active set).
if nargin >= 1 && (ischar(H) || (isstring(H) && isscalar(H)))
    fname = char(H);
    if nargin > 2, error('brisk:input', 'brisk_qp: usage brisk_qp(''problem.qps'', opts)'); end
    if nargin < 2, f = struct(); end
    [~, ~, ext] = fileparts(fname);
    if ~strcmpi(ext, '.qps')
        error('brisk:input', 'brisk_qp: a QPS file (.qps) is expected; brisk_sdpa reads SDPA files, brisk_sedumi MAT-files');
    end
    if ~exist(fname, 'file'), error('brisk:input', 'brisk_qp: file not found: %s', fname); end
    [x, y, z, info] = brisk_mex('qp_file', fname, brisk_opts2args(f, 'brisk_qp'));
    info.y = y;
    info.z = z;
    return
end
if nargin < 7, error('brisk:input', 'brisk_qp: usage brisk_qp(H, f, A, l, u, lb, ub, opts)'); end
if nargin < 8, opts = struct(); end
[H, f, A, l, u, lb, ub] = brisk_qp_data(H, f, A, l, u, lb, ub);
[x, y, z, info] = brisk_mex('qp', H, f, A, l, u, lb, ub, brisk_opts2args(opts, 'brisk_qp'));
info.y = y;
info.z = z;
end
