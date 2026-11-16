function [x, info] = brisk_lp(f, A, l, u, lb, ub, opts)
%BRISK_LP  Solve a linear program with BRISK.
%
%   [x, info] = brisk_lp(f, A, l, u, lb, ub)
%   [x, info] = brisk_lp(f, A, l, u, lb, ub, opts)
%
%   min  f'x   s.t.  l <= A x <= u,  lb <= x <= ub
%
% A (m x n) full or sparse. An absent bound is -Inf or Inf; [] for l, u, lb or ub: no bounds of that kind
% (note: no default lower bound 0 as in an MPS file); an equality row has l(i) = u(i). The LP takes the same
% path as an MPS file of the same LP: by default the interior-point method and the dual simplex method side
% by side, the presolves and the crossover; opts.lpmethod = 1-5 as on the command line (2 the dual simplex
% method, 3 the primal simplex method, 4 the interior-point method followed by a crossover); other options:
% verbose, tol, timelimit, threads (see brisk_opts2args).
%
% info: status (string), statuscode (0 optimal, 1 primal infeasible, 2 dual infeasible, 3 iteration limit,
% 4 numerical difficulties, 5 reduced accuracy, 6 time limit), obj, y (the rows' multipliers: >= 0 at a row's
% lower side, <= 0 at its upper side) and z = f - A'y (the bounds' multipliers, the same signs), both [] when
% the method used gives no exact duals (the simplex methods and the crossover always give them; the
% interior-point method's point only when its duals are exact to 1e-6; opts.lpmethod = 4 gives them always),
% iter, time. x is [] when no point is returned (an infeasible LP).
if nargin < 6, error('brisk:input', 'brisk_lp: usage brisk_lp(f, A, l, u, lb, ub, opts)'); end
if nargin < 7, opts = struct(); end
n = numel(f);
[~, f, A, l, u, lb, ub] = brisk_qp_data(sparse(n, n), f, A, l, u, lb, ub);
[x, y, z, info] = brisk_mex('lp', sparse(n, n), f, A, l, u, lb, ub, brisk_opts2args(opts, 'brisk_lp'));
info.y = y;
info.z = z;
end
