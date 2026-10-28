classdef BriskQpModel < handle
%BRISKQPMODEL  A convex QP kept for re-solves after changes of its bounds or cost, each solve from the
% previous active set (BRISK's dual active-set method): model predictive control, branch and bound.
%
%   mdl = BriskQpModel(H, f, A, l, u, lb, ub)          the data as in brisk_qp
%   mdl = BriskQpModel(H, f, A, l, u, lb, ub, opts)    opts.tol, maxit, timelimit, verbose;
%                                                      opts.qpipm = true: the interior-point method,
%                                                      every solve from the start
%   [x, info] = mdl.solve()                            info as in brisk_qp
%   mdl.set_var_bounds(j, lo, hi)                      lb(j) = lo, ub(j) = hi
%   mdl.set_row_bounds(i, lo, hi)                      l(i) = lo, u(i) = hi
%   mdl.set_cost(f)                                    the linear cost
%   delete(mdl)                                        frees the model (also when it is cleared)
%
% Example: [x, info] = mdl.solve(); mdl.set_var_bounds(1, 3, Inf); [x, info] = mdl.solve();
    properties (SetAccess = private)
        handle = 0;
    end
    methods
        function obj = BriskQpModel(H, f, A, l, u, lb, ub, opts)
            if nargin < 7, error('brisk:input', 'BriskQpModel: usage BriskQpModel(H, f, A, l, u, lb, ub, opts)'); end
            if nargin < 8, opts = struct(); end
            [H, f, A, l, u, lb, ub] = brisk_qp_data(H, f, A, l, u, lb, ub);
            obj.handle = brisk_mex('qp_create', H, f, A, l, u, lb, ub, brisk_opts2args(opts, 'BriskQpModel'));
        end
        function [x, info] = solve(obj)
            [x, y, z, info] = brisk_mex('qp_solve', obj.handle);
            info.y = y;
            info.z = z;
        end
        function set_var_bounds(obj, j, lo, hi)
            brisk_mex('qp_set_var_bounds', obj.handle, j, lo, hi);
        end
        function set_row_bounds(obj, i, lo, hi)
            brisk_mex('qp_set_row_bounds', obj.handle, i, lo, hi);
        end
        function set_cost(obj, f)
            brisk_mex('qp_set_cost', obj.handle, full(double(f(:))));
        end
        function delete(obj)
            if obj.handle > 0
                try, brisk_mex('qp_free', obj.handle); catch, end
                obj.handle = 0;
            end
        end
    end
end
