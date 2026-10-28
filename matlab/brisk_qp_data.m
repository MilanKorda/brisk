function [H, f, A, l, u, lb, ub] = brisk_qp_data(H, f, A, l, u, lb, ub)
%BRISK_QP_DATA  The QP data of brisk_qp and BriskQpModel in the MEX file's form (internal).
n = numel(f);
if isempty(A), A = sparse(0, n); end
m = size(A, 1);
H = sparse(double(H)); A = sparse(double(A)); f = full(double(f(:)));
l = vec(l, m, -Inf, 'l'); u = vec(u, m, Inf, 'u'); lb = vec(lb, n, -Inf, 'lb'); ub = vec(ub, n, Inf, 'ub');
if ~isequal(size(H), [n n]), error('brisk:input', 'brisk_qp: H must be %d x %d (f has %d entries)', n, n, n); end
if size(A, 2) ~= n, error('brisk:input', 'brisk_qp: A must have %d columns', n); end
end

function v = vec(v, k, d, name)
if isempty(v), v = repmat(d, k, 1); return; end
v = full(double(v(:)));
if numel(v) ~= k, error('brisk:input', 'brisk_qp: %s must have %d entries', name, k); end
end
