function test_brisk()
%TEST_BRISK  Checks of the BRISK MATLAB interface (runs in MATLAB and GNU Octave).
%   Run after build_brisk_mex. Every check prints PASS or raises an error.
here = fileparts(mfilename('fullpath'));
ex1 = fullfile(here, 'examples', 'example1.dat-s');
ex2 = fullfile(here, 'examples', 'example2.dat-s');
nfail = 0;

% ---- 1. SDPA file ------------------------------------------------------------------
[objVal, x, X, Y, INFO] = brisk_sdpa(ex1);
[m, bs, c, F] = read_sdpa_cells(ex1);
nfail = nfail + check('file input: status', strcmp(INFO.status, 'OPTIMAL'));
nfail = nfail + check('file input: SDPA manual value -41.9', abs(objVal(1) + 41.9) < 1e-6 && abs(objVal(2) + 41.9) < 1e-6);
nfail = nfail + check('file input: X = sum F_i x_i - F_0', relres_sdpa(bs, F, x, X) < 1e-8);
nfail = nfail + check('file input: F_i . Y = c_i', norm(dotFY(bs, F, Y) - c) / (1 + norm(c)) < 1e-7);

% ---- 2. SDPA-M data (cell F) and triplets agree with the file --------------------------
[o2f, x2f] = brisk_sdpa(ex2);
[m, bs, c, F, T] = read_sdpa_cells(ex2);
[o2c, x2c, X2c, Y2c, I2c] = brisk_sdpa(m, numel(bs), bs, c, F);
[o2t, x2t] = brisk_sdpa(m, numel(bs), bs, c, T);
nfail = nfail + check('SDPA-M cells: same value as the file', abs(o2c(1) - o2f(1)) < 1e-7 * (1 + abs(o2f(1))));
nfail = nfail + check('SDPA triplets: same value as the file', abs(o2t(1) - o2f(1)) < 1e-7 * (1 + abs(o2f(1))));
nfail = nfail + check('SDPA-M: same x as the file', norm(x2c - x2f) < 1e-5 * (1 + norm(x2f)) && norm(x2t - x2f) < 1e-5 * (1 + norm(x2f)));
nfail = nfail + check('SDPA-M: LP block returned as a vector', isvector(Y2c{3}) && numel(Y2c{3}) == 2 && all(Y2c{3} >= -1e-9));
nfail = nfail + check('SDPA-M: primal/dual value agree', abs(o2c(1) - o2c(2)) < 1e-6 * (1 + abs(o2c(1))) && max(abs(I2c.dimacs)) < 1e-6);

% ---- 3. SeDuMi form of example 2: min <C,X>, <A_i,X> = b_i ------------------------------
[A, b, cs, K] = sdpa2sedumi(m, bs, c, F);
[xs, ys, is] = brisk_sedumi(A, b, cs, K);
nfail = nfail + check('SeDuMi: example 2 value (= -F_0.Y of SDPA)', abs(is.pobj + o2f(2)) < 1e-6 * (1 + abs(o2f(2))));
nfail = nfail + check('SeDuMi: numerr = 0', is.numerr == 0 && is.pinf == 0 && is.dinf == 0);
nfail = nfail + check('SeDuMi: A x = b', norm(A * xs - b) / (1 + norm(b)) < 1e-8);
nfail = nfail + check('SeDuMi: y = -x of SDPA', norm(ys + x2f) < 1e-5 * (1 + norm(x2f)));

% ---- 4. SeDuMi with free, nonnegative, 1x1 and PSD blocks (random, strictly feasible) ---
rng_seed(7);
K = struct('f', 3, 'l', 4, 's', [1 3 5]);
N = K.f + K.l + sum(K.s .^ 2);
m = 20;
A = sprandn(m, N, 0.4);
A(:, 1:K.f) = randn(m, K.f);
x0 = interior_point(K, 1);
b = A * x0;
y0 = randn(m, 1);
z0 = interior_point(K, 0);           % zero on the free part
c = A' * y0 + z0;
[x, y, info, z] = brisk_sedumi(A, b, c, K, struct('acc', 'high'));
nfail = nfail + check('SeDuMi mixed cone: status', info.numerr == 0);
nfail = nfail + check('SeDuMi mixed cone: A x = b', norm(A * x - b) / (1 + norm(b)) < 1e-9);
nfail = nfail + check('SeDuMi mixed cone: gap', abs(c' * x - b' * y) / (1 + abs(b' * y)) < 1e-8);
nfail = nfail + check('SeDuMi mixed cone: z = c - A''y in K', cone_min(K, z, 0) > -1e-8 && norm(z(1:K.f)) < 1e-7 * (1 + norm(c)));
nfail = nfail + check('SeDuMi mixed cone: x in K', cone_min(K, x, 1) > -1e-8);
nfail = nfail + check('SeDuMi mixed cone: transposed A accepted', abs(brisk_obj(A', b, c, K) - c' * x) < 1e-7 * (1 + abs(c' * x)));

% ---- 5. options and errors -------------------------------------------------------------
[~, ~, ~, ~, Ih] = brisk_sdpa(ex2, struct('acc', 'high'));
nfail = nfail + check('opts.acc = high: DIMACS errors <= 1e-9', max(abs(Ih.dimacs)) <= 1e-9);
ok = false;
try
    brisk_sdpa(ex2, struct('args', '-nosuchoption'));
catch err
    ok = ~isempty(strfind(err.identifier, 'brisk')); %#ok<STREMP>
end
nfail = nfail + check('an invalid option raises an error (and MATLAB keeps running)', ok);
% second-order cones (K.q, K.r): the cone solver, the semidefinite solver (mixed), high precision
Kq = struct('f', 2, 'l', 3, 'q', [4 3], 'r', 4);
Aq = sprandn(5, 16, 0.6);
pt = @() [3; 0.5 * randn(3, 1); 2; 0.5 * randn(2, 1); 1; 1; 0.3 * randn(2, 1)];
xq = [randn(2, 1); rand(3, 1) + 0.1; pt()]; zq = [zeros(2, 1); rand(3, 1) + 0.1; pt()];
bq = Aq * xq; cq = Aq' * randn(5, 1) + zq;
[x, y, info, z] = brisk_sedumi(Aq, bq, cq, Kq, struct('verbose', 0));
nfail = nfail + check('SeDuMi second-order cone program (K.q, K.r): cone solver', strcmp(info.status, 'OPTIMAL') && ...
    norm(Aq * x - bq) < 1e-7 * (1 + norm(bq)) && abs(info.pobj - info.dobj) < 1e-7 * (1 + abs(info.pobj)) && ...
    x(6) >= norm(x(7:9)) - 1e-8 && 2 * x(13) * x(14) >= norm(x(15:16))^2 - 1e-8 && norm(z - (cq - Aq' * y)) < 1e-6 * (1 + norm(cq)));    % z is the solver's slack (in K); c - A'y up to the dual residual
pq = info.pobj;
[~, ~, info] = brisk_sedumi(Aq, bq, cq, Kq, struct('verbose', 0, 'conesolver', 0));
nfail = nfail + check('the same through the semidefinite solver (conesolver = 0)', info.numerr <= 1 && abs(info.pobj - pq) < 1e-6 * (1 + abs(pq)));
[~, ~, info] = brisk_sedumi(Aq, bq, cq, Kq, struct('verbose', 0, 'prec', 'dd'));
nfail = nfail + check('the same in double-double', strcmp(info.status, 'OPTIMAL') && max(abs(info.dimacs)) < 1e-18 && abs(info.pobj - pq) < 1e-7 * (1 + abs(pq)));
Km = Kq; Km.s = 3;
Sx = randn(3); Sx = Sx * Sx' + eye(3); Sz = randn(3); Sz = Sz * Sz' + eye(3);
Am = [Aq, sprandn(5, 9, 0.5)]; bm = Am * [xq; Sx(:)]; cm = Am' * randn(5, 1) + [zq; Sz(:)];
[x, ~, info] = brisk_sedumi(Am, bm, cm, Km, struct('verbose', 0));
nfail = nfail + check('SeDuMi second-order cones with a PSD block', info.numerr <= 1 && norm(Am * x - bm) < 1e-6 * (1 + norm(bm)) && ...
    abs(info.pobj - info.dobj) < 1e-6 * (1 + abs(info.pobj)) && x(6) >= norm(x(7:9)) - 1e-7 && min(eig(reshape(x(17:25), 3, 3))) > -1e-7);
[~, ~, info] = brisk_sedumi(sparse([0 1 0]), 0, [1; 0; 2], struct('q', 3), struct('verbose', 0));   % x2 = 0 on the cone, min x1 + 2 x3: unbounded along (1, 0, -1)
nfail = nfail + check('an unbounded second-order cone program is reported (dinf)', info.dinf == 1);
ok = false;
try
    brisk_sedumi(Aq, bq, cq, struct('f', 2, 'l', 3, 'q', [4 4], 'r', 4));
catch err
    ok = ~isempty(strfind(err.identifier, 'brisk')); %#ok<STREMP>
end
nfail = nfail + check('a K that does not match A is rejected', ok);
[~, ~, ~, ~, I3] = brisk_sdpa(ex1);
nfail = nfail + check('a second solve in the same session works', strcmp(I3.status, 'OPTIMAL'));

% ---- bound mode, certification, re-solve and cause fields ---------------------
% BRISK's (P) of example 1 has the optimal value 41.9 (the SDPA value is -41.9)
[ob, xb, Xb, Yb, Ib] = brisk_sdpa(ex1, struct('bound', 'd', 'certify', true));
nfail = nfail + check('bound d: side, validity, certified', strcmp(Ib.bound.side, 'd') && Ib.bound.valid >= 1 && Ib.bound.certified == 1);
nfail = nfail + check('bound d: rigorous lower bound <= 41.9', Ib.bound.rigorous <= 41.9 + 1e-9 && Ib.bound.rigorous > 41.9 - 1e-5);
[ob, xb, Xb, Yb, Ib] = brisk_sdpa(ex1, struct('bound', 'p', 'certify', true));
nfail = nfail + check('bound p: rigorous upper bound >= 41.9', strcmp(Ib.bound.side, 'p') && Ib.bound.certified == 1 && Ib.bound.rigorous >= 41.9 - 1e-9 && Ib.bound.rigorous < 41.9 + 1e-5);
[ob, xb, Xb, Yb, Ib] = brisk_sdpa(ex1, struct('bound', 'p'));
nfail = nfail + check('certify is off by default', Ib.bound.certified == 0 && isnan(Ib.bound.rigorous));
nfail = nfail + check('info.resolves and info.cause present', isfield(Ib, 'resolves') && isfield(Ib, 'cause') && Ib.resolves.n >= 0);
[m2, bs2, c2, F2] = read_sdpa_cells(ex2);
[A2, b2, cs2, K2] = sdpa2sedumi(m2, bs2, c2, F2);
[xs2, ys2, is2] = brisk_sedumi(A2, b2, cs2, K2, struct('bound', 'sos'));
nfail = nfail + check('brisk_sedumi accepts opts.bound = ''sos''', is2.numerr <= 1);

% ---- every command-line option through opts ----------------------------------------
a = brisk_opts2args(struct('chordal', 0, 'nohsd', true, 'nofr', false, 'certify', true, 'dir', 'nt', 'fomstart_x', 'x.txt'));
nfail = nfail + check('opts -> args: numbers, flags, strings, underscore = dash', ...
    isequal(a, {'-q', '-silent', '-chordal', '0', '-nohsd', '-certify', '-dir', 'nt', '-fomstart-x', 'x.txt'}));
% 1.3.2: the commands name themselves to the solver, and the advice at the end of the log is
% written as fields of their options; a SeDuMi problem with one row (find(A) returned rows)
a = brisk_opts2args(struct('verbose', 1), 'brisk_sdpa');
nfail = nfail + check('opts -> args: the calling command', isequal(a, {'-caller', 'matlab:brisk_sdpa'}));
out = evalc('brisk_sdpa(ex1, struct(''verbose'', 1));');
has = @(t) ~isempty(strfind(out, t));
nfail = nfail + check('log: the advice in MATLAB syntax (accuracy, precision, bound)', ...
    has('in the options of brisk_sdpa(..., opts)') && has('opts.acc = ''high''') && has('opts.prec = ''dd''') && ...
    has('opts.bound = ''d''') && has('opts.certify = true'));
[x1, y1, i1] = brisk_sedumi(sparse([1 0 0 1]), 1, [2; 0; 0; 2], struct('s', 2));
nfail = nfail + check('brisk_sedumi: one constraint (min 2 tr X, tr X = 1)', i1.numerr <= 1 && abs([2 0 0 2] * x1 - 2) < 1e-6 && abs(y1 - 2) < 1e-6);
% a max-cut-type SDP on a path (tridiagonal, chordal pattern): chordal decomposition auto / off / forced
n = 120; Tm = zeros(0, 5);
for k = 1:n
    Tm(end + 1, :) = [0 1 k k -0.25 * (1 + (k > 1 && k < n))]; %#ok<AGROW>
    if k < n, Tm(end + 1, :) = [0 1 k k + 1 0.25]; end %#ok<AGROW>
end
for k = 1:n, Tm(end + 1, :) = [k 1 k k 1]; end %#ok<AGROW>
ok = true; vals = zeros(1, 3); cs = [-1 0 1];
for q = 1:3
    o = struct('verbose', 1, 'chordal', cs(q), 'threads', 1);
    txt = evalc('[ov, ~, ~, ~, inf_] = brisk_sdpa(n, 1, n, ones(n, 1), Tm, o);');
    vals(q) = ov(1);
    ok = ok && strcmp(inf_.status, 'OPTIMAL') && (~isempty(strfind(txt, 'chordal decomposition')) == (cs(q) ~= 0));
end
nfail = nfail + check('opts.chordal = -1 / 0 / 1 reaches the solver (auto, off, forced)', ok && max(abs(vals - vals(1))) < 1e-6 * (1 + abs(vals(1))));
% returnx = 0 on a chordal problem: Y (BRISK's X) is not formed, its blocks are empty
ws = warning('off', 'brisk:nox');
[~, ~, ~, Yn, in0] = brisk_sdpa(n, 1, n, ones(n, 1), Tm, struct('chordal', 1, 'returnx', 0, 'threads', 1));
[~, ~, ~, Yy, in1] = brisk_sdpa(n, 1, n, ones(n, 1), Tm, struct('chordal', 1, 'returnx', 1, 'threads', 1));
warning(ws);
nfail = nfail + check('opts.returnx = 0: no Y on a chordal problem (empty block, have_x = 0); 1: Y returned', ...
    strcmp(in0.status, 'OPTIMAL') && in0.have_x == 0 && isempty(Yn{1}) && in1.have_x == 1 && isequal(size(Yy{1}), [n n]));
o = struct('nohsd', 1, 'sym', false, 'dualize', 0, 'freeelim', 0, 'maxit', 50);
[ov, ~, ~, ~, inf_] = brisk_sdpa(ex1, o);
nfail = nfail + check('opts: flags as 1, sym false, further numeric options', strcmp(inf_.status, 'OPTIMAL') && abs(ov(1) + 41.9) < 1e-6);
try
    brisk_sdpa(ex1, struct('nosuchoption', 1)); thrown = false;
catch
    thrown = true;
end
nfail = nfail + check('opts: an unknown option is an error', thrown);
% high precision: the errors are those of the working precision, the result comes back in doubles
[ovq, ~, ~, ~, inq] = brisk_sdpa(ex1, struct('prec', 'qd'));
nfail = nfail + check('opts.prec: quad-double solve (errors below 1e-40, value -41.9)', ...
    strcmp(inq.status, 'OPTIMAL') && max(abs(inq.dimacs)) < 1e-40 && abs(ovq(1) + 41.9) < 1e-12);
% the *-algebra symmetry reduction (hidden symmetry under a random orthogonal basis)
hs = fullfile(here, '..', 'examples', 'hidden_symmetry23.dat-s');
txt1 = evalc('[ov1, ~, ~, ~, i1] = brisk_sdpa(hs, struct(''verbose'', 1, ''threads'', 1));');
txt0 = evalc('[ov0, ~, ~, ~, i0] = brisk_sdpa(hs, struct(''verbose'', 1, ''threads'', 1, ''symalg'', 0));');
nfail = nfail + check('opts.symalg: the algebra symmetry on by default, off with 0, same objective', ...
    strcmp(i1.status, 'OPTIMAL') && strcmp(i0.status, 'OPTIMAL') && ~isempty(strfind(txt1, '23 -> 5 3(x2) 4(x3)')) ...
    && isempty(strfind(txt0, 'algebra symmetry')) && abs(ov1(1) - ov0(1)) < 1e-7 * (1 + abs(ov0(1))));
try
    brisk_sdpa(ex1, struct('y', 'out.txt')); thrown = false;
catch
    thrown = true;
end
nfail = nfail + check('opts: output files are refused (the solution is returned)', thrown);

% ---- 6. against SeDuMi, when installed ---------------------------------------------------
if exist('sedumi', 'file') == 2 || exist('sedumi', 'file') == 3
    pars.fid = 0;
    [xd, yd] = sedumi(A, b, c, K, pars); %#ok<ASGLU>
    nfail = nfail + check('same value as SeDuMi', abs(c' * xd - c' * x) < 1e-6 * (1 + abs(c' * x)));
else
    fprintf('  (SeDuMi not on the path: comparison skipped)\n');
end

if nfail == 0
    fprintf('test_brisk: all checks passed\n');
else
    error('brisk:test', 'test_brisk: %d check(s) failed', nfail);
end
end

% ======================================================================================
function f = check(name, ok)
if ok, fprintf('  PASS  %s\n', name); f = 0; else, fprintf('  FAIL  %s\n', name); f = 1; end
end

function v = brisk_obj(A, b, c, K)
[x, ~, ~] = brisk_sedumi(A, b, c, K);
v = c' * x;
end

function rng_seed(s)
if exist('OCTAVE_VERSION', 'builtin'), rand('seed', s); randn('seed', s); else, rng(s); end %#ok<RAND>
end

function x = interior_point(K, withfree)
x = [withfree * randn(K.f, 1); 1 + rand(K.l, 1)];
for n = K.s
    B = randn(n); S = B * B' + n * eye(n);
    x = [x; S(:)]; %#ok<AGROW>
end
end

function lm = cone_min(K, x, ~)
lm = inf;
o = K.f;
if K.l > 0, lm = min(lm, min(x(o + (1:K.l)))); end
o = o + K.l;
for n = K.s
    S = reshape(x(o + (1:n ^ 2)), n, n); S = (S + S') / 2;
    lm = min(lm, min(eig(S)) / max(1, norm(S, 1)));
    o = o + n ^ 2;
end
end

function [m, bs, c, F, T] = read_sdpa_cells(fname)
% a small reader of SDPA sparse files (comment lines start with " or *)
txt = fileread(fname);
lines = regexp(txt, '\r?\n', 'split');
lines = lines(~cellfun(@is_comment, lines));
tok = @(l) sscanf(regexprep(l, '[,{}()=]', ' '), '%f')';
m = tok(lines{1}); m = m(1);
nb = tok(lines{2}); nb = nb(1);
bs = tok(lines{3}); bs = bs(1:nb);
c = tok(lines{4}); c = c(1:m)';
T = cell2mat(cellfun(@(l) tok(l), lines(5:end)', 'UniformOutput', false));
T = T(:, 1:5);
F = cell(nb, m + 1);
for k = 1:nb
    n = abs(bs(k));
    for i = 0:m
        sel = T(:, 1) == i & T(:, 2) == k;
        if bs(k) < 0
            v = zeros(n, 1); v(T(sel, 3)) = T(sel, 5); F{k, i + 1} = v;
        else
            M = sparse(T(sel, 3), T(sel, 4), T(sel, 5), n, n);
            F{k, i + 1} = M + triu(M, 1)';
        end
    end
end
end

function t = is_comment(l)
l = strtrim(l);
t = isempty(l) || l(1) == '"' || l(1) == '*';
end

function r = relres_sdpa(bs, F, x, X)
r = 0; nx = 0;
for k = 1:numel(bs)
    S = -F{k, 1};
    for i = 1:numel(x), S = S + x(i) * F{k, i + 1}; end
    if bs(k) < 0, S = S(:); end
    r = r + norm(full(S) - X{k}, 'fro') ^ 2; nx = nx + norm(X{k}, 'fro') ^ 2;
end
r = sqrt(r) / (1 + sqrt(nx));
end

function v = dotFY(bs, F, Y)
m = size(F, 2) - 1; v = zeros(m, 1);
for i = 1:m
    for k = 1:numel(bs)
        if bs(k) < 0, v(i) = v(i) + F{k, i + 1}(:)' * Y{k}(:);
        else, v(i) = v(i) + sum(sum(F{k, i + 1} .* Y{k})); end
    end
end
end

function [A, b, c, K] = sdpa2sedumi(m, bs, cs, F)
% (P) min <C,X> s.t. <A_i,X> = b_i with C = -F_0, A_i = F_i, b = c_sdpa, in SeDuMi form
lp = find(bs < 0); sd = find(bs > 0);
K.l = sum(abs(bs(lp))); K.s = bs(sd);
cols = {}; c = [];
for i = 0:m
    row = [];
    for k = lp, row = [row; F{k, i + 1}(:)]; end %#ok<AGROW>
    for k = sd, M = full(F{k, i + 1}); row = [row; M(:)]; end %#ok<AGROW>
    if i == 0, c = -row; else, cols{end + 1} = row; end %#ok<AGROW>
end
A = sparse([cols{:}]');
b = cs(:);
end
