function args = brisk_opts2args(opts, caller)
%BRISK_OPTS2ARGS  BRISK options struct -> command-line arguments (internal).
%   Every command-line option of BRISK can be given as a field: opts.<name> = value gives
%   '-<name> value'. An underscore in the field name stands for a dash (opts.certify_y =
%   'y.txt' gives '-certify-y y.txt'); a logical true gives the bare flag (opts.nohsd = true
%   gives '-nohsd'), false or [] drops it. The full list with explanations is OPTIONS.md in the
%   BRISK directory (or run the command-line binary './brisk' without arguments). Examples:
%     opts.chordal   = 0;        % chordal decomposition off (-1 automatic, the default; 1 force)
%     opts.sym       = 'none';   % symmetry reduction off (also: false); 'auto' (default)
%     opts.acc       = 'high';   % 'low' | 'default' | 'high'  (tolerance 1e-6 / 1e-8 / 1e-10)
%     opts.timelimit = 60;  opts.threads = 2;  opts.fom = 1;  opts.mfipm = 1;  opts.lralm = 1;
%   SeDuMi's fields are taken too (BRISK in SeDuMi's place): fid (0 quiet, else the log),
%   eps (= tol), maxiter (= maxit); its tuning fields (alg, theta, beta, stepdif, w, bigeps,
%   vplot, stopat, denq, denf, numtol, bignumtol, numlvl, chol, cg, sdp, free, errors) are ignored.
%   Fields with a meaning of their own:
%     opts.verbose   0 quiet (default), 1 the log (iterations and summary), 2 the diagnosis, 3 internals
%     opts.bound     'p' | 'd' | 'pd' (both) | 'sos' (= 'p': with SeDuMi/GloptiPoly input the SOS
%                    side is SeDuMi's primal x); always checked rigorously; info.bounds.p / .d
%     opts.args      further options exactly as on the command line: a string
%                    ('-nofr -dir nt') or a cell array ({'-nofr', '-dir', 'nt'})
%   Options the MEX interface handles itself are refused: x, y, z (output files; the solution
%   is returned), certify_x / certify_y (checks of a given certificate: use the command line).
if nargin < 1 || isempty(opts), opts = struct(); end
if nargin < 2, caller = ''; end
if ~isstruct(opts), error('brisk:input', 'options must be a struct'); end
% SeDuMi's own fields (pars of sedumi(A, b, c, K, pars)), so that BRISK can be called in its place (YALMIP, scripts
% written for SeDuMi): fid (0 quiet, otherwise the log, as verbose = 1), eps (the tolerance, as tol) and maxiter (as
% maxit) are taken when BRISK's own field is not given; SeDuMi's tuning fields, which have no meaning in BRISK, are
% ignored. (31 October 2026; until then every one of them reached BRISK as an unknown option and stopped the call)
if isfield(opts, 'fid')
    if ~isfield(opts, 'verbose') && ~isempty(opts.fid), opts.verbose = double(opts.fid ~= 0); end
    opts = rmfield(opts, 'fid');
end
if isfield(opts, 'eps')
    if ~isfield(opts, 'tol') && ~isempty(opts.eps) && opts.eps > 0, opts.tol = opts.eps; end
    opts = rmfield(opts, 'eps');
end
if isfield(opts, 'maxiter')
    if ~isfield(opts, 'maxit') && ~isempty(opts.maxiter), opts.maxit = opts.maxiter; end
    opts = rmfield(opts, 'maxiter');
end
sedumi_only = {'alg', 'theta', 'beta', 'stepdif', 'w', 'bigeps', 'vplot', 'stopat', 'denq', 'denf', 'numtol', ...
               'bignumtol', 'numlvl', 'chol', 'cg', 'sdp', 'free', 'errors'};
for k = 1:numel(sedumi_only)
    if isfield(opts, sedumi_only{k}), opts = rmfield(opts, sedumi_only{k}); end
end
args = {};
verbose = 0;
if isfield(opts, 'verbose'), verbose = opts.verbose; end
if verbose <= 0
    args = [args, {'-q', '-silent'}];
elseif verbose == 2
    args{end + 1} = '-v';      % the diagnosis: every line of the solver's steps
elseif verbose >= 3
    args{end + 1} = '-vv';     % the solver's internals, per iteration
end
if isfield(opts, 'acc')
    a = char(opts.acc);
    if ~any(strcmp(a, {'low', 'default', 'high'}))
        error('brisk:input', 'opts.acc must be ''low'', ''default'' or ''high''');
    end
    args = [args, {'-acc', a}];
end
if isfield(opts, 'bound')
    % 'p' (upper bound from a feasible X of BRISK's (P)), 'd' (lower bound from a
    % feasible y), 'pd' or 'both' (both, in one solve); 'sos' = 'p': with SeDuMi/GloptiPoly
    % input the SOS side is SeDuMi's primal x. A bound is always checked rigorously; the
    % solution returned stays the solve's own point (info.bounds holds the certificates)
    b = lower(char(opts.bound));
    if strcmp(b, 'sos'), b = 'p'; end
    if any(strcmp(b, {'both', 'dp'})), b = 'pd'; end
    if ~any(strcmp(b, {'p', 'd', 'pd'})), error('brisk:input', 'opts.bound must be ''p'', ''d'', ''pd'' (''both'') or ''sos'''); end
    args = [args, {'-bound', b}];
end
if isfield(opts, 'sym')
    v = opts.sym;
    if isnumeric(v) || islogical(v), if v, v = 'auto'; else, v = 'none'; end, end
    args = [args, {'-sym', char(v)}];
end
% every other field is a command-line option
special = {'verbose', 'acc', 'bound', 'sym', 'args'};
refused = {'x', 'y', 'z', 'certify_x', 'certify_y', 'q', 'v', 'silent'};
names = fieldnames(opts);
for k = 1:numel(names)
    f = names{k};
    if any(strcmp(f, special)), continue; end
    if any(strcmp(f, refused))
        error('brisk:input', 'opts.%s is not available through the MEX interface (see brisk_opts2args)', f);
    end
    v = opts.(f);
    flag = ['-', strrep(f, '_', '-')];
    if isempty(v), continue; end
    if islogical(v)
        if ~isscalar(v), error('brisk:input', 'opts.%s must be a scalar', f); end
        if v, args{end + 1} = flag; end %#ok<AGROW>
    elseif isnumeric(v)
        if ~isscalar(v), error('brisk:input', 'opts.%s must be a scalar', f); end
        args = [args, {flag, sprintf('%.17g', double(v))}]; %#ok<AGROW>
    elseif ischar(v) || isstring(v)
        args = [args, {flag, char(v)}]; %#ok<AGROW>
    else
        error('brisk:input', 'opts.%s must be a number, a logical or a string', f);
    end
end
if isfield(opts, 'args')
    extra = opts.args;
    if ischar(extra) || isstring(extra)
        extra = regexp(strtrim(char(extra)), '\s+', 'split');
        extra = extra(~cellfun(@isempty, extra));
    end
    if ~iscell(extra), error('brisk:input', 'opts.args must be a string or a cell array of strings'); end
    args = [args, extra(:)'];
end
% which command called: the advice at the end of the log (more accuracy, a guaranteed bound) is
% then written as fields of that command's options
if ~isempty(caller), args = [args, {'-caller', ['matlab:', caller]}]; end
end
