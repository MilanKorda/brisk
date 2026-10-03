function args = brisk_opts2args(opts)
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
%   Fields with a meaning of their own:
%     opts.verbose   0 quiet (default), 1 summary, 2 iteration log
%     opts.bound     'p' | 'd' | 'sos' (= 'p': with SeDuMi/GloptiPoly input the SOS side is
%                    SeDuMi's primal x)
%     opts.args      further options exactly as on the command line: a string
%                    ('-nofr -dir nt') or a cell array ({'-nofr', '-dir', 'nt'})
%   Options the MEX interface handles itself are refused: x, y, z (output files; the solution
%   is returned), certify_x / certify_y (checks of a given certificate: use the command line).
if nargin < 1 || isempty(opts), opts = struct(); end
if ~isstruct(opts), error('brisk:input', 'options must be a struct'); end
args = {};
verbose = 0;
if isfield(opts, 'verbose'), verbose = opts.verbose; end
if verbose <= 0
    args = [args, {'-q', '-silent'}];
elseif verbose >= 2
    args{end + 1} = '-v';
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
    % feasible y); 'sos' = 'p': with SeDuMi/GloptiPoly input the SOS side is SeDuMi's primal x
    b = lower(char(opts.bound));
    if strcmp(b, 'sos'), b = 'p'; end
    if ~any(strcmp(b, {'p', 'd'})), error('brisk:input', 'opts.bound must be ''p'', ''d'' or ''sos'''); end
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
end
