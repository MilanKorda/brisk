# MathOptInterface wrapper of BRISK.
#
# The model, after MOI's bridges, is
#
#     min / max  a0'x + c0   s.t.  A_k x + b_k in K_k,
#     K_k in {Zeros, Nonnegatives, PositiveSemidefiniteConeTriangle},
#
# and it is handed to BRISK in one of two SDPA forms (`_build_image`, `_build_kernel`; the
# optimizer attribute "form" = "auto" (default), "image" or "kernel"):
#
# IMAGE form - x free, the SDPA primal  min c'x  s.t.  F(x) = sum_j x_j F_j - F0 >= 0: one SDP
#   block per PSD constraint (F_j = the coefficients of x_j, F0 = -b), one LP block with the
#   Nonnegatives rows and, for every Zeros row a'x + b = 0, the pair a'x + b >= 0, -a'x - b >= 0
#   (split free variables of BRISK's (P), which its presolve eliminates). The natural form of
#   moment relaxations (free moments in LMIs). Rows of BRISK's Schur complement: the variables.
#
# KERNEL form - BRISK's (P)  min <C,X>  s.t.  <A_i,X> = b_i,  X in K. A cone constraint whose
#   rows are exactly distinct variables (coefficient 1, no constant; what JuMP creates for
#   `@variable(model, X[1:n,1:n], PSD)` or `x >= 0`) is a VARIABLE BLOCK: its variables are
#   entries of X. Every other cone constraint gets a slack block S = A x + b (one row per
#   entry: S - A x = b); every Zeros row is a row; variables in no variable block are split
#   pairs x = x+ - x- in the LP block. The natural form of SOS programs and of SDPs over
#   matrix variables. Rows: the equalities plus the slack entries.
#
# "auto" counts the rows left after the split pairs are eliminated (BRISK's presolve removes
# one row per pair): image  n_vars - n_zeros,  kernel  n_zeros + n_slack_rows - n_free, and
# takes the form with fewer pairs unless the other has less than half its rows (the rule of
# BRISK's dual form). So JuMP's PSD-variable models (SOS, max-cut) go in the kernel form
# and moment relaxations in the image form. BRISK's own automatic dual form would find the
# cheap side of either file too, but only after its presolve has paid for the expensive one,
# and its automatic first-order rule looks at the row count of the file as given (ISSUES 22),
# so the choice is made here.
#
# Solutions (BRISK: X of (P), y, Z = C - A'y of (D); duals follow MOI's conventions, in which
# PositiveSemidefiniteConeTriangle is self-dual under set_dot, i.e. the dual of entry (i,j) is
# the (i,j) entry of the dual matrix):
#   image:  x = -y;  duals: PSD row (i,j) -> X_ij, Nonnegatives row -> X_lp, Zeros row ->
#           X_lp(+) - X_lp(-)
#   kernel: x from X (a free variable: X_lp(+) - X_lp(-));  duals: variable-block row -> Z at
#           the variable's entry, slack row -> Z at the slack entry, Zeros row -> y_i
#   both:   constraint primals A x + b computed here from the data; the dual objective from
#           the duals.
# Infeasibility: BRISK's statuses refer to its (P) and (D). In the image form (P) is MOI's dual
# (BRISK "PRIMAL INFEASIBLE" = MOI DUAL_INFEASIBLE with the primal ray -y), in the kernel form
# (P) is MOI's primal (BRISK "PRIMAL INFEASIBLE" = MOI INFEASIBLE with the dual ray from y and
# Z - C). BRISK returns the diverging iterate: certificates are scaled to unit max-norm.
# Variables in no constraint are left out; an objective term on one makes the problem unbounded
# when the rest is feasible.

MOI.Utilities.@product_of_sets(
    _Cones,
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.PositiveSemidefiniteConeTriangle,
)

const _SUPPORTED_SETS = Union{MOI.Zeros,MOI.Nonnegatives,MOI.PositiveSemidefiniteConeTriangle}

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{Float64,Int,MOI.Utilities.OneBasedIndexing},
        Vector{Float64},
        _Cones{Float64},
    },
}

"""
    Brisk.SDPAData

The SDPA problem built from an MOI model (the arguments of `Brisk.solve_sdpa_data`), with the
maps back to the model: `form` is `:image` or `:kernel` (see `src/MOI_wrapper.jl`). Kept by
the optimizer after `copy_to` (`Brisk.sdpa_data(optimizer)`).
"""
struct SDPAData
    form::Symbol
    m::Int                          # SDPA m (image: variables; kernel: rows)
    blocksizes::Vector{Int}
    c::Vector{Float64}
    mat::Vector{Int32}
    blk::Vector{Int32}
    i::Vector{Int32}
    j::Vector{Int32}
    v::Vector{Float64}
    nzeros::Int                     # Zeros rows of the model (the first rows of A)
    roww::Vector{Float64}           # model row -> weight in set_dot (2: off-diagonal PSD entry)
    # where each model row's dual lives: image - its entry of X; kernel - its entry of Z
    # (variable-block and slack rows) or its row of y (rowt, Zeros rows)
    rowblk::Vector{Int32}
    rowi::Vector{Int32}
    rowj::Vector{Int32}
    rowt::Vector{Int32}             # kernel: model row -> SDPA row (0: a variable-block row)
    lpblock::Int                    # the LP block (0: none)
    # model variable -> SDPA: image - its index (0: in no constraint); kernel - kind (0 in no
    # constraint, 1 an entry (varblk, vari, varj) of X, 2 a split pair at LP entries vari, vari+1)
    col::Vector{Int}
    varkind::Vector{Int8}
    varblk::Vector{Int32}
    vari::Vector{Int32}
    varj::Vector{Int32}
    cvar::Vector{Float64}           # kernel: C at the variable's entry (for ray duals)
end

mutable struct _Solution
    termination::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    raw_status::String
    x::Vector{Float64}
    s::Vector{Float64}
    dual::Vector{Float64}
    objective::Float64
    dual_objective::Float64
    solve_time::Float64
    iterations::Int
    dimacs::NTuple{6,Float64}
    result::Union{Nothing,Result}
end

"""
    Brisk.Optimizer()

A MathOptInterface optimizer for BRISK: `JuMP.Model(Brisk.Optimizer)`.

Attributes: `MOI.Silent`, `MOI.TimeLimitSec` (BRISK's `-timelimit`), `MOI.NumberOfThreads`
(`-threads`), and any BRISK option as a `MOI.RawOptimizerAttribute`, e.g.
`set_attribute(model, "acc", "high")`, `set_attribute(model, "fom", 1)`,
`set_attribute(model, "nohsd", true)`. Raw attributes of the wrapper itself:

  * `"output"`: `:julia` (default: BRISK's messages through Julia's `stdout`), `:stdout`
    (written by C), `:silent`;
  * `"form"`: `"auto"` (default), `"image"` or `"kernel"`: the SDPA form the model is
    passed in (see `src/MOI_wrapper.jl`);
  * `"write_sdpa"`: a file name; the SDPA file of the model is written there before each
    solve, so that `./brisk file` reproduces it.

After a solve, `Brisk.DIMACSErrors()` gives BRISK's six DIMACS errors and
`Brisk.RawResult()` the `Brisk.Result` in BRISK's own convention.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    options::Dict{String,Any}
    silent::Bool
    time_limit::Union{Nothing,Float64}
    threads::Union{Nothing,Int}
    cache::Union{Nothing,OptimizerCache}
    data::Union{Nothing,SDPAData}
    sense::MOI.OptimizationSense
    a0::Vector{Float64}
    c0::Float64
    solution::Union{Nothing,_Solution}
    function Optimizer()
        return new(Dict{String,Any}(), false, nothing, nothing, nothing, nothing,
                   MOI.FEASIBILITY_SENSE, Float64[], 0.0, nothing)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "BRISK"
MOI.get(::Optimizer, ::MOI.SolverVersion) = version()

function MOI.is_empty(o::Optimizer)
    return o.cache === nothing && o.solution === nothing
end

function MOI.empty!(o::Optimizer)
    o.cache = nothing
    o.data = nothing
    o.solution = nothing
    o.sense = MOI.FEASIBILITY_SENSE
    o.a0 = Float64[]
    o.c0 = 0.0
    return
end

# ---- parameters ---------------------------------------------------------------------------

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent
MOI.set(o::Optimizer, ::MOI.Silent, v::Bool) = (o.silent = v; nothing)

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(o::Optimizer, ::MOI.TimeLimitSec) = o.time_limit
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, v::Real) = (o.time_limit = Float64(v); nothing)
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (o.time_limit = nothing; nothing)

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true
MOI.get(o::Optimizer, ::MOI.NumberOfThreads) = o.threads
MOI.set(o::Optimizer, ::MOI.NumberOfThreads, v::Integer) = (o.threads = Int(v); nothing)
MOI.set(o::Optimizer, ::MOI.NumberOfThreads, ::Nothing) = (o.threads = nothing; nothing)

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true
function MOI.get(o::Optimizer, p::MOI.RawOptimizerAttribute)
    haskey(o.options, p.name) || throw(MOI.GetAttributeNotAllowed(p, "option $(p.name) is not set"))
    return o.options[p.name]
end
function MOI.set(o::Optimizer, p::MOI.RawOptimizerAttribute, v)
    if p.name == "output"
        (v isa Symbol || v isa AbstractString) && Symbol(v) in _OUTPUT_MODES ||
            throw(ArgumentError("output must be one of $(_OUTPUT_MODES)"))
        v = Symbol(v)
    elseif p.name == "form"
        String(v) in ("auto", "image", "kernel") || throw(ArgumentError("form must be \"auto\", \"image\" or \"kernel\""))
        v = String(v)
    end
    o.options[p.name] = v
    return
end

"""
    Brisk.DIMACSErrors()

Optimizer attribute: BRISK's six DIMACS errors of the last solve, on the SDPA problem built
from the model (err1, err2: A(X) = b and lambda_min(X), BRISK's (P); err3, err4:
C - A'y - Z and lambda_min(Z), BRISK's (D); err5, err6: the gaps). (P) is MOI's primal in
the kernel form and MOI's dual in the image form (`Brisk.sdpa_data(optimizer).form`).
"""
struct DIMACSErrors <: MOI.AbstractModelAttribute end
MOI.is_set_by_optimize(::DIMACSErrors) = true

"""
    Brisk.RawResult()

Optimizer attribute: the `Brisk.Result` of the last solve (BRISK's convention, the SDPA
problem of `Brisk.sdpa_data(optimizer)`), or `nothing`.
"""
struct RawResult <: MOI.AbstractModelAttribute end
MOI.is_set_by_optimize(::RawResult) = true

"""
    Brisk.sdpa_data(o::Brisk.Optimizer) -> Brisk.SDPAData

The SDPA problem built from the model at the last `copy_to` (for inspection).
"""
sdpa_data(o::Optimizer) = o.data

# ---- supported model --------------------------------------------------------------------

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorAffineFunction{Float64}}, ::Type{<:_SUPPORTED_SETS}) = true

# ---- the structure shared by both forms -----------------------------------------------------

struct _Structure
    nz::Int                          # Zeros rows
    nn::Int                          # Nonnegatives rows
    psd::Vector{UnitRange{Int}}      # rows of each PSD constraint
    psddim::Vector{Int}
    nncons::Vector{UnitRange{Int}}   # rows of each Nonnegatives constraint
    rowi::Vector{Int32}              # PSD row -> (i, j) in its matrix; Nonnegatives row -> index in its constraint
    rowj::Vector{Int32}
    roww::Vector{Float64}
end

function _structure(cache::OptimizerCache)
    Ab = cache.constraints
    nrow = Ab.coefficients.m
    nz = MOI.Utilities.num_rows(Ab.sets, MOI.Zeros)
    nn = MOI.Utilities.num_rows(Ab.sets, MOI.Nonnegatives)
    rowi = zeros(Int32, nrow); rowj = zeros(Int32, nrow); roww = ones(Float64, nrow)
    F = MOI.VectorAffineFunction{Float64}
    nncons = UnitRange{Int}[]
    for ci in MOI.get(cache, MOI.ListOfConstraintIndices{F,MOI.Nonnegatives}())
        rows = MOI.Utilities.rows(Ab.sets, ci)
        push!(nncons, rows)
        for (t, r) in enumerate(rows)
            rowi[r] = t; rowj[r] = t
        end
    end
    psd = UnitRange{Int}[]; psddim = Int[]
    for ci in MOI.get(cache, MOI.ListOfConstraintIndices{F,MOI.PositiveSemidefiniteConeTriangle}())
        d = MOI.get(cache, MOI.ConstraintSet(), ci).side_dimension
        rows = MOI.Utilities.rows(Ab.sets, ci)
        push!(psd, rows); push!(psddim, d)
        t = 0
        for jj in 1:d, ii in 1:jj            # MOI's triangle: upper, column by column
            t += 1
            r = rows[t]
            rowi[r] = ii; rowj[r] = jj
            ii != jj && (roww[r] = 2.0)
        end
    end
    return _Structure(nz, nn, psd, psddim, nncons, rowi, rowj, roww)
end

# the transpose of A (rows of the constraints)
function _rows_of(A)
    nrow, nvar = A.m, A.n
    nnzA = A.colptr[end] - 1
    rp = zeros(Int, nrow + 1)
    for p in 1:nnzA
        rp[A.rowval[p]+1] += 1
    end
    for r in 1:nrow
        rp[r+1] += rp[r]
    end
    ci = Vector{Int}(undef, nnzA); cv = Vector{Float64}(undef, nnzA)
    pos = rp[1:nrow] .+ 1
    for j in 1:nvar, p in A.colptr[j]:A.colptr[j+1]-1
        r = A.rowval[p]
        ci[pos[r]] = j; cv[pos[r]] = A.nzval[p]; pos[r] += 1
    end
    return rp, ci, cv
end

# variable blocks: cone constraints whose rows are distinct, otherwise unclaimed variables with
# coefficient 1 and no constant (JuMP's PSD and nonnegative variables)
function _variable_blocks(A, b, S::_Structure, rp, ci, cv)
    claimed = falses(A.n)
    isvar_psd = falses(length(S.psd))
    isvar_nn = falses(length(S.nncons))
    function check(rows)
        for r in rows
            (rp[r+1] - rp[r] == 1 && cv[rp[r]+1] == 1.0 && b[r] == 0.0) || return false
            claimed[ci[rp[r]+1]] && return false
        end
        vs = [ci[rp[r]+1] for r in rows]
        length(unique(vs)) == length(vs) || return false
        claimed[vs] .= true
        return true
    end
    for (k, rows) in enumerate(S.psd)
        isvar_psd[k] = check(rows)
    end
    for (k, rows) in enumerate(S.nncons)
        isvar_nn[k] = check(rows)
    end
    return claimed, isvar_psd, isvar_nn
end

# one coefficient of row r in the SDPA matrix t (0: F0); in the image form a Zeros row also
# gets its negated twin in the next LP position
@inline function _put!(mat, blk, ei, ej, v, q, t, bk, ii, jj, val)
    q += 1
    mat[q] = t; blk[q] = bk; ei[q] = ii; ej[q] = jj; v[q] = val
    return q
end

function _build_image(cache::OptimizerCache, S::_Structure, sense::MOI.OptimizationSense, a0::Vector{Float64})
    Ab = cache.constraints
    A = Ab.coefficients
    b = Ab.constants
    nrow, nvar = A.m, A.n
    nz, nn = S.nz, S.nn
    rowblk = zeros(Int32, nrow); rowi = zeros(Int32, nrow); rowj = zeros(Int32, nrow)
    blocksizes = Int[]
    nlp = 2nz + nn
    lpblock = 0
    if nlp > 0
        push!(blocksizes, -nlp)
        lpblock = 1
        for r in 1:nz
            rowblk[r] = 1; rowi[r] = 2r - 1; rowj[r] = 2r - 1
        end
        for r in nz+1:nz+nn
            rowblk[r] = 1; rowi[r] = 2nz + (r - nz); rowj[r] = rowi[r]
        end
    end
    for (k, rows) in enumerate(S.psd)
        push!(blocksizes, S.psddim[k])
        for r in rows
            rowblk[r] = length(blocksizes); rowi[r] = S.rowi[r]; rowj[r] = S.rowj[r]
        end
    end
    # the variables in at least one constraint are the SDPA variables
    col = zeros(Int, nvar)
    m = 0
    for j in 1:nvar
        if A.colptr[j+1] > A.colptr[j]
            m += 1
            col[j] = m
        end
    end
    σ = sense == MOI.MAX_SENSE ? -1.0 : 1.0
    c = zeros(m)
    for j in 1:nvar
        col[j] > 0 && (c[col[j]] = σ * a0[j])
    end
    nnzA = A.colptr[end] - 1
    nent = 2nnzA + 2nrow
    mat = Vector{Int32}(undef, nent); blk = similar(mat); ei = similar(mat); ej = similar(mat)
    v = Vector{Float64}(undef, nent)
    q = 0
    for j in 1:nvar
        t = col[j]
        t == 0 && continue
        for p in A.colptr[j]:A.colptr[j+1]-1
            r = A.rowval[p]; a = A.nzval[p]
            q = _put!(mat, blk, ei, ej, v, q, t, rowblk[r], rowi[r], rowj[r], a)
            r <= nz && (q = _put!(mat, blk, ei, ej, v, q, t, rowblk[r], rowi[r] + 1, rowj[r] + 1, -a))
        end
    end
    for r in 1:nrow                                  # F0 = -b
        b[r] == 0 && continue
        q = _put!(mat, blk, ei, ej, v, q, 0, rowblk[r], rowi[r], rowj[r], -b[r])
        r <= nz && (q = _put!(mat, blk, ei, ej, v, q, 0, rowblk[r], rowi[r] + 1, rowj[r] + 1, b[r]))
    end
    resize!(mat, q); resize!(blk, q); resize!(ei, q); resize!(ej, q); resize!(v, q)
    return SDPAData(:image, m, blocksizes, c, mat, blk, ei, ej, v, nz, S.roww, rowblk, rowi, rowj,
                    zeros(Int32, nrow), lpblock, col, Int8[], Int32[], Int32[], Int32[], Float64[])
end

function _build_kernel(cache::OptimizerCache, S::_Structure, sense::MOI.OptimizationSense, a0::Vector{Float64},
                       rp, ci, cv, claimed, isvar_psd, isvar_nn)
    Ab = cache.constraints
    A = Ab.coefficients
    b = Ab.constants
    nrow, nvar = A.m, A.n
    nz = S.nz
    varkind = zeros(Int8, nvar); varblk = zeros(Int32, nvar); vari = zeros(Int32, nvar); varj = zeros(Int32, nvar)
    rowblk = zeros(Int32, nrow); rowi = zeros(Int32, nrow); rowj = zeros(Int32, nrow); rowt = zeros(Int32, nrow)
    # the LP block: split pairs of the free variables, then the Nonnegatives constraints
    # (variable blocks and slacks alike)
    inrow = falses(nvar)
    for j in 1:nvar
        inrow[j] = A.colptr[j+1] > A.colptr[j]
    end
    nlp = 0
    for j in 1:nvar
        if inrow[j] && !claimed[j]
            varkind[j] = 2; varblk[j] = 1; vari[j] = nlp + 1; varj[j] = nlp + 1
            nlp += 2
        end
    end
    for (k, rows) in enumerate(S.nncons)
        for r in rows
            nlp += 1
            rowblk[r] = 1; rowi[r] = nlp; rowj[r] = nlp
            if isvar_nn[k]
                j = ci[rp[r]+1]
                varkind[j] = 1; varblk[j] = 1; vari[j] = nlp; varj[j] = nlp
            end
        end
    end
    blocksizes = Int[]
    lpblock = 0
    if nlp > 0
        push!(blocksizes, -nlp); lpblock = 1
    end
    for (k, rows) in enumerate(S.psd)
        push!(blocksizes, S.psddim[k])
        kb = length(blocksizes)
        for r in rows
            rowblk[r] = kb; rowi[r] = S.rowi[r]; rowj[r] = S.rowj[r]
            if isvar_psd[k]
                j = ci[rp[r]+1]
                varkind[j] = 1; varblk[j] = kb; vari[j] = S.rowi[r]; varj[j] = S.rowj[r]
            end
        end
    end
    # SDPA rows: the Zeros rows, then the slack rows (in model order)
    isvarrow = falses(nrow)
    for (k, rows) in enumerate(S.psd)
        isvar_psd[k] && (isvarrow[rows] .= true)
    end
    for (k, rows) in enumerate(S.nncons)
        isvar_nn[k] && (isvarrow[rows] .= true)
    end
    m = 0
    for r in 1:nrow
        isvarrow[r] && continue
        m += 1
        rowt[r] = m
    end
    c = zeros(m)
    nnzA = A.colptr[end] - 1
    nent = 2nnzA + nrow + 2nvar
    mat = Vector{Int32}(undef, nent); blk = similar(mat); ei = similar(mat); ej = similar(mat)
    v = Vector{Float64}(undef, nent)
    q = 0
    # coefficient `a` of variable j in SDPA matrix t: the entry of X it is (off-diagonal
    # entries enter <A, X> twice), or +a / -a on its split pair
    @inline function putvar(q, t, j, a)
        if varkind[j] == 1
            return _put!(mat, blk, ei, ej, v, q, t, varblk[j], vari[j], varj[j], vari[j] == varj[j] ? a : a / 2)
        end
        q = _put!(mat, blk, ei, ej, v, q, t, varblk[j], vari[j], vari[j], a)
        return _put!(mat, blk, ei, ej, v, q, t, varblk[j], vari[j] + 1, vari[j] + 1, -a)
    end
    for r in 1:nrow
        t = rowt[r]
        t == 0 && continue
        if r <= nz                                   # a'x + b = 0:  a'x = -b
            c[t] = -b[r]
            for p in rp[r]+1:rp[r+1]
                q = putvar(q, t, ci[p], cv[p])
            end
        else                                         # S = a'x + b:  S - a'x = b
            c[t] = b[r]
            q = _put!(mat, blk, ei, ej, v, q, t, rowblk[r], rowi[r], rowj[r], rowi[r] == rowj[r] ? 1.0 : 0.5)
            for p in rp[r]+1:rp[r+1]
                q = putvar(q, t, ci[p], -cv[p])
            end
        end
    end
    # the objective: C (min), F0 = -C
    σ = sense == MOI.MAX_SENSE ? -1.0 : 1.0
    cvar = zeros(nvar)
    for j in 1:nvar
        (a0[j] == 0 || varkind[j] == 0) && continue
        q = putvar(q, 0, j, -σ * a0[j])
        cvar[j] = varkind[j] == 1 && vari[j] != varj[j] ? σ * a0[j] / 2 : σ * a0[j]
    end
    resize!(mat, q); resize!(blk, q); resize!(ei, q); resize!(ej, q); resize!(v, q)
    col = [varkind[j] == 0 ? 0 : 1 for j in 1:nvar]
    return SDPAData(:kernel, m, blocksizes, c, mat, blk, ei, ej, v, nz, S.roww, rowblk, rowi, rowj, rowt,
                    lpblock, col, varkind, varblk, vari, varj, cvar)
end

function _build(cache::OptimizerCache, sense::MOI.OptimizationSense, a0::Vector{Float64}, form::String)
    A = cache.constraints.coefficients
    b = cache.constraints.constants
    S = _structure(cache)
    form == "image" && return _build_image(cache, S, sense, a0)
    rp, ci, cv = _rows_of(A)
    claimed, isvar_psd, isvar_nn = _variable_blocks(A, b, S, rp, ci, cv)
    nused = count(j -> A.colptr[j+1] > A.colptr[j], 1:A.n)
    nfree = nused - count(claimed)
    nslack = A.m - S.nz - sum(length(S.psd[k]) for k in eachindex(S.psd) if isvar_psd[k]; init = 0) -
             sum(length(S.nncons[k]) for k in eachindex(S.nncons) if isvar_nn[k]; init = 0)
    mkernel = S.nz + nslack
    if form == "auto"
        # rows after BRISK eliminates the split pairs (one row each): the form with fewer
        # pairs, unless the other has less than half its rows (dualize.c's rule)
        effI, effK = nused - S.nz, mkernel - nfree
        form = nfree <= S.nz ? (2effI < effK ? "image" : "kernel") : (2effK < effI ? "kernel" : "image")
    end
    (form == "kernel" && mkernel >= 1) || return _build_image(cache, S, sense, a0)
    return _build_kernel(cache, S, sense, a0, rp, ci, cv, claimed, isvar_psd, isvar_nn)
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    n = MOI.get(cache, MOI.NumberOfVariables())
    sense = MOI.get(cache, MOI.ObjectiveSense())
    a0 = zeros(n)
    c0 = 0.0
    if sense != MOI.FEASIBILITY_SENSE
        f = MOI.get(cache, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        for term in f.terms
            a0[term.variable.value] += term.coefficient
        end
        c0 = f.constant
    end
    dest.cache = cache
    dest.sense = sense
    dest.a0 = a0
    dest.c0 = c0
    dest.data = _build(cache, sense, a0, String(get(dest.options, "form", "auto")))
    return index_map
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    index_map = MOI.copy_to(dest, src)
    MOI.optimize!(dest)
    return index_map, false
end

# ---- solving ----------------------------------------------------------------------------------

_Ax(A, x) = (y = zeros(A.m); for j in 1:A.n, p in A.colptr[j]:A.colptr[j+1]-1; y[A.rowval[p]] += A.nzval[p] * x[j]; end; y)

function _run_options(o::Optimizer)
    opts = Pair{String,Any}[]
    for (k, v) in o.options
        k in ("output", "write_sdpa", "form") && continue
        if k == "bound"
            # 4.38 (untested here: no Julia in the build environment): "primal"/"sos" is the
            # side of the model's variables (the SOS Gram matrices), "dual" the other; MOI's
            # primal is BRISK's (P) in the kernel form and its (D) in the image form
            s = lowercase(String(v))
            if !(s in ("p", "d"))
                s in ("primal", "dual", "sos") || throw(ArgumentError("bound must be \"primal\", \"dual\" or \"sos\""))
                prim = s != "dual"
                img = o.data !== nothing && o.data.form == :image
                s = img ? (prim ? "d" : "p") : (prim ? "p" : "d")
            end
            push!(opts, "bound" => s)
            continue
        end
        push!(opts, k => v)
    end
    o.time_limit !== nothing && push!(opts, "timelimit" => o.time_limit)
    o.threads !== nothing && push!(opts, "threads" => o.threads)
    return opts
end

_output(o::Optimizer) = o.silent ? :silent : Symbol(get(o.options, "output", :julia))

_entry(B::AbstractVector, i, j) = B[i]
_entry(B::AbstractMatrix, i, j) = B[i, j]

# a model whose constraints involve no variable: the constants are feasible or not
function _solve_constant(S::_Structure, b::Vector{Float64})
    nrow = length(b)
    dual = zeros(nrow)
    feasible = true
    for r in 1:S.nz
        if abs(b[r]) > 1e-9 * (1 + abs(b[r]))
            feasible && (dual[r] = -sign(b[r]))
            feasible = false
        end
    end
    for r in S.nz+1:S.nz+S.nn
        if b[r] < -1e-9
            feasible && (dual[r] = 1.0)
            feasible = false
        end
    end
    for (k, rows) in enumerate(S.psd)
        n = S.psddim[k]
        B = zeros(n, n)
        for r in rows
            B[S.rowi[r], S.rowj[r]] = B[S.rowj[r], S.rowi[r]] = b[r]
        end
        ev = LinearAlgebra.eigen(LinearAlgebra.Symmetric(B))
        if ev.values[1] < -1e-9 * (1 + maximum(abs, ev.values))
            if feasible
                u = ev.vectors[:, 1]
                for r in rows
                    dual[r] = u[S.rowi[r]] * u[S.rowj[r]]
                end
            end
            feasible = false
        end
    end
    z6 = ntuple(_ -> 0.0, 6)
    feasible && return _Solution(MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT,
                                 "OPTIMAL (no variables in the constraints)", Float64[], copy(b), zeros(nrow), 0.0, 0.0, 0.0, 0, z6, nothing)
    return _Solution(MOI.INFEASIBLE, MOI.NO_SOLUTION, MOI.INFEASIBILITY_CERTIFICATE,
                     "INFEASIBLE (constant constraints violated)", Float64[], copy(b), dual, 0.0, 0.0, 0.0, 0, z6, nothing)
end

# BRISK's status and errors -> MOI's statuses. (P) is BRISK's primal: MOI's primal in the
# kernel form, MOI's dual in the image form.
function _statuses(st::Int, err::NTuple{6,Float64}, have_x::Bool, kernel::Bool)
    ep = max(abs(err[1]), abs(err[2]))      # (P): A(X) = b, X in K
    ed = max(abs(err[3]), abs(err[4]))      # (D): C - A'y = Z in K
    near(e) = e <= 1e-6 ? MOI.NEARLY_FEASIBLE_POINT : MOI.UNKNOWN_RESULT_STATUS
    pP = have_x ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
    if st == 0
        return kernel ? (MOI.OPTIMAL, pP, MOI.FEASIBLE_POINT) : (MOI.OPTIMAL, MOI.FEASIBLE_POINT, pP)
    elseif st == 1                          # (P) infeasible: a ray y
        return kernel ? (MOI.INFEASIBLE, MOI.NO_SOLUTION, MOI.INFEASIBILITY_CERTIFICATE) :
                        (MOI.DUAL_INFEASIBLE, MOI.INFEASIBILITY_CERTIFICATE, MOI.NO_SOLUTION)
    elseif st == 2                          # (D) infeasible: a ray X
        rx = have_x ? MOI.INFEASIBILITY_CERTIFICATE : MOI.NO_SOLUTION
        return kernel ? (MOI.DUAL_INFEASIBLE, rx, MOI.NO_SOLUTION) : (MOI.INFEASIBLE, MOI.NO_SOLUTION, rx)
    end
    term = st == 5 ? MOI.ALMOST_OPTIMAL : st == 3 ? MOI.ITERATION_LIMIT : st == 4 ? MOI.NUMERICAL_ERROR :
           st == 6 ? MOI.TIME_LIMIT : MOI.OTHER_ERROR
    sP = have_x ? near(ep) : MOI.NO_SOLUTION
    sD = near(ed)
    return kernel ? (term, sP, sD) : (term, sD, sP)
end

function _map_image(D::SDPAData, res::Result, nvar::Int, nrow::Int)
    x = zeros(nvar)
    for j in 1:nvar
        D.col[j] > 0 && (x[j] = -res.y[D.col[j]])
    end
    dual = zeros(nrow)
    if res.X !== nothing
        for r in 1:nrow
            k = D.rowblk[r]
            k == 0 && continue
            Xk = res.X[k]
            dual[r] = k == D.lpblock && r <= D.nzeros ? Xk[D.rowi[r]] - Xk[D.rowi[r]+1] : _entry(Xk, D.rowi[r], D.rowj[r])
        end
    end
    return x, dual
end

function _map_kernel(D::SDPAData, res::Result, nvar::Int, nrow::Int, dual_ray::Bool)
    x = zeros(nvar)
    if res.X !== nothing
        for j in 1:nvar
            k = D.varkind[j]
            k == 0 && continue
            Xk = res.X[D.varblk[j]]
            x[j] = k == 1 ? _entry(Xk, D.vari[j], D.varj[j]) : Xk[D.vari[j]] - Xk[D.vari[j]+1]
        end
    end
    # a ray: Z - C (BRISK's Z = C - A'y includes C)
    cpos = Dict{Tuple{Int32,Int32,Int32},Float64}()
    if dual_ray
        for j in 1:nvar
            D.varkind[j] == 1 && D.cvar[j] != 0 && (cpos[(D.varblk[j], D.vari[j], D.varj[j])] = D.cvar[j])
        end
    end
    dual = zeros(nrow)
    for r in 1:nrow
        if D.rowt[r] > 0 && r <= D.nzeros
            dual[r] = res.y[D.rowt[r]]
        else
            k = D.rowblk[r]
            dual[r] = _entry(res.Z[k], D.rowi[r], D.rowj[r])
            dual_ray && (dual[r] -= get(cpos, (Int32(k), D.rowi[r], D.rowj[r]), 0.0))
        end
    end
    return x, dual
end

function MOI.optimize!(o::Optimizer)
    o.cache === nothing && error("Brisk.Optimizer: no model (copy_to first)")
    D = o.data
    A = o.cache.constraints.coefficients
    b = o.cache.constraints.constants
    nvar, nrow = A.n, A.m
    t0 = time()
    haskey(o.options, "write_sdpa") &&
        write_sdpa(String(o.options["write_sdpa"]), D.m, D.blocksizes, D.c, D.mat, D.blk, D.i, D.j, D.v;
                   comment = "Brisk.jl: SDPA form ($(D.form)) of a MathOptInterface model")
    # objective terms on variables in no constraint: unbounded unless the rest is infeasible
    σ = o.sense == MOI.MAX_SENSE ? -1.0 : 1.0
    free_ray = zeros(nvar)
    for j in 1:nvar
        A.colptr[j+1] == A.colptr[j] && o.a0[j] != 0 && (free_ray[j] = -σ * sign(o.a0[j]))
    end
    local sol::_Solution
    if isempty(D.blocksizes) || D.m == 0
        sol = isempty(D.blocksizes) ?
            _Solution(MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT, "OPTIMAL (no constraints)",
                      Float64[], Float64[], Float64[], 0.0, 0.0, 0.0, 0, ntuple(_ -> 0.0, 6), nothing) :
            _solve_constant(_structure(o.cache), b)
        sol.x = zeros(nvar)
        sol.s = copy(b)
    else
        kernel = D.form == :kernel
        res = _solve_data(D.m, D.blocksizes, D.c, D.mat, D.blk, D.i, D.j, D.v,
                          option_args(_run_options(o)), _output(o))
        term, pst, dst = _statuses(res.status, res.dimacs, res.X !== nothing, kernel)
        x, dual = kernel ? _map_kernel(D, res, nvar, nrow, dst == MOI.INFEASIBILITY_CERTIFICATE) :
                           _map_image(D, res, nvar, nrow)
        s = _Ax(A, x)
        pst == MOI.INFEASIBILITY_CERTIFICATE || (s .+= b)     # a primal ray: A d
        sol = _Solution(term, pst, dst, res.status_string, x, s, dual, 0.0, 0.0, res.solve_time,
                        res.iterations, res.dimacs, res)
    end
    if any(!iszero, free_ray) && sol.termination in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        sol.termination = MOI.DUAL_INFEASIBLE
        sol.primal_status = MOI.INFEASIBILITY_CERTIFICATE
        sol.dual_status = MOI.NO_SOLUTION
        sol.raw_status *= "; unbounded: an objective variable is in no constraint"
        sol.x = free_ray
        sol.s = zeros(nrow)
    end
    # certificates to unit max-norm: BRISK returns the diverging iterate itself (e.g. X of
    # norm 1e8 with A(X) = c, not A(X) = 0), which is a ray only after scaling
    ray_p = sol.primal_status == MOI.INFEASIBILITY_CERTIFICATE
    ray_d = sol.dual_status == MOI.INFEASIBILITY_CERTIFICATE
    if ray_p && (sx = maximum(abs, sol.x; init = 0.0)) > 0
        sol.x ./= sx; sol.s ./= sx
    end
    if ray_d && (sd = maximum(abs, sol.dual; init = 0.0)) > 0
        sol.dual ./= sd
    end
    # objective values from the returned points (constants dropped for certificates; a Farkas
    # ray has <b, dual> < 0: dual objective -<b, dual> > 0 when minimizing, <b, dual> when
    # maximizing, as for a dual point)
    sol.objective = LinearAlgebra.dot(o.a0, sol.x) + (ray_p ? 0.0 : o.c0)
    wbd = sum(D.roww[r] * b[r] * sol.dual[r] for r in eachindex(b); init = 0.0)
    sol.dual_objective = (o.sense == MOI.MAX_SENSE ? wbd : -wbd) + (ray_d ? 0.0 : o.c0)
    sol.solve_time = time() - t0
    o.solution = sol
    return
end

# ---- results -------------------------------------------------------------------------------------

MOI.get(o::Optimizer, ::MOI.TerminationStatus) = o.solution === nothing ? MOI.OPTIMIZE_NOT_CALLED : o.solution.termination
MOI.get(o::Optimizer, ::MOI.RawStatusString) = o.solution === nothing ? "optimize not called" : o.solution.raw_status
MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.solution === nothing ? NaN : o.solution.solve_time
MOI.get(o::Optimizer, ::MOI.BarrierIterations) = o.solution === nothing ? 0 : o.solution.iterations
MOI.get(o::Optimizer, ::DIMACSErrors) = o.solution === nothing ? nothing : o.solution.dimacs
MOI.get(o::Optimizer, ::RawResult) = o.solution === nothing ? nothing : o.solution.result
# (the caching layers map indices in attribute values: these hold none)
MOI.Utilities.map_indices(::Function, r::Result) = r
MOI.Utilities.map_indices(::AbstractDict{T,T}, r::Result) where {T<:Union{MOI.VariableIndex,MOI.ConstraintIndex}} = r
# (and the bridge layer would try to substitute bridged variables in it)
MOI.Bridges.unbridged_function(::MOI.Bridges.AbstractBridgeOptimizer, r::Result) = r

function MOI.get(o::Optimizer, ::MOI.ResultCount)
    s = o.solution
    s === nothing && return 0
    return s.primal_status == MOI.NO_SOLUTION && s.dual_status == MOI.NO_SOLUTION ? 0 : 1
end

function MOI.get(o::Optimizer, a::MOI.PrimalStatus)
    (o.solution === nothing || a.result_index != 1) && return MOI.NO_SOLUTION
    return o.solution.primal_status
end

function MOI.get(o::Optimizer, a::MOI.DualStatus)
    (o.solution === nothing || a.result_index != 1) && return MOI.NO_SOLUTION
    return o.solution.dual_status
end

function MOI.get(o::Optimizer, a::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(o, a)
    return o.solution.objective
end

function MOI.get(o::Optimizer, a::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(o, a)
    return o.solution.dual_objective
end

function MOI.get(o::Optimizer, a::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(o, a)
    return o.solution.x[vi.value]
end

function MOI.get(o::Optimizer, a::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},S}) where {S<:_SUPPORTED_SETS}
    MOI.check_result_index_bounds(o, a)
    return o.solution.s[MOI.Utilities.rows(o.cache.constraints.sets, ci)]
end

function MOI.get(o::Optimizer, a::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},S}) where {S<:_SUPPORTED_SETS}
    MOI.check_result_index_bounds(o, a)
    return o.solution.dual[MOI.Utilities.rows(o.cache.constraints.sets, ci)]
end
