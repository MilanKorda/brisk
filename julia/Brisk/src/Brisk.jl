"""
    Brisk

Julia interface to BRISK, the interior-point / first-order SDP solver, through its shared
library `libbrisk` (built by `make libbrisk` in the BRISK source directory; the C interface
is `capi.c`).

Two levels:

  * `Brisk.solve_sdpa(file; options...)` and `Brisk.solve_sdpa_data(m, blocksizes, c, mat,
    blk, i, j, v; options...)`: the problem as an SDPA file or as the numbers of one, the
    result as a [`Brisk.Result`](@ref); `Brisk.solve_sedumi(A, b, c, K; options...)`: a
    problem in SeDuMi format (linear, second-order and semidefinite cones).
  * `Brisk.Optimizer`: a MathOptInterface optimizer, so `JuMP.Model(Brisk.Optimizer)` works
    (see `src/MOI_wrapper.jl`).

Options are BRISK's command-line options without the dash: `acc = "high"`, `fom = 1`,
`threads = 2`, `timelimit = 60`, `nohsd = true` (a flag), ... (`./brisk` with no arguments
lists them).

The library is found at `ENV["BRISK_LIBRARY"]` if set, else at `../../../libbrisk.so`
(`.dylib`) relative to this file, i.e. in the BRISK source directory that contains
`julia/Brisk`. `Brisk.load_library(path)` loads another one.
"""
module Brisk

import Libdl
import LinearAlgebra
import SparseArrays
import MathOptInterface as MOI

const _LIB = Ref{Ptr{Cvoid}}(C_NULL)
const _LIBPATH = Ref{String}("")
const _FN = Dict{Symbol,Ptr{Cvoid}}()
# the solver has process-wide state (capi.c): one solve at a time
const _LOCK = ReentrantLock()

function _default_library_path()
    p = get(ENV, "BRISK_LIBRARY", "")
    isempty(p) || return p
    return normpath(joinpath(@__DIR__, "..", "..", "..", "libbrisk." * Libdl.dlext))
end

"""
    Brisk.load_library(path::AbstractString = <default>)

Load `libbrisk` from `path` (done automatically on first use). On Linux the library is
opened with `RTLD_DEEPBIND`, so that its BLAS/LAPACK calls go to the (LP64) BLAS it was
linked with and not to Julia's `libblastrampoline`, which exports the same symbol names.
Like the command-line solver, BRISK wants its BLAS single-threaded (OpenBLAS's worker
threads spin next to BRISK's OpenMP threads): `OPENBLAS_NUM_THREADS=1` is set while the
library and its BLAS are loaded, unless the variable (or `BRISK_BLASMT`) is already set.
"""
function load_library(path::AbstractString = _default_library_path())
    if !isfile(path)
        error("libbrisk not found at $path: build it with `make libbrisk` in the BRISK " *
              "source directory, or set ENV[\"BRISK_LIBRARY\"] to its path")
    end
    set_env = !haskey(ENV, "OPENBLAS_NUM_THREADS") && !haskey(ENV, "BRISK_BLASMT")
    set_env && (ENV["OPENBLAS_NUM_THREADS"] = "1")
    flags = Libdl.RTLD_NOW | Libdl.RTLD_LOCAL
    Sys.islinux() && (flags |= Libdl.RTLD_DEEPBIND)
    h = try
        Libdl.dlopen(path, flags)
    finally
        set_env && delete!(ENV, "OPENBLAS_NUM_THREADS")
    end
    _LIB[] = h
    _LIBPATH[] = String(path)
    empty!(_FN)
    return path
end

function _fn(name::Symbol)
    _LIB[] == C_NULL && load_library()
    return get!(() -> Libdl.dlsym(_LIB[], name), _FN, name)
end

"""
    Brisk.version() -> String

The version of the loaded `libbrisk`.
"""
version() = unsafe_string(ccall(_fn(:brisk_version), Cstring, ()))

"""
    Brisk.library_path() -> String

The path of the loaded `libbrisk` (loading it if needed).
"""
function library_path()
    _LIB[] == C_NULL && load_library()
    return _LIBPATH[]
end

# ---- output ---------------------------------------------------------------------------

function _print_to_julia(s::Cstring, is_err::Cint)::Cint
    print(is_err != 0 ? stderr : stdout, unsafe_string(s))
    return Cint(0)
end

const _OUTPUT_MODES = (:julia, :stdout, :silent)

# :julia  - BRISK's output through Julia's stdout/stderr (redirect_stdout, notebooks, ...)
# :stdout - written by C to the process's stdout/stderr (as the command line does)
# :silent - nothing but error messages
function _set_output(mode::Symbol)
    mode in _OUTPUT_MODES || throw(ArgumentError("output must be one of $(_OUTPUT_MODES), not :$mode"))
    if mode == :julia
        cb = @cfunction(_print_to_julia, Cint, (Cstring, Cint))
        ccall(_fn(:brisk_set_output), Cvoid, (Cint, Ptr{Cvoid}), 2, cb)
    else
        ccall(_fn(:brisk_set_output), Cvoid, (Cint, Ptr{Cvoid}), mode == :silent ? 1 : 0, C_NULL)
    end
    return
end

# ---- options --------------------------------------------------------------------------

"""
    Brisk.option_args(options) -> Vector{String}

BRISK command-line arguments from `name => value` pairs: `"acc" => "high"` gives
`["-acc", "high"]`, `"nohsd" => true` gives `["-nohsd"]` (a flag), `"x" => false` or
`nothing` gives nothing. A leading dash in the name is optional. A value that is a vector
of strings is appended verbatim (`"args" => ["-acc", "high"]`).
"""
function option_args(options)
    args = String[]
    for (k, v) in options
        name = String(k)
        if v isa AbstractVector{<:AbstractString}
            append!(args, String.(v))
            continue
        end
        (v === false || v === nothing) && continue
        flag = startswith(name, "-") ? name : "-" * replace(name, "_" => "-")   # certify_y -> -certify-y
        push!(args, flag)
        v === true || push!(args, v isa AbstractFloat ? repr(Float64(v)) : string(v))
    end
    return args
end

# ---- results --------------------------------------------------------------------------

"""
    Brisk.Result

The solution of a problem in BRISK's convention (the SDPA file's, with C = -F0,
A_i = F_i, b = c):

    (P) min <C,X>  s.t. <A_i,X> = b_i, X in K
    (D) max b'y    s.t. C - sum_i y_i A_i = Z in K

(the SDPA primal variable is x = -y; the SDPA optimal value is -primal_objective).

Fields: `status` (0 optimal, 1 primal infeasible, 2 dual infeasible - of (P) and (D), i.e.
swapped with respect to the SDPA names -, 3 iteration limit, 4 numerical difficulties,
5 reduced accuracy, 6 time limit; -1 not solved), `status_string`, `exit_code` (the command
line's), `iterations`, `primal_objective` <C,X>, `dual_objective` b'y, `dimacs` (errors 1-6
on the data given), `solve_time`, `blocksizes`, `y`, `X` (per block: a matrix for an SDP
block, a vector for an LP block; `nothing` when the engine returned no X), `Z`, and `bound`
(`nothing` unless the option `bound = "p"` or `"d"` was given): a named tuple `(side, value,
resid, lammin, valid, certified, rigorous)` — `side` `:p` (an upper bound ⟨C,X⟩) or `:d` (a
lower bound b'y), `rigorous` the certified bound of `certify = true` (NaN when not run).
"""
struct Result
    status::Int
    status_string::String
    exit_code::Int
    iterations::Int
    primal_objective::Float64
    dual_objective::Float64
    dimacs::NTuple{6,Float64}
    solve_time::Float64
    blocksizes::Vector{Int}
    y::Vector{Float64}
    X::Union{Nothing,Vector{Array{Float64}}}
    Z::Vector{Array{Float64}}
    bound::Union{Nothing,NamedTuple}   # with "bound" (BRISK's -bound p|d), see below
end

function Base.show(io::IO, r::Result)
    print(io, "Brisk.Result(", r.status_string, ", ", r.iterations, " iterations, <C,X> = ",
          r.primal_objective, ", b'y = ", r.dual_objective, ", DIMACS max ",
          maximum(abs, r.dimacs), ", ", round(r.solve_time; digits = 3), " s)")
end

function _block(p::Ptr{Cdouble}, s::Int)
    n = abs(s)
    p == C_NULL && return s < 0 ? zeros(0) : zeros(0, 0)   # a block that is not returned (a large chordal block): empty, not n x n zeros
    v = copy(unsafe_wrap(Array, p, s < 0 ? n : n * n))
    return s < 0 ? v : reshape(v, n, n)
end

function _read_result(r::Ptr{Cvoid})
    st = Int(ccall(_fn(:brisk_result_status), Cint, (Ptr{Cvoid},), r))
    sstr = unsafe_string(ccall(_fn(:brisk_result_status_str), Cstring, (Ptr{Cvoid},), r))
    ec = Int(ccall(_fn(:brisk_result_exit_code), Cint, (Ptr{Cvoid},), r))
    it = Int(ccall(_fn(:brisk_result_iterations), Cint, (Ptr{Cvoid},), r))
    m = Int(ccall(_fn(:brisk_result_m), Cint, (Ptr{Cvoid},), r))
    nb = Int(ccall(_fn(:brisk_result_nblk), Cint, (Ptr{Cvoid},), r))
    bs = [Int(ccall(_fn(:brisk_result_blocksize), Cint, (Ptr{Cvoid}, Cint), r, k)) for k in 0:nb-1]
    pobj = ccall(_fn(:brisk_result_pobj), Cdouble, (Ptr{Cvoid},), r)
    dobj = ccall(_fn(:brisk_result_dobj), Cdouble, (Ptr{Cvoid},), r)
    t = ccall(_fn(:brisk_result_time), Cdouble, (Ptr{Cvoid},), r)
    err = zeros(6)
    ccall(_fn(:brisk_result_dimacs), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), r, err)
    yp = ccall(_fn(:brisk_result_y), Ptr{Cdouble}, (Ptr{Cvoid},), r)
    y = yp == C_NULL ? zeros(m) : copy(unsafe_wrap(Array, yp, m))
    havex = ccall(_fn(:brisk_result_have_x), Cint, (Ptr{Cvoid},), r) != 0
    X = havex ? Array{Float64}[_block(ccall(_fn(:brisk_result_X), Ptr{Cdouble}, (Ptr{Cvoid}, Cint), r, k - 1), bs[k]) for k in 1:nb] : nothing
    Z = Array{Float64}[_block(ccall(_fn(:brisk_result_Z), Ptr{Cdouble}, (Ptr{Cvoid}, Cint), r, k - 1), bs[k]) for k in 1:nb]
    bo = zeros(6)
    side = Int(ccall(_fn(:brisk_result_bound), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), r, bo))
    bound = side == 0 ? nothing :
        (side = side == 1 ? :p : :d, value = bo[1], resid = bo[2], lammin = bo[3], valid = bo[4] != 0,
         certified = bo[5] != 0, rigorous = bo[6])
    return Result(st, sstr, ec, it, pobj, dobj, Tuple(err), t, bs, y, X, Z, bound)
end

"""
    Brisk.BriskError

Thrown when BRISK returns no solution: invalid options or data (the messages are printed
by BRISK), or an abort (out of memory).
"""
struct BriskError <: Exception
    code::Int
    msg::String
end
Base.showerror(io::IO, e::BriskError) = print(io, "BriskError(", e.code, "): ", e.msg)

"""
    Brisk.hp_solution() -> NamedTuple or nothing

All the digits of the last high-precision solve (option `prec = "dd"`, `"qd"` or a number of
digits): `(digits, blocksizes, pobj, dobj, y, X, Z)` as `BigFloat`s of sufficient precision
(`pobj` is `<C,X>`, `dobj` is `b'y`; `X[k]`, `Z[k]` are matrices, or vectors for LP blocks).
The `Result` of that solve holds the same solution rounded to `Float64`. Returns `nothing`
when the last solve was not a high-precision one. With `bound = "p"` or `"d"` and a
certificate, `bound` is the rigorous bound on the optimal value of (P) (upper for p, lower
for d; rounded outward) and the certificate is the `X` or the `y` returned; otherwise
`bound` is `nothing`.
"""
function hp_solution()
    digits = Int(ccall(_fn(:brisk_hp_digits), Cint, ()))
    digits <= 0 && return nothing
    prec = ceil(Int, digits * log2(10)) + 16
    function get(what, block = 0)
        n = -ccall(_fn(:brisk_hp_text), Clonglong, (Cint, Cint, Ptr{UInt8}, Clonglong), what, block, C_NULL, 0)
        n <= 0 && return BigFloat[]
        buf = Vector{UInt8}(undef, n)
        len = ccall(_fn(:brisk_hp_text), Clonglong, (Cint, Cint, Ptr{UInt8}, Clonglong), what, block, buf, n)
        return [parse(BigFloat, t; precision = prec) for t in split(String(buf[1:len]))]
    end
    nb = Int(ccall(_fn(:brisk_hp_nblk), Cint, ()))
    bs = [Int(ccall(_fn(:brisk_hp_blocksize), Cint, (Cint,), k - 1)) for k in 1:nb]
    blk(what, k) = bs[k] < 0 ? get(what, k - 1) : permutedims(reshape(get(what, k - 1), bs[k], bs[k]))
    first_or_nothing(v) = isempty(v) ? nothing : v[1]
    return (digits = digits, blocksizes = bs, pobj = first_or_nothing(get(0)), dobj = first_or_nothing(get(1)), y = get(2),
            X = [blk(3, k) for k in 1:nb], Z = [blk(4, k) for k in 1:nb], bound = first_or_nothing(get(5)))
end

function _check(rc, res)
    rc < 0 && throw(BriskError(rc, "BRISK stopped (out of memory, or an option it rejected late); see the messages above"))
    rc == 1 && throw(BriskError(rc, "invalid options; see the messages above"))
    rc == 2 && throw(BriskError(rc, "the problem could not be read (invalid data); see the messages above"))
    res.status < 0 && throw(BriskError(rc, "BRISK returned no solution; see the messages above"))
    return res
end

function _with_result(f, reader = _read_result)
    r = ccall(_fn(:brisk_result_new), Ptr{Cvoid}, ())
    r == C_NULL && throw(OutOfMemoryError())
    try
        rc = f(r)
        return rc, reader(r)
    finally
        ccall(_fn(:brisk_result_delete), Cvoid, (Ptr{Cvoid},), r)
    end
end

# 4.39 (ISSUES 23): Ctrl-C. With more than one Julia thread the solve runs in a spawned task and
# the calling task waits; an InterruptException there calls brisk_interrupt(), and the solve
# returns its current iterate (status TIME LIMIT, `Brisk.interrupted()` true). With one thread a
# ccall cannot be interrupted (Julia delivers the exception when it returns).
function _interruptible(g)
    (Threads.nthreads() > 1 && Threads.threadid() == 1) || return g()
    # the solve on another thread than the calling one (the root task on thread 1 receives the
    # InterruptException; a ccall on thread 1 would block its delivery): ThreadPools.jl's way
    t = Task(g); t.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, Threads.nthreads() - 1)
    schedule(t)
    try
        # a timed wait: the sleeping root task reaches a safepoint, where Julia delivers the
        # InterruptException (an untimed fetch would sleep until the solve ends)
        while !istaskdone(t)
            sleep(0.05)
        end
    catch e
        e isa InterruptException || rethrow()
        ccall(_fn(:brisk_interrupt), Cvoid, ())
    end
    return fetch(t)
end

"""
    Brisk.interrupted() -> Bool

Whether the last solve was stopped by an interrupt (Ctrl-C with Julia started with `-t 2` or
more threads, or `Brisk.interrupt()` from another task).
"""
interrupted() = ccall(_fn(:brisk_interrupted), Cint, ()) != 0

"""
    Brisk.interrupt()

Stop the running solve as if its time limit were reached now (callable from any task or thread).
"""
interrupt() = ccall(_fn(:brisk_interrupt), Cvoid, ())

# The option pair that tells libbrisk which interface and command called it: the advice at the
# end of the log (more accuracy, a guaranteed bound) is then written in that command's syntax.
# Libraries before 1.3.2 do not know the option and are not given it.
function _tagged(args::Vector{String}, tag::String)
    v = tryparse(VersionNumber, version())
    (v === nothing || v < v"1.3.2") && return args
    return vcat(args, ["-caller", tag])
end

function _call(f, args::Vector{String}, output::Symbol, reader = _read_result)
    lock(_LOCK) do
        _set_output(output)
        flush(stdout); flush(stderr)
        cargs = [Base.unsafe_convert(Cstring, a) for a in args]
        GC.@preserve args cargs begin
            rc, res = _interruptible(() -> _with_result(r -> f(r, cargs), reader))
        end
        _set_output(:stdout)
        return _check(rc, res)
    end
end

"""
    Brisk.solve_sdpa(file::AbstractString; output = :julia, options...) -> Brisk.Result

Solve an SDPA sparse file (`.dat-s`) exactly as the command line `./brisk file options...`
does. `output`: `:julia` (BRISK's messages through Julia's `stdout`), `:stdout` (written by
C directly) or `:silent`. Other keywords are BRISK options (`acc = "high"`, `fom = 1`, ...).
"""
function solve_sdpa(file::AbstractString; output::Symbol = :julia, options...)
    isfile(file) || throw(ArgumentError("no such file: $file"))
    args = _tagged(option_args(options), "julia:solve_sdpa")
    return _call(args, output) do r, cargs
        Int(ccall(_fn(:brisk_solve_file), Cint, (Cstring, Cint, Ptr{Cstring}, Ptr{Cvoid}),
                  file, length(cargs), cargs, r))
    end
end

"""
    Brisk.solve_sdpa_data(m, blocksizes, c, mat, blk, i, j, v; output = :julia, options...) -> Brisk.Result

Solve the problem whose SDPA file would have `m` constraints, the block sizes `blocksizes`
(LP blocks negative), the objective `c` (length `m`) and the entries
`(mat[q], blk[q], i[q], j[q], v[q])`, 1-based as in the file (`mat` 0 for F0, 1..m for
F_i; entries with i > j are swapped; zeros are skipped). Nothing is written to disk; the
result is the same as `solve_sdpa` on that file.
"""
function solve_sdpa_data(m::Integer, blocksizes::AbstractVector{<:Integer}, c::AbstractVector{<:Real},
                         mat::AbstractVector{<:Integer}, blk::AbstractVector{<:Integer},
                         i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer}, v::AbstractVector{<:Real};
                         output::Symbol = :julia, options...)
    length(c) == m || throw(DimensionMismatch("length(c) = $(length(c)) != m = $m"))
    nnz = length(v)
    all(x -> length(x) == nnz, (mat, blk, i, j)) || throw(DimensionMismatch("mat, blk, i, j, v must have the same length"))
    return _solve_data(m, blocksizes, c, mat, blk, i, j, v, _tagged(option_args(options), "julia:solve_sdpa_data"), output)
end

# the same with the options already as command-line arguments (the MOI wrapper)
function _solve_data(m, blocksizes, c, mat, blk, i, j, v, args::Vector{String}, output::Symbol)
    nnz = length(v)
    bs = Vector{Cint}(blocksizes)
    cc = Vector{Cdouble}(c)
    mat32, blk32, i32, j32 = Vector{Cint}(mat), Vector{Cint}(blk), Vector{Cint}(i), Vector{Cint}(j)
    vv = Vector{Cdouble}(v)
    return _call(args, output) do r, cargs
        Int(ccall(_fn(:brisk_solve_data), Cint,
                  (Cint, Cint, Ptr{Cint}, Ptr{Cdouble}, Int64, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                   Cint, Ptr{Cstring}, Ptr{Cvoid}),
                  m, length(bs), bs, cc, nnz, mat32, blk32, i32, j32, vv, length(cargs), cargs, r))
    end
end

"""
    Brisk.write_sdpa(file, m, blocksizes, c, mat, blk, i, j, v; comment = "")

Write the numbers of `solve_sdpa_data` as an SDPA sparse file (full precision), e.g. to
reproduce a JuMP model's solve with the command-line solver.
"""
function write_sdpa(file::AbstractString, m, blocksizes, c, mat, blk, i, j, v; comment::AbstractString = "")
    open(file, "w") do io
        println(io, "\"", isempty(comment) ? "written by Brisk.jl" : replace(comment, '\n' => ' '))
        println(io, m)
        println(io, length(blocksizes))
        println(io, join(blocksizes, " "))
        println(io, join((repr(Float64(x)) for x in c), " "))
        for q in eachindex(v)
            v[q] == 0 && continue
            println(io, mat[q], " ", blk[q], " ", i[q], " ", j[q], " ", repr(Float64(v[q])))
        end
    end
    return file
end

# ---- problems in SeDuMi format -------------------------------------------------------------------

"""
    Brisk.ConeResult

The solution of a problem in SeDuMi format (`Brisk.solve_sedumi`): `status`, `status_string`,
`exit_code`, `iterations`, `primal_objective` (c'x), `dual_objective` (b'y), `dimacs`,
`solve_time`, and the vectors `x` (primal), `y`, `z = c - A'y` (dual slack), in the order of
`K`. For an infeasible problem `y` (status 1, b'y = 1) or `x` (status 2, c'x = -1) is the
certificate. `x` is `nothing` when the engine returned no primal point.
"""
struct ConeResult
    status::Int
    status_string::String
    exit_code::Int
    iterations::Int
    primal_objective::Float64
    dual_objective::Float64
    dimacs::NTuple{6,Float64}
    solve_time::Float64
    x::Union{Nothing,Vector{Float64}}
    y::Vector{Float64}
    z::Vector{Float64}
end

function Base.show(io::IO, r::ConeResult)
    print(io, "Brisk.ConeResult(", r.status_string, ", ", r.iterations, " iterations, c'x = ",
          r.primal_objective, ", b'y = ", r.dual_objective, ", max error ",
          maximum(abs, r.dimacs), ", ", round(r.solve_time; digits = 3), " s)")
end

function _read_cone_result(r::Ptr{Cvoid})
    st = Int(ccall(_fn(:brisk_result_status), Cint, (Ptr{Cvoid},), r))
    sstr = unsafe_string(ccall(_fn(:brisk_result_status_str), Cstring, (Ptr{Cvoid},), r))
    ec = Int(ccall(_fn(:brisk_result_exit_code), Cint, (Ptr{Cvoid},), r))
    it = Int(ccall(_fn(:brisk_result_iterations), Cint, (Ptr{Cvoid},), r))
    m = Int(ccall(_fn(:brisk_result_m), Cint, (Ptr{Cvoid},), r))
    n = Int(ccall(_fn(:brisk_result_sn), Cint, (Ptr{Cvoid},), r))
    pobj = ccall(_fn(:brisk_result_pobj), Cdouble, (Ptr{Cvoid},), r)
    dobj = ccall(_fn(:brisk_result_dobj), Cdouble, (Ptr{Cvoid},), r)
    t = ccall(_fn(:brisk_result_time), Cdouble, (Ptr{Cvoid},), r)
    err = zeros(6)
    ccall(_fn(:brisk_result_dimacs), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), r, err)
    vec(p, k) = p == C_NULL ? nothing : copy(unsafe_wrap(Array, p, k))
    x = vec(ccall(_fn(:brisk_result_sx), Ptr{Cdouble}, (Ptr{Cvoid},), r), n)
    y = vec(ccall(_fn(:brisk_result_y), Ptr{Cdouble}, (Ptr{Cvoid},), r), m)
    z = vec(ccall(_fn(:brisk_result_sz), Ptr{Cdouble}, (Ptr{Cvoid},), r), n)
    return ConeResult(st, sstr, ec, it, pobj, dobj, Tuple(err), t, x, y === nothing ? zeros(m) : y, z === nothing ? zeros(n) : z)
end

_kget(K, k::Symbol) = K isa AbstractDict ? get(K, k, get(K, String(k), nothing)) : (hasproperty(K, k) ? getproperty(K, k) : nothing)
_kcount(K, k) = (v = _kget(K, k); v === nothing || isempty(v) ? 0 : Int(sum(v)))
_klist(K, k) = (v = _kget(K, k); v === nothing ? Cint[] : Cint[Int(d) for d in v if d > 0])

"""
    Brisk.solve_sedumi(A, b, c, K; output = :julia, options...) -> Brisk.ConeResult
    Brisk.solve_sedumi(file; output = :julia, options...)

Solve a problem in SeDuMi format,

    min c'x  s.t.  A x = b,  x in K        max b'y  s.t.  c - A'y = z in K*,

with `A` an m x n matrix (sparse or dense) and `K` a named tuple or dictionary with `f` (free
variables), `l` (nonnegative variables), `q` (second-order cones x0 >= |x(1:)|: a vector of
dimensions), `r` (rotated cones 2 x0 x1 >= |x(2:)|^2) and `s` (semidefinite blocks, each as
its d*d entries by columns), in this order: `K = (f = 2, l = 10, q = [4, 7])`. Without
semidefinite blocks the problem is solved by BRISK's second-order cone solver, with them by
the semidefinite solver. `file` is a MAT-file (version 5 to 7) with `A` (or `At`), `b`, `c`, `K`.
"""
function solve_sedumi(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, c::AbstractVector{<:Real}, K;
                      output::Symbol = :julia, options...)
    m, n = length(b), length(c)
    size(A) == (m, n) || (size(A) == (n, m) && m != n ? (A = permutedims(A)) :
        throw(DimensionMismatch("A is $(size(A, 1)) x $(size(A, 2)), b has $m entries, c $n")))
    nf, nl, q, r, s = _kcount(K, :f), _kcount(K, :l), _klist(K, :q), _klist(K, :r), _klist(K, :s)
    nf + nl + sum(q; init = 0) + sum(r; init = 0) + sum(d -> Int(d)^2, s; init = 0) == n ||
        throw(DimensionMismatch("K does not match the number of columns of A"))
    Ap = Cint[0]; Ai = Cint[]; Ax = Cdouble[]
    if A isa SparseArrays.SparseMatrixCSC
        Ap = Vector{Cint}(A.colptr .- 1); Ai = Vector{Cint}(A.rowval .- 1); Ax = Vector{Cdouble}(A.nzval)
    else
        for j in 1:n
            for i in 1:m
                A[i, j] != 0 && (push!(Ai, i - 1); push!(Ax, A[i, j]))
            end
            push!(Ap, length(Ai))
        end
    end
    return _solve_sedumi(m, n, Ap, Ai, Ax, Vector{Cdouble}(b), Vector{Cdouble}(c), nf, nl, q, r, s,
                         _tagged(option_args(options), "julia:solve_sedumi"), output)
end

function _solve_sedumi(m, n, Ap::Vector{Cint}, Ai::Vector{Cint}, Ax::Vector{Cdouble}, b::Vector{Cdouble}, c::Vector{Cdouble},
                       nf, nl, q::Vector{Cint}, r::Vector{Cint}, s::Vector{Cint}, args::Vector{String}, output::Symbol)
    return _call(args, output, _read_cone_result) do res, cargs
        Int(ccall(_fn(:brisk_solve_sedumi), Cint,
                  (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint,
                   Cint, Ptr{Cint}, Cint, Ptr{Cint}, Cint, Ptr{Cint}, Cint, Ptr{Cstring}, Ptr{Cvoid}),
                  m, n, Ap, Ai, Ax, b, c, nf, nl, length(q), q, length(r), r, length(s), s, length(cargs), cargs, res))
    end
end

function solve_sedumi(file::AbstractString; output::Symbol = :julia, options...)
    isfile(file) || throw(ArgumentError("no such file: $file"))
    args = _tagged(option_args(options), "julia:solve_sedumi")
    return _call(args, output, _read_cone_result) do r, cargs
        Int(ccall(_fn(:brisk_solve_file), Cint, (Cstring, Cint, Ptr{Cstring}, Ptr{Cvoid}),
                  file, length(cargs), cargs, r))
    end
end

include("MOI_wrapper.jl")

end # module
