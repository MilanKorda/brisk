# Tests of Brisk.jl: the low-level interface against the SDPA-file path, the MOI wrapper on
# small models with known answers, and MathOptInterface's own test suite.
#
#   julia --project=<env with Brisk developed> -e 'using Pkg; Pkg.test("Brisk")'
#   BRISK_TEST_MOI=0 skips the MOI suite (about 10 minutes on 2 cores).
using Brisk, JuMP, LinearAlgebra, Test
import MathOptInterface as MOI

# theta3: the package's examples/ directory, else the development tree's SDPLIB copy
const SDPLIB = get(ENV, "BRISK_SDPLIB",
    let ex = joinpath(@__DIR__, "..", "..", "..", "examples")
        isfile(joinpath(ex, "theta3.dat-s")) ? ex :
            joinpath(@__DIR__, "..", "..", "..", "..", "inst", "brisk-instances-4.13", "sdplib")
    end)

# the numbers of an SDPA file (for the in-memory path)
function read_sdpa_numbers(file)
    lines = [l for l in eachline(file) if !startswith(l, "\"") && !startswith(l, "*")]
    tok(l) = split(replace(l, r"[,{}()]" => " "))
    m = parse(Int, tok(lines[1])[1]); nb = parse(Int, tok(lines[2])[1])
    bs = parse.(Int, tok(lines[3]))[1:nb]
    c = parse.(Float64, tok(lines[4]))[1:m]
    mat, blk, ii, jj, v = Int[], Int[], Int[], Int[], Float64[]
    for l in lines[5:end]
        t = tok(l)
        length(t) < 5 && continue
        push!(mat, parse(Int, t[1])); push!(blk, parse(Int, t[2])); push!(ii, parse(Int, t[3])); push!(jj, parse(Int, t[4]))
        push!(v, parse(Float64, t[5]))
    end
    return m, bs, c, mat, blk, ii, jj, v
end

# the models of the JuMP tests, in each SDPA form the wrapper can build
const FORMS = ("auto", "image", "kernel")
function brisk_model(form)
    model = Model(Brisk.Optimizer)
    set_silent(model)
    set_attribute(model, "form", form)
    return model
end

@testset "Brisk.jl" begin

@testset "high precision" begin
    f = joinpath(@__DIR__, "..", "..", "..", "examples", "truss1.dat-s")
    if isfile(f)
        r = Brisk.solve_sdpa(f; prec = "qd", output = :silent)
        @test r.status == 0
        @test maximum(abs, r.dimacs) < 1e-45
        h = Brisk.hp_solution()
        @test h.digits >= 63 && length(h.X) == length(h.blocksizes)
        @test abs(h.pobj - h.dobj) < big"1e-43" * (1 + abs(h.dobj))
        @test h.y isa Vector{BigFloat} && size(h.X[1]) == (h.blocksizes[1], h.blocksizes[1])
        r2 = Brisk.solve_sdpa(f; prec = "dd", lralm = 1, output = :silent)
        @test r2.status == 0 && abs(r2.dual_objective - r.dual_objective) < 1e-12
        @test Brisk.hp_solution().bound === nothing
        # ex-post certification in high precision: rigorous bounds that enclose the optimum
        rd = Brisk.solve_sdpa(f; prec = "dd", bound = "d", output = :silent)
        hd = Brisk.hp_solution()
        rp = Brisk.solve_sdpa(f; prec = "dd", bound = "p", output = :silent)
        hp = Brisk.hp_solution()
        @test hd.bound isa BigFloat && hp.bound isa BigFloat
        @test hd.bound <= hp.bound && hp.bound - hd.bound < big"1e-20" * (1 + abs(hd.bound))
        @test hd.digits >= 90
        Brisk.solve_sdpa(f; output = :silent)
        @test Brisk.hp_solution() === nothing
    end
end

@testset "low level" begin
    @test Brisk.version() isa String
    @test isfile(Brisk.library_path())
    @test Brisk.option_args(["acc" => "high", "nohsd" => true, "q" => false, "-maxit" => 50, "tol" => 1e-9]) ==
          ["-acc", "high", "-nohsd", "-maxit", "50", "-tol", "1.0e-9"]
    f = joinpath(SDPLIB, "theta3.dat-s")
    if isfile(f)
        r1 = Brisk.solve_sdpa(f; output = :silent)
        @test r1.status == 0
        @test -r1.primal_objective ≈ 42.16698 atol = 1e-4       # SDPLIB: 4.216698e+01
        r2 = Brisk.solve_sdpa_data(read_sdpa_numbers(f)...; output = :silent)
        @test r2.y == r1.y && r2.primal_objective == r1.primal_objective   # the same pipeline
        @test r2.X == r1.X && r2.Z == r1.Z
        @test maximum(abs, r2.dimacs) < 1e-7
        tmp = tempname() * ".dat-s"
        Brisk.write_sdpa(tmp, read_sdpa_numbers(f)...)
        r3 = Brisk.solve_sdpa(tmp; output = :silent)
        @test r3.y == r1.y
        rm(tmp)
        # options reach the solver
        r4 = Brisk.solve_sdpa(f; output = :silent, maxit = 3)
        @test r4.status == 3                                      # ITERATION LIMIT (the retry has its own 3)
        # output through Julia's stdout
        out = mktemp() do path, io
            redirect_stdout(io) do
                Brisk.solve_sdpa(f; output = :julia)
            end
            flush(io)
            read(path, String)
        end
        @test occursin("status: OPTIMAL", out)
    else
        @warn "SDPLIB not found at $SDPLIB (set BRISK_SDPLIB): file tests skipped"
    end
    # errors: bad data, bad option
    @test_throws Brisk.BriskError Brisk.solve_sdpa_data(1, [2], [1.0], [1], [1], [3], [1], [1.0]; output = :silent)
    @test_throws Brisk.BriskError Brisk.solve_sdpa_data(1, [2], [1.0], [1], [1], [1], [1], [1.0]; output = :silent, nosuchoption = 1)
end

@testset "JuMP models, form = $form" for form in FORMS
    # lambda_max as an SDP over a PSD variable, a constant in the objective
    C = [2.0 1 0; 1 3 1; 0 1 1]
    model = brisk_model(form)
    @variable(model, X[1:3, 1:3], PSD)
    c1 = @constraint(model, tr(X) == 1)
    @objective(model, Max, dot(C, X) + 5)
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test primal_status(model) == FEASIBLE_POINT && dual_status(model) == FEASIBLE_POINT
    @test objective_value(model) ≈ eigmax(C) + 5 atol = 1e-6
    @test dual_objective_value(model) ≈ eigmax(C) + 5 atol = 1e-6
    @test dual(c1) ≈ -eigmax(C) atol = 1e-6                    # JuMP's sign convention for Max
    @test tr(value.(X)) ≈ 1 atol = 1e-7
    @test Brisk.sdpa_data(unsafe_backend(model)).form == (form == "image" ? :image : :kernel)
    @test MOI.get(model, Brisk.DIMACSErrors()) isa NTuple{6,Float64}
    @test MOI.get(model, Brisk.RawResult()) isa Brisk.Result
    @test solver_name(model) == "BRISK"

    # the moment form: free variables in an LMI; min y2 - 2 y1 over y2 >= y1^2: y = (1, 1)
    model = brisk_model(form)
    @variable(model, y[1:2])
    M = [1 y[1]; y[1] y[2]]
    lmi = @constraint(model, M in PSDCone())
    @objective(model, Min, y[2] - 2y[1])
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test value.(y) ≈ [1, 1] atol = 1e-3                      # a quadratic face: y to sqrt(tol)
    @test objective_value(model) ≈ -1 atol = 1e-6
    D = dual(lmi)                                              # the dual matrix: PSD, <D, M> = 0
    @test eigmin(Symmetric(D)) > -1e-7
    @test abs(dot(D, value.(M))) < 1e-6
    @test D ≈ [1 -1; -1 1] atol = 1e-4
    form == "auto" && @test Brisk.sdpa_data(unsafe_backend(model)).form == :image

    # a mixed model: a PSD variable, a free variable in an LMI, equalities and bounds
    model = brisk_model(form)
    @variable(model, P[1:2, 1:2], PSD)
    @variable(model, t)
    @variable(model, 0 <= u <= 3)
    @constraint(model, P[1, 1] + P[2, 2] == 2)
    @constraint(model, [t P[1, 2]; P[1, 2] 1] in PSDCone())    # t >= P12^2
    @constraint(model, u >= P[1, 2])
    @objective(model, Min, t - 2P[1, 2] + u)                   # p^2 - 2p + max(p, 0): p = 1/2, -1/4
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test objective_value(model) ≈ -0.25 atol = 1e-6
    @test value(P[1, 2]) ≈ 0.5 atol = 1e-3
    @test dual_objective_value(model) ≈ -0.25 atol = 1e-6

    # SOC and bounds through MOI's bridges
    model = brisk_model(form)
    @variable(model, t); @variable(model, x[1:2] >= 0.5)
    @constraint(model, [t; x .- [1, 2]] in SecondOrderCone())
    @objective(model, Min, t)
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test objective_value(model) ≈ 0 atol = 1e-6
    @test value.(x) ≈ [1, 2] atol = 1e-4

    # infeasible: x >= 1 and x <= -1 in an LMI
    model = brisk_model(form)
    @variable(model, x)
    @constraint(model, [x - 1 0; 0 -1 - x] in PSDCone())
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == INFEASIBLE
    @test dual_status(model) == INFEASIBILITY_CERTIFICATE

    # infeasible over a PSD variable: tr X = -1
    model = brisk_model(form)
    @variable(model, X[1:2, 1:2], PSD)
    @constraint(model, X[1, 1] + X[2, 2] == -1)
    @objective(model, Min, X[1, 2])
    optimize!(model)
    @test termination_status(model) == INFEASIBLE
    @test dual_status(model) == INFEASIBILITY_CERTIFICATE

    # unbounded: min -x s.t. [1 0; 0 x] PSD
    model = brisk_model(form)
    @variable(model, x)
    @constraint(model, [1 0; 0 x] in PSDCone())
    @objective(model, Min, -x)
    optimize!(model)
    @test termination_status(model) == DUAL_INFEASIBLE
    @test primal_status(model) == INFEASIBILITY_CERTIFICATE
    @test value(x) > 0

    # unbounded over a PSD variable: max X12 s.t. X11 = X22 (the ray [1 1; 1 1])
    model = brisk_model(form)
    @variable(model, X[1:2, 1:2], PSD)
    @constraint(model, X[1, 1] == X[2, 2])
    @objective(model, Max, X[1, 2])
    optimize!(model)
    @test termination_status(model) == DUAL_INFEASIBLE
    @test primal_status(model) == INFEASIBILITY_CERTIFICATE
    d = value.(X)
    @test d[1, 2] > 0.5 && eigmin(Symmetric(d)) > -1e-6 && abs(d[1, 1] - d[2, 2]) < 1e-6

    # an objective variable in no constraint: unbounded
    model = brisk_model(form)
    @variable(model, x); @variable(model, z)
    @constraint(model, [x 0; 0 1] in PSDCone())
    @objective(model, Min, x + z)
    optimize!(model)
    @test termination_status(model) == DUAL_INFEASIBLE
end

@testset "options" begin
    model = Model(Brisk.Optimizer); set_silent(model)
    set_attribute(model, "acc", "high")
    set_time_limit_sec(model, 100)
    MOI.set(model, MOI.NumberOfThreads(), 1)
    tmp = tempname() * ".dat-s"
    set_attribute(model, "write_sdpa", tmp)
    @variable(model, X[1:2, 1:2], PSD)
    @constraint(model, X[1, 1] + X[2, 2] == 2)
    @objective(model, Max, 2X[1, 2])
    optimize!(model)
    @test objective_value(model) ≈ 2 atol = 1e-8
    @test isfile(tmp)
    r = Brisk.solve_sdpa(tmp; output = :silent, acc = "high", threads = 1)   # the same problem, from the file
    @test abs(r.primal_objective) ≈ 2 atol = 1e-7
    @test r.y == MOI.get(model, Brisk.RawResult()).y
    rm(tmp)
    @test MOI.get(model, Brisk.RawResult()).bound === nothing
    # bound mode and the rigorous certifier through the low-level call and the model
    tmp2 = tempname() * ".dat-s"
    write(tmp2, "1\n1\n2\n1.0\n0 1 1 2 1.0\n1 1 1 1 1.0\n1 1 2 2 1.0\n")   # max 2 X12 s.t. tr X = 1: value 1
    rb = Brisk.solve_sdpa(tmp2; output = :silent, threads = 1, bound = "d", certify = true)
    @test rb.bound !== nothing && rb.bound.side == :d && rb.bound.valid && rb.bound.certified
    @test rb.bound.rigorous <= -1 + 1e-9 && rb.bound.rigorous >= -1 - 1e-6   # (P) min -2 X12 = -1: a lower bound
    rn = Brisk.solve_sdpa(tmp2; output = :silent, threads = 1, bound = "d")
    @test rn.bound !== nothing && !rn.bound.certified && isnan(rn.bound.rigorous)   # certify is off by default
    rm(tmp2)
    set_attribute(model, "bound", "dual")
    set_attribute(model, "certify", true)
    optimize!(model)
    @test objective_value(model) ≈ 2 atol = 1e-6
    bb = MOI.get(model, Brisk.RawResult()).bound
    @test bb !== nothing && bb.certified
    @test_throws ArgumentError set_attribute(model, "form", "other")
    @test_throws ArgumentError set_attribute(model, "output", :nowhere)
end

@testset "options reach the solver" begin
    @test Brisk.option_args(["certify_y" => "y.txt", "nohsd" => 1]) == ["-certify-y", "y.txt", "-nohsd", "1"]
    # a max-cut-type SDP on a path (tridiagonal, chordal pattern): chordal decomposition auto / off / forced
    n = 120
    mat = Int[]; blk = Int[]; ii = Int[]; jj = Int[]; vv = Float64[]
    for k in 1:n
        push!(mat, 0); push!(blk, 1); push!(ii, k); push!(jj, k); push!(vv, -0.25 * (k in (1, n) ? 1 : 2))
        k < n && (push!(mat, 0); push!(blk, 1); push!(ii, k); push!(jj, k + 1); push!(vv, 0.25))
    end
    for k in 1:n
        push!(mat, k); push!(blk, 1); push!(ii, k); push!(jj, k); push!(vv, 1.0)
    end
    run(c) = mktemp() do path, io
        r = redirect_stdout(io) do
            Brisk.solve_sdpa_data(n, [n], ones(n), mat, blk, ii, jj, vv; output = :julia, chordal = c, threads = 1)
        end
        flush(io)
        (r, read(path, String))
    end
    for (c, conv) in ((-1, true), (0, false), (1, true))
        r, out = run(c)
        @test r.status == 0
        @test occursin("chordal decomposition", out) == conv
    end
    # returnx = 0 on a chordal problem: no X (verified on the cliques); 1: X returned
    rx0 = Brisk.solve_sdpa_data(n, [n], ones(n), mat, blk, ii, jj, vv; output = :silent, chordal = 1, returnx = 0, threads = 1)
    rx1 = Brisk.solve_sdpa_data(n, [n], ones(n), mat, blk, ii, jj, vv; output = :silent, chordal = 1, returnx = 1, threads = 1)
    @test rx0.status == 0 && rx0.X === nothing
    @test rx1.status == 0 && size(rx1.X[1]) == (n, n)
    @test isapprox(rx0.primal_objective, rx1.primal_objective; atol = 1e-6)
    # flags with 0/1 and further options through the model
    model = Model(Brisk.Optimizer); set_silent(model)
    for (k, v) in ("nohsd" => 1, "sym" => "none", "chordal" => 0, "dir" => "nt", "nofr" => true, "dualize" => 0)
        set_attribute(model, k, v)
    end
    @variable(model, X[1:3, 1:3], PSD)
    @constraint(model, tr(X) == 1)
    @objective(model, Max, dot([2.0 1 0; 1 3 1; 0 1 1], X))
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    # the *-algebra symmetry reduction on a hidden symmetry: on by default, off with symalg = 0
    hs = joinpath(@__DIR__, "..", "..", "..", "examples", "hidden_symmetry23.dat-s")
    runf(; o...) = mktemp() do path, io
        r = redirect_stdout(io) do
            Brisk.solve_sdpa(hs; output = :julia, threads = 1, o...)
        end
        flush(io)
        (r, read(path, String))
    end
    r1, o1 = runf()
    r0, o0 = runf(symalg = 0)
    @test r1.status == 0 && r0.status == 0
    @test occursin("algebra symmetry: block 1: 23 -> 5 3(x2) 4(x3)", o1) && !occursin("algebra symmetry", o0)
    @test isapprox(r1.primal_objective, r0.primal_objective; rtol = 1e-7)
    set_attribute(model, "nosuchoption", 1)
    @test_throws Exception optimize!(model)
end

if get(ENV, "BRISK_TEST_MOI", "1") != "0"
    function moi_model(acc)
        inner = Brisk.Optimizer()
        MOI.set(inner, MOI.Silent(), true)
        MOI.set(inner, MOI.RawOptimizerAttribute("acc"), acc)
        cached = MOI.Utilities.CachingOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), inner)
        return MOI.Bridges.full_bridge_optimizer(cached, Float64)
    end
    config = MOI.Test.Config(atol = 1e-4, rtol = 1e-4,
                             exclude = Any[MOI.ConstraintBasisStatus, MOI.VariableBasisStatus, MOI.ObjectiveBound])
    # On the curved faces of these bridged SOC problems the duals are accurate to about
    # sqrt(tol): 1e-4 at the default tol 1e-8, which is the suite's tolerance. They run with
    # acc = "high" (tol 1e-10); at that level some other bridged problems end at
    # SOLVED TO REDUCED ACCURACY (ALMOST_OPTIMAL), so the rest runs at the default.
    curved = ["test_conic_SecondOrderCone_Nonnegatives", "test_conic_SecondOrderCone_Nonpositives",
              "test_conic_RotatedSecondOrderCone_INFEASIBLE_2"]
    @testset "MOI.Test" begin
        MOI.Test.runtests(moi_model("default"), config; exclude = curved)
    end
    @testset "MOI.Test, acc = high" begin
        MOI.Test.runtests(moi_model("high"), config; include = curved)
    end
end

end # testset
