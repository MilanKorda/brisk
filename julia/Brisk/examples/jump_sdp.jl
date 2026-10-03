# Small JuMP models solved with BRISK: julia --project=<env> jump_sdp.jl
using JuMP, Brisk, LinearAlgebra

# 1. lambda_max(C) = max <C,X> s.t. tr X = 1, X PSD (X is a PSD variable: BRISK's dual form
#    takes it back to one row)
C = [2.0 1 0; 1 3 1; 0 1 1]
model = Model(Brisk.Optimizer)
@variable(model, X[1:3, 1:3], PSD)
@constraint(model, tr(X) == 1)
@objective(model, Max, dot(C, X))
optimize!(model)
println(solution_summary(model))
println("lambda_max = ", eigmax(C), ", BRISK: ", objective_value(model))

# 2. a moment relaxation: min_x x^4 - 3x^2 + x over R, order 2 (free moments in an LMI,
#    the form BRISK takes as it is)
model = Model(Brisk.Optimizer)
set_silent(model)
@variable(model, y[1:4])                       # y_k = E[x^k], y_0 = 1
M = [1 y[1] y[2]; y[1] y[2] y[3]; y[2] y[3] y[4]]
@constraint(model, M in PSDCone())
@objective(model, Min, y[4] - 3y[2] + y[1])
optimize!(model)
println("moment bound: ", objective_value(model), " (the minimum of x^4 - 3x^2 + x is ",
        minimum(x -> x^4 - 3x^2 + x, range(-3, 3; length = 600001)), ")")
println("DIMACS errors: ", MOI.get(model, Brisk.DIMACSErrors()))

# 3. options and the SDPA file of a model
model = Model(Brisk.Optimizer)
set_attribute(model, "acc", "high")            # any BRISK option: -acc high
set_attribute(model, "output", :silent)
set_attribute(model, "write_sdpa", "lmax.dat-s")
@variable(model, Z[1:3, 1:3], PSD)
@constraint(model, tr(Z) == 1)
@objective(model, Max, dot(C, Z))
optimize!(model)
println("acc high: ", objective_value(model), "; the model as an SDPA file: lmax.dat-s")
r = Brisk.solve_sdpa("lmax.dat-s"; output = :silent)
println(r)
