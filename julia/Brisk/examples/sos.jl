# An SOS program through SumOfSquares.jl (needs SumOfSquares and DynamicPolynomials in the
# environment): the global minimum of the Motzkin-like polynomial p + gamma certificate.
#   julia --project=<env> sos.jl
using SumOfSquares, DynamicPolynomials, Brisk
@polyvar x y
p = x^4 * y^2 + x^2 * y^4 - 3x^2 * y^2 + 1           # Motzkin: nonnegative, not SOS
model = SOSModel(Brisk.Optimizer)
set_silent(model)
@variable(model, γ)
@constraint(model, (x^2 + y^2 + 1) * (p - γ) >= 0)  # SOS after multiplying by x^2 + y^2 + 1
@objective(model, Max, γ)
optimize!(model)
println(termination_status(model), ": γ = ", objective_value(model), " (the minimum is 0)")
println("DIMACS errors: ", MOI.get(model, Brisk.DIMACSErrors()))
