include("models.jl")
using JLD2
default(fontfamily = "Computer Modern", framestyle = :box)

# Load the data
X_b = load_object("Data/X_two_types.jld2")
X_f = load_object("Data/X_fundamentalist.jld2")
X_r = load_object("Data/X_rational.jld2")
plot_evolution_and_fraction(X_b, false; savefile = "Figures/evolution_and_fraction_two_types.png")
plot_evolution_and_fraction(X_f, false; savefile = "Figures/evolution_and_fraction_fundamentalist.png")
plot_evolution_and_fraction(X_r, true; savefile = "Figures/evolution_and_fraction_rational.png")

