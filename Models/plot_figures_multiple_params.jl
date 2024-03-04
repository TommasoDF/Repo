include("models.jl")
using JLD2, JSON

# Set the default font
default(fontfamily = "Helvetica", tickfont = font(24), legendfont = font(24), guidefont = font(24))

function speculations_effect()
    lw = 4
    #Compute the mean and variance of the trajectories
    mean_b = [mean(y_b[i]) for i in eachindex(y_b)]
    var_b = [var(y_b[i]) for i in eachindex(y_b)]
    mean_f = [mean(y_f[i]) for i in eachindex(y_f)]
    var_f = [var(y_f[i]) for i in eachindex(y_f)]
    mean_r = [mean(y_r[i]) for i in eachindex(y_r)]
    var_r = [var(y_r[i]) for i in eachindex(y_r)]
    #Plot the mean and variance
    A = plot(x_b, mean_b, label = "Two types", linewidth = lw, xlabel = L"\beta", ylabel = "Mean", legend = :topright, size = (800, 600))
    plot!(x_b, mean_f, label = "Fundamentalist", linewidth = lw)
    plot!(x_b, mean_r, label = "Rational", linewidth = lw)
    B = plot(x_b, var_b, label = "Two types", linewidth = lw, xlabel = L"\beta", ylabel = "Variance", legend = :topright, size = (800, 600))
    plot!(x_b, var_f, label = "Fundamentalist", linewidth = lw)
    plot!(x_b, var_r, label = "Rational", linewidth = lw)
    savefig(A, "Figures/mean_comparison.png")
    savefig(B, "Figures/variance_comparison.png")
end

# import the parameters
dict = JSON.parsefile("params.json")
beta = dict["beta"]
g =  dict["g"]
b =  dict["b"]
R =  dict["R"]
parameters = [beta, g, b, R]

# Load the data
x_b, y_b = load_object("Data/bifurcation_two_types.jld2")
x_f, y_f = load_object("Data/bifurcation_fundamentalist.jld2")
x_r, y_r = load_object("Data/bifurcation_rational.jld2")
ds_b = load_object("Data/ds_two_types.jld2")
ds_f = load_object("Data/ds_fundamentalist.jld2")

#plot bifurcation
scatter_bifurcation(x_b, y_b, savefile = "Figures/bifurcation_two_types.png")
scatter_bifurcation(x_f, y_f, savefile = "Figures/bifurcation_fundamentalist.png")
scatter_bifurcation(x_r, y_r, savefile = "Figures/bifurcation_rational.png")

#plot phase_diagram
phase_plots(x_b, y_b, savefile = "Figures/phase_diagram_two_types.png")
phase_plots(x_f, y_f, savefile = "Figures/phase_diagram_fundamentalist.png")
phase_plots(x_r, y_r, savefile = "Figures/phase_diagram_rational.png")

#plot the eigenvalues
eigenvalues_plot(parameters, x_b, savefile = "Figures/eigenvalues_two_types.png")
eigenvalues_plot(parameters, x_f, savefile = "Figures/eigenvalues_fundamentalist.png")

#Lyapunov exponent
plot_lyapunov(ds_b, x_b, savefile = "Figures/lyapunov_two_types.png")
plot_lyapunov(ds_f, x_f, savefile = "Figures/lyapunov_fundamentalist.png")

#compare the mean and var over the 3 models
speculations_effect()