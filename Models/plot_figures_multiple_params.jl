include("models.jl")
using JLD2, JSON, Colors

# Define colors
c1 = RGB(10/255, 9/255, 8/255)
c2 = RGB(34/255, 51/255, 59/255)
c3 = RGB(234/255, 224/255, 213/255)
c4 = RGB(198/255, 172/255, 143/255)
c5 = RGB(94/255, 80/255, 63/255)

function plot_error()
    plot(x_r, error_r, linewidth = 2, xlabel = L"\beta", color = c2, label = "")
    savefig("Figures/error.pdf")
end

function speculations_effect()
    lw = 2
    #Compute the mean and variance of the trajectories
    mean_b = [mean(y_b[i]) for i in eachindex(y_b)]
    var_b = [var(y_b[i]) for i in eachindex(y_b)]
    mean_f = [mean(y_f[i]) for i in eachindex(y_f)]
    var_f = [var(y_f[i]) for i in eachindex(y_f)]
    mean_r = [mean(y_r[i]) for i in eachindex(y_r)]
    var_r = [var(y_r[i]) for i in eachindex(y_r)]
    #Plot the mean and variance
    A = plot(x_b, mean_b, label = "Two types", linewidth = lw, xlabel = L"\beta", 
    legend = :topright, linestyle=:solid, color = c1, ylims = (0, 2))
    # add shaded area for one and minus one standard deviation
    plot!(x_b, mean_b, ribbon = sqrt.(var_b), fillalpha = 0.2, color = c1, label = "")
    plot!(x_f, mean_f, label = "Fundamentalist", linewidth = lw, color = c3, marker = :circle)
    plot!(x_f, mean_f, ribbon = + sqrt.(var_f), color = c3, fillalpha = 0.2, label = "")
    plot!(x_r, mean_r, label = "Rational", linewidth = lw, color = c5, marker = :square)
    plot!(x_r, mean_r, ribbon =  sqrt.(var_r), color = c5, fillalpha = 0.2, label = "")
    savefig(A, "Figures/mean.pdf")
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
x_r, y_r, error_r = load_object("Data/bifurcation_rational.jld2")
ds_b = load_object("Data/ds_two_types.jld2")
ds_f = load_object("Data/ds_fundamentalist.jld2")


# # #plot bifurcation
#gr(size=(400, 400))
# scatter_bifurcation(x_b, y_b, savefile = "Figures/bifurcation_two_types.pdf")
# scatter_bifurcation(x_f, y_f, savefile = "Figures/bifurcation_fundamentalist.pdf")
# scatter_bifurcation(x_r, y_r, savefile = "Figures/bifurcation_rational.pdf")

# #plot phase_diagram
# gr(size=(300, 300))
# phase_plots(x_b, y_b, savefile = "Figures/phase_diagram_two_types.pdf")
# phase_plots(x_f, y_f, savefile = "Figures/phase_diagram_fundamentalist.pdf")
# phase_plots(x_r, y_r, savefile = "Figures/phase_diagram_rational.pdf")

# #plot the eigenvalues
# eigenvalues_plot(parameters, x_b, model = "trend_plus_bias", savefile = "Figures/eigenvalues_two_types.pdf")
# eigenvalues_plot(parameters, x_f, savefile = "Figures/eigenvalues_fundamentalist.pdf")

# #Lyapunov exponent
# plot_lyapunov(ds_b, x_b, savefile = "Figures/lyapunov_two_types.pdf")
# plot_lyapunov(ds_f, x_f, savefile = "Figures/lyapunov_fundamentalist.pdf")

#compare the mean and var over the 3 models
#speculations_effect()
plot_error()