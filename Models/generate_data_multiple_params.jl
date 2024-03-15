using JLD2, JSON
include("models.jl")



# Generate bifurcation data for the rational speculators model
function generate_bifurcation_data_rational()
    initial_conditions = [1.3, 1.2, 1.1]
    time_series_list, error_list = compute_bifurcation_ft(beta_range, parameters, initial_conditions, generate_dynamics_rational_speculators, N= 10)
    data = [beta_range, time_series_list, error_list]
    save_object("Data/bifurcation_rational.jld2", data)
end

# Generate bifurcation data for the two types model
function generate_bifurcation_data_two_types()
    initial_conditions = [1.3, 1.2, 1.1, 0.5, 0.5]
    ds = DeterministicIteratedMap(trend_plus_bias, initial_conditions, parameters)
    y = computeorbit(ds, beta_range)
    data = [beta_range, y]
    save_object("Data/bifurcation_two_types.jld2", data)
    save_object("Data/ds_two_types.jld2", ds)
end

# Generate bifurcation data for the fundamentalist model
function generate_bifurcation_data_fundamentalist()
    initial_conditions = [1.3, 1.2, 1.1, 0.3, 0.3, 0.4]
    ds = DeterministicIteratedMap(trend_plus_bias_plus_fundamentalist, initial_conditions, parameters)
    y = computeorbit(ds, beta_range)
    data = [beta_range, y]
    save_object("Data/bifurcation_fundamentalist.jld2", data)
    save_object("Data/ds_fundamentalist.jld2", ds)
end

# import the parameters
dict = JSON.parsefile("params.json")
beta_range = range(dict["beta_low"], dict["beta_high"], length = dict["bifurcation_L"])
beta = dict["beta"]
g =  dict["g"]
b =  dict["b"]
R =  dict["R"]
parameters = [beta, g, b, R]

#Bifurcation data
generate_bifurcation_data_two_types()
generate_bifurcation_data_fundamentalist()
generate_bifurcation_data_rational()