using JLD2, JSON
include("models.jl")

### Generate data for the two type model
function generate_data_two_types()
    # Initial conditions
    initial_conditions = [1.3, 1.2, 1.1, 0.5, 0.5]
    # Generate the dynamical system
    ds = DeterministicIteratedMap(trend_plus_bias, initial_conditions, parameters)
    # Generate a trajectory, that is a simulation of the model
    X, t = trajectory(ds, T)
    # save X to a file in the Data folder
    save_object("Data/X_two_types.jld2", X)
end

function generate_data_fundamentalist()
    # Initial conditions
    initial_conditions = [1.3, 1.2, 1.1, 0.3, 0.3, 0.4]
    # Generate the dynamical system
    ds = DeterministicIteratedMap(trend_plus_bias_plus_fundamentalist, initial_conditions, parameters)
    # Generate a trajectory, that is a simulation of the model
    X, t = trajectory(ds, T)
    # save X to a file in the Data folder
    save_object("Data/X_fundamentalist.jld2", X)
end

function generate_data_rational()
    initial_conditions = [1.3, 1.2, 1.1]
    expectations = fair_taylor_iteration(generate_dynamics_rational_speculators, parameters, initial_conditions, 100, 10, T)
    X = generate_dynamics_rational_speculators(expectations, parameters, initial_conditions)
    save_object("Data/X_rational.jld2", X)
end

# import the parameters
dict = JSON.parsefile("params.json")
beta = dict["beta"]
g =  dict["g"]
b =  dict["b"]
R =  dict["R"]
parameters = [beta, g, b, R]
T = dict["T"]

generate_data_two_types()
generate_data_fundamentalist()
generate_data_rational()
