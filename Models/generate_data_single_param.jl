using JLD2, JSON, CSV, DataFrames
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
    save_object("Data/X_rational_unstable.jld2", X)
end

function generate_data_simulated(noise; filename = "Data/simulated_data.csv")
    initial_conditions = [1.3, 1.2, 1.1]
    expectations = fair_taylor_iteration(generate_dynamics_rational_speculators, parameters, initial_conditions, 100, 10, T)
    X = generate_dynamics_rational_speculators(expectations, parameters, initial_conditions, noise = noise)
    data = DataFrame(actual = X[1])
    # add expectations to the data
    data[!, :expectations] = expectations
    # save data to a CSV file in the Data folder
    CSV.write(filename, data)
    return data

end

# import the parameters
dict = JSON.parsefile("params.json")
beta = dict["beta"]
g =  dict["g"]
b =  dict["b"]
R =  dict["R"]
parameters = [beta, g, b, R]
T = dict["T"]

#generate_data_two_types()
#generate_data_fundamentalist()

data = generate_data_simulated(0.0, filename = "Data/simulated_data_b7n0.csv")
#Compute the mean of the squared of the column actual of data
mean_squared = mean(data[!, :actual].^2)
print(mean_squared)

data100 = generate_data_simulated(sqrt(mean_squared/100), filename = "Data/simulated_data_b7n100.csv")
data10 = generate_data_simulated(sqrt(mean_squared/10), filename = "Data/simulated_data_b7n10.csv")