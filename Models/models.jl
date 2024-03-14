using Random, Statistics, DynamicalSystems, LaTeXStrings, Plots, NLsolve

#### Functions to generate the three models
function trend_plus_bias(states, params, n) # here `n` is "time", but we don't use it.
    p_t1, p_t2, p_t3, n1_0, n2_0 = states # system states
    β, g, b, R = params # system parameters
    prof1 = (p_t1 - R*p_t2)*(g*p_t3 - R*p_t2)
    prof2 = (p_t1 - R*p_t2)*(b -R*p_t2)
    num1 = exp(β*(prof1))
    num2 = exp(β*(prof2))
    den = num1 + num2
    frac_1 = num1/den
    frac_2 = num2/den
    nextstate_1 = (frac_1 * g * p_t1 + frac_2 * b)/(R)	#Next pₜ
    nextstate_2 = p_t1 #Next pₜ_₁
    nextstate_3 = p_t2 #Next pₜ_2
    nextstate_4 = frac_1
    nextstate_5 = frac_2
    return SVector(nextstate_1, nextstate_2, nextstate_3, nextstate_4, nextstate_5)
end

function trend_plus_bias_plus_fundamentalist(states, params, n) # here `n` is "time", but we don't use it.
    p_t1, p_t2, p_t3, n1_0, n2_0, n3_0 = states # system states
    β, g, b, R = params # system parameters
    prof1 = (p_t1 - R*p_t2)*(g*p_t3 - R*p_t2)
    prof2 = (p_t1 - R*p_t2)*(b -R*p_t2)
    prof3 = (p_t1 - R*p_t2)*(0 -R*p_t2)
    num1 = exp(β*(prof1))
    num2 = exp(β*(prof2))
    num3 = exp(β*(prof3))
    den = num1 + num2 + num3
    frac_1 = num1/den
    frac_2 = num2/den
    frac_3 = num3/den
    nextstate_1 = (frac_1 * g * p_t1 + frac_2 * b)/(R)	#Next pₜ
    nextstate_2 = p_t1 #Next pₜ_₁
    nextstate_3 = p_t2 #Next pₜ_2
    nextstate_4 = frac_1
    nextstate_5 = frac_2
    nextstate_6 = frac_3
    return SVector(nextstate_1, nextstate_2, nextstate_3, nextstate_4, nextstate_5, nextstate_6)
end

function generate_dynamics_rational_speculators(expectations, params, inital_conditions; noise = 0)
    # Define the parameters
    β, g, b, R = params
    # Define the initial conditions
    x_minus_1, x_minus_2, x_minus_3 = inital_conditions
    # Define the lenght of the time series
    N = length(expectations)
    # Initialize empty arrays to store the dynamics
    x_array = zeros(N)
    x_minus_1_array = zeros(N+1)
    x_minus_2_array = zeros(N+1)
    x_minus_3_array = zeros(N+1)
    # add the inital_conditions to the arrays
    x_minus_1_array[1] = x_minus_1
    x_minus_2_array[1] = x_minus_2
    x_minus_3_array[1] = x_minus_3
    profits_1_array = zeros(N)
    profits_2_array = zeros(N)
    profits_3_array = zeros(N)
    fractions_1_array = zeros(N)
    fractions_2_array = zeros(N)
    fractions_3_array = zeros(N)

    # Loop over the time series
    for t in 1:N
        # Define the current expectations
        exp_plus_1 = expectations[t]
        # Define the current profits
        profits_1_array[t] = (x_minus_1_array[t] - R * x_minus_2_array[t]) * (g * x_minus_3_array[t] - R * x_minus_2_array[t])
        profits_2_array[t] = (x_minus_1_array[t] - R * x_minus_2_array[t]) * (b - R * x_minus_2_array[t])
        profits_3_array[t] = (x_minus_1_array[t] - R * x_minus_2_array[t]) ^2
        # Define the current fractions
        Z = exp(β * profits_1_array[t]) + exp(β * profits_2_array[t]) + exp(β * profits_3_array[t])
        fractions_1_array[t] = exp(β * profits_1_array[t])/Z
        fractions_2_array[t] = exp(β * profits_2_array[t])/Z
        fractions_3_array[t] = exp(β * profits_3_array[t])/Z
        # Define the current state
        x_array[t] = randn()*noise + (fractions_1_array[t] * g * x_minus_1_array[t] + fractions_2_array[t] * b + fractions_3_array[t] * exp_plus_1)/R
        # Update the state
        x_minus_3_array[t+1] = x_minus_2_array[t]
        x_minus_2_array[t+1] = x_minus_1_array[t]
        x_minus_1_array[t+1] = x_array[t]
    end
    #genrate the array x_plus_1 by shifting x_array and adding a 0 as the last element
    x_plus_1_array = [x_array[2:end]; expectations[end]]
    #Remove the last element of the arrays -1, -2, -3
    x_minus_1_array = x_minus_1_array[1:end-1]
    x_minus_2_array = x_minus_2_array[1:end-1]
    x_minus_3_array = x_minus_3_array[1:end-1]
    return x_array, x_plus_1_array, x_minus_1_array, x_minus_2_array, x_minus_3_array, fractions_1_array, fractions_2_array, fractions_3_array
end

#### Functions to solve for rational expectations

function fair_taylor_iteration(model, params, initial_conditions, k, N, T; 
    tolerance = 1e-14, max_iter = 1000)
    #### Function to find a rational solution to the non-linear model using the Fair and Taylor (Econometrica 1983) algorithm
    # Define the initial expectations vector of length k+1
    expectations_old = ones(T+k) + randn(T+k)*0.1
    expectations_k_plus_1_old = ones(T+k) + randn(T+k)*0.1
    #Define the main loop over N
    for n in 1:N
        if sum(abs.(expectations_k_plus_1_old[1:T+k] - expectations_old[1:T+k]))/k < tolerance
            #exit the main loop
            break
        else
            expectations_k_plus_1_old = [expectations_k_plus_1_old; 1.0 + randn()*0.1]
        end
        #Define the inner loop over the max_iter
        for i in 1:max_iter
            # Generate the dynamics
            x_plus_1_array = model(expectations_k_plus_1_old, params, initial_conditions)[2]
            # Define the expectations vector
            expectations_new = x_plus_1_array
            # Check for nans and replace them with the old expectations plus stop the loop
            if any(isnan.(expectations_new))
                expectations_new = expectations_old
                print("Nans in the expectations vector")
                break
            end
            # Check for convergence
            if sum(abs.(expectations_new[1:T+k] - expectations_k_plus_1_old[1:T+k]))/k < tolerance
                expectations_k_plus_1_old = expectations_new
                #exit the inner loop
                break
            else
                expectations_k_plus_1_old = 0.5.*(expectations_new) .+ 0.5.*(expectations_k_plus_1_old)
            end
        end
    end
    return expectations_k_plus_1_old[1:T]
end

function compute_bifurcation_ft(param_range, params, initial_conditions,model;k = 100, N = 100, T = 2000, burnout = 100, tolerance = 1e-14, max_ite = 10000)
    # Define the inital conditions
    time_series_list = []
    error_list = []
    j = 0
    # Loop over the parameter range
    for β in param_range
        j += 1
        params[1] = β
        # Generate the expectations
        expectations = fair_taylor_iteration(model, params, initial_conditions, k, N, T, tolerance = tolerance, max_iter = max_ite)
        x_array, x_plus_1_array = model(expectations, params, initial_conditions)[1:2]
        error = sum(abs.(x_plus_1_array - expectations))/k
        push!(time_series_list, x_array[burnout:end])
        push!(error_list, error)
        if j % 10 == 0
            print("Iteration: ", j, " of ", length(param_range), " completed")
        end
    end
    return time_series_list, error_list
end

#### Functions to generate the bifurcation data
function computeorbit(
    ds, beta_range; 
    n = 1_000, 	# Number of points
    xindex = 1, 	# Index of the state variables in the dynamical system
    aindex = 1, 	# Index of the parameter in the dynamical system
    kwargs...
)
P = beta_range
L = length(P)
orbits = orbitdiagram(
    ds, xindex, aindex, P; 
    n = n, Ttr = 2000, kwargs...
)
# x = Vector{Float64}(undef, n*L) # Empty vector to store points
# y = copy(x)
# for j in 1:L
#     x[(1 + (j-1)*n):j*n] .= P[j]
#     y[(1 + (j-1)*n):j*n] .= orbits[j]
# end
return orbits
end


#### Plotting functions 
function phase_plots(x,y;
    L = 252,
    savefile = nothing
    )
    a0,a1 = x[1], x[end]
    #Get L uniformely distributed values from x and y 
    x1 = x[1:end]
    y1 = y[1:end]

    s = scatter()
    xlabel!(L"x_{t-1}")
    ylabel!(L"x_t")          
    #create a color palette, similar to viridis
    l = @layout [a{0.95w} [b{0.001h}; c]]
    cmap = cgrad(:thermal, length(x)) #cgrad(:thermal)[1:L]
    #reverse the palette
    cmap = reverse(cmap)
    
    for i in range(1,length(x))
        scatter!(y1[i][end-301:end-1], y1[i][end-300:end], markersize = 0.5, color = cmap[i],legend=false,markerstrokewidth=0)
    end
 
    p2 = plot([NaN], lims=(0,1), framestyle=:none, legend = false)
    annotate!(-0.2, -0.2, text("β", :black))
    
    p3 = heatmap(rand(2,2), clims=(a0,a1), framestyle=:none, c=cmap, cbar=true, lims=(-1,0), yflip=true)
    S = plot(s, p2, p3, layout=l)
    if savefile !== nothing
        savefig(S, savefile)
    end
    #visualize the first and last color in the palette
    return S
end

#Generate the Lyapunov Exponents plot
function plot_lyapunov(ds, beta_range; 
    n = 8_000, 	# Number of points
    aindex = 1, 	# Index of the parameter in the dynamical DynamicalSystem
    figsize = (900, 600),
    linewidth = 4,
    savefile = nothing
)   
    aspace = beta_range
    L = length(aspace)
    λs = zeros(length(aspace))
    for (i, a) in enumerate(aspace)
        set_parameter!(ds, aindex, a)
        λs[i] = lyapunov(ds, n; Ttr = 500)
    end

    lyap = plot(
        aspace, λs, 
        xlabel = "β", ylabel = "Lyapunov Exponent", label = nothing, color=:black, 
        linewidth=2, 
        linestyle=:solid, 
        marker=(:circle, 4),
        grid=true, 
        legend=:topright,
    )
    if savefile !== nothing
        savefig(lyap, savefile)
    end
    return lyap
end

function eigenvalues_plot(params, beta_range; 
    param_index = 1,
    linewidth = 2, 
    model = "fundamentalist", #Model to use
    savefile = nothing) #Name of the file to save the plot
    L = length(beta_range) # Number of points
    # Check that the model is either "fundamentalist" or "trend_plus_bias"
    if model != "fundamentalist" && model != "trend_plus_bias"
        error("The model must be either 'fundamentalist' or 'trend_plus_bias'")
    end
    P = beta_range # Parameter range
    rows = [] # Empty matrix to store the parameters
    for p in P 
        params[param_index] = p # Change the parameter we want to vary
        push!(rows, copy(params)) # Store the parameters
    end
  # Create a matrix of parameters
    if model == "fundamentalist" 
        eigenvalues = compute_eigenvalues_fundamentalist.(rows) # Compute eigenvalues for each parameter combination
    else
        eigenvalues = compute_eigenvalues_trend_plus_bias.(rows)
    end
    λ₁ = first.(eigenvalues) # Extract the first eigenvalue
    mod_λ₁ = abs.(λ₁) # Compute the modulus
    λ₂ = [t[2] for t in eigenvalues] # Extract the second eigenvalue
    mod_λ₂ = abs.(λ₂) # Compute the modulus
    λ₃ = last.(eigenvalues) # Extract the third eigenvalue
    mod_λ₃ = abs.(λ₃) # Compute the modulus
    # Plot the data
    eigen = plot(
        P, mod_λ₁, label = L"|(\lambda_1)|", linewidth = linewidth, 
        xlabel = "β", ylabel = "Modulus of eigenvalues", alpha = 0.5,
        grid=true, 
        legend=:topright,)
    plot!(P, mod_λ₂, label = L"\|(\lambda_2)|", linewidth = linewidth, alpha = 1.0)
    plot!(P, mod_λ₃, label = L"|(\lambda_3)|", linewidth = linewidth, alpha = 0.5)
    hline!([1], label = nothing, linewidth = linewidth, linestyle = :dash, linecolor = :black)
    #save the figure
    if savefile !== nothing
        savefig(eigen, savefile)
    end
    return eigen
end

function plot_evolution_and_fraction(X, rational;
    figsize = (900, 600),
    linewidth = 4,
    savefile = nothing)
    #Plot

    if rational
        A = plot(X[1][end-100:end],label = nothing, color = :black, linewidth = linewidth, ylabel = "xₜ", size = figsize,margin = 15Plots.mm, xticks = nothing)
        B = plot(X[6][end-100:end],  label = "trend following", linewidth = linewidth, xlabel = "t", ylabel = "fractions", size = figsize,margin = 15Plots.mm)
        plot!(X[7][end-100:end], label = "bias", linewidth = linewidth)
        plot!(X[8][end-100:end], label = "rational", linewidth = linewidth)
    else
        A = plot(X[:,1][end-100:end],label = nothing, color = :black, linewidth = linewidth, ylabel = "xₜ", size = figsize,margin = 15Plots.mm, xticks = nothing)
        B = plot(X[:,4][end-100:end],  label = "trend following", linewidth = linewidth, xlabel = "t", ylabel = "fractions", size = figsize,margin = 15Plots.mm)
        plot!(X[:,5][end-100:end], label = "bias", linewidth = linewidth)
        if size(X)[2] == 6
             plot!(X[:,6][end-100:end], label = "fundamentalist", linewidth = linewidth)
        end
    end
   C = plot(A,B, layout = (2,1))

   #Saving the plot into the results folder
   if savefile !== nothing
       savefig(C, savefile)
   end
   return C
end

function scatter_bifurcation(x,y;
    savefile = nothing
    )
    bifurcation = scatter(fill(x[1], length(y[1])), y[1],
    	xaxis = L"$\beta$", ylabel = L"x", 
    	ms = 0.5, color = :black, legend = nothing,
    	alpha = .050
    )
    for j in 2:length(x)
        scatter!(bifurcation, fill(x[j], length(y[j])), y[j],ms = 0.5, color = :black, legend = nothing,
    	alpha = .050)
    end

    if savefile !== nothing
        savefig(bifurcation, savefile)
    end
    return bifurcation
end

function overlay_bifurcations(x, y1, y2, y3;
color1 = :black, color2 = :black, color3 = :black,
size = (300, 200),
savefile = nothing)
# add x to itself to make it from lenght 500 to 500*1000
x = repeat(x, inner = length(y1[1]))
y1 = vcat(y1...)
y2 = vcat(y2...)
#make every element of y3 of length 1000
for j in 1:length(y3)
    y3[j] = y3[j][end-999:end]
end
y = vcat(y_r...)
bifurcation1 = scatter(x, y1,
    	xaxis = L"$\beta$", ylabel = L"x", 
    	ms = 0.5, color = color1, legend = nothing,
    	alpha = .1, size = size
    )
    scatter!(bifurcation1, x, y, ms = 0.5, color = color3, legend = nothing,
    	alpha = .2)
    # Change x_f and y_f shape from a vector of lenght 500 with elements of lenght 1000 to a vector of lenght 500*1000
    bifurcation = scatter!(twinx(), x, y2,ms = 0.5, color = color2, legend = nothing,
    	alpha = .050)

    # bifurcation2 = scatter(fill(x[1], length(y2[1])), y2[1],
    # xaxis = L"$\beta$", ylabel = L"x", 
    # ms = 0.5, color = color2, legend = nothing,
    # alpha = .050)

    # for j in 2:length(x)
    #     scatter!(bifurcation2, fill(x[j], length(y2[j])), y2[j],ms = 0.5, color = color2, legend = nothing,
    # 	alpha = .050)
    #     # scatter by twinx the axes
    # end

    # # overlay the two plots side by side
    # overlay_plot = plot(bifurcation1, bifurcation2, layout = (1,2))

    if savefile !== nothing
        savefig(bifurcation, savefile)
    end
    return bifurcation
end

function compute_eigenvalues_trend_plus_bias(params)
    β, g, b, R = params
    function f!(F,x) #Make it in-place. Takes a vector x and returns a vector F such that F(x) = 0
        F[1] = R*x[1] - g*x[1]*(1+exp(β*(x[1]-R*x[1])*(b-g*x[1])))^(-1) - b* (1-(1+exp(β*(x[1]-R*x[1])*(b-g*x[1])))^(-1))
    end
    results = nlsolve(f! , [b/(2*R - g)]) #Solve the system, initial guess is ss of the case β = 0 
    ss =  results.zero[1] #Get the steady state
    i = sqrt(complex(-1)) #Define i
    #Compute the eigenvalues
    A = (((β* g^2 *ss - b*β*g)*ss - b*β*g*ss + g + b^2*β)*(exp(β*(b-g*ss)*(ss-R*ss)))+g)/(R*(exp(β*(b-g*ss)*(ss-R*ss))+1)^2)
    B = -(β*(g*ss -b)*(g*ss-b)*(exp(β*(b-g*ss)*(ss-R*ss))))/((exp(β*(b-g*ss)*(ss-R*ss))+1)^2)
    C = (β*g*(g*ss -b)*(ss - R*ss)*(exp(β*(b-g*ss)*(ss-R*ss))))/(R*(exp(β*(b-g*ss)*(ss-R*ss))+1)^2)
    λ₁ = ((2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3))/(3*2^(1/3)) - (2^(1/3)*(-A^2 - 3*B))/(3*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)) +A/3
    λ₂ = (-((1 - i*sqrt(3))*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)))/(6*2^(1/3)) + ((1 + i*sqrt(3))*(-A^2 - 3*B))/(3*2^(2/3)*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)) + A/3
    λ₃ = (-((1 + i*sqrt(3))*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)))/(6*2^(1/3)) + ((1 - i*sqrt(3))*(-A^2 - 3*B))/(3*2^(2/3)*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)) + A/3
    return λ₁, λ₂, λ₃
end

function compute_eigenvalues_fundamentalist(params)
    beta, g, b, R = params
    function fund!(F,x) #Make it in-place. Takes a vector x and returns a vector F such that F(x) = 0
        num1 = exp(beta*(x[1] - R*x[1])*(g*x[1]- R*x[1]))
        num2 = exp(beta*(x[1] - R*x[1])*(b -R*x[1]))
        num3 = exp(beta*(x[1] - R*x[1])*(0 -R*x[1]))
        den = num1 + num2 + num3
        F[1] = R*x[1] - g*x[1]*(num1/den) - b* (num2/den)
    end
    results = nlsolve(fund! , [b/(3*R - g)])   #Solve the system, initial guess is ss of the case β = 0
    ss =  results.zero[1]
    i = sqrt(complex(-1))
    A = (exp(R*beta*ss*(ss-R*ss))*(g*exp(2*beta*(g*ss-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+exp(beta*(g*ss-R*ss)*(ss-R*ss))*(((beta*g^2*ss-b*beta*g)*ss-b*beta*g*ss+g+b^2*beta)*exp(beta*(b-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+beta*g^2*ss*ss+g)+b^2*beta*exp(beta*(b-R*ss)*(ss-R*ss))))/(R*(exp(beta*(g*ss-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+exp(beta*(b-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+1)^2)
    B = -(beta*exp(R*beta*ss*(ss-R*ss))*(exp(beta*(ss-R*ss)*(g*ss-R*ss))*(((g^2*ss-b*g)*ss-b*g*ss+b^2)*exp(beta*(b-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+g^2*ss*ss)+b^2*exp(beta*(b-R*ss)*(ss-R*ss))))/(exp(beta*(ss-R*ss)*(g*ss-R*ss)+R*beta*ss*(ss-R*ss))+exp(beta*(b-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+1)^2
    C = (beta*g*(ss-R*ss)*((g*ss-b)*exp(beta*(b-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+g*ss)*exp(beta*(ss-R*ss)*(g*ss-R*ss)+R*beta*ss*(ss-R*ss)))/(R*(exp(beta*(ss-R*ss)*(g*ss-R*ss)+R*beta*ss*(ss-R*ss))+exp(beta*(b-R*ss)*(ss-R*ss)+R*beta*ss*(ss-R*ss))+1)^2)
    λ₁ = ((2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3))/(3*2^(1/3)) - (2^(1/3)*(-A^2 - 3*B))/(3*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)) +A/3
    λ₂ = (-((1 - i*sqrt(3))*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)))/(6*2^(1/3)) + ((1 + i*sqrt(3))*(-A^2 - 3*B))/(3*2^(2/3)*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)) + A/3
    λ₃ = (-((1 + i*sqrt(3))*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)))/(6*2^(1/3)) + ((1 - i*sqrt(3))*(-A^2 - 3*B))/(3*2^(2/3)*(2*A^3 + 3*sqrt(3)*sqrt(Complex(4*A^3*C - A^2*B^2 + 18*A*B*C - 4*B^3 + 27*C^2)) + 9*A*B + 27*C)^(1/3)) + A/3
    return λ₁, λ₂, λ₃
end