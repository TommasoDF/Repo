import numpy as np
from scipy.optimize import curve_fit
import estimation_functions as ef
import pandas as pd
import matplotlib.pyplot as plt

def Bootstrap_confidence_interval(model, x_data, y_data,
                      N_bootstraps = 2000, bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])):
    #y = p(t)/p(t-2), x0 = w(t-1)/p(t-2), x1 = p(t-1)/p(t-2), x2 = w(t-3)
    # x3 = p(t-3), x4 = p(t-1), x5 = R
    
    # Estimate the parameters of the full model
    popt, pcov = ef.estimate_params(model, x_data, y_data)
    # Compute the residuals
    residuals = y_data - model(x_data, *popt)

    # Initialize the arrays to store the bootstrapped parameters
    N_params = len(popt)
    bootstrapped_params = np.zeros((N_bootstraps, N_params))

    # Perform the bootstrapping
    for i in range(N_bootstraps):
        if i % 100 == 0:
            print('Bootstrapping iteration', i)
        # Resample the residuals
        resample = np.random.choice(residuals, size=len(residuals), replace=True)

        #Multiply the residuals by a random number from a normal distribution (0,1)
        resample = resample * np.random.normal(0,1, len(residuals))

        # Generate the bootstrapped data
        bootstrapped_data = model(x_data, *popt) + resample

        # Estimate the parameters of the bootstrapped data
        try:
            bootstrapped_params[i] = curve_fit(model, x_data, bootstrapped_data
                                               , p0= [*popt])[0]
        except Exception as e:
            print('Error in iteration', i)
            print(e)
            bootstrapped_params[i] = popt
   
    return popt, bootstrapped_params
    
def main():
    # Load data
    data_folder = "Data"
    filename = '/df_estimation.csv'
    df = pd.read_csv(data_folder + filename ,index_col=0,parse_dates=True)
    df = df.dropna()

    #Define x_data and y_data
    x_data = [df['x0'], df['x1'], df['x2'], df['x3'], df['x4'], df['x5'], df['x6'], df['x7'], df['x8'], df['x9'], df['x10']]
    y_data = df['y']

    # Bootstrap the standard errors
    popt1, bootstrapped_params1 = Bootstrap_confidence_interval(ef.trend_follower_plus_bias, x_data, y_data, N_bootstraps = 2_000)
    popt2, bootstrapped_params2 = Bootstrap_confidence_interval(ef.trend_follower_plus_bias_plus_fundamentalists, x_data, y_data, N_bootstraps = 2_000)
    popt3, bootstrapped_params3 = Bootstrap_confidence_interval(ef.trend_follower_plus_bias_plus_LSTM, x_data, y_data, N_bootstraps = 2_000)
    
    
    # Comput the confidence intervals for 95% confidence
    lower_bound1 = np.percentile(bootstrapped_params1, 2.5, axis=0)
    upper_bound1 = np.percentile(bootstrapped_params1, 97.5, axis=0)
    lower_bound2 = np.percentile(bootstrapped_params2, 2.5, axis=0)
    upper_bound2 = np.percentile(bootstrapped_params2, 97.5, axis=0)
    lower_bound3 = np.percentile(bootstrapped_params3, 2.5, axis=0)
    upper_bound3 = np.percentile(bootstrapped_params3, 97.5, axis=0)

    print('bias', popt1, 'lower_bound', lower_bound1, 'upper_bound', upper_bound1)
    print('undamentalists', popt2, 'lower_bound', lower_bound2, 'upper_bound', upper_bound2)
    print('LSTM', popt3, 'lower_bound', lower_bound3, 'upper_bound', upper_bound3)


    #Save the results in a dataframe. One row for each model and we save only the lower and upper bounds
    #columns are called lower_bound and upper_bound
    # Keep only two decimal points after the dot
    lower_bound1 = np.round(lower_bound1, 2)
    upper_bound1 = np.round(upper_bound1, 2)
    lower_bound2 = np.round(lower_bound2, 2)
    upper_bound2 = np.round(upper_bound2, 2)
    lower_bound3 = np.round(lower_bound3, 2)
    upper_bound3 = np.round(upper_bound3, 2)
    df = pd.DataFrame({'lower_bound': [lower_bound1, lower_bound2, lower_bound3],
                       'upper_bound': [upper_bound1, upper_bound2, upper_bound3]},
                      index = ['bias', 'fundamentalists', 'LSTM'])

    df.to_csv(data_folder + '/bootstrap_confidence_errors.csv')

    #Plot the distribution of the parameters, layout 1x3
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(bootstrapped_params1[:, 0], bins=30, color='blue', alpha=0.5, label='bias')
    # add vertical lines for the confidence intervals
    ax[0].axvline(lower_bound1[0], color='blue', linestyle='dashed', linewidth=2)
    ax[0].axvline(upper_bound1[0], color='blue', linestyle='dashed', linewidth=2)
    # ax[0].hist(bootstrapped_params2[:, 0], bins=30, color='red', alpha=0.5, label='fundamentalists')
    # ax[0].hist(bootstrapped_params3[:, 0], bins=30, color='green', alpha=0.5, label='LSTM')
    ax[1].hist(bootstrapped_params1[:, 1], bins=30, color='red', alpha=0.5, label='bias')
    #ax[1].hist(bootstrapped_params2[:, 1], bins=30, color='blue', alpha=0.5, label='fundamentalists')
    #ax[1].hist(bootstrapped_params3[:, 1], bins=30, color='green', alpha=0.5, label='LSTM')
    ax[2].hist(bootstrapped_params1[:, 2], bins=30, color='green', alpha=0.5, label='bias')
    #ax[2].hist(bootstrapped_params2[:, 2], bins=30, color='red', alpha=0.5, label='fundamentalists')
    #ax[2].hist(bootstrapped_params3[:, 2], bins=30, color='blue', alpha=0.5, label='LSTM')

    #titles
    ax[0].set_title('g')
    ax[1].set_title('b')
    ax[2].set_title('beta')

    #labels

    plt.show()
if __name__ == '__main__':
    main()