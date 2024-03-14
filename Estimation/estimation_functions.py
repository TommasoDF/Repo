import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy import stats

data_folder = 'Data/'
figures_folder = 'Figures/'
# Define default plot parameters
# Set the default font family
plt.rcParams['font.family'] = 'Helvetica'

# Set font size for tick labels
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# Set font size for legend
plt.rcParams['legend.fontsize'] = 24

# Set font size for guide (axis labels, titles, etc.)
plt.rcParams['axes.labelsize'] = 24

#Define the colors to match Julia plots
color_blue = '#75BDF7'
color_orange = '#E57650'

def trend_follower_plus_bias(x, g, b, beta):
    #y = x(t), x0 = x(t-1)/F, x1 = x(t-2), x2 = x(t-3)
    # x3 = Bitsi(t-1)/F, x4 = Bitsi(t-2), x5 = Bitsi(t-3), x6 = R_t, x7 = R(t-1), x8 = R(t-2),
    # x9 = LSTM(t+1)/F, x10 = LSTM(t-2), x11 = x(t-2)**2, x12 = x(t-1)
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[12] - x[8]*x[1]) * (g * x[2] - x[8]*x[1])/ x[11]
    fitness_2 = (x[12] - x[8]*x[1]) * (b * x[5] - x[8]*x[1])/ x[11]
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)))
    return  (g* x[0] * n1 + b*x[3] * (1-n1))/x[6]


def trend_follower_plus_bias_fractions(x, g, b, beta, a = 5.5):
    #y = x(t), x0 = x(t-1)/F, x1 = x(t-2), x2 = x(t-3)
    # x3 = Bitsi(t-1)/F, x4 = Bitsi(t-2), x5 = Bitsi(t-3), x6 = R_t, x7 = R(t-1), x8 = R(t-2),
    # x9 = LSTM(t+1)/F, x10 = LSTM(t-2), x11 = x(t-2)**2 x12 = x(t-1)
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[12] - x[8]*x[1]) * (g * x[2] - x[8]*x[1])/ x[11]
    fitness_2 = (x[12] - x[8]*x[1]) * (b * x[5] - x[8]*x[1])/ x[11]
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)))
    return  n1, (1-n1)

def trend_follower_plus_bias_plus_fundamentalists(x, g, b, beta):
   #y = x(t), x0 = x(t-1)/F, x1 = x(t-2), x2 = x(t-3)
    # x3 = Bitsi(t-1)/F, x4 = Bitsi(t-2), x5 = Bitsi(t-3), x6 = R_t, x7 = R(t-1), x8 = R(t-2),
    # x9 = LSTM(t+1)/F, x10 = LSTM(t-2), x11 = x(t-2)**2, x12 = x(t-1)
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[12] - x[8]*x[1]) * (g * x[2] - x[8]*x[1])/ x[11]
    fitness_2 = (x[12] - x[8]*x[1]) * (b * x[5] - x[8]*x[1])/ x[11]
    fitness_3 = (x[12] - x[8]*x[1]) * (0 - x[8]*x[1])/ x[11]
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n2 = np.exp(beta * (fitness_2))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n3 = np.exp(beta * (fitness_3))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    return  (g* x[0] * n1 + b*x[3] * (n2))/x[6]

def trend_follower_plus_bias_plus_fundamentalists_fractions(x, g, b, beta):
#y = x(t), x0 = x(t-1)/F, x1 = x(t-2), x2 = x(t-3)
    # x3 = Bitsi(t-1)/F, x4 = Bitsi(t-2), x5 = Bitsi(t-3), x6 = R_t, x7 = R(t-1), x8 = R(t-2),
    # x9 = LSTM(t+1)/F, x10 = LSTM(t-2), x11 = x(t-2)**2, x12 = x(t-1)
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[12] - x[8]*x[1]) * (g * x[2] - x[8]*x[1])/ x[11]
    fitness_2 = (x[12] - x[8]*x[1]) * (b * x[5] - x[8]*x[1])/ x[11]
    fitness_3 = (x[12] - x[8]*x[1]) * (0 - x[8]*x[1])/ x[11]
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n2 = np.exp(beta * (fitness_2))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n3 = np.exp(beta * (fitness_3))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    return  n1, n2, n3


def trend_follower_plus_bias_plus_LSTM(x, g, b, beta):
    #y = x(t), x0 = x(t-1)/F, x1 = x(t-2), x2 = x(t-3)
    # x3 = Bitsi(t-1)/F, x4 = Bitsi(t-2), x5 = Bitsi(t-3), x6 = R_t, x7 = R(t-1), x8 = R(t-2),
    # x9 = LSTM(t+1)/F, x10 = LSTM(t-2), x11 = x(t-2)**2, x12 = x(t-1)
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[12] - x[8]*x[1]) * (g * x[2] - x[8]*x[1])/ x[11]
    fitness_2 = (x[12] - x[8]*x[1]) * (b * x[5] - x[8]*x[1])/ x[11]
    fitness_3 = (x[12] - x[8]*x[1]) * (x[10] - x[8]*x[1])/ x[11]
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n2 = np.exp(beta * (fitness_2))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n3 = np.exp(beta * (fitness_3))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    return  (g* x[0] * n1 + b*x[3] * (n2) + x[9]*n3)/x[6]

def trend_follower_plus_bias_plus_LSTM_simulated(x, g, b, beta):
    #y = x(t), x0 = x(t-1), x1 = x(t-2), x2 = x(t-3)
    # x3 =  LSTM(t+1), x4 = LSTM(t-2), x5 = R
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[0] - x[5]*x[1]) * (g * x[2] - x[5]*x[1])
    fitness_2 = (x[0] - x[5]*x[1]) * (b - x[5]*x[1])
    fitness_3 = (x[0] - x[5]*x[1]) * (x[4] - x[5]*x[1])
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n2 = np.exp(beta * (fitness_2))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n3 = np.exp(beta * (fitness_3))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    return  (g* x[0] * n1 + b * (n2) + x[3]*n3)/x[5]


def trend_follower_plus_bias_plus_LSTM_fractions(x, g, b, beta):
    #y = x(t), x0 = x(t-1)/F, x1 = x(t-2), x2 = x(t-3)
    # x3 = Bitsi(t-1)/F, x4 = Bitsi(t-2), x5 = Bitsi(t-3), x6 = R_t, x7 = R(t-1), x8 = R(t-2),
    # x9 = LSTM(t+1)/F, x10 = LSTM(t-2), x11 = x(t-2)**2, x12 = x(t-1)
    # g = params[0]
    # b = params[1]
    # beta = params[2]
    #Chatch the overflow error
    fitness_1 = (x[12] - x[8]*x[1]) * (g * x[2] - x[8]*x[1])/ x[11]
    fitness_2 = (x[12] - x[8]*x[1]) * (b * x[5] - x[8]*x[1])/ x[11]
    fitness_3 = (x[12] - x[8]*x[1]) * (x[10] - x[8]*x[1])/ x[11]
    n1 = np.exp(beta * (fitness_1))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n2 = np.exp(beta * (fitness_2))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    n3 = np.exp(beta * (fitness_3))/(np.exp(beta * (fitness_1)) + np.exp(beta * (fitness_2)) + np.exp(beta * (fitness_3)))
    return  n1, n2, n3

def estimate_params(model, x_data, y_data, bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])):
#Estimate the parameters
    popt, pcov = curve_fit(model, x_data, y_data, bounds=bounds)
    #print('paramters:',popt, 't-stats:', popt / np.sqrt(np.diagonal(pcov)))
    return popt, pcov

def compute_r2_and_residuals(model, popt, x_data, y_data):
    yhat = model(x_data, *popt)
    ybar = np.sum(y_data)/len(y_data)
    residuals = y_data - yhat
    ssres = np.sum(residuals**2)
    sstot = np.sum((y_data - ybar)**2)
    r_squared = 1 - (ssres / sstot)
    #print('r_squared:', r_squared)
    n = len(y_data)
    k = len(popt)
    adj_r_squared = 1 - (1-r_squared)*(n-1)/(n-k-1)
    #print('adj_r_squared:', adj_r_squared)
    # fig, ax = plt.subplots(figsize=(15,5))
    # ax.hist(residuals, bins = 30, color=color_blue)
    # #add lightgrey lines in the background
    # ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)
    return residuals, ssres, adj_r_squared

def fit_AR_model(df):
    X = [df['x0'], df['x3']]
    X = np.transpose(X)
    y = df['y']*df['x6']
    model = sm.OLS(y, X)
    results = model.fit()
    return results.ssr

def fit_AR_model_LSTM(df):
    X = [df['x0'], df['x3']]
    X = np.transpose(X)
    y = (df['y']*df['x6']) - (1/3)*df['x9']
    model = sm.OLS(y, X)
    results = model.fit()
    return results.ssr

def f_test_lag_and_bitsi(RSS_U, RSS_R, n):
    # Get number of parameters
    params_R = 2
    params_U = 3

    # Calculate F-statistic
    F = ((RSS_R - RSS_U) / (params_U - params_R)) / (RSS_U / (n - params_U))

    # Get p-value
    p = 1 - stats.f.cdf(F, params_U - params_R, n - params_U)

    print("F-stat:, ", F, "p-value: ", p)
    return F, p

def test_and_plot_residuals(residuals, label = None, savefile = None):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(residuals, color=color_blue, label = label)
    #add lightgrey lines in the background
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)
    #legend
    ax.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    if savefile is not None:
        plt.savefig(savefile)
    # Statiosnarity of the residuals
    adf = adfuller(residuals)
    print('p-value:', adf[1], 'for residuals')

    # Hetroskedasticity of the residuals
    het_arch(residuals)
    print('p-value:', het_arch(residuals)[3], 'for hetroskedasticity')
    return het_arch(residuals)[3]

def plot_fractions(model, x_data, y_data, popt, savefile = None, LSTM = False):
    fig, ax = plt.subplots(figsize=(15,5))
    fractions = model(x_data, *popt)
    ax.plot(fractions[0], color=color_orange, label = 'trend_follower')
    ax.plot(fractions[1], color='green', label = 'bias')
    if len(fractions) == 3:
        if LSTM:
            ax.plot(fractions[2], color= color_blue, label = 'LSTM')
        else:
            ax.plot(fractions[2], color= color_blue, label = 'fundamentalist')
    ax.set_ylabel('Fractions')
    #add lightgrey lines in the background
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    if savefile is not None:
        plt.savefig(savefile)
#plot acf and pacf for the residuals
def plot_acf_residuals(residuals, label = None):
    fig, ax = plt.subplots(figsize=(15,5), label = label)
    sm.graphics.tsa.plot_acf(residuals, lags=20, ax=ax)
    #add lightgrey lines in the background
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.5)
    #legend
    ax.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    

def main():
    # Load data
    filename =  'df_estimation.csv'
    df = pd.read_csv(data_folder + filename ,index_col=0,parse_dates=True)
    df = df.dropna()

    #Define x_data and y_da
    x_data = [df['x0'], df['x1'], df['x2'], df['x3'], df['x4'], df['x5'], df['x6'], df['x7'], df['x8'], df['x9'], df['x10'], df['x11'], df['x12']]
    y_data = df['y']

    # Estimate the parameters
    popt_1, pcov_1 = estimate_params(trend_follower_plus_bias, x_data, y_data)
    # print('bias', popt_1, popt_1 / np.sqrt(np.diagonal(pcov_1)))
    popt_2, pcov_2 = estimate_params(trend_follower_plus_bias_plus_fundamentalists, x_data, y_data)
    # print('fund', popt_2, popt_2 / np.sqrt(np.diagonal(pcov_2)))
    popt_3, pcov_3 = estimate_params(trend_follower_plus_bias_plus_LSTM, x_data, y_data)
    tstat_1, tstat_2, tstat_3 = popt_1 / np.sqrt(np.diagonal(pcov_1)), popt_2 / np.sqrt(np.diagonal(pcov_2)), popt_3 / np.sqrt(np.diagonal(pcov_3))
    #print(popt_1 / np.sqrt(np.diagonal(pcov_1)))
    # #Compute r_squared and residuals
    residuals_1, ssres_1, adj_r2_1 = compute_r2_and_residuals(trend_follower_plus_bias, popt_1, x_data, y_data)
    residuals_2, ssres_2, adj_r2_2 = compute_r2_and_residuals(trend_follower_plus_bias_plus_fundamentalists, popt_2, x_data, y_data)
    residuals_3, ssres_3, adj_r2_3 = compute_r2_and_residuals(trend_follower_plus_bias_plus_LSTM, popt_3, x_data, y_data)
    print(residuals_1.mean())
    # #Plot residuals
    plot_acf_residuals(residuals_1, label = "trend_follower_plus_bias_ret")
    plot_acf_residuals(residuals_2, label = "trend_follower_plus_bias_plus_fundamentalists")
    plot_acf_residuals(residuals_3, label= "trend_follower_plus_bias_plus_LSTM")

    # Compute ssr for restricted model
    RSS_R = fit_AR_model(df)
    RSS_R_LSTM = fit_AR_model_LSTM(df)
    # # Compute F-test
    F_1, p_1 = f_test_lag_and_bitsi(ssres_1, RSS_R, len(y_data))
    F_2, p_2 = f_test_lag_and_bitsi(ssres_2, RSS_R, len(y_data))
    F_3, p_3 = f_test_lag_and_bitsi(ssres_3, RSS_R_LSTM, len(y_data))

    # #Test and plot residuals
    het_1 = test_and_plot_residuals(residuals_1, label = "trend_follower_plus_bias", savefile=figures_folder + '/'  + 'residuals_trend_follower_plus_bias.png')
    het_2 = test_and_plot_residuals(residuals_2, label = "trend_follower_plus_bias_plus_fundamentalists", savefile=figures_folder + '/'  + 'residuals_trend_follower_plus_bias_plus_fundamentalists.png')
    het_3 = test_and_plot_residuals(residuals_3, label= "trend_follower_plus_bias_plus_LSTM", savefile=figures_folder + '/' + 'residuals_trend_follower_plus_bias_plus_LSTM.png')

    # #Plot fractions
    plot_fractions(trend_follower_plus_bias_fractions, x_data, y_data, popt_1, savefile=figures_folder + '/' + 'fractions_trend_follower_plus_bias.png')
    plot_fractions(trend_follower_plus_bias_plus_fundamentalists_fractions, x_data, y_data, popt_2, savefile=figures_folder + '/' + 'fractions_trend_follower_plus_bias_plus_fundamentalists.png')
    plot_fractions(trend_follower_plus_bias_plus_LSTM_fractions, x_data, y_data, popt_3, savefile=figures_folder + '/' + 'fractions_trend_follower_plus_bias_plus_LSTM.png', LSTM = True)

    # #Store the values in a dataframe. The columns are the models and the rows are the parameters
    params_df = pd.DataFrame()
    # We make sure to unpack the list of parameters and to have the index be given by 
    #g, tstatg, b, tstatb, beta, tstat_beta, adj_r2, F, p, het
    params_df['trend_follower_plus_bias'] = [popt_1[0], tstat_1[0], popt_1[1], tstat_1[1], popt_1[2], tstat_1[2], adj_r2_1, F_1, p_1, het_1]
    params_df['trend_follower_plus_bias_plus_fundamentalists'] = [popt_2[0], tstat_2[0], popt_2[1], tstat_2[1], popt_2[2], tstat_2[2], adj_r2_2, F_2, p_2, het_2]
    params_df['trend_follower_plus_bias_plus_LSTM'] = [popt_3[0], tstat_3[0], popt_3[1], tstat_3[1], popt_3[2], tstat_3[2], adj_r2_3, F_3, p_3, het_3]
    params_df.index = ['g', 'tstatg', 'b', 'tstatb', 'beta', 'tstat_beta', 'adj_r2', 'F', 'p', 'het']

    # #Keep only two decimals for every value
    params_df = params_df.round(2)

    # #Add parenthesis to the tstat values and the p val, making sure to access the whole row
    params_df.iloc[1] = params_df.iloc[1].apply(lambda x: '(' + str(x) + ')')
    params_df.iloc[3] = params_df.iloc[3].apply(lambda x: '(' + str(x) + ')')
    params_df.iloc[5] = params_df.iloc[5].apply(lambda x: '(' + str(x) + ')')
    params_df.iloc[8] = params_df.iloc[8].apply(lambda x: '(' + str(x) + ')')


    # #Save the dataframe to a csv file
    params_df.to_csv(data_folder + '/' + 'params_df.csv')


    #plt.show()

if __name__ == "__main__":
    main()

