import yfinance as yf # import ticker data
from scipy.optimize import minimize # import optimiser
from scipy.stats import norm
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import requests
from io import StringIO
import os
import hmmlearn.hmm as hmm

def build_universe():
    # Build the ticker universe by grabbing a list of S&P 500 Companies

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Define a header to mimic a standard desktop web browser

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Fetch the webpage content using equests with the custom header
    response = requests.get(url, headers=headers)

    df = pd.read_html(StringIO(response.text), flavor="bs4")[0]
    # Formatting to allow compatibility with the rest of the project
    df["Symbol"] = (
        df["Symbol"].str.replace(".", "-", regex=False)
    )

    return df

def build_industry_universe(df):
    # Sort into industry groups to have eocnomically linked pairs and avoid brute force pairing
    return (
        df.groupby("GICS Sub-Industry")["Symbol"].apply(list).to_dict()
    )

def generate_pairs(industry_dict):

    pairs = []

    for industry, tickers in industry_dict.items():
        if len(tickers) < 2:
            continue
        
        for pair in combinations(tickers,2):
            pairs.append((industry, pair[0], pair[1]))
    
    return pairs

def align_prices(price_a, price_b):
    pair = pd.concat(
        [price_a, price_b],
        axis=1
    ).dropna()

    return pair.iloc[:,0], pair.iloc[:,1]

def correlation_filter(price_a, price_b, threshold=0.6):
    
    returns = pd.concat(
        [
            np.log(price_a).diff(),
            np.log(price_b).diff()
        ],
        axis=1
    ).dropna()

    correlation = (
        returns.iloc[:,0].corr(returns.iloc[:,1])
    )

    return correlation > threshold

def cointegration_test(price_a, price_b):
    score, p_value, _ = coint(np.log(price_a), np.log(price_b))

    return p_value

def estimate_beta(price_a, price_b):
    y = np.log(price_a)

    x = sm.add_constant(np.log(price_b))

    model = sm.OLS(
        y,
        x
    ).fit()

    return model.params.iloc[1]

def spread_construction(price_a,price_b,beta):

    spread = np.log(price_a) - beta*np.log(price_b)

    return spread

def adf_test(spread):

    result = adfuller(spread.dropna())

    return result[1]

#def test_pair(price_a, price_b):
    #score, pvalue, critical_values = coint(
        #price_a,
        #price_b
    #)

    #return pvalue

#def ordinary_least_squares(adj_close):
    # calculating the offset using linear regression
    # utilise log_prices
    #ko = np.log(adj_close["KO"])
    #pep = np.log(adj_close["PEP"])

    # correlation
    #corr = np.corrcoef(
        #np.log(adj_close["KO"]),
        #np.log(adj_close["PEP"])
    #)[0,1]

    #print("Correlation:", corr)

    # adding a constant
    #X = sm.add_constant(pep)

    #model = sm.OLS(ko, X).fit()

    #print("Result: ", model.summary())

    #beta = model.params.iloc[1]
    #print("Beta:", beta) # the gradient/offset

    #spread = ko - beta * pep

    #print("Spread:", spread)

    # matplotlib of spread
    #spread.plot(figsize=(12,6))

    #plt.title("KO-PEP Spread")
    #plt.savefig("plot.png", dpi=300, bbox_inches="tight")

    # Engle-Granger Cointegration Test
    #score, pvalue, critical_values = coint(
        #ko,
        #pep
    #)

    #print(pvalue)

    #if pvalue < 0.05:
        #print("cointegrated")
    #else:
        #print("not cointegrated")

    # augmented dickey fuller test
    #result = adfuller(spread)
    #print("Dickey Fuller Test:")
    
    #adf_stat = result[0]
    #p_value = result[1]

    #print("ADF Statistic:", adf_stat)
    #print("p-value:", p_value)

    #if p_value < 0.05:
        #print("Stationary")
   # else:
        #print("Not stationary")

def neg_log_likelihood(params, spread):

    kappa, mu, sigma = params # where params is a tuple of those parameters
    # kappa - speed of mean reversion
    # mu - long run mean
    # sigma - volatility
    
    if kappa <= 0:
        return np.inf

    # the job of otpimisation is to try as many parameter values and find the ones that make the negative
    # log-likelihood as small as possible

    # daily data observation spacing
    dt = 1.0 # one day units

    # observations:
    x_prev = spread[:-1]
    x_next = spread[1:]

    # conditional means
    phi = np.exp(-kappa * dt)

    mean = mu + (x_prev - mu) * phi

    # conditional variance
    variance = (sigma**2 / (2 * kappa)) * (1 - phi**2)

    # residuals
    residuals = x_next - mean

    # Gaussian log likelihood
    # Every observation transition we need to apply the formula for the log density of a normal random variable with variance v
    log_lik_terms = (
        -0.5 * np.log(2 * np.pi * variance)
        -0.5 * residuals**2 / variance
    )

    # total log-likelihood
    log_likelihood = np.sum(log_lik_terms)

    return -log_likelihood

def calibrate_ou(spread):

    result = minimize(
        neg_log_likelihood,
        x0=(
            1.0,
            spread.mean(),
            spread.std()
        ),
        args=(spread.values,),
        bounds=[
            (1e-06, 20),
            (None, None),
            (1e-6, None)
        ],
        method="L-BFGS-B"
    )

    if not result.success:
        return None

    return result.x

def reversion_probability(St, kappa, mu, sigma, horizon=5):
    mean = (mu + (St - mu)*np.exp(-kappa*horizon))

    variance = (
        sigma**2 / (2 * kappa) * (1-np.exp(-2*kappa*horizon))
    )

    if variance <= 1e-12:
        return np.nan

    std = np.sqrt(variance)

    T = (St + mu) / 2

    z = (T - mean) / std

    if St > mu:
        return norm.cdf(z)

    return 1 - norm.cdf(z)

def compute_z_score(St, kappa, mu, sigma):
    if kappa <= 1e-8:
        return np.nan

    sigma_stationary = (sigma / np.sqrt(2 * kappa))

    return (St - mu) / sigma_stationary

def generate_signal(z_score, probability, z_threshold=2, probability_threshold=0.60):

    if (z_score > z_threshold and probability > probability_threshold):
        return "SHORT_SPREAD"

    elif (z_score < -z_threshold and probability > probability_threshold):
        return "LONG_SPREAD"
    
    return "NO_TRADE"

#def monte_carlo_parameter_recovery():
    # Synthetic OU Data
    #for i in range(20):
        #true_kappa = 1.5
        #true_mu = 2.0
        #true_sigma = 0.5

        #dt = 1.0

        #phi = np.exp(-true_kappa * dt)

        #variance = (
            #true_sigma**2
            #/ (2 * true_kappa)
            #* (1 - phi**2)
        #)

        #print(np.sqrt(variance))

        #std = np.sqrt(variance)

        #n = 5000

        #spread = np.empty(n)

        #spread[0] = true_mu

        # generate observations
        #for t in range(1, n):
            #mean = (
                #true_mu
                #+ (spread[t-1] - true_mu) * phi
            #)

            #spread[t] = np.random.normal(
                #mean,
                #std
            #)

        # estimation
        #result = minimize(
            #neg_log_likelihood,
            #x0=(1.0, spread.mean(), spread.std()),
            #args=(spread,),
            #bounds=[
                #(1e-6, 20.0),
                #(None, None),
                #(1e-6, None)
            #],
            #method="L-BFGS-B"
        #)

        #results.append(result.x)

    #results = np.array(results)

    #print("All estimates:")
    #print(results)

    #column_means = results.mean(axis=0)
    #print("\nAverage estimates:")
    #print(f"kappa: {column_means[0]:.4f}")
    #print(f"mu:    {column_means[1]:.4f}")
    #print(f"sigma: {column_means[2]:.4f}")
