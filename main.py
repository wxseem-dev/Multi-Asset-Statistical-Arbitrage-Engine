# imports
import yfinance as yf # import ticker data
from scipy.optimize import minimize # import optimiser
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

def build_universe():
    # Build the ticker universe by grabbing a list of S&P 500 Companies

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    df = pd.read_html(url, flavor="bs4")[0]
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

def correlation_filter(price_a, price_b, threshold=0.8):
    
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
    score, pvalue, critical_values = coint(
        price_a,
        price_b
    )

    return pvalue

#def ordinary_least_squares(adj_close):
    # calculating the offset using linear regression
    # utilise log_prices
    ko = np.log(adj_close["KO"])
    pep = np.log(adj_close["PEP"])

    # correlation
    corr = np.corrcoef(
        np.log(adj_close["KO"]),
        np.log(adj_close["PEP"])
    )[0,1]

    print("Correlation:", corr)

    # adding a constant
    X = sm.add_constant(pep)

    model = sm.OLS(ko, X).fit()

    print("Result: ", model.summary())

    beta = model.params.iloc[1]
    print("Beta:", beta) # the gradient/offset

    spread = ko - beta * pep

    print("Spread:", spread)

    # matplotlib of spread
    spread.plot(figsize=(12,6))

    plt.title("KO-PEP Spread")
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")

    # Engle-Granger Cointegration Test
    score, pvalue, critical_values = coint(
        ko,
        pep
    )

    print(pvalue)

    if pvalue < 0.05:
        print("cointegrated")
    else:
        print("not cointegrated")

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

    return result.x

#def monte_carlo_parameter_recovery():
    # Synthetic OU Data
    for i in range(20):
        true_kappa = 1.5
        true_mu = 2.0
        true_sigma = 0.5

        dt = 1.0

        phi = np.exp(-true_kappa * dt)

        variance = (
            true_sigma**2
            / (2 * true_kappa)
            * (1 - phi**2)
        )

        print(np.sqrt(variance))

        std = np.sqrt(variance)

        n = 5000

        spread = np.empty(n)

        spread[0] = true_mu

        # generate observations
        for t in range(1, n):
            mean = (
                true_mu
                + (spread[t-1] - true_mu) * phi
            )

            spread[t] = np.random.normal(
                mean,
                std
            )

        # estimation
        result = minimize(
            neg_log_likelihood,
            x0=(1.0, spread.mean(), spread.std()),
            args=(spread,),
            bounds=[
                (1e-6, 20.0),
                (None, None),
                (1e-6, None)
            ],
            method="L-BFGS-B"
        )

        results.append(result.x)

    results = np.array(results)

    print("All estimates:")
    print(results)

    column_means = results.mean(axis=0)
    print("\nAverage estimates:")
    print(f"kappa: {column_means[0]:.4f}")
    print(f"mu:    {column_means[1]:.4f}")
    print(f"sigma: {column_means[2]:.4f}")

df = build_universe()

industry_universe = build_industry_universe(df)

pairs = generate_pairs(industry_universe)

all_tickers = df["Symbol"].tolist()

data = yf.download(
    all_tickers,
    start="2021-05-29", 
    auto_adjust=False
)

prices = data["Adj Close"]

results = []

for industry, a, b in pairs:
    if not correlation_filter(prices[a], prices[b]):
        continue

    cointegration_p = cointegration_test(prices[a], prices[b])

    if cointegration_p >= 0.05:
        continue

    beta = estimate_beta(prices[a], prices[b])

    spread = spread_construction(prices[a], prices[b], beta)

    adf_p = adf_test(spread)

    if adf_p >= 0.05:
        continue

    kappa, mu, sigma = calibrate_ou(spread)

    results.append({

        "industry": industry,

        "ticker_a": a,

        "ticker_b": b,

        "cointegration_p": cointegration_p,

        "adf_p": adf_p,

        "beta": beta,

        "kappa": kappa,

        "mu": mu,

        "sigma": sigma
    })

results_df = pd.DataFrame(results)

results_df.sort_values(
    "kappa",
    ascending=False
)

print(results_df)




