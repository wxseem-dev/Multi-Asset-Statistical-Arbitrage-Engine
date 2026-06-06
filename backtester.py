import pandas as pd
import numpy as np
import requests
from quant_math import build_universe, build_industry_universe, generate_pairs, align_prices, correlation_filter, cointegration_test, estimate_beta, spread_construction, adf_test, neg_log_likelihood, calibrate_ou, reversion_probability, compute_z_score, generate_signal
from tqdm import tqdm
import yfinance as yf

class KalmanPairTracker:
    def __init__(self, intial_beta, intial_mu):
        # State vector: [beta, mu] -> [hedge_ratio, intercept]
        self.state_mean = np.array([[intial_beta], [intial_mu]])

        # State covariance matrix (how uncertain are we about our state?)
        self.state_cov = np.zeros((2, 2))

        # Process Noise (how fast are the true beta/mu allowed to change over time)
        # Set very low so it doesn't overreact to daily noise
        self.Vw = np.array([[1e-5, 0],
        [0, 1e-5]])

        # Measurement Noise (how noisy is the market data?)
        self.Vv = 1e-3

    def update(self, price_a, price_b):
        # Takes today's raw prices, updates the dynamic hedge ratio and returns the instantaneous adaptive Z-score

        # We work in log prices
        y_t = np.log(price_a)
        x_t = np.log(price_b)

        # Observation matrix: [ln(Asset B), 1]
        F_t = np.array([[x_t, 1.0]])

        # 1. PREDICTION STEP
        # State remians the same from yesterday
        predicted_state_mean = self.state_mean
        # Uncertainty increases slightly by the process noise
        predicted_state_cov = self.state_cov + self.Vw

        # What do we expect Asset A's log price ot be?
        predicted_y = F_t.dot(predicted_state_mean)[0,0]

        # 2. MEASUREMENT UPDATE
        # The 'error' is our instantaneous spread
        error = y_t - predicted_y

        # Variance of the prediction
        Q_t = F_t.dot(predicted_state_cov).dot(F_t.T)[0, 0] + self.Vv

        # Kalman Gain (how much should we care about today's error?)
        kalman_gain = predicted_state_cov.dot(F_t.T) / Q_t

        # Update our hidden state [beta, mu]
        self.state_mean = predicted_state_mean + kalman_gain * error
        self.state_cov = predicted_state_cov - kalman_gain.dot(F_t).dot(predicted_state_cov)

        # The adaptive Z-score is simply the error divided by its standard deviation
        adaptive_z_score = error / np.sqrt(Q_t)

        return adaptive_z_score, self.state_mean[0,0] # Returns Z-score and the new Beta

class WalkForwardBacktester:
    def __init__(self, historical_price_data):
        # historical_price_data - pandas datafrime where the index is dates and columns are tickers

        self.prices = historical_price_data
        self.all_dates = sorted(self.prices.index.unique())

        # Portfolio State
        self.active_order_book = [] # Will hold top 10 pairs
        self.rebalance_frequency = 21 # Run the heavy maths every 21 days (~ 1 month)
        self.lookback_window = 252 # Use 1 year of data to recalibrate

    def run_structural_rebalance(self, current_date):
        # Monthly clock: stop the belt, look back 1 year and pick the top 10 again

        print(f"\n[STRUCTURAL CLOCK] Rebalancing on {current_date.date()}...")

        # 1. Slice the data safely (no lookahead bias)
        current_idx = self.all_dates.index(current_date)
        start_idx = max(0, current_idx - self.lookback_window)
        start_date = self.all_dates[start_idx]

        historical_slice = self.prices.loc[start_date:current_date]

        # Mathematics
        print(" -> Building Universe & Generating Pairs...")
        
        sp500_df = build_universe()
        industry_dict = build_industry_universe(sp500_df)
        all_possible_pairs = generate_pairs(industry_dict)

        results = []
        print(f" -> Scanning {len(all_possible_pairs)} pairs.")

        for industry, a, b in tqdm(all_possible_pairs, desc="Analyzing pairs"):
            # check if both tickers actually exist in our historical slice
            if a not in historical_slice.columns or b not in historical_slice.columns:
                continue

            price_a, price_b = align_prices(historical_slice[a], historical_slice[b])

            # Filter 1: Correlation
            if not correlation_filter(price_a, price_b, threshold=0.6):
                continue

            # Filter 2: Cointegration
            coint_p = cointegration_test(price_a, price_b)
            if coint_p > 0.05:
                continue

            # Filter 3: Ornstein-Uhlenbeck Calibation
            beta = estimate_beta(price_a, price_b)
            spread = spread_construction(price_a, price_b, beta)

            # Make sure spread is stationary
            adf_p = adf_test(spread)
            if adf_p > 0.05:
                continue

            ou_params = calibrate_ou(spread)
            if ou_params is None:
                continue

            kappa, mu, sigma = ou_params
            half_life = np.log(2) / kappa

            # Calculate final metrics for ranking
            latest_spread = spread.iloc[-1]
            z_score = compute_z_score(latest_spread, kappa, mu, sigma)
            prob = reversion_probability(latest_spread, kappa, mu, sigma)

            # Only keep things with a decent z_score
            if abs(z_score) < 1.5:
                continue
                
            results.append({
                "ticker_a": a, "ticker_b": b, "industry": industry,
                "beta": beta, "mu": mu, "sigma": sigma, "kappa": kappa,
                "z_score": z_score, "reversion_probability": prob,
                "signal_strength": abs(z_score * prob)
            })

        # 3. RANK AND SAVE THE TOP 10
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("signal_strength", ascending=False)

            # Save our top 10 as a list of dictionaries for the Daily Clock to monitor
            self.active_order_book = results_df.head(10).to_dict('records')

            for pair in self.active_order_book:
                pair['kalman'] = KalmanPairTracker(intial_beta=pair['beta'], intial_mu=pair['mu'])

            print(f" -> SUCCESS! Selected new Top {len(self.active_order_book)} pairs.")
        else:
            self.active_order_book = []
            print(" -> WARNING: No pairs passed the strict mathematical filters this month.")

        # For now, let's just pretend it returned a new Top 10 list
        # self.active_order_book = your_top_10_dataframe.to_dict('records')
        print(f" -> Selected new Top 10 pairs based on data from {start_date.date()} to {current_date.date()}")

    def run_tactical_daily_check(self, current_date):
        # Daily Clock: Just check today's prices against the chosen top 10
        # If our monthly searches found nothing, there is nothing to track today
        if not self.active_order_book:
            return
        
        today_prices = self.prices.loc[current_date]

        print(f" [Daily Clock] {current_date.date()} > Monitoring {len(self.active_order_book)} active target(s):")
        
        for pair in self.active_order_book:
            a = pair["ticker_a"]
            b = pair["ticker_b"]
            
            # The static Z-Score using frozen monthly math for comparison
            static_spread = np.log(today_prices[a]) - pair['beta'] * np.log(today_prices[b])
            static_z = (static_spread - pair['mu']) / pair['sigma']

            kf = pair['kalman']
            adaptive_z, dynamic_beta = kf.update(today_prices[a], today_prices[b])

            print(f"    -> Pair {a}-{b} | STATIC Z: {static_z: 5.2f} | ADAPTIVE Z: {adaptive_z: 5.2f} | Dynamic Beta: {dynamic_beta:.3f}")
            # Future home of execution:
            # if today_z > 2.0: Trigger SHORT_SPREAD
            # if today < -2.0: Trigger LONG_SPREAD
            # if trade is active and today_z crosses 0: CLOSE_TRADE (Take Profit)

        # In the future, this is where you loop through self.active_order_book
        # update the Kalman Filter, and check if Z-scores crossed +/- 2.0
        # print(f"  [Daily Clock] Checking positions for {current_date.date()}...")

    def execute_simulation(self):
        # conveyor belt: starts the loop through time
        print("Starting Walk-Forward Simulation..")

        # Start at day 252, because we need 1 year of history for the first calibration!
        for i in range(self.lookback_window, len(self.all_dates)):
            today = self.all_dates[i]
            
            # Check which clock needs to run today
            is_rebalance_day = (i - self.lookback_window) % self.rebalance_frequency == 0
            
            if is_rebalance_day:
                self.run_structural_rebalance(today)
            
            # The daily check runs EVERY day, even on rebalance days
            self.run_tactical_daily_check(today)
            
        print("\nSimulation Complete.")

if __name__ == "__main__":
    print("Fetching historical data...")
    test_tickers = ["JPM", "BAC", "C", "WFC", "GS", "XOM", "CVX", "COP", "EOG", "OXY"]

    # Download 3 years of daily close data
    real_prices = yf.download(test_tickers, start="2020-01-01", end="2023-01-01", auto_adjust=False)['Adj Close']

    # Drop columns that failed to download or have massive missing data
    real_prices = real_prices.dropna(axis=1)

    # Start the engine with real data
    backtester = WalkForwardBacktester(real_prices)
    backtester.execute_simulation()
