import pandas as pd
import numpy as np
import requests
from quant_math import build_universe, build_industry_universe, generate_pairs, align_prices, correlation_filter, cointegration_test, estimate_beta, spread_construction, adf_test, neg_log_likelihood, calibrate_ou, reversion_probability, compute_z_score, generate_signal, detect_market_regime, calculate_copula_probability
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

        # Initialise our portfolio manager ledger
        self.portfolio = PortfolioManager(initial_capital=1000000.0, allocation_per_pair=100000.0)

    def run_structural_rebalance(self, current_date):
        # Monthly clock: stop the belt, look back 1 year and pick the top 10 again

        print(f"\n[STRUCTURAL CLOCK] Rebalancing on {current_date.date()}...")

        # 1. Slice the data safely (no lookahead bias)
        current_idx = self.all_dates.index(current_date)
        start_idx = max(0, current_idx - self.lookback_window)
        start_date = self.all_dates[start_idx]

        historical_slice = self.prices.loc[start_date:current_date]

        # Detect Market Regime
        # Assuming you have SPY data in your historical slice, or use a proxy like JPM currently
        market_proxy = historical_slice["JPM"].pct_change().dropna()
        self.current_regime = detect_market_regime(market_proxy)

        print(f" -> Detected Market Regime: {self.current_regime}")

        # Mathematics
        print(" -> Building Universe & Generating Pairs...")
        
        sp500_df = build_universe()
        industry_dict = build_industry_universe(sp500_df)
        all_possible_pairs = generate_pairs(industry_dict)

        results = []

        # DIAGNOSTIC FUNNEL #
        stats = {
            "total_scanned": 0,
            "corr_pass": 0,
            "coint_pass": 0,
            "adf_pass": 0,
            "ou_pass": 0,
            "hl_pass": 0
        }

        print(f" -> Scanning {len(all_possible_pairs)} pairs.")

        for industry, a, b in tqdm(all_possible_pairs, desc="Analyzing pairs"):
            # check if both tickers actually exist in our historical slice
            if a not in historical_slice.columns or b not in historical_slice.columns:
                continue
            
            stats["total_scanned"] += 1
            price_a, price_b = align_prices(historical_slice[a], historical_slice[b])

            # Filter 1: Correlation
            if not correlation_filter(price_a, price_b, threshold=0.6):
                continue
            stats["corr_pass"] += 1

            # Filter 2: Cointegration
            coint_pass, coint_p = cointegration_test(price_a, price_b)
            
            if not coint_pass:
                continue
            stats["coint_pass"] += 1

            # Filter 3: Ornstein-Uhlenbeck Calibation
            beta = estimate_beta(price_a, price_b)
            spread = spread_construction(price_a, price_b, beta)

            # Make sure spread is stationary
            adf_pass, adf_p = adf_test(spread)
            if not adf_pass:
                continue
            stats["adf_pass"] += 1

            ou_params = calibrate_ou(spread)
            if ou_params is None:
                continue
            stats["ou_pass"] += 1

            kappa, mu, sigma = ou_params
            half_life = np.log(2) / kappa

            # Filter 4: Half-life bounds (10 to 60 days)
            if not (1 <= half_life <= 90):
                continue
            stats["hl_pass"] += 1

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

        # PRINT FUNNEL RESULTS
        print("\n--- STRUCTURAL FUNNEL DIAGNOSTICS ---")
        for key, val in stats.items():
            print(f" {key.ljust(15)}: {val}")
        print("------------------------")

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
        
        # Dictionary container to pass cleanly to the Portfolio Ledger
        daily_signals = {}

        for pair in self.active_order_book:
            a = pair["ticker_a"]
            b = pair["ticker_b"]
            pair_key = f"{a}-{b}"

            # 1. Kalman Filter Adaptive Z-Score
            kf = pair['kalman']
            adaptive_z, dynamic_beta = kf.update(today_prices[a], today_prices[b])

            # 2. Re-create the dynamic spread for copula evaluation
            today_spread = np.log(today_prices[a]) - dynamic_beta * np.log(today_prices[b])

            # Fetch the historical slice for this pair up to today to ft the distribution
            current_idx = self.all_dates.index(current_date)
            start_idx = max(0, current_idx - self.lookback_window)
            start_date = self.all_dates[start_idx]
            hist_prices = self.prices.loc[start_date:current_date]

            hist_spread = np.log(hist_prices[a]) - dynamic_beta * np.log(hist_prices[b])

            # 3. Get the Copula CDF Probability
            copula_prob = calculate_copula_probability(hist_spread, today_spread)

            # Path dependent execution hysteresis state machine
            is_currently_holding = pair_key in self.portfolio.active_positions
            signal = "WATCH"

            if is_currently_holding:
                # path a: if we are already in this trade, we hold until it mean reverts back to zero (Z=0)
                position_details = self.portfolio.active_positions[pair_key]
                entry_signal = position_details["signal"]

                if entry_signal == "SHORT SPREAD":
                    # We shorted the spread; stay short until it falls back below mean zero
                    if adaptive_z <= 0.0:
                        signal = "WATCH" # Hits exit rule
                    else:
                        signal = "SHORT SPREAD" # Continue holding position open
                elif entry_signal == "LONG SPREAD":
                    # We went long the spread; stay long until it rises back above mean zero
                    if adaptive_z >= 0.0:
                        signal = "WATCH"  # Hits exit rule
                    else:
                        signal = "LONG SPREAD"  # Continue holding position open

            else:
                # PATH B: If we do NOT have an open position, evaluate standard regime-aware entry triggers
                if self.current_regime == "NORMAL":
                    if adaptive_z > 1.75 and copula_prob > 0.95:
                        signal = "SHORT SPREAD"
                    elif adaptive_z < -1.75 and copula_prob < 0.05:
                        signal = "LONG REDIRECT"
                        signal = "LONG SPREAD"
                
                elif self.current_regime == "PANIC":
                    # Demand high margins of safety in structural high vol panic regimes
                    if adaptive_z > 2.0 and copula_prob > 0.99:
                        signal = "SHORT SPREAD"
                    elif adaptive_z < -2.0 and copula_prob < 0.01:
                        signal = "LONG SPREAD"

            # Store mapping detais for portfolio lifecycle calculations
            daily_signals[pair_key] = {
                "signal": signal,
                "beta": dynamic_beta
            }

            print(f"    -> Pair {a}-{b} | Regime: {self.current_regime:6s} | ADAPTIVE Z: {adaptive_z: 5.2f} | Copula Prob: {copula_prob:.3f} | Decision: {signal}")

        # Dispatch daily signals batch cleanly into the portfolio accounting engine
        self.portfolio.execute_signals(current_date, daily_signals, today_prices)

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

        # Print accounting performance engine metrics
        final_equity = self.portfolio.pnl_history[-1]["total_equity"] if self.portfolio.pnl_history else self.portfolio.capital
        print("\n" + "="*30 + " BACKTEST METRIC PERFORMANCE REPORT " + "="*30)
        print(f" Initial Portfolio Capital:  $1,000,000.00")
        print(f" Final Mark-to-Market Equity: ${final_equity:,.2f}")
        print(f" Strategy Net Absolute Return: {((final_equity / 1000000.0) - 1) * 100:.2f}%")
        print(f" Total Closed Completed Trades: {len(self.portfolio.trade_log)}")
        print("="*96)

        if len(self.portfolio.trade_log) > 0:
            trades_df = pd.DataFrame(self.portfolio.trade_log)
            print("\n--- COMPLETED TRADE LOG AUDIT TRAIL ---")
            print(trades_df.to_string(index=False))

class PortfolioManager:
    def __init__(self, initial_capital=1000000.0, allocation_per_pair=100000.0):
        self.capital = initial_capital
        self.allocation = allocation_per_pair
        self.active_positions = {} # Key: "A-B", Value: entry details
        self.pnl_history = [] # Tracks total portfolio value over time
        self.trade_log = [] # For auditing every trade later

    def execute_signals(self, current_date, daily_signals, today_prices):
            # Processes today's tactical signals to open, hold, or close positions
            # 1. Check existing positions for exits
            active_keys = list(self.active_positions.keys())

            for pair_key in active_keys:
                pos = self.active_positions[pair_key]
                a, b = pair_key.split("-")

                # Calculate the physical cash PnL for this pair today
                pnl_a = (today_prices[a] - pos["entry_price_a"]) * pos["shares_a"]
                pnl_b = (today_prices[b] - pos["entry_price_b"]) * pos["shares_b"]

                trade_pnl = ((-pnl_a) + pnl_b) if pos["signal"] == "SHORT SPREAD" else (pnl_a + (-pnl_b))

                # Risk Management: Set a strict -5% Stop loss on deployed capital
                max_loss_limit = -(pos["allocation"] * 0.05)

                current_signal = daily_signals.get(pair_key, {}).get("signal", "WATCH")

                # Force exit if statistical mean-reversion completes (WATCH) OR if Risk limit is breached
                if current_signal == "WATCH" or trade_pnl <= max_loss_limit:
                    self._close_position(pair_key, today_prices[a], today_prices[b], current_date)
                    
                    # Optional: Print a warning to the console so you can see the risk manager working
                    if trade_pnl <= max_loss_limit:
                        print(f"    [RISK MGR] STOP-LOSS TRIGGERED on {pair_key}: ${trade_pnl:.2f}")
            
            # 2. Check for new trade entries
            for pair_key, data in daily_signals.items():
                if pair_key in self.active_positions:
                    continue
                    
                signal = data["signal"]
                beta = data["beta"]
                a, b = pair_key.split("-")

                if signal in ["SHORT SPREAD", "LONG SPREAD"]:
                    self._open_position(pair_key, signal, beta, today_prices[a], today_prices[b], current_date)
            
            # 3. Mark to market total equity for today
            today_equity = self.capital + self._calculate_unrealised_pnl(today_prices)
            self.pnl_history.append({"date": current_date, "total_equity": today_equity})

            
        
    def _open_position(self, pair_key, signal, beta, price_a, price_b, date):
            # Calculate exactly how many shares we handle based on our allocation budget
            # For simplicity, allocating standard capital to Leg A, weighted by beta to Leg B
            
            # 1. Allocate 10% of current available equity to this trade
            allocation = self.capital * 0.10

            # 2. Beta-Weighted Dollar Allocation
            # We need Value_A + Value_B = allocation, where Value_B = Value_A * abs(beta)
            value_a = allocation / (1 + abs(beta))
            value_b = allocation - value_a # This equals value_a * abs(beta)

            # 3. Convert locked dollar allocations into physical shares
            shares_a = value_a / price_a
            shares_b = value_b / price_b

            self.active_positions[pair_key] = {
                "signal": signal,
                "beta": beta,
                "entry_price_a": price_a,
                "entry_price_b": price_b,
                "shares_a": shares_a,
                "shares_b": shares_b,
                "allocation": allocation,
                "entry_date": date
            }
        
    def _close_position(self, pair_key, price_a, price_b, date):
            pos = self.active_positions.pop(pair_key)

            # Calculate individual leg returns
            pnl_a = (price_a - pos["entry_price_a"]) * pos["shares_a"]
            pnl_b = (price_b - pos["entry_price_b"]) * pos["shares_b"]

            # Invert returns if we were shorting that specific leg
            if pos["signal"] == "SHORT SPREAD":
                total_trade_pnl = (-pnl_a) + pnl_b
            else: # LONG SPREAD
                total_trade_pnl = pnl_a + (-pnl_b)

            # Update physical cash account balance
            self.capital += total_trade_pnl

            self.trade_log.append({
                "pair": pair_key,
                "type": pos["signal"],
                "entry_date": pos["entry_date"],
                "exit_date": date,
                "pnl": total_trade_pnl
            })
        
    def _calculate_unrealised_pnl(self, today_prices):
            unrealised = 0.0
            for pair_key, pos in self.active_positions.items():
                a, b = pair_key.split("-")
                pnl_a = (today_prices[a] - pos["entry_price_a"]) * pos["shares_a"]
                pnl_b = (today_prices[b] - pos["entry_price_b"]) * pos["shares_b"]

                if pos["signal"] == "SHORT_SPREAD":
                    unrealised += (-pnl_a) + pnl_b
                else:
                    unrealised += pnl_a + (-pnl_b)
                
            return unrealised

if __name__ == "__main__":
    import os

    CACHE_FILENAME = "sp500_historical_data.csv"

    if not os.path.exists(CACHE_FILENAME):
        print(f"ERROR: '{CACHE_FILENAME}' not found.")
        print("Please run 'data_manager.py' first to download the S&P 500 dataset.")
    else:
        print(f"Loading institutional dataset from local cache: {CACHE_FILENAME}...")
        # Load the massive CSV instantly into Pandas
        real_prices = pd.read_csv(CACHE_FILENAME, index_col=0, parse_dates=True)

        # Start the engine with S&P 500 data
        backtester = WalkForwardBacktester(real_prices)
        backtester.execute_simulation()
