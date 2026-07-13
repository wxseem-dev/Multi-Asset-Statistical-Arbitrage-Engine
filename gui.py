import customtkinter as ctk # dark themed widget library
import tkinter.messagebox as mb # pop-up dialos for validation errors
import threading # runs backtest in background thread
import queue # message passing
import sys # to swap out sys.stdout with queue.Queue
import re # to parse dates out of log lines for the program
import os # to check whether csv files actually exist
import numpy as np # to populate _populate_results for sharpe calculation
import pandas as pd # to load the proce csv for rebalance count estimate

DEFAULTS = {
    "data_path": "sp500_historical_data.csv",
    "constituents_path": "sp500_constituents.csv",
    "initial_capital": 1000000.0,
    "lookback_window": 252,
    "rebal_freq": 21,
    "corr_threshold": 0.60,
    "coint_p": 0.05,
    "min_hl": 5.0,
    "max_hl": 40.0,
    "max_z": 3.5,
    "normal_z": 1.75,
    "panic_z": 2.25,
    "hmm_window": 126,
    "hmm_restarts": 20,
    "stop_loss_pct": 0.05,
    "position_size_pct": 0.10,
    "panic_size_mult": 0.50,
    "max_concurrent": 10
}

class _QueueStream:
    # bridging threads safely (between worker thread which runs backtest and the main thread which runs GUI)
    # replacing sys.stdout with an insance of this class in print() will mean every print from the backtester
    # will be put into queue.Queue instead of the terminal. 

    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str):
        if text:
            self._q.put(text)

    def flush(self):
        pass # to avoid attribute error

class _ParamRow(ctk.CTkFrame):
    # adding and removing parameters

    def __init__(self, parent, label: str, default: str, tooltip: str = "", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text=label,
            font=("Helvetica", 11),
            text_color="#8090a0",
            anchor="w",
            width=170
        ).grid(row=0, column=0, sticky="w")

        self.entry = ctk.CTkEntry(
            self,
            height=26,
            width=8,
            font=("Courier New", 11),
            border_color="#1e3a5f",
            fg_color="#0a1628",
            text_color="#e0e8f0"
        )

        self.entry.insert(0, default)
        self.entry.grid(row=0, column=1, sticky="e")

        if tooltip:
            ctk.CTkLabel(
                self,
                text=tooltip,
                font=("Helvetica", 9),
                text_color="#445566",
                anchor="w"
            ).grid(row=1, column=0, columnspan=2, sticky="", pady=(0, 2))

    def get(self) -> str:
        return self.entry.get().strip()

    def set(self, value: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)

class BacktestApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Statistical Arbitrage Engine")
        self.geometry("1320x860")
        self.minsize(900, 640)
        self.configure(fg_color="#060f1a")

        # Runtime state
        self._log_queue: queue.Queue = queue.Queue()
        self._backtester_ref = None
        self._rebalance_count = 0
        self._total_rebalances = 24

        self._build_layout()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=310) # weight - means the sidebar never grows when you resize the window
        self.grid_columnconfigure(1, weight = 1) # weight 1 means the main absorbs all extra space
        self.grid_rowconfigure(0, weight = 1)

        self._build_sidebar()
        self._build_main_panel()

    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(
            self,
            width=290,
            corner_radius=0,
            fg_color="#0a1628",
            scrollbar_button_color="#1e3a5f",
            scrollbar_button_hover_color="#2a5080"
        )
        sidebar.grid(row = 0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="CONFIGURATION",
            font=("Courier New", 12, "bold"),
            text_color="#3a7dc9"
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(18, 6))

        row = 1
        self._params = {}

        def section(title):
            nonlocal row
            ctk.CTkFrame(sidebar, height=1, fg_color="#1e3a5f").grid(
                row=row, column=0, sticky="ew", padx=16, pady=(12,4)
            )
            row += 1
            ctk.CTkLabel(
                sidebar,
                text=title,
                font=("Helvetica", 9, "bold"),
                text_color="#3a6090"
            ).grid(row=row, column=0, sticky="w", padx=16, pady=(0, 6))
            row += 1

        def param(key, label, tooltip=""):
            nonlocal row
            p = _ParamRow(sidebar, label, DEFAULTS[key], tooltip)
            p.grid(row=row, column=0, sticky="ew", padx=16, pady=2)
            self._params[key] = p
            row += 1

        section("DATA & SIMULATION")
        param("data_path",         "Price data CSV",         "Path to sp500_historical_data.csv")
        param("constituents_path", "Constituents CSV",       "Path to sp500_constituents.csv")
        param("initial_capital",   "Initial capital ($)",    "Starting portfolio value")
        param("lookback_window",   "Lookback window (days)", "Historical days for pair calibration")
        param("rebal_freq",        "Rebalance freq (days)",  "Trading days between monthly rebalances")

        section("SIGNAL THRESHOLDS")
        param("normal_z", "Normal entry |Z|", "Min OU-normalised Z in NORMAL regime")
        param("panic_z",  "Panic entry |Z|",  "Min OU-normalised Z in PANIC regime")
        param("max_z",    "Max Z ceiling",    "Knife-catch guard and structural break exit")

        section("PAIR FILTERS")
        param("min_hl",         "Min half-life (days)", "Minimum mean-reversion speed")
        param("max_hl",         "Max half-life (days)", "Maximum mean-reversion speed")
        param("corr_threshold", "Correlation threshold","Min log-return correlation")
        param("coint_p",        "Cointegration p-value","Engle-Granger significance level")

        section("RISK MANAGEMENT")
        param("stop_loss_pct",     "Stop-loss (%)",           "Max loss per position as % of allocation")
        param("position_size_pct", "Position size (%)",       "Allocation per trade as % of capital")
        param("panic_size_mult",   "PANIC size mult (%)",     "Position scaling when regime = PANIC")
        param("max_concurrent",    "Max concurrent trades",   "Hard cap on simultaneous open positions")

        section("REGIME FILTER (HMM)")
        param("hmm_window",   "Training window (days)", "Rolling SPY RV window fed to the HMM")
        param("hmm_restarts", "Restarts",               "EM restarts; best log-likelihood wins")

        ctk.CTkFrame(sidebar, height=1, fg_color="#1e3a5f").grid(
            row=row, column=0, sticky="ew", padx=16, pady=(16, 8)
        )

        row += 1
        ctk.CTkButton(
            sidebar,
            text="Reset to Defaults",
            fg_color="transparent",
            border_width=1,
            border_color="#1e3a5f",
            text_color="#5580a0",
            height=28,
            font=("Helvetica", 10),
            hover_color="#0d1f33",
            command=self._reset_defaults
        ).grid(row = row, column=0, padx = 16, pady=(0, 24), sticky="ew")

    def _build_main_panel(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # control bar 
        top = ctk.CTkFrame(main, height=76, corner_radius=0, fg_color="#0a1628")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        self._run_btn = ctk.CTkButton(
            top,
            text="▶  RUN BACKTEST",
            width=190, height=46,
            font=("Helvetica", 13, "bold"),
            fg_color="#1a4a8a",
            hover_color="#0d3366",
            corner_radius=6,
            command=self._on_run
        )
        self._run_btn.grid(row=0, column=0, padx=20, pady=15)

        prog_col = ctk.CTkFrame(top, fg_color="transparent")
        prog_col.grid(row=0, column=1, sticky="ew", padx=(0, 20))
        prog_col.grid_columnconfigure(0, weight=1)

        self._status_lbl = ctk.CTkLabel(
            prog_col,
            text="Ready — configure parameters and press Run.",
            font=("Courier New", 10),
            text_color="#4a6080",
            anchor="w"
        )
        self._status_lbl.grid(row=0, column=0, sticky="w", pady=(14, 2))

        self._progress = ctk.CTkProgressBar(
            prog_col,
            height=6,
            corner_radius=3,
            fg_color="#0d1f33",
            progress_color="#1a6abf"
        )
        self._progress.set(0)
        self._progress.grid(row=1, column=0, sticky="ew", pady=(0, 14))

        # log box
        log_outer = ctk.CTkFrame(main, corner_radius=0, fg_color="#060f1a")
        log_outer.grid(row=1, column=0, sticky="nsew")
        log_outer.grid_rowconfigure(1, weight=1)
        log_outer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_outer,
            text="SIMULATION LOG",
            font=("Courier New", 9),
            text_color="#2a3f52",
            anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(8, 0))

        self._log_box = ctk.CTkTextbox(
            log_outer,
            font=("Courier New", 10),
            fg_color="#060f1a",
            text_color="#5aafaf",
            corner_radius=0,
            border_width=0,
            activate_scrollbars=True
        )
        self._log_box.grid(row=1, column=0, sticky="nsew")
        self._log_box.configure(state="disabled")

        # results strip
        results = ctk.CTkFrame(main, corner_radius=0, fg_color="#0a1628", height=110)
        results.grid(row=2, column=0, sticky="ew")
        results.grid_propagate(False)   # prevents the frame from shrinking to fit its children

        metric_defs = [
            "Net Return", "Sharpe", "Sortino",
            "Max Drawdown", "Win Rate", "Profit Factor",
            "Round-Trips", "Borrow Fees"
        ]
        for c in range(len(metric_defs)):
            results.grid_columnconfigure(c, weight=1)

        ctk.CTkLabel(
            results,
            text="PERFORMANCE SUMMARY",
            font=("Courier New", 9),
            text_color="#2a3f52",
            anchor="w"
        ).grid(row=0, column=0, columnspan=len(metric_defs), sticky="w", padx=14, pady=(8, 0))

        self._metric_lbls = {}
        for col, name in enumerate(metric_defs):
            ctk.CTkLabel(
                results,
                text=name,
                font=("Helvetica", 9),
                text_color="#3a5570"
            ).grid(row=1, column=col, padx=6, pady=(4, 0))

            lbl = ctk.CTkLabel(
                results,
                text="--",
                font=("Courier New", 15, "bold"),
                text_color="#2a4a6a"
            )
            lbl.grid(row=2, column=col, padx=6, pady=(2, 10))
            self._metric_lbls[name] = lbl

    def _reset_defaults(self):
        for key, widget in self._params.items():
            widget.set(DEFAULTS[key])

    def _collect_config(self):
        raw = {k: w.get() for k, w in self._params.items()}
        try:
            cfg = {
               "data_path":          raw["data_path"],
                "constituents_path":  raw["constituents_path"],
                "initial_capital":    float(raw["initial_capital"].replace(",", "")),
                "lookback_window":    int(raw["lookback_window"]),
                "rebal_freq":         int(raw["rebal_freq"]),
                "normal_z":           float(raw["normal_z"]),
                "panic_z":            float(raw["panic_z"]),
                "max_z":              float(raw["max_z"]),
                "min_hl":             float(raw["min_hl"]),
                "max_hl":             float(raw["max_hl"]),
                "corr_threshold":     float(raw["corr_threshold"]),
                "coint_p":            float(raw["coint_p"]),
                "stop_loss_pct":      float(raw["stop_loss_pct"])     / 100.0,
                "position_size_pct":  float(raw["position_size_pct"]) / 100.0,
                "panic_size_mult":    float(raw["panic_size_mult"])   / 100.0,
                "max_concurrent":     int(raw["max_concurrent"]),
                "hmm_window":         int(raw["hmm_window"]),
                "hmm_restarts":       int(raw["hmm_restarts"]), 
            }
        except ValueError as e:
            mb.showerror("Invalid Input", f"Check parameter values:\n{e}")
            return None
        
        if not (0 < cfg["normal_z"] < cfg["panic_z"] < cfg["max_z"]):
            mb.showerror(
                "Invalid Thresholds",
                "Z-score thresholds must satisfy:\n"
                " Normal Entry Z < Panic Entry Z < Max Z Ceiling\n\n"
                f"Got: {cfg['normal_z']} < {cfg['panic_z']} < {cfg['max_z']}"
            )
            return None
        
        if cfg["min_hl"] >= cfg["max_hl"]:
            mb.showerror("Invalid Half-life range",
                f"Min half life ({cfg['min_hl']}) must be less than max ({cfg['max_hl']}).")
            return None

        if not os.path.isfile(cfg["data_path"]):
            mb.showerror("File Not Found",
                f"Price data not found:\n  {cfg['data_path']}\n\nRun data_manager.py first.")
            return None

        if not os.path.isfile(cfg["constituents_path"]):
            mb.showerror("File Not Found",
                f"Constituents file not found:\n  {cfg['constituents_path']}")
            return None

        return cfg

    def _on_run(self):
        cfg = self._collect_config()
        if cfg is None:
            return
        
        try:
            prices = pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
            n_days = len(prices)
            self._total_rebalances = max(1,
                (n_days - cfg["lookback_window"]) // cfg["rebal_freq"])
        except Exception:
            self._total_rebalances = 24
        
        # reset all UI state
        self._run_btn.configure(state="disabled", text=" Running...")
        self._progress.set(0)
        self._progress.configure(progress_color="#1a6abf")
        self._rebalance_count = 0
        self._backtester_ref = None
        self._status_lbl.configure(text="Loading data...", text_color="#6090b0")

        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

        for lbl in self._metric_lbls.values():
            lbl.configure(text="--", text_color="#2a4a6a")

        # drain any messages left in the queue from the previous run
        while not self._log_queue_empty():
            try:
                self._log_queue.get_nowait()
            except queue.Empty:
                break
        
        # launch the background thread and start polling
        threading.Thread(target=self._worker_thread, args=(cfg,), daemon=True).start()
        self.after(80, self._poll_queue)

    def _worker_thread(self, cfg: dict):
        old_stdout = sys.stdout
        sys.stdout = _QueueStream(self._log_queue)
        try:
            from backtester import WalkForwardBacktester
            prices = pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
            bt = WalkForwardBacktester(prices, config=cfg)
            self._backtester_ref = bt
            bt.execute_simulation()
            self._log_queue.put("\x00DONE")
        except Exception as exc:
            import traceback
            self._log_queue.put(f"\n[ERROR] {type(exc).__name__}: {exc}\n")
            self._log_queue.put(traceback.format_exc())
            self._log_queue.put("\x00ERROR")
        finally:
            sys.stdout = old_stdout
    
    def _poll_queue(self):
        for _ in range(40):
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty():
                break

            if msg == "\x00DONE":
                self._on_complete()
                return
            if msg == "\x00ERROR":
                self._on_error()
                return

            # Update progress bar when a rebalance line appears
            if "[STRUCTURAL CLOCK]" in msg:
                self._rebalance_count += 1
                frac = min(self._rebalance_count / self._total_rebalances, 1.0)
                self._progress.set(frac)
                date_m = re.search(r"\d{4}-\d{2}-\d{2}", msg)
                date_s = date_m.group(0) if date_m else ""
                self._status_lbl.configure(
                    text=f"Rebalancing  {date_s}  "
                         f"({self._rebalance_count} / {self._total_rebalances})",
                    text_color="#c8a040"
                )

            # Append to log
            self._log_box.configure(state="normal")
            self._log_box.insert("end", msg if msg.endswith("\n") else msg + "\n")
            self._log_box.see("end")
            self._log_box.configure(state="disabled")
        
        self.after(60, self._poll_queue)
    
    def _on_complete(self):
        self._progress.set(1.0)
        self._progress.configure(progress_color="#2e7d32")
        self._status_lbl.configure(text="Simulation complete", text_color="#66bb6a")
        self._run_btn.configure(state="normal", text="RUN BACKTEST")
        self._populate_results()
    
    def _on_error(self):
        self._progress.set(0)
        self._progress.configure(progress_color="#b71c1c")
        self._status_lbl.configure(
            text="Simulation failed — see log", text_color="#ef5350")
        self._run_btn.configure(state="normal", text="RUN BACKTEST")
    
    def _populate_results(self):
        bt = self._backtester_ref
        if bt is None or not bt.portfolio.trade_log:
            return

        trades = pd.DataFrame(bt.portfolio.trade_log)
        eq     = pd.DataFrame(bt.equity_curve).set_index("date")

        ending    = eq["equity"].iloc[-1]
        total_ret = (ending - bt.initial_capital) / bt.initial_capital * 100

        eq["r"]  = eq["equity"].pct_change()
        mu, sig  = eq["r"].mean(), eq["r"].std()
        down_sig = eq[eq["r"] < 0]["r"].std()
        sharpe   = (mu / sig)      * np.sqrt(252) if sig      > 0 else 0.0
        sortino  = (mu / down_sig) * np.sqrt(252) if down_sig > 0 else 0.0

        eq["peak"] = eq["equity"].cummax()
        eq["dd"]   = (eq["equity"] - eq["peak"]) / eq["peak"]
        max_dd     = eq["dd"].min() * 100

        wins     = trades[trades["pnl"] > 0]
        win_rate = len(wins) / len(trades) * 100
        gross_p  = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_l  = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        pf       = gross_p / gross_l if gross_l > 0 else float("inf")
        fees     = trades.get("borrow_fees_paid", pd.Series(dtype=float)).sum()

        def clr(val, positive_good=True):
            if positive_good:
                return "#66bb6a" if val > 0 else "#ef5350"
            return "#ef5350" if val < 0 else "#66bb6a"

        self._metric_lbls["Net Return"].configure(
            text=f"{total_ret:+.2f}%",  text_color=clr(total_ret))
        self._metric_lbls["Sharpe"].configure(
            text=f"{sharpe:.2f}",        text_color=clr(sharpe))
        self._metric_lbls["Sortino"].configure(
            text=f"{sortino:.2f}",       text_color=clr(sortino))
        self._metric_lbls["Max Drawdown"].configure(
            text=f"{max_dd:.2f}%",       text_color=clr(max_dd, positive_good=False))
        self._metric_lbls["Win Rate"].configure(
            text=f"{win_rate:.1f}%",     text_color=clr(win_rate - 50))
        self._metric_lbls["Profit Factor"].configure(
            text=f"{pf:.2f}",            text_color=clr(pf - 1.0))
        self._metric_lbls["Round-Trips"].configure(
            text=str(len(trades)),        text_color="#4fc3f7")
        self._metric_lbls["Borrow Fees"].configure(
            text=f"${fees:,.0f}",         text_color="#ffb74d")
    
if __name__ == "__main__":
    app = BacktestApp()
    app.mainloop()

