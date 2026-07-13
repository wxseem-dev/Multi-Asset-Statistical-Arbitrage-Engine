import customtkinter as ctk

DEFAULTS = {
            "constituents_path": "sp500_constituents.csv",
            "initial_capital": 1000000.0,
            "lookback_window": 252,
            "rebal_freq": 21,
            "corr_threshold": 0.60,
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
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str):
        if text:
            self._q.put(text)

    def flush(self):
        pass

class _ParamRow(ctk.CTkFrame):
    def __init__(self, parent, label, default, tooltip="", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text=label, ..., width=170).grid(row=0, column=0, sticky="w")

        self.entry = ctk.CTkEntry(self, height=26, width=88, ...)
        self.entry.insert(0, default)
        self.entry.grid(row=0, column=1, sticky="e")

        if tooltip:
            ctk.CTkLabel(self, text=tooltip, ...).grid(row=1, ...)
    
    def get(self) -> str:
        return self.entry.get().strip()
    
    def set(set, value: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)

