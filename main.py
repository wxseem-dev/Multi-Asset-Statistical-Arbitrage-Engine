# imports
import yfinance as yf # import ticker data

data = yf.download(["KO", "PEP"], start="2021-05-29", auto_adjust=False) # ticker data for PEPSI and Coke
adj_close = data["Adj Close"] # grab only the adjusted close column prices
# (Why do we pick and work with the Adjusted Close data?)

print(adj_close.head())