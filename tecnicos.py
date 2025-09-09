import yfinance as yf
import pandas as pd
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# 1. Baixar dados de PETR4 para 2019
df = yf.download("PETR4.SA", start="2019-01-01", end="2019-12-31", auto_adjust=True)

# 2. Calcular MACD e sua signal line
macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
df["MACD"]    = macd.macd()
df["Signal"]  = macd.macd_signal()

# 3. Calcular RSI
df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

# 4. Calcular CCI
df["CCI"] = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()

# 5. Calcular ADX
df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()

# 6. Calcular Turbulência aproximada (erro quadrático padronizado dos retornos diários)
ret = df["Close"].pct_change().dropna()
mu, var = ret.mean(), ret.var()
df["Turbulence"] = ((ret - mu)**2 / var).reindex(df.index).fillna(0)

# 7. Exibir um exemplo: primeiras 10 linhas com todos os indicadores
print(df[["Close","MACD","Signal","RSI","CCI","ADX","Turbulence"]].head(10))
