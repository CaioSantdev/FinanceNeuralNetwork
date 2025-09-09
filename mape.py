import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1) baixar dados e simular previsão (aqui um exemplo simples: média móvel)
df = yf.download("PETR4.SA", start="2019-01-01", end="2020-01-01", auto_adjust=True)
df["pred"] = df["Close"].rolling(5).mean().shift(1).fillna(method="bfill")

# 2) calcular métricas
y, yhat = df["Close"], df["pred"]
mae = mean_absolute_error(y, yhat)
rmse = np.sqrt(mean_squared_error(y, yhat))
mape = np.mean(np.abs((y - yhat) / y)) * 100

# 3) plot série temporal
plt.figure(figsize=(10,4))
plt.plot(df.index, y, label="Real")
plt.plot(df.index, yhat, label="Previsto")
plt.fill_between(df.index, yhat-mae, yhat+mae, color="gray", alpha=0.3, 
                 label=f"± MAE ({mae:.2f})")
plt.title("Preço Real vs Previsto (PETR4 em 2019)")
plt.legend()
plt.show()

# 4) histograma dos resíduos
resid = y - yhat
plt.figure(figsize=(6,4))
plt.hist(resid, bins=30, edgecolor="k")
plt.axvline(mae, color="r", linestyle="--", label=f"MAE ({mae:.2f})")
plt.axvline(-mae, color="r", linestyle="--")
plt.title("Histograma dos Resíduos")
plt.legend()
plt.show()

# 5) scatter predito x real
plt.figure(figsize=(5,5))
sc = plt.scatter(yhat, y, c=np.abs((y-yhat)/y)*100, cmap="viridis", s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.colorbar(sc, label="Erro (%)")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Predito vs Real")
plt.show()

# 6) barra comparativa
plt.figure(figsize=(4,3))
plt.bar(["MAE","RMSE","MAPE"], [mae, rmse, mape])
plt.ylabel("Valor")
plt.title("Comparativo de Métricas")
plt.show()
