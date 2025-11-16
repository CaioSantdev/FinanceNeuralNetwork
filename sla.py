# =========================================================
# LSTM Zanotto + Multi-estratégias + Ranking + Short-selling
# =========================================================
import os, math, random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------
# Reprodutibilidade
# ---------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 1) DOWNLOAD + PRÉ-PROCESSAMENTO ZANOTTO
# =========================================================
def fetch_hist(ticker, years=10):
    end = datetime.today()
    start = end - timedelta(days=365*years)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     auto_adjust=False, progress=False)

    if df.empty:
        raise RuntimeError(f"Nenhum dado encontrado para {ticker}")

    df = df.rename(columns={"Adj Close":"AdjClose"})
    df = df[["Open","High","Low","Close","AdjClose","Volume"]]
    return df

def zanotto_preprocess(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    factor = df["AdjClose"] / df["Close"]
    out = df.copy()
    out["Open"] *= factor
    out["High"] *= factor
    out["Low"]  *= factor
    out["Close"] = out["AdjClose"]
    out = out.drop(columns=["AdjClose"])

    out[["Open","High","Low","Close"]] = out[["Open","High","Low","Close"]].interpolate(
        method="linear", limit_direction="both"
    )
    out["Volume"] = out["Volume"].replace(0, np.nan).ffill().bfill()

    span = max(5, min(int(len(out)*0.2), 60))
    out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean().fillna(out["Close"])

    return out, span

def train_test_split_ordered(df, test_ratio=0.2):
    n = len(df)
    n_train = int((1 - test_ratio) * n)
    return df.iloc[:n_train], df.iloc[n_train:]

def make_windows(arr, target_idx, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i])
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)


def rmse(y_true, y_pred): return math.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred):
    eps=1e-8
    return np.mean(np.abs((y_true-y_pred)/(np.abs(y_true)+eps))) * 100

# =========================================================
# 2) ARQUITETURA LSTM
# =========================================================
def build_lstm(n_steps, n_features, units=500, dropout=0.3):
    model = models.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# =========================================================
# 3) ESTRATÉGIAS
# =========================================================
def strategy_gap_pct(pred, real, pct_gap=0.01):
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        diff_pct = (pred[i] - real[i-1]) / real[i-1]
        if diff_pct >= pct_gap:
            sig.append("BUY")
        elif diff_pct <= -pct_gap:
            sig.append("SELL")
        else:
            sig.append("HOLD")
    return sig

def strategy_trend_filter(pred, real, ema, index):
    sig=["HOLD"]
    for i in range(1,len(pred)):
        if pred[i] > ema.iloc[i] and real[i-1] > ema.iloc[i-1]:
            sig.append("BUY")
        elif pred[i] < ema.iloc[i] and real[i-1] < ema.iloc[i-1]:
            sig.append("SELL")
        else:
            sig.append("HOLD")
    return sig

def strategy_mean_reversion(pred, real, ema, index):
    sig=["HOLD"]
    for i in range(1,len(pred)):
        if real[i-1] < ema.iloc[i-1]*0.98:
            sig.append("BUY")
        elif real[i-1] > ema.iloc[i-1]*1.02:
            sig.append("SELL")
        else:
            sig.append("HOLD")
    return sig

def strategy_always_in(pred, real):
    sig = []
    up = pred[-1] > real[-1]
    for _ in range(len(pred)):
        sig.append("BUY" if up else "SELL")
    return sig


# =========================================================
# 4) BACKTEST — LONG + SHORT
# =========================================================
def backtest_discrete_short(prices, signals,
                            initial_capital=5000,
                            fee_rate=0.0003,
                            stop_loss=0.02,
                            take_profit=0.05):
    cash = initial_capital
    shares = 0
    entry_price=None
    equity=[cash]
    trades=[]

    for i, sig in enumerate(signals):
        price = float(prices[i])

        # Stop-loss / Take profit
        if shares != 0 and entry_price:
            var = (price-entry_price)/entry_price

            # Long
            if shares>0:
                if var<=-stop_loss or var>=take_profit:
                    revenue=shares*price
                    cash+=revenue-revenue*fee_rate
                    trades.append((i,"STOP/TP_LONG",shares,price))
                    shares=0; entry_price=None

            # Short
            elif shares<0:
                if -var<=-stop_loss or -var>=take_profit:
                    qty=abs(shares)
                    pnl=qty*(entry_price-price)
                    cash+=pnl-qty*price*fee_rate
                    trades.append((i,"STOP/TP_SHORT",shares,price))
                    shares=0; entry_price=None

        # Execução dos sinais
        if sig=="BUY" and shares==0:
            qty=int(cash//price)
            if qty>0:
                cost=qty*price
                cash-=cost+cost*fee_rate
                shares=qty
                entry_price=price
                trades.append((i,"BUY",qty,price))

        elif sig=="SELL" and shares>0:
            revenue=shares*price
            cash+=revenue-revenue*fee_rate
            trades.append((i,"SELL",shares,price))
            shares=0; entry_price=None

        elif sig=="SELL_SHORT" and shares==0:
            qty = int(cash//price)
            if qty>0:
                shares = -qty
                entry_price=price
                trades.append((i,"SELL_SHORT",-qty,price))

        elif sig=="CLOSE_SHORT" and shares<0:
            qty=abs(shares)
            pnl=qty*(entry_price-price)
            cash+=pnl-qty*price*fee_rate
            trades.append((i,"CLOSE_SHORT",shares,price))
            shares=0; entry_price=None

        equity.append(cash + shares*price if shares!=0 else cash)

    # liquidação
    if shares!=0:
        price=float(prices[-1])
        if shares>0:
            cash+=shares*price - shares*price*fee_rate
        else:
            qty=abs(shares)
            pnl=qty*(entry_price-price)
            cash+=pnl-qty*price*fee_rate
        shares=0
        equity[-1]=cash

    return np.array(equity), pd.DataFrame(trades,columns=["idx","action","shares","price"])


# =========================================================
# 5) PIPELINE COMPLETO
# =========================================================
def run_pipeline(ticker, window=50, test_ratio=0.2, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {ticker} ===")

    raw = fetch_hist(ticker)
    df, ema_span = zanotto_preprocess(raw)
    features=["Open","High","Low","Close","Volume",f"EMA_{ema_span}"]
    target="Close"
    target_idx = features.index(target)

    train_df, test_df = train_test_split_ordered(df[features], test_ratio)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled  = scaler.transform(test_df)

    X_train,y_train = make_windows(train_scaled,target_idx,window)
    X_test,y_test   = make_windows(test_scaled,target_idx,window)

    model = build_lstm(window, len(features))
    ckpt=f"{outdir}/best_{ticker.replace('.','_')}.keras"

    hist = model.fit(X_train,y_train,epochs=50,batch_size=64,
                     validation_split=0.1,shuffle=False,
                     callbacks=[callbacks.EarlyStopping(patience=10,restore_best_weights=True),
                                callbacks.ModelCheckpoint(ckpt,save_best_only=True)])

    model.save(ckpt)
    y_pred_scaled=model.predict(X_test)

    def invert_scaling(v):
        zeros=np.zeros((len(v),len(features)))
        zeros[:,target_idx]=v.ravel()
        return scaler.inverse_transform(zeros)[:,target_idx]

    y_true = invert_scaling(y_test.reshape(-1,1))
    y_pred = invert_scaling(y_pred_scaled)

    m_rmse = rmse(y_true,y_pred)
    m_mae  = mean_absolute_error(y_true,y_pred)
    m_mape = mape(y_true,y_pred)
    m_r2   = r2_score(y_true,y_pred)

    print(f"\nMÉTRICAS:")
    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    # Estratégias
    test_idx = df.iloc[len(train_df):].index[window:]
    ema_test = df[f"EMA_{ema_span}"].iloc[len(train_df):].iloc[window:]

    strategies = {
        "gap_1%": lambda: strategy_gap_pct(y_pred,y_true,0.01),
        "gap_0.7%": lambda: strategy_gap_pct(y_pred,y_true,0.007),
        "trend_filter": lambda: strategy_trend_filter(y_pred,y_true,ema_test,test_idx),
        "mean_reversion": lambda: strategy_mean_reversion(y_pred,y_true,ema_test,test_idx),
        "always_in": lambda: strategy_always_in(y_pred,y_true),
    }

    print("\n=== Testando Estratégias ===")
    results_all={}

    for name,fn in strategies.items():
        print(f"\n→ Estratégia: {name}")
        signals=fn()
        equity,trades=backtest_discrete_short(y_true,signals)
        roi=(equity[-1]/5000-1)*100
        print(f"ROI: {roi:.2f}% | Trades: {len(trades)}")
        results_all[name]={"roi":roi,"equity":equity,"signals":signals,"trades":trades}

    # Ranking
    rank=pd.DataFrame([
        {"estrategia":k,"roi":v["roi"],"trades":len(v["trades"])}
        for k,v in results_all.items()
    ]).sort_values("roi",ascending=False)

    rank_path=f"{outdir}/ranking_{ticker.replace('.','_')}.csv"
    rank.to_csv(rank_path,index=False)
    print(f"\n🏆 Ranking salvo em: {rank_path}")

    # Gráfico comparativo
    plt.figure(figsize=(16,8))
    for name,r in results_all.items():
        plt.plot(r["equity"],label=f"{name} ({r['roi']:.2f}%)")
    plt.legend(); plt.grid(True)
    comp=f"{outdir}/comparison_{ticker.replace('.','_')}.png"
    plt.savefig(comp,dpi=140)
    print(f"📊 Comparação salva em: {comp}")

    # Melhor
    best_name=rank.iloc[0]["estrategia"]
    best_equity=results_all[best_name]["equity"]

    plt.figure(figsize=(16,5))
    plt.plot(best_equity,label=best_name)
    plt.legend(); plt.grid(True)
    best_path=f"{outdir}/best_{ticker.replace('.','_')}.png"
    plt.savefig(best_path,dpi=140)
    print(f"🏆 Melhor estratégia salva em: {best_path}")

    best_signals = results_all[best_name]["signals"]

    # ----------------------------------------------
    # 1) Preço real vs previsto + BUY/SELL da melhor
    # ----------------------------------------------
    plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(test_idx, y_true, label="Real", linewidth=1.5)
    ax1.plot(test_idx, y_pred, label="Previsto", linestyle="--")
    buy_idx  = [i for i,s in enumerate(best_signals) if s=="BUY"]
    sell_idx = [i for i,s in enumerate(best_signals) if s=="SELL"]
    ax1.scatter(test_idx[buy_idx],  y_true[buy_idx],  marker='^', color='green', s=60, label='Compra')
    ax1.scatter(test_idx[sell_idx], y_true[sell_idx], marker='v', color='red',   s=60, label='Venda')
    ax1.set_title(f"{ticker} — Preço Real vs Previsto ({best_name})")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # ----------------------------------------------
    # 2) Evolução da Loss
    # ----------------------------------------------
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(hist.history['loss'], label='Treino')
    ax2.plot(hist.history['val_loss'], label='Validação')
    ax2.set_title("Evolução da Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ----------------------------------------------
    # 3) Erros de previsão
    # ----------------------------------------------
    ax3 = plt.subplot(3, 2, 3)
    errors = y_true - y_pred
    ax3.plot(test_idx, errors, color='crimson')
    ax3.axhline(0, color='black', linestyle='--')
    ax3.set_title("Erros de Previsão")
    ax3.grid(True, alpha=0.3)

    # ----------------------------------------------
    # 4) Distribuição dos erros
    # ----------------------------------------------
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(errors, bins=40, alpha=0.8)
    ax4.set_title("Distribuição dos Erros")
    ax4.grid(True, alpha=0.3)

    # ----------------------------------------------
    # 5) Evolução do patrimônio (somente da melhor)
    # ----------------------------------------------
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(best_equity, label=best_name, linewidth=1.5)
    ax5.axhline(5000, linestyle=':', color='gray', label='Capital Inicial')
    ax5.set_title("Evolução do Patrimônio (Melhor Estratégia)")
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # ----------------------------------------------
    # 6) Retorno Final
    # ----------------------------------------------
    ax6 = plt.subplot(3, 2, 6)
    ax6.bar(["LSTM"], [rank.iloc[0]["roi"]])
    ax6.set_title("Retorno Final (%)")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    detailed_path = f"{outdir}/detailed_{ticker.replace('.','_')}.png"
    plt.savefig(detailed_path, dpi=140)
    print(f"📊 Gráficos detalhados salvos em: {detailed_path}")

# =========================================================
# MENU PRINCIPAL
# =========================================================
if __name__=="__main__":
    bancos={
        "1":("ITUB4.SA","Itaú"),
        "2":("BBDC4.SA","Bradesco"),
        "3":("BBAS3.SA","Banco do Brasil"),
        "4":("SANB11.SA","Santander"),
        "5":("BPAC11.SA","BTG Pactual"),
    }

    print("\n=== LSTM MULTI-ESTRATÉGIAS ===")
    for k,(t,n) in bancos.items():
        print(f"{k}) {t} — {n}")

    c=input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos: c="1"

    ticker=bancos[c][0]
    run_pipeline(ticker)
