import os
import math
import random
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
# 1) DOWNLOAD + PRÉ-PROCESSAMENTO
# =========================================================
def fetch_hist(ticker, years=10):
    end   = datetime.strptime("2020-12-31", "%Y-%m-%d")
    start = end - timedelta(days=365 * years)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        raise RuntimeError(f"Nenhum dado encontrado para {ticker}")
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df = df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]]
    return df


def zanotto_preprocess(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    factor = df["AdjClose"] / df["Close"]
    out = df.copy()
    out["Open"]  *= factor
    out["High"]  *= factor
    out["Low"]   *= factor
    out["Close"]  = out["AdjClose"]
    out = out.drop(columns=["AdjClose"])

    out[["Open", "High", "Low", "Close"]] = (
        out[["Open", "High", "Low", "Close"]]
        .interpolate(method="linear", limit_direction="both")
    )
    out["Volume"] = out["Volume"].replace(0, np.nan).ffill().bfill()
    return out, None


def compute_ema_no_leakage(train_series: pd.Series, test_series: pd.Series, span: int):
    ema_train = train_series.ewm(span=span, adjust=False).mean()

    alpha        = 2.0 / (span + 1)
    last_ema_val = ema_train.iloc[-1]
    ema_test_values = []

    for val in test_series:
        last_ema_val = alpha * val + (1 - alpha) * last_ema_val
        ema_test_values.append(last_ema_val)

    return ema_train, pd.Series(ema_test_values, index=test_series.index)


def train_test_split_ordered(df, test_ratio=0.2):
    n       = len(df)
    n_train = int((1 - test_ratio) * n)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def make_windows(arr, target_idx, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


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
        diff_pct = (pred[i] - real[i - 1]) / real[i - 1]
        if diff_pct >= pct_gap:
            sig.append("BUY")
        elif diff_pct <= -pct_gap:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_trend_filter(pred, real, ema, index):
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        if pred[i] > ema.iloc[i] and real[i - 1] > ema.iloc[i - 1]:
            sig.append("BUY")
        elif pred[i] < ema.iloc[i] and real[i - 1] < ema.iloc[i - 1]:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_mean_reversion(pred, real, ema, index):
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        if real[i - 1] < ema.iloc[i - 1] * 0.98:
            sig.append("BUY")
        elif real[i - 1] > ema.iloc[i - 1] * 1.02:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_always_in(pred, real):
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        if pred[i] > real[i - 1]:
            sig.append("BUY")
        else:
            sig.append("SELL_SHORT")
    return sig


# =========================================================
# 4) BACKTEST
# =========================================================
def backtest_discrete_short(
    prices, signals,
    initial_capital=5000, fee_rate=0.0003,
    stop_loss=0.02, take_profit=0.05, min_volume=100
):
    cash          = initial_capital
    shares        = 0
    entry_price   = None
    position_type = None
    equity        = []
    trades        = []

    for i, (price, sig) in enumerate(zip(prices, signals)):
        price = float(price)

        if shares != 0 and entry_price is not None and position_type is not None:
            var = (price - entry_price) / entry_price

            if position_type == 'LONG':
                if var <= -stop_loss:
                    revenue = shares * price
                    cash += revenue - revenue * fee_rate
                    trades.append((i, "STOP_LOSS_LONG", shares, price, entry_price))
                    shares = 0; entry_price = None; position_type = None
                elif var >= take_profit:
                    revenue = shares * price
                    cash += revenue - revenue * fee_rate
                    trades.append((i, "TAKE_PROFIT_LONG", shares, price, entry_price))
                    shares = 0; entry_price = None; position_type = None

            elif position_type == 'SHORT':
                if var >= stop_loss:
                    qty  = abs(shares)
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    trades.append((i, "STOP_LOSS_SHORT", shares, price, entry_price))
                    shares = 0; entry_price = None; position_type = None
                elif var <= -take_profit:
                    qty  = abs(shares)
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    trades.append((i, "TAKE_PROFIT_SHORT", shares, price, entry_price))
                    shares = 0; entry_price = None; position_type = None

        if shares == 0:
            if sig == "BUY":
                max_qty = int(cash // price)
                qty     = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    cost    = qty * price
                    cash   -= cost + cost * fee_rate
                    shares  = qty; entry_price = price; position_type = 'LONG'
                    trades.append((i, "BUY", qty, price, price))
            elif sig == "SELL_SHORT":
                max_qty = int(cash // price)
                qty     = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    proceeds = qty * price
                    cash    += proceeds - proceeds * fee_rate
                    shares   = -qty; entry_price = price; position_type = 'SHORT'
                    trades.append((i, "SELL_SHORT", -qty, price, price))

        elif shares > 0 and sig == "SELL":
            revenue = shares * price
            cash   += revenue - revenue * fee_rate
            trades.append((i, "SELL", shares, price, entry_price))
            shares = 0; entry_price = None; position_type = None

        elif shares < 0 and sig == "BUY":
            qty  = abs(shares)
            cost = qty * price
            cash -= cost + cost * fee_rate
            trades.append((i, "BUY_TO_COVER", shares, price, entry_price))
            shares = 0; entry_price = None; position_type = None

        equity.append(cash + (shares * price if shares != 0 else 0))

    if shares != 0 and len(prices) > 0:
        price = float(prices[-1])
        if shares > 0:
            cash += shares * price - shares * price * fee_rate
        else:
            qty   = abs(shares)
            cost  = qty * price
            cash -= cost + cost * fee_rate
        equity[-1] = cash

    return np.array(equity), pd.DataFrame(
        trades, columns=["idx", "action", "shares", "price", "entry_price"]
    )


def backtest_buy_and_hold(prices, initial_capital=5000, fee_rate=0.0003, min_volume=100):
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return np.array([initial_capital])

    entry_price = prices[0]
    max_qty     = int(initial_capital // entry_price)
    qty         = (max_qty // min_volume) * min_volume
    cash        = initial_capital

    if qty >= min_volume:
        cost  = qty * entry_price
        cash -= cost + cost * fee_rate

    equity = [cash + qty * p for p in prices]

    if qty >= min_volume:
        equity[-1] = cash + qty * prices[-1] - qty * prices[-1] * fee_rate

    return np.array(equity)


# =========================================================
# 5) GRÁFICOS INDIVIDUAIS
# =========================================================
def save_plot_loss(history, ticker, years, outdir):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"],     label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"{ticker} — Evolução da Loss ({years} anos)")
    plt.xlabel("Época"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/loss_{ticker.replace('.', '_')}_{years}anos.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_errors(test_idx, errors, ticker, years, outdir):
    plt.figure(figsize=(16, 5))
    plt.plot(test_idx, errors, color="#d62728")
    plt.axhline(0, linestyle="--", color="black")
    plt.title(f"{ticker} — Erros de Previsão ({years} anos)")
    plt.xlabel("Data"); plt.ylabel("Erro (R$)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/errors_{ticker.replace('.', '_')}_{years}anos.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_error_distribution(errors, ticker, years, outdir):
    plt.figure(figsize=(12, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title(f"{ticker} — Distribuição dos Erros ({years} anos)")
    plt.xlabel("Erro (R$)"); plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/error_distribution_{ticker.replace('.', '_')}_{years}anos.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


# =========================================================
# 6) GRÁFICOS COMPARATIVOS 5 ANOS VS 10 ANOS
# =========================================================
def save_plot_compare_real_vs_pred_clean(exp5, exp10, ticker, outdir):
    """Preço real vs previsto — sem sinais, apenas as curvas."""
    plt.figure(figsize=(16, 6))

    # Série real: usa o período de teste do experimento de 10 anos (mais longo)
    plt.plot(exp10["test_idx"], exp10["y_true"],
             label="Real", color="tab:gray", linewidth=2.4)

    # Previsto 5 anos — só existe no subconjunto de datas do teste de 5 anos
    plt.plot(exp5["test_idx"], exp5["y_pred"],
             label="Previsto (5 anos)", linestyle="--",
             color="tab:blue", linewidth=2.0)

    # Previsto 10 anos
    plt.plot(exp10["test_idx"], exp10["y_pred"],
             label="Previsto (10 anos)", linestyle="--",
             color="tab:orange", linewidth=2.0)

    plt.title(f"{ticker} — Real vs Previsto: 5 anos vs 10 anos")
    plt.xlabel("Data"); plt.ylabel("Preço (R$)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/comparison_real_vs_pred_clean_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_compare_real_vs_pred(exp5, exp10, ticker, outdir):
    """Preço real vs previsto — com sinais BUY/SHORT de cada experimento."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    fig.suptitle(f"{ticker} — Real vs Previsto com Sinais: 5 anos vs 10 anos",
                 fontsize=13, fontweight="bold")

    for ax, exp, label, c_pred, c_buy, c_sell in [
        (axes[0], exp10, "10 anos — teste: set. 2018 a dez. 2020",
         "tab:orange", "green", "red"),
        (axes[1], exp5,  "5 anos  — teste: mar. 2020 a dez. 2020",
         "tab:blue",   "green", "red"),
    ]:
        ax.plot(exp["test_idx"], exp["y_true"],
                label="Real", color="tab:gray", linewidth=1.8)
        ax.plot(exp["test_idx"], exp["y_pred"],
                label="Previsto", color=c_pred, linestyle="--", linewidth=1.6)

        # [CORREÇÃO] converter lista para array numpy antes de indexar
        signals = np.array(exp["best_signals"])
        buy_mask  = signals == "BUY"
        sell_mask = signals == "SELL_SHORT"

        if buy_mask.any():
            ax.scatter(exp["test_idx"][buy_mask], exp["y_true"][buy_mask],
                       marker="^", color=c_buy, s=50, alpha=0.6, label="Compra (LONG)")
        if sell_mask.any():
            ax.scatter(exp["test_idx"][sell_mask], exp["y_true"][sell_mask],
                       marker="v", color=c_sell, s=50, alpha=0.6, label="Venda (SHORT)")

        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Preço (R$)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    axes[1].set_xlabel("Data")
    plt.tight_layout()
    path = f"{outdir}/comparison_real_vs_pred_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_compare_strategies(exp5, exp10, ticker, outdir):
    """Evolução do patrimônio de todas as estratégias — 5 e 10 anos."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"{ticker} — Comparação entre Estratégias: 5 anos vs 10 anos",
                 fontsize=13, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, exp, label in [
        (axes[0], exp10, "10 anos — teste: set. 2018 a dez. 2020"),
        (axes[1], exp5,  "5 anos  — teste: mar. 2020 a dez. 2020"),
    ]:
        min_len   = min(len(v["equity"]) for v in exp["results_all"].values())
        first_val = list(exp["results_all"].values())[0]["equity"][0]

        for (name, r), color in zip(exp["results_all"].items(), colors):
            eq = r["equity"][:min_len]
            ax.plot(np.arange(len(eq)), eq,
                    label=f"{name} | ROI: {r['roi']:.2f}%",
                    color=color, linewidth=1.5)

        ax.axhline(first_val, color="gray", linestyle=":", alpha=0.7,
                   label=f"Capital inicial (R${first_val:,.2f})")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Patrimônio (R$)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R${x:,.0f}"))
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Dias de trading")
    plt.tight_layout()
    path = f"{outdir}/comparison_strategies_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_compare_regression_metrics(exp5, exp10, ticker, outdir):
    """Gráfico de barras duplas com métricas de regressão."""
    metrics   = ["RMSE", "MAE", "MAPE", "R²"]
    values_5  = [exp5["metrics_dict"][m]  for m in metrics]
    values_10 = [exp10["metrics_dict"][m] for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width / 2, values_5,  width, label="5 anos",  color="#1f77b4")
    bars2 = plt.bar(x + width / 2, values_10, width, label="10 anos", color="#ff7f0e")

    plt.xticks(x, metrics)
    plt.title(f"{ticker} — Métricas de Regressão: 5 anos vs 10 anos")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = f"{outdir}/comparison_regression_metrics_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_compare_best_equity(exp5, exp10, ticker, outdir):
    """Melhor estratégia de cada experimento vs Buy & Hold do experimento de 10 anos."""
    initial_capital = 5000

    eq5   = exp5["best_equity"]
    eq10  = exp10["best_equity"]
    bh10  = exp10["buyhold_equity"]

    roi_eq5  = (eq5[-1]  / initial_capital - 1) * 100
    roi_eq10 = (eq10[-1] / initial_capital - 1) * 100
    roi_bh10 = (bh10[-1] / initial_capital - 1) * 100

    plt.figure(figsize=(16, 6))

    plt.plot(np.arange(len(eq5)),  eq5,
             label=f"{exp5['best_name']} (5 anos) | ROI: {roi_eq5:.2f}%",
             linewidth=2, color="tab:blue")

    plt.plot(np.arange(len(eq10)), eq10,
             label=f"{exp10['best_name']} (10 anos) | ROI: {roi_eq10:.2f}%",
             linewidth=2, linestyle="--", color="tab:orange")

    plt.plot(np.arange(len(bh10)), bh10,
             label=f"Buy & Hold | ROI: {roi_bh10:.2f}%",
             linewidth=1.8, linestyle=":", color="tab:green")

    # Valores finais anotados
    plt.text(len(eq5)  - 1, eq5[-1],  f"R${eq5[-1]:,.0f}",  color="tab:blue",   fontsize=9, va="bottom")
    plt.text(len(eq10) - 1, eq10[-1], f"R${eq10[-1]:,.0f}", color="tab:orange", fontsize=9, va="bottom")
    plt.text(len(bh10) - 1, bh10[-1], f"R${bh10[-1]:,.0f}", color="tab:green",  fontsize=9, va="bottom")

    plt.axhline(y=initial_capital, color="gray", linestyle=":", linewidth=1.2,
                alpha=0.8, label=f"Capital inicial (R${initial_capital:,.2f})")

    # Linha vertical marcando o fim do período de teste de 5 anos
    plt.axvline(x=len(eq5) - 1, color="tab:blue", linestyle=":", linewidth=1.2,
                alpha=0.7, label="Fim do período de teste — 5 anos")

    plt.title(f"{ticker} — Melhor estratégia vs Buy & Hold: 5 anos vs 10 anos")
    plt.xlabel("Dias de trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R${x:,.0f}"))
    plt.tight_layout()
    path = f"{outdir}/comparison_best_equity_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


# =========================================================
# 7) PIPELINE DE UM EXPERIMENTO
# =========================================================
def run_single_experiment(
    ticker, years, window=50, test_ratio=0.2,
    outdir="resultados", initial_capital=5000
):
    print(f"\n{'='*50}")
    print(f"{ticker} | Experimento com {years} anos")
    print(f"{'='*50}")

    raw     = fetch_hist(ticker, years=years)
    df, _   = zanotto_preprocess(raw)

    ema_span   = max(5, min(int(len(df) * 0.2), 60))
    features   = ["Open", "High", "Low", "Close", "Volume", f"EMA_{ema_span}"]
    target     = "Close"
    target_idx = features.index(target)

    train_df_raw, test_df_raw = train_test_split_ordered(
        df[["Open", "High", "Low", "Close", "Volume"]], test_ratio
    )

    ema_train, ema_test = compute_ema_no_leakage(
        train_df_raw["Close"], test_df_raw["Close"], ema_span
    )
    train_df_raw[f"EMA_{ema_span}"] = ema_train.values
    test_df_raw[f"EMA_{ema_span}"]  = ema_test.values

    train_df = train_df_raw[features]
    test_df  = test_df_raw[features]

    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled  = scaler.transform(test_df)

    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test,  y_test  = make_windows(test_scaled,  target_idx, window)

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(
            f"Janelas insuficientes para {ticker} com {years} anos."
        )

    model = build_lstm(window, len(features))
    hist  = model.fit(
        X_train, y_train,
        epochs=50, batch_size=64,
        validation_split=0.1, shuffle=False,
        callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    target_mean   = scaler.mean_[target_idx]
    target_std    = scaler.scale_[target_idx]
    y_true = y_test                * target_std + target_mean
    y_pred = y_pred_scaled.ravel() * target_std + target_mean

    m_rmse = rmse(y_true, y_pred)
    m_mae  = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2   = r2_score(y_true, y_pred)

    print(f"\nMÉTRICAS ({years} anos):")
    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    test_start_idx   = len(train_df) + window
    test_idx         = df.index[test_start_idx:]
    ema_test_aligned = test_df_raw.loc[test_idx, f"EMA_{ema_span}"]

    strategies = {
        "gap_1%":         lambda: strategy_gap_pct(y_pred, y_true, 0.01),
        "gap_0.7%":       lambda: strategy_gap_pct(y_pred, y_true, 0.007),
        "trend_filter":   lambda: strategy_trend_filter(y_pred, y_true, ema_test_aligned, test_idx),
        "mean_reversion": lambda: strategy_mean_reversion(y_pred, y_true, ema_test_aligned, test_idx),
        "always_in":      lambda: strategy_always_in(y_pred, y_true),
    }

    print("\n=== Testando Estratégias ===")
    results_all = {}
    for name, fn in strategies.items():
        signals = fn()
        equity, trades = backtest_discrete_short(
            y_true, signals, initial_capital=initial_capital
        )
        roi = (equity[-1] / initial_capital - 1) * 100
        print(f"  {name:20s} ROI: {roi:+.2f}%  "
              f"BUY: {signals.count('BUY')}  "
              f"SHORT: {signals.count('SELL_SHORT')}  "
              f"HOLD: {signals.count('HOLD')}")
        results_all[name] = {
            "roi": roi, "equity": equity,
            "signals": signals, "trades": trades
        }

    rank = pd.DataFrame([
        {"estrategia": k, "roi": v["roi"], "trades": len(v["trades"])}
        for k, v in results_all.items()
    ]).sort_values("roi", ascending=False).reset_index(drop=True)

    print("\n=== Ranking ===")
    print(rank.to_string(index=False))

    best_name      = rank.iloc[0]["estrategia"]
    best_equity    = results_all[best_name]["equity"]
    best_signals   = results_all[best_name]["signals"]   # lista Python
    buyhold_equity = backtest_buy_and_hold(y_true, initial_capital=initial_capital)
    buyhold_roi    = (buyhold_equity[-1] / initial_capital - 1) * 100

    print(f"\nMelhor estratégia : {best_name} | ROI: {rank.iloc[0]['roi']:+.2f}%")
    print(f"Buy & Hold ROI    : {buyhold_roi:+.2f}%")

    errors       = y_true - y_pred
    metrics_dict = {"RMSE": m_rmse, "MAE": m_mae, "MAPE": m_mape, "R²": m_r2}

    # alinhamento — test_idx como DatetimeIndex, restante como arrays
    min_len         = min(len(test_idx), len(y_true), len(y_pred),
                          len(errors), len(best_signals),
                          len(best_equity), len(buyhold_equity))
    test_idx_al     = test_idx[:min_len]
    y_true_al       = y_true[:min_len]
    y_pred_al       = y_pred[:min_len]
    errors_al       = errors[:min_len]
    # [CORREÇÃO] armazena best_signals como array numpy para indexação segura
    best_signals_al = np.array(best_signals[:min_len])
    best_equity_al  = best_equity[:min_len]
    buyhold_eq_al   = buyhold_equity[:min_len]

    # gráficos individuais
    save_plot_loss(hist, ticker, years, outdir)
    save_plot_errors(test_idx_al, errors_al, ticker, years, outdir)
    save_plot_error_distribution(errors_al, ticker, years, outdir)

    return {
        "years":          years,
        "ticker":         ticker,
        "test_idx":       test_idx_al,
        "y_true":         y_true_al,
        "y_pred":         y_pred_al,
        "errors":         errors_al,
        "results_all":    results_all,
        "rank":           rank,
        "metrics_dict":   metrics_dict,
        "best_name":      best_name,
        "best_signals":   best_signals_al,   # numpy array
        "best_equity":    best_equity_al,
        "buyhold_equity": buyhold_eq_al,
        "buyhold_roi":    buyhold_roi,
    }


# =========================================================
# 8) PIPELINE COMPARATIVO 5 VS 10 ANOS
# =========================================================
def run_comparison_pipeline(
    ticker, outdir="resultados", initial_capital=5000,
    window=50, test_ratio=0.2
):
    os.makedirs(outdir, exist_ok=True)

    exp5  = run_single_experiment(ticker=ticker, years=5,  window=window,
                                  test_ratio=test_ratio, outdir=outdir,
                                  initial_capital=initial_capital)

    exp10 = run_single_experiment(ticker=ticker, years=10, window=window,
                                  test_ratio=test_ratio, outdir=outdir,
                                  initial_capital=initial_capital)

    print("\n=== GERANDO GRÁFICOS COMPARATIVOS ===")
    save_plot_compare_real_vs_pred_clean(exp5, exp10, ticker, outdir)
    save_plot_compare_real_vs_pred(exp5, exp10, ticker, outdir)
    save_plot_compare_strategies(exp5, exp10, ticker, outdir)
    save_plot_compare_regression_metrics(exp5, exp10, ticker, outdir)
    save_plot_compare_best_equity(exp5, exp10, ticker, outdir)

    # CSV resumo
    summary = pd.DataFrame([
        {
            "Experimento":              "5 anos",
            "RMSE":                     exp5["metrics_dict"]["RMSE"],
            "MAE":                      exp5["metrics_dict"]["MAE"],
            "MAPE":                     exp5["metrics_dict"]["MAPE"],
            "R²":                       exp5["metrics_dict"]["R²"],
            "Melhor Estratégia":        exp5["best_name"],
            "ROI Melhor Estratégia (%)":exp5["rank"].iloc[0]["roi"],
            "ROI Buy & Hold (%)":       exp5["buyhold_roi"],
        },
        {
            "Experimento":              "10 anos",
            "RMSE":                     exp10["metrics_dict"]["RMSE"],
            "MAE":                      exp10["metrics_dict"]["MAE"],
            "MAPE":                     exp10["metrics_dict"]["MAPE"],
            "R²":                       exp10["metrics_dict"]["R²"],
            "Melhor Estratégia":        exp10["best_name"],
            "ROI Melhor Estratégia (%)":exp10["rank"].iloc[0]["roi"],
            "ROI Buy & Hold (%)":       exp10["buyhold_roi"],
        },
    ])

    csv_path = f"{outdir}/resumo_comparativo_{ticker.replace('.', '_')}.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n=== RESUMO FINAL ===")
    print(summary.to_string(index=False))
    print(f"\nCSV salvo em: {csv_path}")
    print(f"Gráficos salvos em: {outdir}")
    print("\nProcessamento concluído com sucesso.")

    return exp5, exp10, summary


# =========================================================
# MENU PRINCIPAL
# =========================================================
if __name__ == "__main__":
    bancos = {
        "1": ("ITUB4.SA",  "Itaú"),
        "2": ("BBDC4.SA",  "Bradesco"),
        "3": ("BBAS3.SA",  "Banco do Brasil"),
        "4": ("SANB11.SA", "Santander"),
        "5": ("BPAC11.SA", "BTG Pactual"),
    }

    print("\n=== LSTM MULTI-ESTRATÉGIAS — COMPARAÇÃO 5 ANOS VS 10 ANOS ===")
    for k, (t, n) in bancos.items():
        print(f"  {k}) {t} — {n}")

    c = input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos:
        c = "1"
    ticker, nome = bancos[c]

    outdir = f"resultados/comparativo_{nome.replace(' ', '_')}"

    run_comparison_pipeline(
        ticker=ticker,
        outdir=outdir,
        initial_capital=5000
    )