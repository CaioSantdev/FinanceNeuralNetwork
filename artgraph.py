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

    last_alpha = 2.0 / (span + 1)
    last_ema_val = ema_train.iloc[-1]

    ema_test_values = []
    for val in test_series:
        last_ema_val = last_alpha * val + (1 - last_alpha) * last_ema_val
        ema_test_values.append(last_ema_val)

    ema_test = pd.Series(ema_test_values, index=test_series.index)
    return ema_train, ema_test


def train_test_split_ordered(df, test_ratio=0.2):
    n = len(df)
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
    cash = initial_capital
    shares = 0
    entry_price = None
    position_type = None
    equity = []
    trades = []

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
                    qty = abs(shares)
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    trades.append((i, "STOP_LOSS_SHORT", shares, price, entry_price))
                    shares = 0; entry_price = None; position_type = None
                elif var <= -take_profit:
                    qty = abs(shares)
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    trades.append((i, "TAKE_PROFIT_SHORT", shares, price, entry_price))
                    shares = 0; entry_price = None; position_type = None

        if shares == 0:
            if sig == "BUY":
                max_qty = int(cash // price)
                qty = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    shares = qty; entry_price = price; position_type = 'LONG'
                    trades.append((i, "BUY", qty, price, price))
            elif sig == "SELL_SHORT":
                max_qty = int(cash // price)
                qty = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    proceeds = qty * price
                    cash += proceeds - proceeds * fee_rate
                    shares = -qty; entry_price = price; position_type = 'SHORT'
                    trades.append((i, "SELL_SHORT", -qty, price, price))

        elif shares > 0 and sig == "SELL":
            revenue = shares * price
            cash += revenue - revenue * fee_rate
            trades.append((i, "SELL", shares, price, entry_price))
            shares = 0; entry_price = None; position_type = None

        elif shares < 0 and sig == "BUY":
            qty = abs(shares)
            cost = qty * price
            cash -= cost + cost * fee_rate
            trades.append((i, "BUY_TO_COVER", shares, price, entry_price))
            shares = 0; entry_price = None; position_type = None

        current_value = cash + (shares * price if shares != 0 else 0)
        equity.append(current_value)

    if shares != 0 and len(prices) > 0:
        price = float(prices[-1])
        if shares > 0:
            cash += shares * price - shares * price * fee_rate
        else:
            qty = abs(shares)
            cost = qty * price
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
    max_qty = int(initial_capital // entry_price)
    qty = (max_qty // min_volume) * min_volume
    cash = initial_capital

    if qty >= min_volume:
        cost = qty * entry_price
        cash -= cost + cost * fee_rate

    equity = [cash + qty * p for p in prices]

    if qty >= min_volume:
        equity[-1] = cash + qty * prices[-1] - qty * prices[-1] * fee_rate

    return np.array(equity)


# =========================================================
# 5) GRÁFICOS INDIVIDUAIS
# =========================================================
def save_plot_real_vs_pred(test_idx, y_true, y_pred, signals, ticker, strategy_name, outdir):
    plt.figure(figsize=(16, 6))
    plt.plot(test_idx, y_true, label="Real", linewidth=1.5)
    plt.plot(test_idx, y_pred, label="Previsto", linestyle="--")

    buy_idx   = [i for i, s in enumerate(signals) if s == "BUY"]
    short_idx = [i for i, s in enumerate(signals) if s == "SELL_SHORT"]

    if buy_idx:
        plt.scatter(test_idx[buy_idx], y_true[buy_idx],
                    marker="^", color="green", s=60, label="Compra (LONG)")
    if short_idx:
        plt.scatter(test_idx[short_idx], y_true[short_idx],
                    marker="v", color="red", s=60, label="Venda (SHORT)")

    plt.title(f"{ticker} — Preço Real vs Previsto ({strategy_name})")
    plt.xlabel("Data"); plt.ylabel("Preço (R$)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/real_vs_pred_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_real_vs_pred_clean(test_idx, y_true, y_pred, ticker, strategy_name, outdir):
    plt.figure(figsize=(16, 6))
    plt.plot(test_idx, y_true, label="Real", linewidth=1.5)
    plt.plot(test_idx, y_pred, label="Previsto", linestyle="--")
    plt.title(f"{ticker} — Preço Real vs Previsto ({strategy_name})")
    plt.xlabel("Data"); plt.ylabel("Preço (R$)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/real_vs_pred_clean_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_loss(history, ticker, outdir):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"],     label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"{ticker} — Evolução da Loss")
    plt.xlabel("Época"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/loss_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_errors(test_idx, errors, ticker, outdir):
    plt.figure(figsize=(16, 5))
    plt.plot(test_idx, errors, color="crimson")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"{ticker} — Erros de Previsão")
    plt.xlabel("Data"); plt.ylabel("Erro (R$)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/errors_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_error_distribution(errors, ticker, outdir):
    plt.figure(figsize=(12, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title(f"{ticker} — Distribuição dos Erros")
    plt.xlabel("Erro (R$)"); plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/error_distribution_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_best_equity(best_equity, ticker, strategy_name, initial_capital, outdir):
    plt.figure(figsize=(14, 5))
    days = np.arange(len(best_equity))
    plt.plot(days, best_equity, label=strategy_name, linewidth=2, color='blue')
    plt.axhline(initial_capital, linestyle=":", color="gray", alpha=0.7,
                label=f"Capital Inicial (R${initial_capital:,.2f})")
    peak = np.maximum.accumulate(best_equity)
    drawdown = (best_equity - peak) / peak * 100
    max_drawdown = drawdown.min()
    plt.title(f"{ticker} — Evolução do Patrimônio ({strategy_name}) | Max DD: {max_drawdown:.2f}%")
    plt.xlabel("Dias de Trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    plt.tight_layout()
    path = f"{outdir}/best_equity_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_all_strategies(results_all, ticker, outdir):
    plt.figure(figsize=(16, 8))
    min_len = min(len(r["equity"]) for r in results_all.values())
    for name, r in results_all.items():
        eq = r["equity"][:min_len]
        plt.plot(np.arange(len(eq)), eq, label=f"{name} (ROI: {r['roi']:.2f}%)", linewidth=1.5)
    first_val = list(results_all.values())[0]["equity"][0]
    plt.axhline(y=first_val, color='gray', linestyle=':', alpha=0.7,
                label=f'Capital Inicial (R${first_val:,.2f})')
    plt.title(f"{ticker} — Comparação entre Estratégias", fontsize=14, fontweight='bold')
    plt.xlabel("Dias de Trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(loc='best', fontsize=9); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    plt.tight_layout()
    path = f"{outdir}/comparison_strategies_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_strategy_vs_buyhold(best_equity, buyhold_equity, ticker, strategy_name, outdir):
    min_len = min(len(best_equity), len(buyhold_equity))
    best_equity    = best_equity[:min_len]
    buyhold_equity = buyhold_equity[:min_len]
    roi_strategy = (best_equity[-1]    / best_equity[0]    - 1) * 100
    roi_buyhold  = (buyhold_equity[-1] / buyhold_equity[0] - 1) * 100

    plt.figure(figsize=(16, 8))
    days = np.arange(min_len)
    plt.plot(days, best_equity,    label=f"{strategy_name} (ROI: {roi_strategy:.2f}%)",
             linewidth=2, color='blue')
    plt.plot(days, buyhold_equity, label=f"Buy & Hold (ROI: {roi_buyhold:.2f}%)",
             linewidth=2, color='green', linestyle='--')
    plt.axhline(y=best_equity[0], color='gray', linestyle=':', alpha=0.7,
                label=f'Capital Inicial (R${best_equity[0]:,.2f})')
    plt.title(f"{ticker} — Melhor Estratégia vs Buy & Hold", fontsize=14, fontweight='bold')
    plt.xlabel("Dias de Trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(loc='best', fontsize=10); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    plt.tight_layout()
    path = f"{outdir}/strategy_vs_buyhold_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_regression_metrics(metrics_dict, ticker, outdir):
    names  = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, values)
    plt.title(f"{ticker} — Métricas de Regressão")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    path = f"{outdir}/regression_metrics_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo em: {path}")


# =========================================================
# 6) GRÁFICOS COMBINADOS (5 vs 10 anos)
# =========================================================
def save_combined_real_vs_pred(
    test_idx_5, y_true_5, y_pred_5,
    test_idx_10, y_true_10, y_pred_10,
    ticker, outdir
):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        f"{ticker} — Preço Real vs Previsto: 5 anos vs 10 anos",
        fontsize=14, fontweight='bold'
    )

    configs = [
        (axes[0], test_idx_10, y_true_10, y_pred_10,
         "10 anos — período de teste: set. 2018 a dez. 2020"),
        (axes[1], test_idx_5,  y_true_5,  y_pred_5,
         "5 anos — período de teste: set. 2019 a dez. 2020"),
    ]

    for ax, test_idx, y_true, y_pred, label in configs:
        ax.plot(test_idx, y_true, label="Preço real",     color="#1f77b4", linewidth=1.5)
        ax.plot(test_idx, y_pred, label="Preço previsto", color="#ff7f0e", linewidth=1.5,
                linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Preço (R$)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    axes[1].set_xlabel("Data")
    plt.tight_layout()
    path = f"{outdir}/combined_real_vs_pred_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico combinado salvo em: {path}")
    return path


def save_combined_strategy_vs_buyhold(
    best_equity_5,  buyhold_5,  best_name_5,  roi_s5,  roi_bh5,  maxdd_5,
    best_equity_10, buyhold_10, best_name_10, roi_s10, roi_bh10, maxdd_10,
    ticker, initial_capital, outdir
):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        f"{ticker} — Melhor estratégia vs Buy & Hold: 5 anos vs 10 anos",
        fontsize=14, fontweight='bold'
    )

    configs = [
        (axes[0], best_equity_10, buyhold_10, best_name_10, roi_s10, roi_bh10, maxdd_10,
         "10 anos — período de teste: set. 2018 a dez. 2020"),
        (axes[1], best_equity_5,  buyhold_5,  best_name_5,  roi_s5,  roi_bh5,  maxdd_5,
         "5 anos — período de teste: set. 2019 a dez. 2020"),
    ]

    for ax, eq, bh, name, roi_s, roi_bh, maxdd, label in configs:
        min_len = min(len(eq), len(bh))
        days = np.arange(min_len)
        ax.plot(days, eq[:min_len],
                label=f"{name} | ROI: {roi_s:.2f}% | Max DD: {maxdd:.2f}%",
                color="#1f4e9e", linewidth=2)
        ax.plot(days, bh[:min_len],
                label=f"Buy & Hold | ROI: {roi_bh:.2f}%",
                color="#2ca02c", linewidth=2, linestyle="--")
        ax.axhline(initial_capital, color="gray", linestyle=":", alpha=0.7,
                   label=f"Capital inicial (R${initial_capital:,.2f})")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Patrimônio (R$)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Dias de trading")
    plt.tight_layout()
    path = f"{outdir}/combined_strategy_vs_buyhold_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico combinado salvo em: {path}")
    return path


def save_combined_comparison_strategies(
    results_all_5, results_all_10,
    ticker, initial_capital, outdir
):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        f"{ticker} — Comparação entre estratégias: 5 anos vs 10 anos",
        fontsize=14, fontweight='bold'
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    configs = [
        (axes[0], results_all_10, "10 anos — período de teste: set. 2018 a dez. 2020"),
        (axes[1], results_all_5,  "5 anos — período de teste: set. 2019 a dez. 2020"),
    ]

    for ax, results, label in configs:
        min_len   = min(len(r["equity"]) for r in results.values())
        first_val = list(results.values())[0]["equity"][0]

        for (name, r), color in zip(results.items(), colors):
            eq = r["equity"][:min_len]
            ax.plot(np.arange(len(eq)), eq,
                    label=f"{name} | ROI: {r['roi']:.2f}%",
                    color=color, linewidth=1.5)

        ax.axhline(first_val, color="gray", linestyle=":", alpha=0.7,
                   label=f"Capital inicial (R${initial_capital:,.2f})")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Patrimônio (R$)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Dias de trading")
    plt.tight_layout()
    path = f"{outdir}/combined_comparison_strategies_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico combinado salvo em: {path}")
    return path


# =========================================================
# 7) PIPELINE
# =========================================================
def run_pipeline(ticker, window=50, test_ratio=0.2, outdir="resultados",
                 initial_capital=5000, years=10):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"{ticker} — {years} anos")
    print(f"{'='*50}")

    raw = fetch_hist(ticker, years=years)
    df, _ = zanotto_preprocess(raw)

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
        print(f"  {name:20s} ROI: {roi:+.2f}%")
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
    best_signals   = results_all[best_name]["signals"]
    buyhold_equity = backtest_buy_and_hold(y_true, initial_capital=initial_capital)
    buyhold_roi    = (buyhold_equity[-1] / initial_capital - 1) * 100

    print(f"\nMelhor estratégia : {best_name} | ROI: {rank.iloc[0]['roi']:+.2f}%")
    print(f"Buy & Hold ROI    : {buyhold_roi:+.2f}%")

    errors       = y_true - y_pred
    metrics_dict = {"RMSE": m_rmse, "MAE": m_mae, "MAPE": m_mape, "R²": m_r2}

    # alinhamento
    min_len         = min(len(test_idx), len(y_true), len(best_signals), len(best_equity))
    test_idx_al     = test_idx[:min_len]
    y_true_al       = y_true[:min_len]
    y_pred_al       = y_pred[:min_len]
    errors_al       = errors[:min_len]
    best_signals_al = best_signals[:min_len]
    best_equity_al  = best_equity[:min_len]
    buyhold_eq_al   = buyhold_equity[:min_len]

    print("\n=== Gerando gráficos individuais ===")
    save_plot_real_vs_pred(test_idx_al, y_true_al, y_pred_al,
                           best_signals_al, ticker, best_name, outdir)
    save_plot_real_vs_pred_clean(test_idx_al, y_true_al, y_pred_al,
                                 ticker, best_name, outdir)
    save_plot_loss(hist, ticker, outdir)
    save_plot_errors(test_idx_al, errors_al, ticker, outdir)
    save_plot_error_distribution(errors_al, ticker, outdir)
    save_plot_best_equity(best_equity_al, ticker, best_name, initial_capital, outdir)
    save_plot_all_strategies(results_all, ticker, outdir)
    save_plot_strategy_vs_buyhold(best_equity_al, buyhold_eq_al,
                                  ticker, best_name, outdir)
    save_plot_regression_metrics(metrics_dict, ticker, outdir)

    print(f"\nExperimento {years} anos concluído.")

    return {
        "results_all":    results_all,
        "rank":           rank,
        "best_name":      best_name,
        "best_equity":    best_equity_al,
        "buyhold_equity": buyhold_eq_al,
        "buyhold_roi":    buyhold_roi,
        "test_idx":       test_idx_al,
        "y_true":         y_true_al,
        "y_pred":         y_pred_al,
    }


# =========================================================
# 8) MENU PRINCIPAL
# =========================================================
if __name__ == "__main__":
    bancos = {
        "1": ("ITUB4.SA",  "Itaú"),
        "2": ("BBDC4.SA",  "Bradesco"),
        "3": ("BBAS3.SA",  "Banco do Brasil"),
        "4": ("SANB11.SA", "Santander"),
        "5": ("BPAC11.SA", "BTG Pactual"),
    }

    print("\n=== LSTM MULTI-ESTRATÉGIAS (LONG/SHORT) ===")
    for k, (t, n) in bancos.items():
        print(f"  {k}) {t} — {n}")

    c = input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos:
        c = "1"
    ticker, nome = bancos[c]

    initial_capital = 5000
    outdir_5        = f"resultados/5anos_{nome.replace(' ', '_')}"
    outdir_10       = f"resultados/10anos_{nome.replace(' ', '_')}"
    outdir_combined = f"resultados/combinado_{nome.replace(' ', '_')}"
    os.makedirs(outdir_combined, exist_ok=True)

    # ── experimento 5 anos ───────────────────────────────────
    res5 = run_pipeline(
        ticker, outdir=outdir_5, years=5,
        initial_capital=initial_capital
    )

    # ── experimento 10 anos ──────────────────────────────────
    res10 = run_pipeline(
        ticker, outdir=outdir_10, years=10,
        initial_capital=initial_capital
    )

    # ── max drawdown ─────────────────────────────────────────
    def calc_maxdd(equity):
        peak = np.maximum.accumulate(equity)
        return ((equity - peak) / peak * 100).min()

    maxdd_5  = calc_maxdd(res5["best_equity"])
    maxdd_10 = calc_maxdd(res10["best_equity"])

    # ── gráficos combinados ───────────────────────────────────
    print("\n=== Gerando gráficos combinados ===")

    save_combined_real_vs_pred(
        res5["test_idx"],  res5["y_true"],  res5["y_pred"],
        res10["test_idx"], res10["y_true"], res10["y_pred"],
        ticker, outdir_combined
    )

    save_combined_strategy_vs_buyhold(
        res5["best_equity"],  res5["buyhold_equity"],
        res5["best_name"],
        res5["results_all"][res5["best_name"]]["roi"],
        res5["buyhold_roi"],  maxdd_5,
        res10["best_equity"], res10["buyhold_equity"],
        res10["best_name"],
        res10["results_all"][res10["best_name"]]["roi"],
        res10["buyhold_roi"], maxdd_10,
        ticker, initial_capital, outdir_combined
    )

    save_combined_comparison_strategies(
        res5["results_all"], res10["results_all"],
        ticker, initial_capital, outdir_combined
    )

    # ── resumo final ─────────────────────────────────────────
    print("\n" + "="*50)
    print("RESUMO FINAL")
    print("="*50)
    print(f"\n5 anos")
    print(f"  Melhor estratégia : {res5['best_name']}")
    print(f"  ROI estratégia    : {res5['results_all'][res5['best_name']]['roi']:+.2f}%")
    print(f"  ROI Buy & Hold    : {res5['buyhold_roi']:+.2f}%")
    print(f"  Max Drawdown      : {maxdd_5:.2f}%")
    print(f"\n10 anos")
    print(f"  Melhor estratégia : {res10['best_name']}")
    print(f"  ROI estratégia    : {res10['results_all'][res10['best_name']]['roi']:+.2f}%")
    print(f"  ROI Buy & Hold    : {res10['buyhold_roi']:+.2f}%")
    print(f"  Max Drawdown      : {maxdd_10:.2f}%")
    print(f"\nGráficos individuais 5 anos  : {outdir_5}")
    print(f"Gráficos individuais 10 anos : {outdir_10}")
    print(f"Gráficos combinados          : {outdir_combined}")