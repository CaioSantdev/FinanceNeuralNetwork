
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
# 1) DOWNLOAD, SELEÇÃO DE DADOS E PRÉ-PROCESSAMENTO
# =========================================================
def fetch_hist(ticker, years=10):
    end = datetime.strptime("2020-12-31", "%Y-%m-%d")
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
    out["Open"] *= factor
    out["High"] *= factor
    out["Low"] *= factor
    out["Close"] = out["AdjClose"]
    out = out.drop(columns=["AdjClose"])

    out[["Open", "High", "Low", "Close"]] = (
        out[["Open", "High", "Low", "Close"]]
        .interpolate(method="linear", limit_direction="both")
    )
    out["Volume"] = out["Volume"].replace(0, np.nan).ffill().bfill()
    return out, None


def compute_ema_no_leakage(train_series: pd.Series, test_series: pd.Series, span: int):
    ema_train = train_series.ewm(span=span, adjust=False).mean()

    alpha = 2.0 / (span + 1)
    last_ema_val = ema_train.iloc[-1]
    ema_test_values = []

    for val in test_series:
        last_ema_val = alpha * val + (1 - alpha) * last_ema_val
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
# 4) BACKTEST — LONG + SHORT
# =========================================================
def backtest_discrete_short(
    prices,
    signals,
    initial_capital=5000,
    fee_rate=0.0003,
    stop_loss=0.02,
    take_profit=0.05,
    min_volume=100
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
                    shares = 0
                    entry_price = None
                    position_type = None
                elif var >= take_profit:
                    revenue = shares * price
                    cash += revenue - revenue * fee_rate
                    trades.append((i, "TAKE_PROFIT_LONG", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None

            elif position_type == 'SHORT':
                if var >= stop_loss:
                    qty = abs(shares)
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    trades.append((i, "STOP_LOSS_SHORT", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None
                elif var <= -take_profit:
                    qty = abs(shares)
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    trades.append((i, "TAKE_PROFIT_SHORT", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None

        if shares == 0:
            if sig == "BUY":
                max_qty = int(cash // price)
                qty = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    shares = qty
                    entry_price = price
                    position_type = 'LONG'
                    trades.append((i, "BUY", qty, price, price))
            elif sig == "SELL_SHORT":
                max_qty = int(cash // price)
                qty = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    proceeds = qty * price
                    cash += proceeds - proceeds * fee_rate
                    shares = -qty
                    entry_price = price
                    position_type = 'SHORT'
                    trades.append((i, "SELL_SHORT", -qty, price, price))

        elif shares > 0 and sig == "SELL":
            revenue = shares * price
            cash += revenue - revenue * fee_rate
            trades.append((i, "SELL", shares, price, entry_price))
            shares = 0
            entry_price = None
            position_type = None

        elif shares < 0 and sig == "BUY":
            qty = abs(shares)
            cost = qty * price
            cash -= cost + cost * fee_rate
            trades.append((i, "BUY_TO_COVER", shares, price, entry_price))
            shares = 0
            entry_price = None
            position_type = None

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
# 5) WALK-FORWARD VALIDATION
# =========================================================
def walk_forward_validation(df, features, target_idx, window, n_splits=5):
    split_size = len(df) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        print(f"\n--- Walk-Forward Split {i+1}/{n_splits} ---")

        train_end = (i + 1) * split_size
        test_end = train_end + split_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        X_train, y_train = make_windows(train_scaled, target_idx, window)
        X_test, y_test = make_windows(test_scaled, target_idx, window)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = build_lstm(window, len(features))
        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=64,
            validation_split=0.1,
            shuffle=False,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        y_pred_scaled = model.predict(X_test, verbose=0)

        target_mean = scaler.mean_[target_idx]
        target_std = scaler.scale_[target_idx]
        y_true = y_test * target_std + target_mean
        y_pred = y_pred_scaled.ravel() * target_std + target_mean

        results.append({
            "split": i + 1,
            "rmse": rmse(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        })

    return pd.DataFrame(results)


# =========================================================
# 6) GRÁFICOS INDIVIDUAIS
# =========================================================
def save_plot_loss(history, ticker, years, outdir):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"{ticker} — Evolução da Loss ({years} anos)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{outdir}/loss_{ticker.replace('.', '_')}_{years}anos.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_errors(test_idx, errors, ticker, years, outdir):
    plt.figure(figsize=(16, 5))
    plt.plot(test_idx, errors)
    plt.axhline(0, linestyle="--")
    plt.title(f"{ticker} — Erros de Previsão ({years} anos)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{outdir}/errors_{ticker.replace('.', '_')}_{years}anos.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_error_distribution(errors, ticker, years, outdir):
    plt.figure(figsize=(12, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title(f"{ticker} — Distribuição dos Erros ({years} anos)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{outdir}/error_distribution_{ticker.replace('.', '_')}_{years}anos.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")


# =========================================================
# 7) GRÁFICOS COMPARATIVOS 5 ANOS VS 10 ANOS
# =========================================================
def save_plot_compare_real_vs_pred(exp5, exp10, ticker, outdir):
    plt.figure(figsize=(16, 6))

    # Linhas principais
    plt.plot(
        exp10["test_idx"], exp10["y_true"],
        label="Real",
        color="tab:gray",
        linewidth=2.2
    )

    plt.plot(
        exp5["test_idx"], exp5["y_pred"],
        label="Previsto (5 anos)",
        color="tab:blue",
        linestyle="--",
        linewidth=1.8
    )

    plt.plot(
        exp10["test_idx"], exp10["y_pred"],
        label="Previsto (10 anos)",
        color="tab:orange",
        linestyle="--",
        linewidth=1.8
    )

    # -------------------------
    # Sinais 5 anos
    # -------------------------
    buy_idx_5 = [i for i, s in enumerate(exp5["best_signals"]) if s == "BUY"]
    sell_idx_5 = [i for i, s in enumerate(exp5["best_signals"]) if s == "SELL_SHORT"]

    if buy_idx_5:
        plt.scatter(
            exp5["test_idx"][buy_idx_5],
            exp5["y_true"][buy_idx_5],
            marker="^",
            color="tab:green",
            s=45,
            alpha=0.35,
            label="Compra 5 anos"
        )

    if sell_idx_5:
        plt.scatter(
            exp5["test_idx"][sell_idx_5],
            exp5["y_true"][sell_idx_5],
            marker="v",
            color="tab:red",
            s=45,
            alpha=0.35,
            label="Venda/Short 5 anos"
        )

    # -------------------------
    # Sinais 10 anos
    # -------------------------
    buy_idx_10 = [i for i, s in enumerate(exp10["best_signals"]) if s == "BUY"]
    sell_idx_10 = [i for i, s in enumerate(exp10["best_signals"]) if s == "SELL_SHORT"]

    if buy_idx_10:
        plt.scatter(
            exp10["test_idx"][buy_idx_10],
            exp10["y_true"][buy_idx_10],
            marker="^",
            color="tab:olive",
            s=45,
            alpha=0.30,
            label="Compra 10 anos"
        )

    if sell_idx_10:
        plt.scatter(
            exp10["test_idx"][sell_idx_10],
            exp10["y_true"][sell_idx_10],
            marker="v",
            color="tab:purple",
            s=45,
            alpha=0.30,
            label="Venda/Short 10 anos"
        )

    plt.title(f"{ticker} — Real vs Previsto: 5 anos vs 10 anos")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/comparison_real_vs_pred_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {path}")


def save_plot_compare_real_vs_pred_clean(exp5, exp10, ticker, outdir):
    plt.figure(figsize=(16, 6))

    plt.plot(
        exp10["test_idx"], exp10["y_true"],
        label="Real",
        color="tab:gray",
        linewidth=2.4
    )

    plt.plot(
        exp5["test_idx"], exp5["y_pred"],
        label="Previsto (5 anos)",
        linestyle="--",
        color="tab:blue",
        linewidth=2.0
    )

    plt.plot(
        exp10["test_idx"], exp10["y_pred"],
        label="Previsto (10 anos)",
        linestyle="--",
        color="tab:orange",
        linewidth=2.0
    )

    plt.title(f"{ticker} — Real vs Previsto Clean: 5 anos vs 10 anos")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/comparison_real_vs_pred_clean_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_compare_strategies(exp5, exp10, ticker, outdir):
    plt.figure(figsize=(16, 8))

    min_len_5 = min(len(v["equity"]) for v in exp5["results_all"].values())
    for name, r in exp5["results_all"].items():
        eq = r["equity"][:min_len_5]
        plt.plot(np.arange(len(eq)), eq, linewidth=1.3, label=f"{name} - 5 anos ({r['roi']:.2f}%)")

    min_len_10 = min(len(v["equity"]) for v in exp10["results_all"].values())
    for name, r in exp10["results_all"].items():
        eq = r["equity"][:min_len_10]
        plt.plot(np.arange(len(eq)), eq, linestyle="--", linewidth=1.3, label=f"{name} - 10 anos ({r['roi']:.2f}%)")

    plt.title(f"{ticker} — Comparison Strategies: 5 anos vs 10 anos")
    plt.xlabel("Dias de trading")
    plt.ylabel("Patrimônio (R$)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{outdir}/comparison_strategies_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")


def save_plot_compare_regression_metrics(exp5, exp10, ticker, outdir):
    metrics = ["RMSE", "MAE", "MAPE", "R²"]
    values_5 = [exp5["metrics_dict"][m] for m in metrics]
    values_10 = [exp10["metrics_dict"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width/2, values_5, width, label="5 anos")
    bars2 = plt.bar(x + width/2, values_10, width, label="10 anos")

    plt.xticks(x, metrics)
    plt.title(f"{ticker} — Regression Metrics: 5 anos vs 10 anos")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    path = f"{outdir}/comparison_regression_metrics_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")

def save_plot_compare_best_equity(exp5, exp10, ticker, outdir):
    initial_capital = 5000

    eq5 = exp5["best_equity"]
    eq10 = exp10["best_equity"]
    bh10 = exp10["buyhold_equity"]

    # ROI
    roi_eq5 = (eq5[-1] / initial_capital - 1) * 100
    roi_eq10 = (eq10[-1] / initial_capital - 1) * 100
    roi_bh10 = (bh10[-1] / initial_capital - 1) * 100

    plt.figure(figsize=(16, 6))

    # 🔵 Estratégia 5 anos
    plt.plot(
        np.arange(len(eq5)),
        eq5,
        label=f"{exp5['best_name']} (5 anos) | ROI: {roi_eq5:.2f}%",
        linewidth=2,
        color="tab:blue"
    )

    # 🟠 Estratégia 10 anos
    plt.plot(
        np.arange(len(eq10)),
        eq10,
        label=f"{exp10['best_name']} (10 anos) | ROI: {roi_eq10:.2f}%",
        linewidth=2,
        linestyle="--",
        color="tab:orange"
    )

    # 🟢 Buy & Hold
    plt.plot(
        np.arange(len(bh10)),
        bh10,
        label=f"Buy & Hold | ROI: {roi_bh10:.2f}%",
        linewidth=1.8,
        linestyle=":",
        color="tab:green"
    )

    # 🔥 VALORES FINAIS NO GRÁFICO
    plt.text(len(eq5)-1, eq5[-1], f"{eq5[-1]:.0f}", color="tab:blue", fontsize=9)
    plt.text(len(eq10)-1, eq10[-1], f"{eq10[-1]:.0f}", color="tab:orange", fontsize=9)
    plt.text(len(bh10)-1, bh10[-1], f"{bh10[-1]:.0f}", color="tab:green", fontsize=9)

    # Capital inicial
    plt.axhline(
        y=initial_capital,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label=f"Capital Inicial (R${initial_capital:,.2f})"
    )

    # Fim 5 anos
    plt.axvline(
        x=len(eq5) - 1,
        color="tab:blue",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="Fim 5 anos"
    )

    plt.title(f"{ticker} — Estratégias vs Buy & Hold (ROI e valor final)")
    plt.xlabel("Dias de trading")
    plt.ylabel("Patrimônio (R$)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/comparison_best_equity_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {path}")

# def save_plot_compare_best_equity(exp5, exp10, ticker, outdir):
#     eq5 = exp5["best_equity"]
#     eq10 = exp10["best_equity"]
#     bh10 = exp10["buyhold_equity"]

#     plt.figure(figsize=(16, 6))

#     # Melhor estratégia - 5 anos
#     plt.plot(
#         np.arange(len(eq5)),
#         eq5,
#         label=f"Melhor estratégia 5 anos ({exp5['best_name']})",
#         linewidth=2,
#         color="tab:blue"
#     )

#     # Melhor estratégia - 10 anos
#     plt.plot(
#         np.arange(len(eq10)),
#         eq10,
#         label=f"Melhor estratégia 10 anos ({exp10['best_name']})",
#         linewidth=2,
#         linestyle="--",
#         color="tab:orange"
#     )

#     # Um único Buy & Hold: usar o de 10 anos
#     plt.plot(
#         np.arange(len(bh10)),
#         bh10,
#         label="Buy & Hold",
#         linewidth=1.8,
#         linestyle=":",
#         color="tab:green"
#     )

#     # Capital inicial
#     plt.axhline(
#         y=5000,
#         color="gray",
#         linestyle=":",
#         linewidth=1.2,
#         alpha=0.8,
#         label="Capital inicial"
#     )

#     # Marca o fim do experimento de 5 anos
#     plt.axvline(
#         x=len(eq5) - 1,
#         color="tab:blue",
#         linestyle=":",
#         linewidth=1.2,
#         alpha=0.8,
#         label="Fim 5 anos"
#     )

#     plt.title(f"{ticker} — Melhor equity vs Buy & Hold: 5 anos vs 10 anos")
#     plt.xlabel("Dias de trading")
#     plt.ylabel("Patrimônio (R$)")
#     plt.legend(loc="best", fontsize=9)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     path = f"{outdir}/comparison_best_equity_{ticker.replace('.', '_')}.png"
#     plt.savefig(path, dpi=140, bbox_inches="tight")
#     plt.close()

#     print(f"Gráfico salvo em: {path}")

# def save_plot_compare_best_equity(exp5, exp10, ticker, outdir):
#     eq5 = exp5["best_equity"]
#     eq10 = exp10["best_equity"]
#     bh5 = exp5["buyhold_equity"]
#     bh10 = exp10["buyhold_equity"]

#     plt.figure(figsize=(16, 6))

#     # Melhor estratégia - 5 anos
#     plt.plot(
#         np.arange(len(eq5)),
#         eq5,
#         label=f"Melhor estratégia 5 anos ({exp5['best_name']})",
#         linewidth=2,
#         color="tab:blue"
#     )

#     # Melhor estratégia - 10 anos
#     plt.plot(
#         np.arange(len(eq10)),
#         eq10,
#         label=f"Melhor estratégia 10 anos ({exp10['best_name']})",
#         linewidth=2,
#         linestyle="--",
#         color="tab:orange"
#     )

#     # Buy & Hold - 5 anos
#     plt.plot(
#         np.arange(len(bh5)),
#         bh5,
#         label="Buy & Hold 5 anos",
#         linewidth=1.8,
#         linestyle=":",
#         color="tab:green"
#     )

#     # Buy & Hold - 10 anos
#     plt.plot(
#         np.arange(len(bh10)),
#         bh10,
#         label="Buy & Hold 10 anos",
#         linewidth=1.8,
#         linestyle="-.",
#         color="tab:red"
#     )

#     # Linha horizontal do capital inicial
#     plt.axhline(
#         y=5000,
#         color="gray",
#         linestyle=":",
#         linewidth=1.2,
#         alpha=0.8,
#         label="Capital inicial"
#     )

#     # Marca o fim do experimento de 5 anos
#     plt.axvline(
#         x=len(eq5) - 1,
#         color="tab:blue",
#         linestyle=":",
#         linewidth=1.2,
#         alpha=0.8,
#         label="Fim 5 anos"
#     )

#     plt.title(f"{ticker} — Melhor equity + Buy & Hold: 5 anos vs 10 anos")
#     plt.xlabel("Dias de trading")
#     plt.ylabel("Patrimônio (R$)")
#     plt.legend(loc="best", fontsize=9)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     path = f"{outdir}/comparison_best_equity_{ticker.replace('.', '_')}.png"
#     plt.savefig(path, dpi=140, bbox_inches="tight")
#     plt.close()

#     print(f"Gráfico salvo em: {path}")


# =========================================================
# 8) PIPELINE DE UM EXPERIMENTO
# =========================================================
def run_single_experiment(
    ticker,
    years,
    window=50,
    test_ratio=0.2,
    outdir="resultados",
    initial_capital=5000,
    do_walk_forward=True
):
    print(f"\n=== {ticker} | Experimento com {years} anos ===")

    raw = fetch_hist(ticker, years=years)
    df, _ = zanotto_preprocess(raw)

    ema_span = max(5, min(int(len(df) * 0.2), 60))
    features = ["Open", "High", "Low", "Close", "Volume", f"EMA_{ema_span}"]
    target = "Close"
    target_idx = features.index(target)

    train_df_raw, test_df_raw = train_test_split_ordered(
        df[["Open", "High", "Low", "Close", "Volume"]],
        test_ratio
    )

    ema_train, ema_test = compute_ema_no_leakage(
        train_df_raw["Close"],
        test_df_raw["Close"],
        ema_span
    )
    train_df_raw[f"EMA_{ema_span}"] = ema_train.values
    test_df_raw[f"EMA_{ema_span}"] = ema_test.values

    train_df = train_df_raw[features]
    test_df = test_df_raw[features]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test, y_test = make_windows(test_scaled, target_idx, window)

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(f"Não foi possível gerar janelas suficientes para {ticker} com {years} anos.")

    model = build_lstm(window, len(features))
    hist = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        shuffle=False,
        callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test, verbose=0)

    target_mean = scaler.mean_[target_idx]
    target_std = scaler.scale_[target_idx]
    y_true = y_test * target_std + target_mean
    y_pred = y_pred_scaled.ravel() * target_std + target_mean

    m_rmse = rmse(y_true, y_pred)
    m_mae = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2 = r2_score(y_true, y_pred)

    print("\nMÉTRICAS:")
    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    test_start_idx = len(train_df) + window
    test_idx = df.index[test_start_idx:]
    ema_test_aligned = test_df_raw.loc[test_idx, f"EMA_{ema_span}"]

    strategies = {
        "gap_1%": lambda: strategy_gap_pct(y_pred, y_true, 0.01),
        "gap_0.7%": lambda: strategy_gap_pct(y_pred, y_true, 0.007),
        "trend_filter": lambda: strategy_trend_filter(y_pred, y_true, ema_test_aligned, test_idx),
        "mean_reversion": lambda: strategy_mean_reversion(y_pred, y_true, ema_test_aligned, test_idx),
        "always_in": lambda: strategy_always_in(y_pred, y_true),
    }

    print("\n=== Testando Estratégias (LONG/SHORT) ===")
    results_all = {}

    for name, fn in strategies.items():
        print(f"\n→ Estratégia: {name}")
        signals = fn()
        equity, trades = backtest_discrete_short(
            y_true,
            signals,
            initial_capital=initial_capital
        )
        roi = (equity[-1] / initial_capital - 1) * 100
        buy_c = signals.count("BUY")
        short_c = signals.count("SELL_SHORT")
        hold_c = signals.count("HOLD")
        print(f"ROI: {roi:.2f}% | Trades: {len(trades)} | BUY: {buy_c} | SHORT: {short_c} | HOLD: {hold_c}")
        results_all[name] = {
            "roi": roi,
            "equity": equity,
            "signals": signals,
            "trades": trades
        }

    rank = pd.DataFrame([
        {"estrategia": k, "roi": v["roi"], "trades": len(v["trades"])}
        for k, v in results_all.items()
    ]).sort_values("roi", ascending=False).reset_index(drop=True)

    print("\n=== Ranking das Estratégias ===")
    print(rank)

    best_name = rank.iloc[0]["estrategia"]
    best_equity = results_all[best_name]["equity"]
    best_signals = results_all[best_name]["signals"]

    buyhold_equity = backtest_buy_and_hold(y_true, initial_capital=initial_capital)
    buyhold_roi = (buyhold_equity[-1] / initial_capital - 1) * 100

    print(f"\n🏆 Melhor estratégia: {best_name} | ROI: {rank.iloc[0]['roi']:.2f}%")
    print(f"📌 Buy & Hold ROI: {buyhold_roi:.2f}%")

    errors = y_true - y_pred
    metrics_dict = {
        "RMSE": m_rmse,
        "MAE": m_mae,
        "MAPE": m_mape,
        "R²": m_r2
    }

    wf_results = None
    if do_walk_forward:
        print("\n=== Walk-Forward Validation ===")
        wf_results = walk_forward_validation(
            pd.concat([train_df, test_df]),
            features,
            target_idx,
            window,
            n_splits=5
        )
        if not wf_results.empty:
            print(wf_results.to_string(index=False))
            print(f"\nMédia RMSE: {wf_results['rmse'].mean():.4f} ± {wf_results['rmse'].std():.4f}")
            print(f"Média R²:   {wf_results['r2'].mean():.4f} ± {wf_results['r2'].std():.4f}")

    min_len = min(len(test_idx), len(y_true), len(y_pred), len(errors), len(best_signals), len(best_equity), len(buyhold_equity))
    test_idx_al = test_idx[:min_len]
    y_true_al = y_true[:min_len]
    y_pred_al = y_pred[:min_len]
    errors_al = errors[:min_len]
    best_signals_al = best_signals[:min_len]
    best_equity_al = best_equity[:min_len]
    buyhold_equity_al = buyhold_equity[:min_len]

    save_plot_loss(hist, ticker, years, outdir)
    save_plot_errors(test_idx_al, errors_al, ticker, years, outdir)
    save_plot_error_distribution(errors_al, ticker, years, outdir)

    return {
        "years": years,
        "ticker": ticker,
        "df": df,
        "test_idx": test_idx_al,
        "y_true": y_true_al,
        "y_pred": y_pred_al,
        "errors": errors_al,
        "results_all": results_all,
        "rank": rank,
        "wf_results": wf_results,
        "metrics_dict": metrics_dict,
        "best_name": best_name,
        "best_signals": best_signals_al,
        "best_equity": best_equity_al,
        "buyhold_equity": buyhold_equity_al
    }


# =========================================================
# 9) PIPELINE COMPARATIVO 5 ANOS VS 10 ANOS
# =========================================================
def run_comparison_pipeline(
    ticker,
    outdir="resultados",
    initial_capital=5000,
    do_walk_forward=True,
    window=50,
    test_ratio=0.2
):
    os.makedirs(outdir, exist_ok=True)

    exp5 = run_single_experiment(
        ticker=ticker,
        years=5,
        window=window,
        test_ratio=test_ratio,
        outdir=outdir,
        initial_capital=initial_capital,
        do_walk_forward=do_walk_forward
    )

    exp10 = run_single_experiment(
        ticker=ticker,
        years=10,
        window=window,
        test_ratio=test_ratio,
        outdir=outdir,
        initial_capital=initial_capital,
        do_walk_forward=do_walk_forward
    )

    print("\n=== GERANDO GRÁFICOS COMPARATIVOS ===")
    save_plot_compare_real_vs_pred(exp5, exp10, ticker, outdir)
    save_plot_compare_real_vs_pred_clean(exp5, exp10, ticker, outdir)
    save_plot_compare_strategies(exp5, exp10, ticker, outdir)
    save_plot_compare_regression_metrics(exp5, exp10, ticker, outdir)
    save_plot_compare_best_equity(exp5, exp10, ticker, outdir)

    summary = pd.DataFrame([
        {
            "Experimento": "5 anos",
            "RMSE": exp5["metrics_dict"]["RMSE"],
            "MAE": exp5["metrics_dict"]["MAE"],
            "MAPE": exp5["metrics_dict"]["MAPE"],
            "R²": exp5["metrics_dict"]["R²"],
            "Melhor Estratégia": exp5["best_name"],
            "ROI Melhor Estratégia (%)": exp5["rank"].iloc[0]["roi"]
        },
        {
            "Experimento": "10 anos",
            "RMSE": exp10["metrics_dict"]["RMSE"],
            "MAE": exp10["metrics_dict"]["MAE"],
            "MAPE": exp10["metrics_dict"]["MAPE"],
            "R²": exp10["metrics_dict"]["R²"],
            "Melhor Estratégia": exp10["best_name"],
            "ROI Melhor Estratégia (%)": exp10["rank"].iloc[0]["roi"]
        }
    ])

    csv_path = f"{outdir}/resumo_comparativo_{ticker.replace('.', '_')}.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nResumo comparativo:")
    print(summary.to_string(index=False))
    print(f"\nCSV salvo em: {csv_path}")

    print("\nProcessamento concluído com sucesso.")
    return exp5, exp10, summary


# =========================================================
# MENU PRINCIPAL
# =========================================================
if __name__ == "__main__":
    bancos = {
        "1": ("ITUB4.SA", "Itaú"),
        "2": ("BBDC4.SA", "Bradesco"),
        "3": ("BBAS3.SA", "Banco do Brasil"),
        "4": ("SANB11.SA", "Santander"),
        "5": ("BPAC11.SA", "BTG Pactual"),
    }

    print("\n=== LSTM MULTI-ESTRATÉGIAS (LONG/SHORT) — COMPARAÇÃO 5 ANOS VS 10 ANOS ===")
    for k, (t, n) in bancos.items():
        print(f"{k}) {t} — {n}")

    c = input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos:
        c = "1"
    ticker = bancos[c][0]
    nome_ticker = bancos[c][1].replace(" ", "_")

    wf = input("\nExecutar walk-forward validation? (s/n): ").strip().lower()
    do_wf = (wf == "s")

    outdir = f"resultados/comparativo_5_vs_10_{nome_ticker}"

    run_comparison_pipeline(
        ticker=ticker,
        outdir=outdir,
        do_walk_forward=do_wf
    )