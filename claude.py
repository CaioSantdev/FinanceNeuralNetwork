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
# 1) DOWNLOAD E PRÉ-PROCESSAMENTO
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
    return out


def train_test_split_ordered(df, test_ratio=0.2):
    n = len(df)
    n_train = int((1 - test_ratio) * n)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


# =========================================================
# 2) INDICADORES TÉCNICOS SEM DATA LEAKAGE
#
#  Princípio geral:
#   - Cada indicador é calculado APENAS com dados de treino
#     usando ewm/rolling normais.
#   - Para o teste, o estado final do treino é propagado
#     passo a passo (online), sem olhar para frente.
# =========================================================

# ----------------------------------------------------------
# EMA
# ----------------------------------------------------------
def compute_ema_no_leakage(train_series: pd.Series,
                            test_series: pd.Series,
                            span: int):
    """EMA sem leakage: treino normal, teste propaga estado."""
    ema_train = train_series.ewm(span=span, adjust=False).mean()

    alpha = 2.0 / (span + 1)
    last_val = ema_train.iloc[-1]

    test_vals = []
    for v in test_series:
        last_val = alpha * v + (1 - alpha) * last_val
        test_vals.append(last_val)

    return ema_train, pd.Series(test_vals, index=test_series.index)


# ----------------------------------------------------------
# RSI
# ----------------------------------------------------------
def compute_rsi_no_leakage(train_series: pd.Series,
                            test_series: pd.Series,
                            period: int = 14):
    """
    RSI de Wilder (EWM com alpha=1/period) sem leakage.
    Guarda avg_gain e avg_loss do último passo do treino
    e propaga para o teste passo a passo.
    """
    alpha = 1.0 / period

    # --- Treino ---
    delta_train = train_series.diff().fillna(0)
    gain_train  = delta_train.clip(lower=0)
    loss_train  = (-delta_train).clip(lower=0)

    avg_gain_train = gain_train.ewm(alpha=alpha, adjust=False).mean()
    avg_loss_train = loss_train.ewm(alpha=alpha, adjust=False).mean()

    rs_train   = avg_gain_train / (avg_loss_train + 1e-8)
    rsi_train  = 100 - 100 / (1 + rs_train)

    # Estado final do treino
    last_avg_gain = avg_gain_train.iloc[-1]
    last_avg_loss = avg_loss_train.iloc[-1]
    last_price    = train_series.iloc[-1]

    # --- Teste (online) ---
    rsi_test_vals = []
    prev_price = last_price

    for price in test_series:
        delta = price - prev_price
        gain  = max(delta, 0.0)
        loss  = max(-delta, 0.0)

        last_avg_gain = alpha * gain + (1 - alpha) * last_avg_gain
        last_avg_loss = alpha * loss + (1 - alpha) * last_avg_loss

        rs  = last_avg_gain / (last_avg_loss + 1e-8)
        rsi = 100 - 100 / (1 + rs)
        rsi_test_vals.append(rsi)
        prev_price = price

    return rsi_train, pd.Series(rsi_test_vals, index=test_series.index)


# ----------------------------------------------------------
# MACD Line  (EMA_fast − EMA_slow)
# ----------------------------------------------------------
def compute_macd_no_leakage(train_series: pd.Series,
                             test_series: pd.Series,
                             fast: int = 12,
                             slow: int = 26):
    """
    Retorna apenas a MACD Line (EMA_fast - EMA_slow).
    Propaga os dois estados de EMA do treino para o teste.
    """
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)

    ema_fast_tr = train_series.ewm(span=fast, adjust=False).mean()
    ema_slow_tr = train_series.ewm(span=slow, adjust=False).mean()
    macd_train  = ema_fast_tr - ema_slow_tr

    # Estado final
    lf = ema_fast_tr.iloc[-1]
    ls = ema_slow_tr.iloc[-1]

    macd_test_vals = []
    for v in test_series:
        lf = alpha_fast * v + (1 - alpha_fast) * lf
        ls = alpha_slow * v + (1 - alpha_slow) * ls
        macd_test_vals.append(lf - ls)

    return macd_train, pd.Series(macd_test_vals, index=test_series.index)


# ----------------------------------------------------------
# ATR (Average True Range)
# ----------------------------------------------------------
def compute_atr_no_leakage(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            period: int = 14):
    """
    ATR com suavização EWM (Wilder) sem leakage.
    train_df / test_df devem ter colunas High, Low, Close.
    """
    alpha = 1.0 / period

    def _true_range(df):
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1).fillna(close.iloc[0])
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr

    tr_train = _true_range(train_df)
    atr_train = tr_train.ewm(alpha=alpha, adjust=False).mean()

    # Estado final
    last_atr   = atr_train.iloc[-1]
    last_close = train_df["Close"].iloc[-1]

    atr_test_vals = []
    prev_close = last_close

    for _, row in test_df.iterrows():
        h, l, c = row["High"], row["Low"], row["Close"]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        last_atr = alpha * tr + (1 - alpha) * last_atr
        atr_test_vals.append(last_atr)
        prev_close = c

    return atr_train, pd.Series(atr_test_vals, index=test_df.index)


# ----------------------------------------------------------
# OBV (On-Balance Volume)
# ----------------------------------------------------------
def compute_obv_no_leakage(train_df: pd.DataFrame,
                            test_df: pd.DataFrame):
    """
    OBV acumulado: propaga o valor final do treino para o teste.
    train_df / test_df devem ter colunas Close e Volume.
    """
    def _obv(df, start_obv=0.0):
        obv_vals = [start_obv]
        closes  = df["Close"].values
        volumes = df["Volume"].values
        for i in range(1, len(df)):
            if closes[i] > closes[i - 1]:
                obv_vals.append(obv_vals[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv_vals.append(obv_vals[-1] - volumes[i])
            else:
                obv_vals.append(obv_vals[-1])
        return pd.Series(obv_vals, index=df.index)

    obv_train = _obv(train_df, start_obv=0.0)
    obv_test  = _obv(test_df,  start_obv=obv_train.iloc[-1])

    return obv_train, obv_test


# ----------------------------------------------------------
# Largura das Bandas de Bollinger  (BB_width)
# ----------------------------------------------------------
def compute_bb_width_no_leakage(train_series: pd.Series,
                                  test_series: pd.Series,
                                  period: int = 20,
                                  n_std: float = 2.0):
    """
    BB_width = (upper - lower) / middle  (razão adimensional)
    Para o teste: janela deslizante combinando o final do treino
    com os valores do teste (sem olhar para frente).
    """
    # Treino: rolling normal
    rolling_mean = train_series.rolling(period).mean()
    rolling_std  = train_series.rolling(period).std(ddof=0)
    upper = rolling_mean + n_std * rolling_std
    lower = rolling_mean - n_std * rolling_std
    bb_width_train = (upper - lower) / (rolling_mean + 1e-8)
    bb_width_train = bb_width_train.fillna(method="bfill")

    # Buffer: últimos (period-1) valores do treino
    buffer = list(train_series.iloc[-(period - 1):])

    bb_width_test_vals = []
    for v in test_series:
        window = buffer + [v]
        arr = np.array(window[-period:])
        m   = arr.mean()
        s   = arr.std(ddof=0)
        bw  = (2 * n_std * s) / (m + 1e-8)
        bb_width_test_vals.append(bw)
        buffer.append(v)
        if len(buffer) > period - 1:
            buffer.pop(0)

    return bb_width_train, pd.Series(bb_width_test_vals, index=test_series.index)


# =========================================================
# 3) CONSTRUÇÃO DO DATAFRAME DE FEATURES
# =========================================================
def build_features(train_raw: pd.DataFrame,
                   test_raw: pd.DataFrame,
                   ema_span: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    """
    Calcula todos os indicadores sem leakage e monta
    os DataFrames de treino e teste prontos para o scaler.

    Retorna: train_feat, test_feat, feature_list, target_idx
    """
    close_tr = train_raw["Close"]
    close_te = test_raw["Close"]

    # EMA
    ema_tr, ema_te = compute_ema_no_leakage(close_tr, close_te, ema_span)

    # RSI
    rsi_tr, rsi_te = compute_rsi_no_leakage(close_tr, close_te, period=14)

    # MACD Line
    macd_tr, macd_te = compute_macd_no_leakage(close_tr, close_te, fast=12, slow=26)

    # ATR
    atr_tr, atr_te = compute_atr_no_leakage(train_raw, test_raw, period=14)

    # OBV
    obv_tr, obv_te = compute_obv_no_leakage(train_raw, test_raw)

    # BB_width
    bb_tr, bb_te = compute_bb_width_no_leakage(close_tr, close_te, period=20)

    # Montagem
    def _assemble(raw, ema, rsi, macd, atr, obv, bb):
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df[f"EMA_{ema_span}"] = ema.values
        df["RSI_14"]          = rsi.values
        df["MACD_line"]       = macd.values
        df["ATR_14"]          = atr.values
        df["OBV"]             = obv.values
        df["BB_width"]        = bb.values
        return df

    train_feat = _assemble(train_raw, ema_tr, rsi_tr, macd_tr, atr_tr, obv_tr, bb_tr)
    test_feat  = _assemble(test_raw,  ema_te, rsi_te, macd_te, atr_te, obv_te, bb_te)

    # Limpeza de NaN residuais (início da série rolling)
    train_feat = train_feat.fillna(method="bfill").fillna(method="ffill")
    test_feat  = test_feat.fillna(method="bfill").fillna(method="ffill")

    feature_list = list(train_feat.columns)
    target_idx   = feature_list.index("Close")

    print(f"\nFeatures ({len(feature_list)}): {feature_list}")
    return train_feat, test_feat, feature_list, target_idx


# =========================================================
# 4) JANELAS DESLIZANTES
# =========================================================
def make_windows(arr, target_idx, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)


# =========================================================
# 5) MÉTRICAS
# =========================================================
def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


# =========================================================
# 6) ARQUITETURA LSTM
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
# 7) ESTRATÉGIAS
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


def strategy_rsi_filter(pred, real, rsi, index):
    """
    Nova estratégia: usa RSI como filtro de confirmação.
    BUY apenas quando previsão sobe E RSI < 70 (não sobrecomprado).
    SHORT apenas quando previsão cai E RSI > 30 (não sobrevendido).
    """
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        diff_pct = (pred[i] - real[i - 1]) / real[i - 1]
        rsi_val  = rsi.iloc[i]
        if diff_pct >= 0.005 and rsi_val < 70:
            sig.append("BUY")
        elif diff_pct <= -0.005 and rsi_val > 30:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_macd_confirm(pred, real, macd, index):
    """
    Nova estratégia: MACD como confirmação de direção.
    BUY quando previsão sobe E MACD > 0 (momentum positivo).
    SHORT quando previsão cai E MACD < 0.
    """
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        diff_pct = (pred[i] - real[i - 1]) / real[i - 1]
        macd_val = macd.iloc[i]
        if diff_pct >= 0.005 and macd_val > 0:
            sig.append("BUY")
        elif diff_pct <= -0.005 and macd_val < 0:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_bb_breakout(pred, real, bb_width, ema, index):
    """
    Nova estratégia: opera somente quando BB_width está elevada
    (mercado em movimento), evitando entrar em lateralizações.
    Threshold dinâmico: mediana da largura histórica.
    """
    median_bw = bb_width.median()
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        diff_pct = (pred[i] - real[i - 1]) / real[i - 1]
        in_movement = bb_width.iloc[i] > median_bw
        if diff_pct >= 0.005 and in_movement:
            sig.append("BUY")
        elif diff_pct <= -0.005 and in_movement:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


# =========================================================
# 8) BACKTEST — LONG + SHORT
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
# 9) WALK-FORWARD VALIDATION
# =========================================================
def walk_forward_validation(train_feat: pd.DataFrame,
                             test_feat: pd.DataFrame,
                             train_raw: pd.DataFrame,
                             test_raw: pd.DataFrame,
                             feature_list: list,
                             target_idx: int,
                             window: int,
                             ema_span: int,
                             n_splits: int = 5):
    """
    Walk-forward com recálculo completo dos indicadores em cada split,
    garantindo zero leakage mesmo na validação cruzada.
    """
    # Concatena os DataFrames RAW para dividir em splits
    full_raw = pd.concat([train_raw, test_raw])
    split_size = len(full_raw) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        print(f"\n--- Walk-Forward Split {i+1}/{n_splits} ---")

        train_end = (i + 1) * split_size
        test_end  = train_end + split_size

        wf_train_raw = full_raw.iloc[:train_end]
        wf_test_raw  = full_raw.iloc[train_end:test_end]

        if len(wf_train_raw) < window + 50 or len(wf_test_raw) < window + 5:
            print("  Split muito pequeno, pulando.")
            continue

        # Recalcula indicadores sem leakage para este split
        wf_train_feat, wf_test_feat, _, _ = build_features(
            wf_train_raw, wf_test_raw, ema_span
        )

        # Scaler fitado apenas no treino deste split
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(wf_train_feat)
        test_scaled  = scaler.transform(wf_test_feat)

        X_train, y_train = make_windows(train_scaled, target_idx, window)
        X_test,  y_test  = make_windows(test_scaled,  target_idx, window)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = build_lstm(window, len(feature_list))
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
        target_std  = scaler.scale_[target_idx]
        y_true = y_test                 * target_std + target_mean
        y_pred = y_pred_scaled.ravel()  * target_std + target_mean

        results.append({
            'split': i + 1,
            'rmse':  rmse(y_true, y_pred),
            'mae':   mean_absolute_error(y_true, y_pred),
            'mape':  mape(y_true, y_pred),
            'r2':    r2_score(y_true, y_pred)
        })

    return pd.DataFrame(results)


# =========================================================
# 10) GRÁFICOS
# =========================================================
def save_plot_real_vs_pred(test_idx, y_true, y_pred, signals, ticker, strategy_name, outdir):
    plt.figure(figsize=(16, 6))
    plt.plot(test_idx, y_true, label="Real", linewidth=1.5)
    plt.plot(test_idx, y_pred, label="Previsto", linestyle="--")

    buy_idx   = [i for i, s in enumerate(signals) if s == "BUY"]
    short_idx = [i for i, s in enumerate(signals) if s == "SELL_SHORT"]

    if buy_idx:
        plt.scatter(test_idx[buy_idx],   y_true[buy_idx],
                    marker="^", color="green", s=60, label="Compra (LONG)")
    if short_idx:
        plt.scatter(test_idx[short_idx], y_true[short_idx],
                    marker="v", color="red",   s=60, label="Venda (SHORT)")

    plt.title(f"{ticker} — Preço Real vs Previsto ({strategy_name})")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/real_vs_pred_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_real_vs_pred_clean(test_idx, y_true, y_pred, ticker, strategy_name, outdir):
    plt.figure(figsize=(16, 6))
    plt.plot(test_idx, y_true, label="Real", linewidth=1.5)
    plt.plot(test_idx, y_pred, label="Previsto", linestyle="--")
    plt.title(f"{ticker} — Preço Real vs Previsto ({strategy_name})")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/real_vs_pred_clean_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()


def save_plot_loss(history, ticker, outdir):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"],     label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"{ticker} — Evolução da Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/loss_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_errors(test_idx, errors, ticker, outdir):
    plt.figure(figsize=(16, 5))
    plt.plot(test_idx, errors, color="crimson")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"{ticker} — Erros de Previsão")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/errors_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_error_distribution(errors, ticker, outdir):
    plt.figure(figsize=(12, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title(f"{ticker} — Distribuição dos Erros")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{outdir}/error_distribution_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_best_equity(best_equity, ticker, strategy_name, initial_capital, outdir):
    plt.figure(figsize=(14, 5))
    days = np.arange(len(best_equity))
    plt.plot(days, best_equity, label=strategy_name, linewidth=2, color='blue')
    plt.axhline(initial_capital, linestyle=":", color="gray", alpha=0.7,
                label=f"Capital Inicial (R${initial_capital:.2f})")
    peak = np.maximum.accumulate(best_equity)
    drawdown = (best_equity - peak) / peak * 100
    max_drawdown = drawdown.min()
    plt.title(f"{ticker} — Patrimônio ({strategy_name}) | Max DD: {max_drawdown:.2f}%")
    plt.xlabel("Dias de Trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    plt.tight_layout()
    path = f"{outdir}/best_equity_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_all_strategies(results_all, ticker, outdir):
    plt.figure(figsize=(16, 8))
    min_len = min(len(r["equity"]) for r in results_all.values())
    for name, r in results_all.items():
        eq = r["equity"][:min_len]
        plt.plot(np.arange(len(eq)), eq, label=f"{name} (ROI: {r['roi']:.2f}%)", linewidth=1.5)
    first_val = list(results_all.values())[0]["equity"][0]
    plt.axhline(y=first_val, color='gray', linestyle=':', alpha=0.7,
                label=f'Capital Inicial (R${first_val:.2f})')
    plt.title(f"{ticker} — Comparação entre Estratégias", fontsize=14, fontweight='bold')
    plt.xlabel("Dias de Trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(loc='best', fontsize=9); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    plt.tight_layout()
    path = f"{outdir}/comparison_strategies_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_strategy_vs_buyhold(best_equity, buyhold_equity, ticker, strategy_name, outdir):
    min_len = min(len(best_equity), len(buyhold_equity))
    best_equity    = best_equity[:min_len]
    buyhold_equity = buyhold_equity[:min_len]
    roi_strategy = (best_equity[-1]    / best_equity[0]    - 1) * 100
    roi_buyhold  = (buyhold_equity[-1] / buyhold_equity[0] - 1) * 100

    plt.figure(figsize=(16, 8))
    days = np.arange(min_len)
    plt.plot(days, best_equity,    label=f"{strategy_name} (ROI: {roi_strategy:.2f}%)", linewidth=2, color='blue')
    plt.plot(days, buyhold_equity, label=f"Buy & Hold (ROI: {roi_buyhold:.2f}%)",       linewidth=2, color='green', linestyle='--')
    plt.axhline(y=best_equity[0], color='gray', linestyle=':', alpha=0.7,
                label=f'Capital Inicial (R${best_equity[0]:.2f})')
    plt.title(f"{ticker} — Melhor Estratégia vs Buy & Hold", fontsize=14, fontweight='bold')
    plt.xlabel("Dias de Trading"); plt.ylabel("Patrimônio (R$)")
    plt.legend(loc='best', fontsize=10); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    plt.tight_layout()
    path = f"{outdir}/strategy_vs_buyhold_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")
    print(f"   Estratégia: R${best_equity[0]:.2f} → R${best_equity[-1]:.2f} ({roi_strategy:.2f}%)")
    print(f"   Buy & Hold: R${buyhold_equity[0]:.2f} → R${buyhold_equity[-1]:.2f} ({roi_buyhold:.2f}%)")


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
    print(f"Gráfico salvo: {path}")


def save_plot_walk_forward(wf_results, ticker, outdir):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(wf_results['split'], wf_results['rmse'], marker='o')
    plt.title('RMSE por Split'); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(wf_results['split'], wf_results['mape'], marker='o', color='orange')
    plt.title('MAPE por Split'); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(wf_results['split'], wf_results['r2'], marker='o', color='green')
    plt.title('R² por Split'); plt.grid(True, alpha=0.3); plt.ylim(-1, 1)

    plt.subplot(2, 2, 4)
    metrics = ['rmse', 'mae', 'mape']
    means = [wf_results[m].mean() for m in metrics]
    plt.bar(metrics, means); plt.title('Média das Métricas'); plt.grid(True, alpha=0.3)

    plt.suptitle(f'{ticker} — Walk-Forward Validation')
    plt.tight_layout()
    path = f"{outdir}/walk_forward_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


def save_plot_indicators(test_idx, y_true, rsi_test, macd_test, atr_test, bb_test, ticker, outdir):
    """Painel com os 4 novos indicadores no período de teste."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    axes[0].plot(test_idx, rsi_test.values, color='purple')
    axes[0].axhline(70, color='red',   linestyle='--', alpha=0.6, label='Sobrecompra (70)')
    axes[0].axhline(30, color='green', linestyle='--', alpha=0.6, label='Sobrevenda (30)')
    axes[0].set_title('RSI 14'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(test_idx, macd_test.values, color='darkorange')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('MACD Line'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(test_idx, atr_test.values, color='teal')
    axes[2].set_title('ATR 14'); axes[2].grid(True, alpha=0.3)

    axes[3].plot(test_idx, bb_test.values, color='navy')
    axes[3].axhline(bb_test.median(), color='gray', linestyle='--', alpha=0.6, label='Mediana')
    axes[3].set_title('BB Width'); axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'{ticker} — Indicadores Técnicos (Período de Teste)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f"{outdir}/indicators_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Gráfico salvo: {path}")


# =========================================================
# 11) PIPELINE PRINCIPAL
# =========================================================
def run_pipeline(ticker, window=50, test_ratio=0.2, outdir="resultados",
                 initial_capital=5000, do_walk_forward=True, years=10):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  LSTM com Indicadores Técnicos — {ticker}")
    print(f"{'='*60}")

    # --- Download e pré-processamento ---
    raw = fetch_hist(ticker, years=years)
    df  = zanotto_preprocess(raw)

    # EMA span adaptativo
    ema_span = max(5, min(int(len(df) * 0.2), 60))

    # Split ANTES de qualquer indicador
    train_raw, test_raw = train_test_split_ordered(
        df[["Open", "High", "Low", "Close", "Volume"]], test_ratio
    )

    # --- Calcula todos os indicadores sem leakage ---
    train_feat, test_feat, feature_list, target_idx = build_features(
        train_raw, test_raw, ema_span
    )

    # --- Scaler fitado APENAS no treino ---
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled  = scaler.transform(test_feat)

    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test,  y_test  = make_windows(test_scaled,  target_idx, window)

    print(f"\nTreino: {X_train.shape} | Teste: {X_test.shape}")

    # --- Treinamento ---
    model = build_lstm(window, len(feature_list))
    hist = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        shuffle=False,
        callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # --- Previsão e inversão de escala ---
    y_pred_scaled = model.predict(X_test, verbose=0)
    target_mean   = scaler.mean_[target_idx]
    target_std    = scaler.scale_[target_idx]
    y_true = y_test                * target_std + target_mean
    y_pred = y_pred_scaled.ravel() * target_std + target_mean

    m_rmse = rmse(y_true, y_pred)
    m_mae  = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2   = r2_score(y_true, y_pred)

    print(f"\n{'='*40}")
    print("MÉTRICAS DE REGRESSÃO")
    print(f"  RMSE : {m_rmse:.4f}")
    print(f"  MAE  : {m_mae:.4f}")
    print(f"  MAPE : {m_mape:.2f}%")
    print(f"  R²   : {m_r2:.4f}")
    print(f"{'='*40}")

    # --- Índice do período de teste (alinhado com as janelas) ---
    test_start_idx = len(train_feat) + window
    test_idx = df.index[test_start_idx:]

    # --- Extrair indicadores do período de teste para estratégias ---
    ema_test_aligned  = test_feat.loc[test_idx, f"EMA_{ema_span}"]
    rsi_test_aligned  = test_feat.loc[test_idx, "RSI_14"]
    macd_test_aligned = test_feat.loc[test_idx, "MACD_line"]
    atr_test_aligned  = test_feat.loc[test_idx, "ATR_14"]
    bb_test_aligned   = test_feat.loc[test_idx, "BB_width"]

    # --- Estratégias ---
    strategies = {
        # Originais
        "gap_1%":         lambda: strategy_gap_pct(y_pred, y_true, 0.01),
        "gap_0.7%":       lambda: strategy_gap_pct(y_pred, y_true, 0.007),
        "trend_filter":   lambda: strategy_trend_filter(y_pred, y_true, ema_test_aligned, test_idx),
        "mean_reversion": lambda: strategy_mean_reversion(y_pred, y_true, ema_test_aligned, test_idx),
        "always_in":      lambda: strategy_always_in(y_pred, y_true),
        # Novas (usam os indicadores adicionados)
        "rsi_filter":     lambda: strategy_rsi_filter(y_pred, y_true, rsi_test_aligned, test_idx),
        "macd_confirm":   lambda: strategy_macd_confirm(y_pred, y_true, macd_test_aligned, test_idx),
        "bb_breakout":    lambda: strategy_bb_breakout(y_pred, y_true, bb_test_aligned, ema_test_aligned, test_idx),
    }

    print("\n=== Testando Estratégias (LONG/SHORT) ===")
    results_all = {}

    for name, fn in strategies.items():
        print(f"\n→ {name}")
        signals = fn()
        equity, trades = backtest_discrete_short(y_true, signals, initial_capital=initial_capital)
        roi     = (equity[-1] / initial_capital - 1) * 100
        buy_c   = signals.count("BUY")
        short_c = signals.count("SELL_SHORT")
        hold_c  = signals.count("HOLD")
        print(f"   ROI: {roi:.2f}% | Trades: {len(trades)} | BUY: {buy_c} | SHORT: {short_c} | HOLD: {hold_c}")
        results_all[name] = {"roi": roi, "equity": equity, "signals": signals, "trades": trades}

    rank = pd.DataFrame([
        {"estrategia": k, "roi": v["roi"], "trades": len(v["trades"])}
        for k, v in results_all.items()
    ]).sort_values("roi", ascending=False).reset_index(drop=True)

    print("\n=== Ranking das Estratégias ===")
    print(rank.to_string(index=False))

    best_name    = rank.iloc[0]["estrategia"]
    best_equity  = results_all[best_name]["equity"]
    best_signals = results_all[best_name]["signals"]

    buyhold_equity = backtest_buy_and_hold(y_true, initial_capital=initial_capital)
    buyhold_roi    = (buyhold_equity[-1] / initial_capital - 1) * 100

    print(f"\n🏆 Melhor estratégia : {best_name} | ROI: {rank.iloc[0]['roi']:.2f}%")
    print(f"📌 Buy & Hold ROI    : {buyhold_roi:.2f}%")

    # --- Walk-Forward ---
    wf_results = None
    if do_walk_forward:
        print("\n=== Walk-Forward Validation ===")
        wf_results = walk_forward_validation(
            train_feat, test_feat,
            train_raw,  test_raw,
            feature_list, target_idx,
            window, ema_span, n_splits=5
        )
        if not wf_results.empty:
            print(wf_results.to_string(index=False))
            print(f"\nMédia RMSE : {wf_results['rmse'].mean():.4f} ± {wf_results['rmse'].std():.4f}")
            print(f"Média R²   : {wf_results['r2'].mean():.4f} ± {wf_results['r2'].std():.4f}")
            save_plot_walk_forward(wf_results, ticker, outdir)

    # --- Alinhamento final ---
    min_len = min(len(test_idx), len(y_true), len(best_signals), len(best_equity))
    test_idx_al       = test_idx[:min_len]
    y_true_al         = y_true[:min_len]
    y_pred_al         = y_pred[:min_len]
    errors_al         = (y_true - y_pred)[:min_len]
    best_signals_al   = best_signals[:min_len]
    best_equity_al    = best_equity[:min_len]
    buyhold_equity_al = buyhold_equity[:min_len]
    rsi_al            = rsi_test_aligned[:min_len]
    macd_al           = macd_test_aligned[:min_len]
    atr_al            = atr_test_aligned[:min_len]
    bb_al             = bb_test_aligned[:min_len]

    # --- Gráficos ---
    print("\n=== Gerando Gráficos ===")
    metrics_dict = {"RMSE": m_rmse, "MAE": m_mae, "MAPE": m_mape, "R²": m_r2}

    save_plot_real_vs_pred(test_idx_al, y_true_al, y_pred_al, best_signals_al, ticker, best_name, outdir)
    save_plot_real_vs_pred_clean(test_idx_al, y_true_al, y_pred_al, ticker, best_name, outdir)
    save_plot_loss(hist, ticker, outdir)
    save_plot_errors(test_idx_al, errors_al, ticker, outdir)
    save_plot_error_distribution(errors_al, ticker, outdir)
    save_plot_best_equity(best_equity_al, ticker, best_name, initial_capital, outdir)
    save_plot_all_strategies(results_all, ticker, outdir)
    save_plot_strategy_vs_buyhold(best_equity_al, buyhold_equity_al, ticker, best_name, outdir)
    save_plot_regression_metrics(metrics_dict, ticker, outdir)
    save_plot_indicators(test_idx_al, y_true_al, rsi_al, macd_al, atr_al, bb_al, ticker, outdir)

    print("\n✅ Processamento concluído com sucesso.")
    return results_all, rank, wf_results


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

    print("\n=== LSTM + INDICADORES TÉCNICOS (LONG/SHORT) — SEM DATA LEAKAGE ===")
    print("\nIndicadores: EMA · RSI · MACD · ATR · OBV · BB_width\n")
    for k, (t, n) in bancos.items():
        print(f"  {k}) {t} — {n}")

    c = input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos:
        c = "1"
    ticker = bancos[c][0]

    wf = input("\nExecutar walk-forward validation? (s/n): ").strip().lower()
    do_wf = (wf == 's')

    print("\nPeríodo dos dados:")
    print("  1) 5 anos  |  2) 10 anos  |  3) 15 anos")
    p = input("Escolha [1-3]: ").strip()
    years = {'1': 5, '2': 10, '3': 15}.get(p, 10)

    outdir = f"resultados/{years}anos_{bancos[c][1].replace(' ', '_')}"

    run_pipeline(ticker, outdir=outdir, do_walk_forward=do_wf, years=years)