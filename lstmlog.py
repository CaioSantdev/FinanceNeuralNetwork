# =========================================================
# LSTM Zanotto — Log Returns + Rolling Refit + Estratégias Revisadas
#
# MUDANÇAS PRINCIPAIS vs versão anterior:
#
# [A] TARGET: log_return em vez de preço absoluto
#     → Série estacionária, sem drift de escala
#     → Retornos ficam em [-0.1, +0.1], scaler mais estável
#     → Preço reconstruído: price[t] = price[t-1] * exp(pred_return)
#
# [B] ROLLING REFIT
#     → A cada `refit_every` dias, o modelo é retreinado com todos
#       os dados disponíveis até aquele ponto
#     → Combate concept drift sem vazar dados futuros
#
# [C] ESTRATÉGIAS REVISADAS
#     → Baseadas em retorno previsto (mais informativo que gap de preço)
#     → Limiar dinâmico baseado em volatilidade recente
#     → Filtro de regime: só opera LONG em tendência de alta, SHORT em baixa
#     → Estratégia de momentum: combina previsão + momentum recente
#
# [D] MÉTRICAS ADICIONAIS
#     → Sharpe Ratio, Calmar Ratio, Max Drawdown por estratégia
#     → Hit Rate (% de previsões de direção corretas)
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

# ── Reprodutibilidade ──────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 1) DOWNLOAD + PRÉ-PROCESSAMENTO
# =========================================================
def fetch_hist(ticker, years=10):
    end   = datetime.today()
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
    return df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]]


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


def compute_ema_no_leakage(train_series, test_series, span):
    ema_train = train_series.ewm(span=span, adjust=False).mean()
    alpha = 2.0 / (span + 1)
    last_val = ema_train.iloc[-1]
    vals = []
    for v in test_series:
        last_val = alpha * v + (1 - alpha) * last_val
        vals.append(last_val)
    return ema_train, pd.Series(vals, index=test_series.index)


def add_features(df, ema_span):
    """
    [A] Adiciona log_return como target e features derivadas.
    Tudo calculado de forma causal (sem lookahead).
    """
    out = df.copy()

    # [A] Target: log return diário
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))

    # Features técnicas causais
    out["log_vol"]    = np.log1p(out["Volume"])
    out["hl_range"]   = (out["High"] - out["Low"]) / out["Close"]   # range relativo
    out["ret_2"]      = out["log_return"].shift(1)                   # retorno D-2
    out["ret_3"]      = out["log_return"].shift(2)                   # retorno D-3
    out["vol_5"]      = out["log_return"].rolling(5).std()           # volatilidade 5d
    out["vol_20"]     = out["log_return"].rolling(20).std()          # volatilidade 20d
    out[f"EMA_{ema_span}"] = out["Close"].ewm(span=ema_span, adjust=False).mean()

    out = out.dropna()
    return out


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


# =========================================================
# 2) MÉTRICAS
# =========================================================
def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def hit_rate(y_true, y_pred):
    """% de previsões que acertaram a DIREÇÃO do movimento"""
    correct = np.sign(y_true) == np.sign(y_pred)
    return correct.mean() * 100

def sharpe_ratio(returns, periods=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return dd.min() * 100

def calmar_ratio(equity, periods=252):
    returns = np.diff(equity) / equity[:-1]
    ann_ret = (equity[-1] / equity[0]) ** (periods / len(equity)) - 1
    mdd     = abs(max_drawdown(equity) / 100)
    return ann_ret / mdd if mdd > 0 else 0.0


# =========================================================
# 3) ARQUITETURA LSTM
# =========================================================
def build_lstm(n_steps, n_features, units=256, dropout=0.3):
    model = models.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")   # Huber: menos sensível a outliers
    return model


# =========================================================
# 4) ROLLING REFIT
# =========================================================
def rolling_refit_predict(
    df_features, features, target_col, window,
    train_size, refit_every=21, epochs=30, batch_size=64
):
    """
    [B] Previsão com rolling refit:
    - Treina no bloco inicial de `train_size` amostras
    - A cada `refit_every` dias, retreina com todos os dados até o momento
    - Nunca usa dados futuros (fit sempre no passado, predict no próximo dia)

    Retorna arrays alinhados com o período de teste.
    """
    target_idx = list(df_features.columns).index(target_col)
    arr_full   = df_features[features].values
    prices_full = df_features["Close"].values

    all_true, all_pred, all_dates = [], [], []
    all_prices_true = []

    n_total = len(arr_full)
    test_start = train_size

    scaler  = None
    model   = None
    last_refit = test_start  # força treino inicial

    print(f"  Total: {n_total} | Treino inicial: {train_size} | Teste: {n_total - train_size}")
    print(f"  Refit a cada {refit_every} dias")

    for i in range(test_start + window, n_total):

        # ── Refit? ──────────────────────────────────────────
        if (i - last_refit) >= refit_every or model is None:
            train_arr = arr_full[:i]   # apenas passado

            scaler      = StandardScaler()
            train_scaled = scaler.fit_transform(train_arr)

            X_tr, y_tr = make_windows(train_scaled, target_idx, window)
            if len(X_tr) == 0:
                continue

            n_features = len(features)
            model = build_lstm(window, n_features)
            model.fit(
                X_tr, y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                shuffle=False,
                callbacks=[
                    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ],
                verbose=0
            )
            last_refit = i
            print(f"  ↺ Refit em t={i} ({df_features.index[i].date()})")

        # ── Predict próximo dia ─────────────────────────────
        window_arr    = arr_full[i - window:i]
        window_scaled = scaler.transform(window_arr)
        X_pred        = window_scaled[np.newaxis, :, :]   # (1, window, features)

        pred_scaled = model.predict(X_pred, verbose=0)[0, 0]

        # Inverter scaling do target
        t_mean = scaler.mean_[target_idx]
        t_std  = scaler.scale_[target_idx]
        pred_return = pred_scaled * t_std + t_mean
        true_return = arr_full[i, target_idx]

        all_pred.append(pred_return)
        all_true.append(true_return)
        all_dates.append(df_features.index[i])
        all_prices_true.append(prices_full[i])

    return (
        np.array(all_true),
        np.array(all_pred),
        pd.DatetimeIndex(all_dates),
        np.array(all_prices_true)
    )


# =========================================================
# 5) RECONSTRUÇÃO DE PREÇO A PARTIR DE LOG RETURNS
# =========================================================
def reconstruct_prices(pred_returns, prices_true, dates):
    """
    [A] Reconstrói série de preços a partir dos log returns previstos.
    Usa o preço real do dia anterior como âncora (sem acumular erro).
    """
    # Preço "previsto" para o dia t = preço real de t-1 * exp(pred_return_t)
    # Isso é o que o modelo realmente afirma: qual será o próximo movimento
    prices_pred = np.zeros(len(pred_returns))
    prices_real_lag = np.roll(prices_true, 1)
    prices_real_lag[0] = prices_true[0]

    for i in range(len(pred_returns)):
        prices_pred[i] = prices_real_lag[i] * np.exp(pred_returns[i])

    return prices_pred


# =========================================================
# 6) ESTRATÉGIAS REVISADAS
# =========================================================
def strategy_return_threshold(pred_returns, prices_true, vol_window=10, k=0.5):
    """
    [C] Limiar dinâmico baseado em volatilidade recente.
    Opera LONG se pred_return > k * vol_recente
    Opera SHORT se pred_return < -k * vol_recente
    Evita operar em mercados laterais (sem sinal claro).
    """
    sig = ["HOLD"]
    for i in range(1, len(pred_returns)):
        start = max(0, i - vol_window)
        vol = np.std(pred_returns[start:i]) if i > 1 else 0.01
        threshold = k * vol
        if pred_returns[i] > threshold:
            sig.append("BUY")
        elif pred_returns[i] < -threshold:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_regime_filter(pred_returns, prices_true, ema):
    """
    [C] Filtro de regime: opera na direção da tendência de longo prazo.
    LONG apenas quando preço > EMA (bull market)
    SHORT apenas quando preço < EMA (bear market)
    HOLD quando previsão contradiz o regime.
    """
    sig = ["HOLD"]
    for i in range(1, len(pred_returns)):
        bull = prices_true[i - 1] > ema.iloc[i - 1]
        bear = prices_true[i - 1] < ema.iloc[i - 1]
        up   = pred_returns[i] > 0
        down = pred_returns[i] < 0

        if bull and up:
            sig.append("BUY")
        elif bear and down:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_momentum_confirm(pred_returns, prices_true, momentum_window=5):
    """
    [C] Confirmação de momentum: sinal gerado apenas quando previsão
    e momentum recente concordam — reduz sinais falsos.
    """
    sig = ["HOLD"]
    for i in range(1, len(pred_returns)):
        start = max(0, i - momentum_window)
        mom   = np.mean(pred_returns[start:i]) if i > 1 else 0.0
        if pred_returns[i] > 0 and mom > 0:
            sig.append("BUY")
        elif pred_returns[i] < 0 and mom < 0:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_top_quantile(pred_returns, q=0.7):
    """
    [C] Opera apenas nos dias com previsões mais extremas (top/bottom quantil).
    Reduz muito o número de trades, foca nos sinais mais fortes.
    """
    sig = []
    for i in range(len(pred_returns)):
        window = pred_returns[:i+1]
        if i < 10:
            sig.append("HOLD")
            continue
        high_thresh = np.quantile(window, q)
        low_thresh  = np.quantile(window, 1 - q)
        if pred_returns[i] >= high_thresh:
            sig.append("BUY")
        elif pred_returns[i] <= low_thresh:
            sig.append("SELL_SHORT")
        else:
            sig.append("HOLD")
    return sig


def strategy_always_in(pred_returns):
    """Sempre posicionado na direção da previsão."""
    return ["BUY" if r > 0 else "SELL_SHORT" for r in pred_returns]


# =========================================================
# 7) BACKTEST
# =========================================================
def backtest_discrete_short(
    prices, signals,
    initial_capital=5000, fee_rate=0.0003,
    stop_loss=0.03, take_profit=0.06, min_volume=100
):
    cash = initial_capital
    shares, entry_price, position_type = 0, None, None
    equity, trades = [], []

    for i, (price, sig) in enumerate(zip(prices, signals)):
        price = float(price)

        # Stop-loss / Take-profit
        if shares != 0 and entry_price is not None:
            var = (price - entry_price) / entry_price
            if position_type == 'LONG':
                if var <= -stop_loss:
                    cash += shares * price * (1 - fee_rate)
                    trades.append((i, "SL_LONG",  shares, price, entry_price))
                    shares, entry_price, position_type = 0, None, None
                elif var >= take_profit:
                    cash += shares * price * (1 - fee_rate)
                    trades.append((i, "TP_LONG",  shares, price, entry_price))
                    shares, entry_price, position_type = 0, None, None
            elif position_type == 'SHORT':
                if var >= stop_loss:
                    qty = abs(shares)
                    cash -= qty * price * (1 + fee_rate)
                    trades.append((i, "SL_SHORT", shares, price, entry_price))
                    shares, entry_price, position_type = 0, None, None
                elif var <= -take_profit:
                    qty = abs(shares)
                    cash -= qty * price * (1 + fee_rate)
                    trades.append((i, "TP_SHORT", shares, price, entry_price))
                    shares, entry_price, position_type = 0, None, None

        # Execução de sinais
        if shares == 0:
            if sig == "BUY":
                qty = (int(cash // price) // min_volume) * min_volume
                if qty >= min_volume:
                    cash -= qty * price * (1 + fee_rate)
                    shares, entry_price, position_type = qty, price, 'LONG'
                    trades.append((i, "BUY", qty, price, price))
            elif sig == "SELL_SHORT":
                qty = (int(cash // price) // min_volume) * min_volume
                if qty >= min_volume:
                    cash += qty * price * (1 - fee_rate)
                    shares, entry_price, position_type = -qty, price, 'SHORT'
                    trades.append((i, "SHORT", -qty, price, price))
        elif shares > 0 and sig == "SELL_SHORT":
            cash += shares * price * (1 - fee_rate)
            trades.append((i, "CLOSE_LONG", shares, price, entry_price))
            shares, entry_price, position_type = 0, None, None
        elif shares < 0 and sig == "BUY":
            qty = abs(shares)
            cash -= qty * price * (1 + fee_rate)
            trades.append((i, "COVER", shares, price, entry_price))
            shares, entry_price, position_type = 0, None, None

        equity.append(cash + (shares * price if shares != 0 else 0))

    # Liquidação final
    if shares != 0:
        price = float(prices[-1])
        if shares > 0:
            cash += shares * price * (1 - fee_rate)
        else:
            cash -= abs(shares) * price * (1 + fee_rate)
        equity[-1] = cash

    return np.array(equity), pd.DataFrame(
        trades, columns=["idx", "action", "shares", "price", "entry_price"]
    )


def backtest_buy_and_hold(prices, initial_capital=5000, fee_rate=0.0003, min_volume=100):
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return np.array([initial_capital])
    qty  = (int(initial_capital // prices[0]) // min_volume) * min_volume
    cash = initial_capital - qty * prices[0] * (1 + fee_rate) if qty >= min_volume else initial_capital
    eq   = [cash + qty * p for p in prices]
    if qty >= min_volume:
        eq[-1] = cash + qty * prices[-1] * (1 - fee_rate)
    return np.array(eq)


# =========================================================
# 8) PLOTS
# =========================================================
def save_plots(outdir, ticker, dates, y_true_ret, y_pred_ret,
               prices_true, prices_pred, hist,
               results_all, best_name, buyhold_equity, initial_capital):

    os.makedirs(outdir, exist_ok=True)

    # ── Loss ──────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(hist.history["loss"],     label="Treino")
    plt.plot(hist.history["val_loss"], label="Validação")
    plt.title(f"{ticker} — Loss (Huber)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{outdir}/loss_{ticker}.png", dpi=140); plt.close()

    # ── Retornos reais vs previstos ────────────────────────
    plt.figure(figsize=(16, 5))
    plt.plot(dates, y_true_ret, label="Retorno Real",    alpha=0.7, linewidth=1)
    plt.plot(dates, y_pred_ret, label="Retorno Previsto", alpha=0.7, linewidth=1, linestyle="--")
    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.title(f"{ticker} — Log Returns: Real vs Previsto")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{outdir}/returns_{ticker}.png", dpi=140); plt.close()

    # ── Hit Rate ao longo do tempo ─────────────────────────
    rolling_hit = pd.Series(
        (np.sign(y_true_ret) == np.sign(y_pred_ret)).astype(float),
        index=dates
    ).rolling(30).mean() * 100
    plt.figure(figsize=(16, 4))
    plt.plot(dates, rolling_hit, color="purple", linewidth=1.5)
    plt.axhline(50, color="gray", linestyle="--", label="Chance aleatória (50%)")
    plt.title(f"{ticker} — Hit Rate Direcional (rolling 30d)")
    plt.ylabel("Hit Rate (%)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{outdir}/hit_rate_{ticker}.png", dpi=140); plt.close()

    # ── Preço real vs reconstruído ─────────────────────────
    plt.figure(figsize=(16, 6))
    plt.plot(dates, prices_true, label="Preço Real",        linewidth=1.5)
    plt.plot(dates, prices_pred, label="Preço Reconstruído", linestyle="--", alpha=0.8)
    plt.title(f"{ticker} — Preço Real vs Reconstruído (via log returns)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{outdir}/price_reconstructed_{ticker}.png", dpi=140); plt.close()

    # ── Comparação de estratégias ──────────────────────────
    plt.figure(figsize=(16, 8))
    min_len = min(len(r["equity"]) for r in results_all.values())
    for name, r in results_all.items():
        eq = r["equity"][:min_len]
        plt.plot(np.arange(min_len), eq,
                 label=f"{name} (ROI:{r['roi']:.1f}% SR:{r['sharpe']:.2f})", linewidth=1.5)
    plt.axhline(initial_capital, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"{ticker} — Comparação entre Estratégias", fontweight="bold")
    plt.xlabel("Dias"); plt.ylabel("Patrimônio (R$)")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{outdir}/strategies_{ticker}.png", dpi=140); plt.close()

    # ── Melhor estratégia vs Buy & Hold ───────────────────
    best_eq = results_all[best_name]["equity"]
    min_len = min(len(best_eq), len(buyhold_equity))
    best_eq = best_eq[:min_len]; bh_eq = buyhold_equity[:min_len]
    roi_s  = (best_eq[-1]  / best_eq[0]  - 1) * 100
    roi_bh = (bh_eq[-1]    / bh_eq[0]    - 1) * 100

    plt.figure(figsize=(16, 7))
    plt.plot(np.arange(min_len), best_eq,
             label=f"{best_name} (ROI:{roi_s:.1f}%)",  linewidth=2, color="blue")
    plt.plot(np.arange(min_len), bh_eq,
             label=f"Buy & Hold (ROI:{roi_bh:.1f}%)", linewidth=2, color="green", linestyle="--")
    plt.axhline(initial_capital, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"{ticker} — Melhor Estratégia vs Buy & Hold", fontweight="bold")
    plt.xlabel("Dias"); plt.ylabel("Patrimônio (R$)")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(f"{outdir}/best_vs_buyhold_{ticker}.png", dpi=140); plt.close()

    # ── Distribuição de retornos previstos ─────────────────
    plt.figure(figsize=(12, 5))
    plt.hist(y_pred_ret, bins=50, alpha=0.6, label="Previsto", color="orange")
    plt.hist(y_true_ret, bins=50, alpha=0.6, label="Real",     color="blue")
    plt.title(f"{ticker} — Distribuição dos Log Returns")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{outdir}/return_dist_{ticker}.png", dpi=140); plt.close()

    print(f"\n📊 Gráficos salvos em: {outdir}/")


# =========================================================
# 9) PIPELINE COMPLETO
# =========================================================
def run_pipeline(ticker, window=30, test_ratio=0.2, outdir="resultados",
                 initial_capital=5000, refit_every=21, years=10):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  LSTM LOG RETURNS + ROLLING REFIT — {ticker}")
    print(f"{'='*55}")

    # ── Dados ──────────────────────────────────────────────
    raw = fetch_hist(ticker, years=years)
    df  = zanotto_preprocess(raw)

    ema_span = max(5, min(int(len(df) * 0.2), 60))
    df = add_features(df, ema_span)

    features = [
        "log_return", "log_vol", "hl_range",
        "ret_2", "ret_3", "vol_5", "vol_20", f"EMA_{ema_span}"
    ]
    target_col = "log_return"

    # Split ordenado (sem leakage)
    n_train = int(len(df) * (1 - test_ratio))

    # ── Rolling Refit + Previsão ───────────────────────────
    print(f"\n[B] Rolling Refit (refit a cada {refit_every} dias)...")
    y_true_ret, y_pred_ret, test_dates, prices_true = rolling_refit_predict(
        df_features=df,
        features=features,
        target_col=target_col,
        window=window,
        train_size=n_train,
        refit_every=refit_every,
        epochs=30,
        batch_size=64
    )

    if len(y_true_ret) == 0:
        print("❌ Sem dados de teste suficientes.")
        return None

    # ── Métricas de regressão ──────────────────────────────
    m_rmse = rmse(y_true_ret, y_pred_ret)
    m_mae  = mean_absolute_error(y_true_ret, y_pred_ret)
    m_hr   = hit_rate(y_true_ret, y_pred_ret)
    m_r2   = r2_score(y_true_ret, y_pred_ret)

    print(f"\n{'─'*45}")
    print(f"  MÉTRICAS DE REGRESSÃO (log returns)")
    print(f"  RMSE:     {m_rmse:.6f}")
    print(f"  MAE:      {m_mae:.6f}")
    print(f"  R²:       {m_r2:.4f}")
    print(f"  Hit Rate: {m_hr:.2f}%  ← % dias com direção correta")
    print(f"{'─'*45}")

    # ── Reconstruir preços ─────────────────────────────────
    prices_pred = reconstruct_prices(y_pred_ret, prices_true, test_dates)

    # ── EMA alinhada com período de teste ──────────────────
    ema_test = df.loc[test_dates, f"EMA_{ema_span}"]

    # ── Estratégias ────────────────────────────────────────
    strats = {
        "ret_threshold_k0.5": lambda: strategy_return_threshold(y_pred_ret, prices_true, k=0.5),
        "ret_threshold_k1.0": lambda: strategy_return_threshold(y_pred_ret, prices_true, k=1.0),
        "regime_filter":      lambda: strategy_regime_filter(y_pred_ret, prices_true, ema_test),
        "momentum_confirm":   lambda: strategy_momentum_confirm(y_pred_ret, prices_true),
        "top_quantile_70":    lambda: strategy_top_quantile(y_pred_ret, q=0.70),
        "always_in":          lambda: strategy_always_in(y_pred_ret),
    }

    print(f"\n{'─'*45}")
    print(f"  ESTRATÉGIAS DE TRADING")
    print(f"{'─'*45}")

    results_all = {}
    for name, fn in strats.items():
        signals = fn()
        equity, trades = backtest_discrete_short(
            prices_true, signals, initial_capital=initial_capital
        )
        roi    = (equity[-1] / initial_capital - 1) * 100
        eq_ret = np.diff(equity) / equity[:-1]
        sr     = sharpe_ratio(eq_ret)
        mdd    = max_drawdown(equity)
        cal    = calmar_ratio(equity)
        n_buy  = signals.count("BUY")
        n_shrt = signals.count("SELL_SHORT")
        n_hold = signals.count("HOLD")

        results_all[name] = {
            "roi": roi, "sharpe": sr, "mdd": mdd, "calmar": cal,
            "equity": equity, "signals": signals, "trades": trades
        }
        print(f"  {name:<25} ROI:{roi:+.1f}%  SR:{sr:.2f}  MDD:{mdd:.1f}%  "
              f"Trades:{len(trades)}  B:{n_buy} S:{n_shrt} H:{n_hold}")

    # ── Ranking (por Sharpe, não só ROI) ──────────────────
    rank = pd.DataFrame([
        {"estrategia": k, "roi": v["roi"], "sharpe": v["sharpe"],
         "mdd": v["mdd"], "calmar": v["calmar"], "trades": len(v["trades"])}
        for k, v in results_all.items()
    ]).sort_values("sharpe", ascending=False).reset_index(drop=True)

    print(f"\n{'─'*45}")
    print(f"  RANKING (por Sharpe Ratio)")
    print(f"{'─'*45}")
    print(rank.to_string(index=False))

    best_name   = rank.iloc[0]["estrategia"]
    best_equity = results_all[best_name]["equity"]

    # ── Buy & Hold ─────────────────────────────────────────
    bh_equity = backtest_buy_and_hold(prices_true, initial_capital=initial_capital)
    bh_roi    = (bh_equity[-1] / initial_capital - 1) * 100
    bh_ret    = np.diff(bh_equity) / bh_equity[:-1]
    bh_sr     = sharpe_ratio(bh_ret)

    print(f"\n  🏆 Melhor estratégia: {best_name}")
    print(f"     ROI: {rank.iloc[0]['roi']:+.2f}%  |  Sharpe: {rank.iloc[0]['sharpe']:.2f}")
    print(f"  📌 Buy & Hold: ROI {bh_roi:+.2f}%  |  Sharpe: {bh_sr:.2f}")

    # ── Treinar modelo final (para histórico de loss) ──────
    print(f"\n[Treinando modelo final para gráfico de loss...]")
    target_idx  = features.index(target_col)
    scaler_full = StandardScaler()
    train_arr   = df[features].values[:n_train]
    tr_scaled   = scaler_full.fit_transform(train_arr)
    X_tr, y_tr  = make_windows(tr_scaled, target_idx, window)
    model_final = build_lstm(window, len(features))
    hist_final  = model_final.fit(
        X_tr, y_tr, epochs=30, batch_size=64,
        validation_split=0.1, shuffle=False,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # ── Plots ──────────────────────────────────────────────
    ticker_safe = ticker.replace(".", "_")
    save_plots(
        outdir, ticker_safe, test_dates,
        y_true_ret, y_pred_ret,
        prices_true, prices_pred,
        hist_final,
        results_all, best_name, bh_equity, initial_capital
    )

    print(f"\n✅ Concluído — {ticker}")
    return results_all, rank


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

    print("\n=== LSTM LOG RETURNS + ROLLING REFIT ===")
    for k, (t, n) in bancos.items():
        print(f"  {k}) {t} — {n}")

    c = input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos:
        c = "1"
    ticker = bancos[c][0]

    print("\nPeríodo dos dados:")
    print("  1) 5 anos | 2) 10 anos | 3) 15 anos")
    p     = input("Escolha [1-3]: ").strip()
    years = {"1": 5, "2": 10, "3": 15}.get(p, 10)

    print(f"\nRefit a cada quantos dias? (padrão: 21 = ~1 mês)")
    r = input("Refit every [21]: ").strip()
    refit_every = int(r) if r.isdigit() else 21

    outdir = f"resultados/{years}anos_{bancos[c][1].replace(' ', '_')}_logret"

    run_pipeline(
        ticker,
        window=30,
        test_ratio=0.2,
        outdir=outdir,
        initial_capital=5000,
        refit_every=refit_every,
        years=years
    )