# =========================================================
# ZanottoTrading.py
# 🧠 LSTM (Zanotto) + Estratégias de Trade + Backtesting
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

# 1) Ddownload e pre processamento
def fetch_hist(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Baixa histórico do Yahoo Finance sem ajuste automático
    e mantém OHLC + AdjClose + Volume.
    """
    start = datetime.strptime("2013-01-01", "%Y-%m-%d")
    end   = datetime.strptime("2025-06-30", "%Y-%m-%d")

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


def zanotto_preprocess(df: pd.DataFrame, ema_span: int = 60):
    """
    - Corrige OHLC usando fator AdjClose/Close
    - Interpola preços (linear) e Volume (ffill/bfill)
    - Adiciona EMA(span) do Close ajustado como feature
    """
    # Corrige MultiIndex do Yahoo se existir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Fator de ajuste
    factor = df["AdjClose"] / df["Close"]
    out = df.copy()
    out["Open"] *= factor
    out["High"] *= factor
    out["Low"] *= factor
    out["Close"] = out["AdjClose"]
    out = out.drop(columns=["AdjClose"])

    # Interpolação de preços
    out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]].interpolate(
        method="linear", limit_direction="both"
    )
    # Volume: zera vira NaN, depois ffill/bfill
    out["Volume"] = out["Volume"].replace(0, np.nan).ffill().bfill()

    # EMA: limita para não passar de 20% do tamanho da série
    span = max(5, min(ema_span, int(len(out) * 0.2)))
    ema_series = out["Close"].ewm(span=span, adjust=False).mean()
    out[f"EMA_{span}"] = ema_series.fillna(out["Close"])

    return out, span


def train_test_split_ordered(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split temporal: primeiros (1 - test_ratio) para treino, resto para teste.
    """
    n = len(df)
    n_train = int((1 - test_ratio) * n)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def make_windows(arr: np.ndarray, target_idx: int, window: int):
    """
    Gera janelas deslizantes (X, y) para LSTM.
    X: [n_samples, window, n_features]
    y: [n_samples]
    """
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i, :])
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


# 2) biuld do modelo LSTM 500 500 0.3
def build_lstm(n_steps: int, n_features: int, units: int = 500, dropout: float = 0.3):
    model = models.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# 3) Duas estrategias de geraçao de sinais
def strategy_gap_pct(pred: np.ndarray,
                     real: np.ndarray,
                     pct_gap: float = 0.004):
    """
    Estratégia 1 — gap_aggressive_0.4% (LONG-only):
        - Se previsão está >= 0.4% acima do preço anterior -> BUY
        - Se já comprado e previsão cai a ponto de ficar <= -0.2% em relação ao preço anterior -> SELL
        - Caso contrário -> HOLD
    """
    signals = ["HOLD"]
    pos = 0  # 0: sem posição, 1: comprado

    for i in range(1, len(pred)):
        ret = (pred[i] - real[i - 1]) / real[i - 1]

        if pos == 0 and ret >= pct_gap:
            signals.append("BUY")
            pos = 1
        elif pos == 1 and ret <= -pct_gap / 2:
            signals.append("SELL")
            pos = 0
        else:
            signals.append("HOLD")

    return signals


def strategy_trend_filter(pred: np.ndarray,
                          real: np.ndarray,
                          ema: np.ndarray):
    """
    Estratégia 2 — trend_filter (LONG-only):
        BUY  quando:
            - previsão > preço anterior
            - preço anterior > EMA atual  (preço acima da média)
        SELL quando:
            - já comprado E (previsão < preço anterior OU preço < EMA atual)
        HOLD caso contrário.
    """
    signals = ["HOLD"]
    pos = 0  # 0: flat, 1: long

    for i in range(1, len(pred)):
        price_prev = real[i - 1]
        pred_i = pred[i]
        ema_i = ema[i]

        # Entrada em tendência de alta
        if pos == 0 and pred_i > price_prev and price_prev > ema_i:
            signals.append("BUY")
            pos = 1
        # Saída quando perder força ou perder EMA
        elif pos == 1 and (pred_i < price_prev or price_prev < ema_i):
            signals.append("SELL")
            pos = 0
        else:
            signals.append("HOLD")

    return signals


# 4) Teste dos sinais gerados & stop loss e take profit 
def backtest_discrete(prices: np.ndarray,
                      signals: list,
                      initial_capital: float = 5000.0,
                      fee_rate: float = 0.0003,
                      stop_loss: float = 0.03,
                      take_profit: float = 0.06):
    """
    Backtest discreto:
      - Apenas posição LONG (BUY/SELL)
      - Usa 100% do capital disponível para comprar
      - Aplica taxa proporcional fee_rate em todas as operações
      - Stop-loss e take-profit relativos ao preço de entrada
      - Retorna curva de patrimônio (equity) e DataFrame de trades
    """
    cash = initial_capital
    shares = 0
    entry_price = None
    equity = [cash]
    trades = []

    for i, sig in enumerate(signals):
        price = float(prices[i])

        # 1) Se há posição, checa SL/TP antes de qualquer sinal
        if shares > 0 and entry_price is not None:
            var = (price - entry_price) / entry_price

            if var <= -stop_loss:
                # STOP LOSS
                revenue = shares * price
                fee = revenue * fee_rate
                cash += revenue - fee
                trades.append((i, "STOP_LOSS", shares, price))
                shares = 0
                entry_price = None

            elif var >= take_profit:
                # TAKE PROFIT
                revenue = shares * price
                fee = revenue * fee_rate
                cash += revenue - fee
                trades.append((i, "TAKE_PROFIT", shares, price))
                shares = 0
                entry_price = None

        # 2) Executa sinais normais
        if sig == "BUY" and shares == 0 and cash >= price:
            qty = int(cash // price)
            if qty > 0:
                cost = qty * price
                fee = cost * fee_rate
                cash -= cost + fee
                shares = qty
                entry_price = price
                trades.append((i, "BUY", qty, price))

        elif sig == "SELL" and shares > 0:
            revenue = shares * price
            fee = revenue * fee_rate
            cash += revenue - fee
            trades.append((i, "SELL", shares, price))
            shares = 0
            entry_price = None

        equity.append(cash + shares * price)

    # 3) Liquidação forçada no final (para comparação com Buy&Hold)
    if shares > 0:
        price = float(prices[-1])
        revenue = shares * price
        fee = revenue * fee_rate
        cash += revenue - fee
        trades.append((len(signals) - 1, "LIQ_FINAL", shares, price))
        shares = 0
        entry_price = None
        equity[-1] = cash

    trades_df = pd.DataFrame(trades, columns=["idx", "action", "shares", "price"])
    return np.array(equity), trades_df


# 5) PIPELINE COMPLETO PARA UM TICKER
def run_pipeline(ticker: str,
                 window: int = 50,
                 test_ratio: float = 0.2,
                 outdir: str = "outputs"):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {ticker} ===")

    # ---------------------
    # 1) Dados + preprocess
    # ---------------------
    raw = fetch_hist(ticker, years=10)
    df, ema_span = zanotto_preprocess(raw)

    features = ["Open", "High", "Low", "Close", "Volume", f"EMA_{ema_span}"]
    target_col = "Close"
    target_idx = features.index(target_col)

    # Split temporal
    train_df, test_df = train_test_split_ordered(df[features], test_ratio=test_ratio)

    # Normalização Z-score (fit só no treino)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    # ---------------------
    # 2) Janelas p/ LSTM
    # ---------------------
    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test, y_test = make_windows(test_scaled, target_idx, window)

    # ---------------------
    # 3) Modelo LSTM
    # ---------------------
    model = build_lstm(window, len(features))
    # ckpt_path = os.path.join(outdir, f"best_{ticker.replace('.','_')}.keras")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        # callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    ]

    hist = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        shuffle=False,
        callbacks=cbs,
        verbose=1
    )

    # model.save(ckpt_path)

    # ---------------------
    # 4) Previsões no teste
    # ---------------------
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)

    def invert_scaling(col_scaled: np.ndarray):
        zeros = np.zeros((len(col_scaled), len(features)))
        zeros[:, target_idx] = col_scaled.ravel()
        inv = scaler.inverse_transform(zeros)
        return inv[:, target_idx]

    y_true = invert_scaling(y_test.reshape(-1, 1))
    y_pred = invert_scaling(y_pred_scaled)

    # ---------------------
    # 5) Métricas de regressão
    # ---------------------
    m_rmse = rmse(y_true, y_pred)
    m_mae = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2 = r2_score(y_true, y_pred)

    print(f"\nMÉTRICAS DE PREVISÃO ({ticker}):")
    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    # ---------------------
    # 6) Índices de tempo + EMA alinhada ao teste
    # ---------------------
    # Índices da parte de teste no df original
    test_index_full = df.iloc[len(train_df):].index     # len = n_test
    test_index = test_index_full[window:]               # len = n_test - window = len(y_true)

    # EMA alinhada com y_true/y_pred
    ema_test_full = df[f"EMA_{ema_span}"].iloc[len(train_df):].values  # len = n_test
    ema_test = ema_test_full[window:]                                   # len = len(y_true)

    # Sanidade: todos com o mesmo tamanho
    assert len(test_index) == len(y_true) == len(y_pred) == len(ema_test)

    # ---------------------
    # 7) Estratégias de sinais
    # ---------------------
    initial_capital = 5000.0
    fee = 0.0003
    stop_loss = 0.03
    take_profit = 0.06

    strategies = {
        "gap_aggressive_0.4%": lambda: strategy_gap_pct(y_pred, y_true, pct_gap=0.004),
        "trend_filter":        lambda: strategy_trend_filter(y_pred, y_true, ema_test),
    }

    print("\n=== Testando Estratégias de Trade ===")
    results_all = {}

    for name, fn in strategies.items():
        print(f"\n→ Estratégia: {name}")
        signals = fn()
        # Sanidade: signals deve ter mesmo tamanho que y_true
        assert len(signals) == len(y_true)
        equity, trades = backtest_discrete(
            y_true,
            signals,
            initial_capital=initial_capital,
            fee_rate=fee,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        roi = (equity[-1] / initial_capital - 1) * 100
        print(f"ROI: {roi:.2f}% | Trades: {len(trades)}")

        # Salva trades desta estratégia
        trades_path = os.path.join(outdir, f"trades_{ticker.replace('.','_')}_{name}.csv")
        trades.to_csv(trades_path, index=False)

        results_all[name] = {
            "roi": roi,
            "equity": equity,
            "signals": signals,
            "trades": trades,
        }

    # ---------------------
    # 8) Buy & Hold (mesmo período)
    # ---------------------
    bh_shares = int(initial_capital // y_true[0])
    entry_cost = bh_shares * y_true[0]
    entry_fee = entry_cost * fee
    bh_cash = initial_capital - entry_cost - entry_fee

    # Equity diária (sem taxas intermediárias)
    bh_equity = bh_cash + bh_shares * y_true
    # Saída final com taxa
    exit_fee = bh_shares * y_true[-1] * fee
    bh_final = bh_cash + bh_shares * y_true[-1] - exit_fee
    roi_bh = (bh_final / initial_capital - 1) * 100

    print(f"\n📊 ROI Buy & Hold ({ticker}): {roi_bh:.2f}%")

    # ---------------------
    # 9) Ranking de estratégias
    # ---------------------
    rank = pd.DataFrame(
        [
            {"estrategia": k, "roi": v["roi"], "trades": len(v["trades"])}
            for k, v in results_all.items()
        ]
    ).sort_values("roi", ascending=False)

    rank_path = os.path.join(outdir, f"ranking_{ticker.replace('.','_')}.csv")
    rank.to_csv(rank_path, index=False)
    print(f"\n🏆 Ranking salvo em: {rank_path}")

    # ---------------------
    # 10) CSV de métricas gerais (melhor estratégia x Buy&Hold)
    # ---------------------
    best_name = rank.iloc[0]["estrategia"]
    best_roi = rank.iloc[0]["roi"]
    best_trades = len(results_all[best_name]["trades"])

    summary_path = os.path.join(outdir, f"summary_{ticker.replace('.','_')}.csv")
    pd.DataFrame([{
        "ticker": ticker,
        "rmse": m_rmse,
        "mae": m_mae,
        "mape(%)": m_mape,
        "r2": m_r2,
        "best_strategy": best_name,
        "roi_lstm_best(%)": best_roi,
        "roi_buyhold(%)": roi_bh,
        "n_trades_best": best_trades
    }]).to_csv(summary_path, index=False)
    print(f"💾 Summary salvo em: {summary_path}")

    # ---------------------
    # 11) Gráfico comparando equity das estratégias
    # ---------------------
    plt.figure(figsize=(16, 6))
    for name, r in results_all.items():
        # equity tem len = len(prices)+1; alinhamos com range
        plt.plot(r["equity"], label=f"{name} ({r['roi']:.2f}%)")
    plt.axhline(initial_capital, linestyle=":", color="gray", label="Capital inicial")
    plt.title(f"{ticker} — Comparação de Estratégias (Equity)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    comp_path = os.path.join(outdir, f"comparison_{ticker.replace('.','_')}.png")
    plt.savefig(comp_path, dpi=140)
    print(f"📊 Comparação de estratégias salva em: {comp_path}")

    # ---------------------
    # 12) Painel detalhado para a melhor estratégia
    # ---------------------
    best_equity = results_all[best_name]["equity"]
    best_signals = results_all[best_name]["signals"]

    plt.figure(figsize=(16, 12))

    # (1) Preço real vs previsto + BUY/SELL da melhor estratégia
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(test_index, y_true, label="Real", linewidth=1.5)
    ax1.plot(test_index, y_pred, label="Previsto", linestyle="--")
    buy_idx = [i for i, s in enumerate(best_signals) if s == "BUY"]
    sell_idx = [i for i, s in enumerate(best_signals) if s == "SELL"]
    if buy_idx:
        ax1.scatter(test_index[buy_idx], y_true[buy_idx], marker="^", color="green", s=60, label="Compra")
    if sell_idx:
        ax1.scatter(test_index[sell_idx], y_true[sell_idx], marker="v", color="red", s=60, label="Venda")
    ax1.set_title(f"{ticker} — Preço Real vs Previsto ({best_name})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Evolução da Loss (treino/val)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(hist.history["loss"], label="Treino")
    ax2.plot(hist.history["val_loss"], label="Validação")
    ax2.set_title("Evolução da Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) Erros de previsão (y_true - y_pred)
    ax3 = plt.subplot(3, 2, 3)
    errors = y_true - y_pred
    ax3.plot(test_index, errors, color="crimson")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.set_title("Erros de Previsão")
    ax3.grid(True, alpha=0.3)

    # (4) Distribuição dos erros (histograma)
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(errors, bins=40, alpha=0.8)
    ax4.set_title("Distribuição dos Erros")
    ax4.grid(True, alpha=0.3)

    # (5) Evolução do patrimônio (melhor estratégia)
    ax5 = plt.subplot(3, 2, 5)
    # Alinha equity com test_index (descarta primeiro ponto de capital inicial)
    ax5.plot(test_index, best_equity[1:], label=f"{best_name} ({best_roi:.2f}%)", linewidth=1.5)
    ax5.axhline(initial_capital, linestyle=":", color="gray", label="Capital inicial")
    ax5.set_title("Evolução do Patrimônio (Melhor Estratégia)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # (6) Retorno Final (%) — LSTM (melhor) vs Buy&Hold
    ax6 = plt.subplot(3, 2, 6)
    ax6.bar(["LSTM (melhor)", "Buy & Hold"], [best_roi, roi_bh])
    ax6.set_title("Retorno Final (%)")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    detailed_path = os.path.join(outdir, f"detailed_{ticker.replace('.','_')}.png")
    plt.savefig(detailed_path, dpi=140)
    print(f"📈 Painel detalhado salvo em: {detailed_path}")


# =========================================================
# 6) MENU — 5 MAIORES BANCOS (B3)
# =========================================================
if __name__ == "__main__":
    bancos = {
        "1": ("ITUB4.SA", "Itaú Unibanco PN"),
        "2": ("BBDC4.SA", "Bradesco PN"),
        "3": ("BBAS3.SA", "Banco do Brasil ON"),
        "4": ("SANB11.SA", "Santander Brasil Units"),
        "5": ("BPAC11.SA", "BTG Pactual Units"),
        "6": ("VALE3.SA", "Vale"),
        "7": ("PETR4.SA", "Petrobras"),
        "8": ("ABEV3.SA", "Ambev"),
        "9": ("WEGE3.SA", "WEG"),
        "10": ("MGLU3.SA", "Magazine Luiza"),
        "11": ("BSLI3.SA", "BRB-Banco de Brasilia SA"),
    }

    print("\n=== LSTM Zanotto + Estratégias de Trade (Bancos do Brasil) ===")
    for k, (t, n) in bancos.items():
        print(f"{k}) {t} — {n}")

    choice = input("\nEscolha o ticker [1-5] (default 1): ").strip()
    if choice not in bancos:
        choice = "1"

    ticker = bancos[choice][0]
    run_pipeline(ticker)
