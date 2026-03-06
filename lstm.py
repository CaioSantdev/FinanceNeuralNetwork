# =========================================================
# LSTM Zanotto + Multi-estratégias + Ranking + Short-selling
# VERSÃO CORRIGIDA
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

    out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]].interpolate(
        method="linear",
        limit_direction="both"
    )
    out["Volume"] = out["Volume"].replace(0, np.nan).ffill().bfill()

    span = max(5, min(int(len(out) * 0.2), 60))
    out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean().fillna(out["Close"])

    return out, span

def train_test_split_ordered(df, test_ratio=0.2):
    n = len(df)
    n_train = int((1 - test_ratio) * n)
    return df.iloc[:n_train], df.iloc[n_train:]

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
# 3) ESTRATÉGIAS (AGORA COM SUPORTE A SHORT)
# =========================================================
def strategy_gap_pct(pred, real, pct_gap=0.01):
    """Gera sinais LONG/SHORT baseado no gap entre previsão e preço atual"""
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        diff_pct = (pred[i] - real[i - 1]) / real[i - 1]
        if diff_pct >= pct_gap:
            sig.append("BUY")  # Entrada LONG
        elif diff_pct <= -pct_gap:
            sig.append("SELL_SHORT")  # Entrada SHORT
        else:
            sig.append("HOLD")
    return sig

def strategy_trend_filter(pred, real, ema, index):
    """Filtro de tendência: LONG em tendência de alta, SHORT em tendência de baixa"""
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
    """Reversão à média: COMPRA quando abaixo da EMA, VENDE quando acima"""
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
    """Sempre posicionado: LONG se previsão > preço atual, SHORT caso contrário"""
    sig = ["HOLD"]
    for i in range(1, len(pred)):
        if pred[i] > real[i - 1]:
            sig.append("BUY")
        else:
            sig.append("SELL_SHORT")
    return sig

# =========================================================
# 4) BACKTEST — LONG + SHORT CORRIGIDO
# =========================================================
def backtest_discrete_short(
    prices,
    signals,
    initial_capital=5000,
    fee_rate=0.0003,
    stop_loss=0.02,
    take_profit=0.05,
    min_volume=100  # Lote mínimo para ações brasileiras
):
    """
    Backtest com suporte a LONG/SHORT
    Retorna equity com mesmo comprimento de prices
    """
    cash = initial_capital
    shares = 0
    entry_price = None
    position_type = None  # 'LONG' ou 'SHORT'
    equity = []  # Lista vazia - será preenchida a cada período
    trades = []

    for i, (price, sig) in enumerate(zip(prices, signals)):
        price = float(price)

        # STOP-LOSS / TAKE-PROFIT (executado antes do novo sinal)
        if shares != 0 and entry_price is not None and position_type is not None:
            var = (price - entry_price) / entry_price

            if position_type == 'LONG':
                if var <= -stop_loss:  # Stop-loss LONG
                    revenue = shares * price
                    cash += revenue - revenue * fee_rate
                    trades.append((i, "STOP_LOSS_LONG", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None
                elif var >= take_profit:  # Take-profit LONG
                    revenue = shares * price
                    cash += revenue - revenue * fee_rate
                    trades.append((i, "TAKE_PROFIT_LONG", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None

            elif position_type == 'SHORT':
                if var >= stop_loss:  # Stop-loss SHORT (preço subiu)
                    qty = abs(shares)
                    cost_to_cover = qty * price
                    cash -= cost_to_cover + cost_to_cover * fee_rate
                    trades.append((i, "STOP_LOSS_SHORT", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None
                elif var <= -take_profit:  # Take-profit SHORT (preço desceu)
                    qty = abs(shares)
                    cost_to_cover = qty * price
                    cash -= cost_to_cover + cost_to_cover * fee_rate
                    trades.append((i, "TAKE_PROFIT_SHORT", shares, price, entry_price))
                    shares = 0
                    entry_price = None
                    position_type = None

        # EXECUÇÃO DOS SINAIS (apenas se não estiver posicionado)
        if shares == 0:
            if sig == "BUY":  # Entrada LONG
                # Respeitar lote mínimo
                max_qty = int(cash // price)
                qty = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    cost = qty * price
                    cash -= cost + cost * fee_rate
                    shares = qty
                    entry_price = price
                    position_type = 'LONG'
                    trades.append((i, "BUY", qty, price, price))

            elif sig == "SELL_SHORT":  # Entrada SHORT
                max_qty = int(cash // price)
                qty = (max_qty // min_volume) * min_volume
                if qty >= min_volume:
                    proceeds = qty * price
                    cash += proceeds - proceeds * fee_rate
                    shares = -qty
                    entry_price = price
                    position_type = 'SHORT'
                    trades.append((i, "SELL_SHORT", -qty, price, price))

        elif shares > 0 and sig == "SELL":  # Fechar LONG manualmente
            revenue = shares * price
            cash += revenue - revenue * fee_rate
            trades.append((i, "SELL", shares, price, entry_price))
            shares = 0
            entry_price = None
            position_type = None

        elif shares < 0 and sig == "BUY":  # Fechar SHORT manualmente
            qty = abs(shares)
            cost_to_cover = qty * price
            cash -= cost_to_cover + cost_to_cover * fee_rate
            trades.append((i, "BUY_TO_COVER", shares, price, entry_price))
            shares = 0
            entry_price = None
            position_type = None

        # Registrar patrimônio deste período
        current_value = cash + (shares * price if shares != 0 else 0)
        equity.append(current_value)

    # Liquidação final (se ainda estiver posicionado)
    if shares != 0 and len(prices) > 0:
        price = float(prices[-1])
        if shares > 0:
            cash += shares * price - shares * price * fee_rate
        else:
            qty = abs(shares)
            cost_to_cover = qty * price
            cash -= cost_to_cover + cost_to_cover * fee_rate
        shares = 0
        # Atualizar último valor do patrimônio
        equity[-1] = cash

    return np.array(equity), pd.DataFrame(trades, columns=["idx", "action", "shares", "price", "entry_price"])

def backtest_buy_and_hold(prices, initial_capital=5000, fee_rate=0.0003, min_volume=100):
    """
    Backtest Buy & Hold com mesmo comprimento de prices
    """
    prices = np.asarray(prices, dtype=float)

    if len(prices) == 0:
        return np.array([initial_capital])

    entry_price = prices[0]
    max_qty = int(initial_capital // entry_price)
    qty = (max_qty // min_volume) * min_volume
    cash = initial_capital

    # Compra inicial
    if qty >= min_volume:
        cost = qty * entry_price
        cash -= cost + cost * fee_rate

    # Calcular patrimônio ao longo do tempo
    equity = []
    for p in prices:
        equity.append(cash + qty * p)

    # Venda final (para consistência com o outro backtest)
    if qty >= min_volume:
        final_cash = cash + qty * prices[-1] - qty * prices[-1] * fee_rate
        equity[-1] = final_cash

    return np.array(equity)

# =========================================================
# 5) WALK-FORWARD VALIDATION
# =========================================================
def walk_forward_validation(df, features, target_idx, window, n_splits=5):
    """
    Validação walk-forward para testar robustez do modelo
    """
    split_size = len(df) // (n_splits + 1)
    results = []
    
    for i in range(n_splits):
        print(f"\n--- Walk-Forward Split {i+1}/{n_splits} ---")
        
        train_end = (i + 1) * split_size
        test_end = train_end + split_size
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Escalonar
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)
        
        # Criar janelas
        X_train, y_train = make_windows(train_scaled, target_idx, window)
        X_test, y_test = make_windows(test_scaled, target_idx, window)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Treinar modelo
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
        
        # Prever
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverter scaling (corrigido)
        target_mean = scaler.mean_[target_idx]
        target_std = scaler.scale_[target_idx]
        y_true = y_test * target_std + target_mean
        y_pred = y_pred_scaled.ravel() * target_std + target_mean
        
        # Métricas
        results.append({
            'split': i+1,
            'rmse': rmse(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mape(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        })
    
    return pd.DataFrame(results)

# =========================================================
# 6) FUNÇÕES DE PLOT
# =========================================================
# [Mantidas as mesmas funções de plot do código original]
def save_plot_real_vs_pred(test_idx, y_true, y_pred, signals, ticker, strategy_name, outdir):
    plt.figure(figsize=(16, 6))
    plt.plot(test_idx, y_true, label="Real", linewidth=1.5)
    plt.plot(test_idx, y_pred, label="Previsto", linestyle="--")

    buy_idx = [i for i, s in enumerate(signals) if s == "BUY"]
    short_idx = [i for i, s in enumerate(signals) if s == "SELL_SHORT"]

    if buy_idx:
        plt.scatter(
            test_idx[buy_idx], y_true[buy_idx],
            marker="^", color="green", s=60, label="Compra (LONG)"
        )
    if short_idx:
        plt.scatter(
            test_idx[short_idx], y_true[short_idx],
            marker="v", color="red", s=60, label="Venda (SHORT)"
        )

    plt.title(f"{ticker} — Preço Real vs Previsto ({strategy_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/real_vs_pred_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📈 Gráfico salvo em: {path}")

def save_plot_real_vs_pred_clean(test_idx, y_true, y_pred, ticker, strategy_name, outdir):
    plt.figure(figsize=(16, 6))
    plt.plot(test_idx, y_true, label="Real", linewidth=1.5)
    plt.plot(test_idx, y_pred, label="Previsto", linestyle="--")

    plt.title(f"{ticker} — Preço Real vs Previsto ({strategy_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/real_vs_pred_clean_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

def save_plot_loss(history, ticker, outdir):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"{ticker} — Evolução da Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/loss_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📉 Gráfico salvo em: {path}")

def save_plot_errors(test_idx, errors, ticker, outdir):
    plt.figure(figsize=(16, 5))
    plt.plot(test_idx, errors, color="crimson")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"{ticker} — Erros de Previsão")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/errors_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📉 Gráfico salvo em: {path}")

def save_plot_error_distribution(errors, ticker, outdir):
    plt.figure(figsize=(12, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title(f"{ticker} — Distribuição dos Erros")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = f"{outdir}/error_distribution_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📊 Gráfico salvo em: {path}")

def save_plot_best_equity(best_equity, ticker, strategy_name, initial_capital, outdir):
    """
    Gráfico da melhor estratégia com eixo X baseado em dias
    """
    plt.figure(figsize=(14, 5))
    
    days = np.arange(len(best_equity))
    
    plt.plot(days, best_equity, 
             label=strategy_name, 
             linewidth=2, color='blue')
    
    plt.axhline(initial_capital, linestyle=":", color="gray", 
                alpha=0.7, label=f"Capital Inicial (R${initial_capital:.2f})")
    
    # Calcular drawdown máximo (opcional)
    peak = np.maximum.accumulate(best_equity)
    drawdown = (best_equity - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    plt.title(f"{ticker} — Evolução do Patrimônio ({strategy_name}) | Max Drawdown: {max_drawdown:.2f}%", 
              fontsize=12)
    plt.xlabel("Dias de Trading", fontsize=10)
    plt.ylabel("Patrimônio (R$)", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Formatar eixo Y como moeda
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    
    plt.tight_layout()

    path = f"{outdir}/best_equity_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"💰 Gráfico salvo em: {path}")

def save_plot_all_strategies(results_all, ticker, outdir):
    """
    Compara todas as estratégias com eixos alinhados
    """
    plt.figure(figsize=(16, 8))
    
    # Encontrar o menor tamanho entre todas as estratégias
    min_len = min([len(r["equity"]) for r in results_all.values()])
    
    for name, r in results_all.items():
        # Truncar para o tamanho mínimo
        equity_aligned = r["equity"][:min_len]
        days = np.arange(len(equity_aligned))
        
        plt.plot(days, equity_aligned, 
                label=f"{name} (ROI: {r['roi']:.2f}%)", 
                linewidth=1.5)
    
    # Linha do capital inicial
    first_equity = list(results_all.values())[0]["equity"][0]
    plt.axhline(y=first_equity, color='gray', linestyle=':', 
                alpha=0.7, label=f'Capital Inicial (R${first_equity:.2f})')
    
    plt.title(f"{ticker} — Comparação entre Estratégias", fontsize=14, fontweight='bold')
    plt.xlabel("Dias de Trading", fontsize=12)
    plt.ylabel("Patrimônio (R$)", fontsize=12)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Formatar eixo Y como moeda
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    
    plt.tight_layout()

    path = f"{outdir}/comparison_strategies_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📊 Gráfico salvo em: {path}")

def save_plot_strategy_vs_buyhold(best_equity, buyhold_equity, ticker, strategy_name, outdir):
    """
    Gráfico comparativo com eixos alinhados
    """
    # VERIFICAÇÃO E ALINHAMENTO DOS TAMANHOS
    if len(best_equity) != len(buyhold_equity):
        print(f"⚠️ Ajustando tamanhos: Estratégia={len(best_equity)}, Buy&Hold={len(buyhold_equity)}")
        
        # Truncar para o menor tamanho
        min_len = min(len(best_equity), len(buyhold_equity))
        best_equity = best_equity[:min_len]
        buyhold_equity = buyhold_equity[:min_len]
        
        print(f"✅ Tamanhos ajustados para: {min_len}")
    
    # Calcular ROIs
    roi_strategy = (best_equity[-1] / best_equity[0] - 1) * 100
    roi_buyhold = (buyhold_equity[-1] / buyhold_equity[0] - 1) * 100

    # Criar figura
    plt.figure(figsize=(16, 8))
    
    # Criar índice baseado no tempo (dias de trading)
    days = np.arange(len(best_equity))
    
    # Plotar linhas
    plt.plot(days, best_equity, 
             label=f"{strategy_name} (ROI: {roi_strategy:.2f}%)", 
             linewidth=2, color='blue')
    plt.plot(days, buyhold_equity, 
             label=f"Buy & Hold (ROI: {roi_buyhold:.2f}%)", 
             linewidth=2, color='green', linestyle='--')
    
    # Linha do capital inicial
    plt.axhline(y=best_equity[0], color='gray', linestyle=':', 
                alpha=0.7, label=f'Capital Inicial (R${best_equity[0]:.2f})')
    
    # Configurações do gráfico
    plt.title(f"{ticker} — Melhor Estratégia vs Buy & Hold", fontsize=14, fontweight='bold')
    plt.xlabel("Dias de Trading", fontsize=12)
    plt.ylabel("Patrimônio (R$)", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Formatar eixo Y como moeda
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R${x:,.0f}'))
    
    plt.tight_layout()

    # Salvar
    path = f"{outdir}/strategy_vs_buyhold_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📊 Gráfico salvo em: {path}")
    
    # Debug: mostrar informações
    print(f"   Estratégia: R${best_equity[0]:.2f} → R${best_equity[-1]:.2f} ({roi_strategy:.2f}%)")
    print(f"   Buy & Hold: R${buyhold_equity[0]:.2f} → R${buyhold_equity[-1]:.2f} ({roi_buyhold:.2f}%)")

def save_plot_regression_metrics(metrics_dict, ticker, outdir):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, values)
    plt.title(f"{ticker} — Métricas de Regressão")
    plt.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    path = f"{outdir}/regression_metrics_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📏 Gráfico salvo em: {path}")

def save_plot_walk_forward(wf_results, ticker, outdir):
    """Salva gráfico dos resultados walk-forward"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(wf_results['split'], wf_results['rmse'], marker='o')
    plt.title('RMSE por Split')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(wf_results['split'], wf_results['mape'], marker='o', color='orange')
    plt.title('MAPE por Split')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(wf_results['split'], wf_results['r2'], marker='o', color='green')
    plt.title('R² por Split')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    
    plt.subplot(2, 2, 4)
    metrics = ['rmse', 'mae', 'mape']
    means = [wf_results[m].mean() for m in metrics]
    plt.bar(metrics, means)
    plt.title('Média das Métricas')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{ticker} — Walk-Forward Validation Results')
    plt.tight_layout()
    
    path = f"{outdir}/walk_forward_{ticker.replace('.', '_')}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📊 Gráfico salvo em: {path}")

# =========================================================
# 7) PIPELINE COMPLETO CORRIGIDO
# =========================================================
def run_pipeline(ticker, window=50, test_ratio=0.2, outdir="resultados", 
                 initial_capital=5000, do_walk_forward=True):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {ticker} ===")

    raw = fetch_hist(ticker)
    df, ema_span = zanotto_preprocess(raw)

    features = ["Open", "High", "Low", "Close", "Volume", f"EMA_{ema_span}"]
    target = "Close"
    target_idx = features.index(target)

    train_df, test_df = train_test_split_ordered(df[features], test_ratio)

    # Escalonamento
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Criar janelas
    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test, y_test = make_windows(test_scaled, target_idx, window)

    # Treinar modelo
    model = build_lstm(window, len(features))

    hist = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        shuffle=False,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test, verbose=0)

    # CORREÇÃO: Inverter scaling corretamente
    target_mean = scaler.mean_[target_idx]
    target_std = scaler.scale_[target_idx]
    
    y_true = y_test * target_std + target_mean
    y_pred = y_pred_scaled.ravel() * target_std + target_mean

    # Métricas
    m_rmse = rmse(y_true, y_pred)
    m_mae = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2 = r2_score(y_true, y_pred)

    print("\nMÉTRICAS:")
    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    # CORREÇÃO: Índices alinhados
    test_start_idx = len(train_df) + window
    test_idx = df.index[test_start_idx:]
    ema_test = df.loc[test_idx, f"EMA_{ema_span}"]

    # Estratégias (agora com suporte a SHORT)
    strategies = {
        "gap_1%": lambda: strategy_gap_pct(y_pred, y_true, 0.01),
        "gap_0.7%": lambda: strategy_gap_pct(y_pred, y_true, 0.007),
        "trend_filter": lambda: strategy_trend_filter(y_pred, y_true, ema_test, test_idx),
        "mean_reversion": lambda: strategy_mean_reversion(y_pred, y_true, ema_test, test_idx),
        "always_in": lambda: strategy_always_in(y_pred, y_true),
    }

    print("\n=== Testando Estratégias (LONG/SHORT) ===")
    results_all = {}

    for name, fn in strategies.items():
        print(f"\n→ Estratégia: {name}")
        signals = fn()
        
        # Verificar tipos de sinais gerados
        buy_count = signals.count("BUY")
        short_count = signals.count("SELL_SHORT")
        hold_count = signals.count("HOLD")
        
        equity, trades = backtest_discrete_short(
            y_true,
            signals,
            initial_capital=initial_capital
        )
        roi = (equity[-1] / initial_capital - 1) * 100
        print(f"ROI: {roi:.2f}% | Trades: {len(trades)} | BUY: {buy_count} | SHORT: {short_count} | HOLD: {hold_count}")

        results_all[name] = {
            "roi": roi,
            "equity": equity,
            "signals": signals,
            "trades": trades
        }

    # Ranking
    rank = pd.DataFrame([
        {"estrategia": k, "roi": v["roi"], "trades": len(v["trades"])}
        for k, v in results_all.items()
    ]).sort_values("roi", ascending=False).reset_index(drop=True)

    print("\n=== Ranking das Estratégias ===")
    print(rank)

    best_name = rank.iloc[0]["estrategia"]
    best_equity = results_all[best_name]["equity"]
    best_signals = results_all[best_name]["signals"]

    # Buy & Hold
    buyhold_equity = backtest_buy_and_hold(y_true, initial_capital=initial_capital)
    buyhold_roi = (buyhold_equity[-1] / initial_capital - 1) * 100

    print(f"\n🏆 Melhor estratégia: {best_name} | ROI: {rank.iloc[0]['roi']:.2f}%")
    print(f"📌 Buy & Hold ROI: {buyhold_roi:.2f}%")

    # Erros
    errors = y_true - y_pred

    # Métricas de regressão
    metrics_dict = {
        "RMSE": m_rmse,
        "MAE": m_mae,
        "MAPE": m_mape,
        "R²": m_r2
    }

    # Walk-Forward Validation
    if do_walk_forward:
        print("\n=== Walk-Forward Validation ===")
        wf_results = walk_forward_validation(
            df[features], features, target_idx, window, n_splits=5
        )
        if not wf_results.empty:
            print("\nResultados Walk-Forward:")
            print(wf_results.to_string(index=False))
            print(f"\nMédia RMSE: {wf_results['rmse'].mean():.4f} ± {wf_results['rmse'].std():.4f}")
            print(f"Média R²: {wf_results['r2'].mean():.4f} ± {wf_results['r2'].std():.4f}")
            save_plot_walk_forward(wf_results, ticker, outdir)

    # Salvar gráficos
    # Salvar gráficos
    print("\n=== Gerando Gráficos ===")
    
    # =====================================================
    # VERIFICAÇÃO E ALINHAMENTO DE TAMANHOS
    # =====================================================
    print("\n=== VERIFICAÇÃO DE TAMANHOS ===")
    print(f"y_true: {len(y_true)}")
    print(f"test_idx: {len(test_idx)}")
    print(f"best_signals: {len(best_signals)}")
    print(f"best_equity (estratégia): {len(best_equity)}")
    print(f"buyhold_equity: {len(buyhold_equity)}")
    print(f"errors: {len(errors)}")

    # ALINHAR TODOS OS ARRAYS PARA O MESMO TAMANHO
    # 1. Alinhar best_equity com test_idx
    if len(best_equity) > len(test_idx):
        best_equity = best_equity[:len(test_idx)]
        print(f"✅ best_equity truncado para {len(best_equity)}")

    # 2. Alinhar buyhold_equity com test_idx
    if len(buyhold_equity) > len(test_idx):
        buyhold_equity = buyhold_equity[:len(test_idx)]
        print(f"✅ buyhold_equity truncado para {len(buyhold_equity)}")

    # 3. Alinhar y_true e y_pred com test_idx
    if len(y_true) > len(test_idx):
        y_true_aligned = y_true[:len(test_idx)]
        y_pred_aligned = y_pred[:len(test_idx)]
        errors_aligned = errors[:len(test_idx)]
        print(f"✅ y_true/y_pred truncados para {len(y_true_aligned)}")
    else:
        y_true_aligned = y_true
        y_pred_aligned = y_pred
        errors_aligned = errors

    # 4. Alinhar best_signals com test_idx
    if len(best_signals) > len(test_idx):
        best_signals_aligned = best_signals[:len(test_idx)]
        print(f"✅ best_signals truncado para {len(best_signals_aligned)}")
    else:
        best_signals_aligned = best_signals

    # 5. Verificação final
    print("\n=== TAMANHOS FINAIS ALINHADOS ===")
    print(f"test_idx: {len(test_idx)}")
    print(f"y_true_aligned: {len(y_true_aligned)}")
    print(f"best_equity: {len(best_equity)}")
    print(f"buyhold_equity: {len(buyhold_equity)}")
    print(f"best_signals_aligned: {len(best_signals_aligned)}")

    # Salvar gráficos com arrays alinhados
    save_plot_real_vs_pred(test_idx, y_true_aligned, y_pred_aligned, best_signals_aligned, ticker, best_name, outdir)
    save_plot_real_vs_pred_clean(test_idx, y_true_aligned, y_pred_aligned, ticker, best_name, outdir)
    save_plot_loss(hist, ticker, outdir)
    save_plot_errors(test_idx, errors_aligned, ticker, outdir)
    save_plot_error_distribution(errors_aligned, ticker, outdir)
    save_plot_best_equity(best_equity, ticker, best_name, initial_capital, outdir)
    save_plot_all_strategies(results_all, ticker, outdir)
    save_plot_strategy_vs_buyhold(best_equity, buyhold_equity, ticker, best_name, outdir)
    save_plot_regression_metrics(metrics_dict, ticker, outdir)

    print("\n✅ Processamento concluído com sucesso.")
    return results_all, rank, wf_results if do_walk_forward else None

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

    print("\n=== LSTM MULTI-ESTRATÉGIAS (LONG/SHORT) ===")
    print("Versão corrigida com walk-forward validation")
    
    for k, (t, n) in bancos.items():
        print(f"{k}) {t} — {n}")

    c = input("\nEscolha o ticker [1-5]: ").strip()
    if c not in bancos:
        c = "1"

    ticker = bancos[c][0]
    
    # Perguntar se quer walk-forward
    wf = input("\nExecutar walk-forward validation? (s/n): ").strip().lower()
    do_wf = wf == 's'
    
    # Escolher período
    print("\nPeríodo dos dados:")
    print("1) 5 anos")
    print("2) 10 anos")
    print("3) 15 anos")
    p = input("Escolha [1-3]: ").strip()
    
    years_map = {'1': 5, '2': 10, '3': 15}
    years = years_map.get(p, 10)
    
    # Modificar fetch_hist temporariamente
    original_fetch = fetch_hist
    def fetch_hist_with_years(ticker):
        return original_fetch(ticker, years=years)
    
    # Substituir temporariamente
    import __main__
    __main__.fetch_hist = fetch_hist_with_years
    
    outdir = f"resultados/{years}anos_{bancos[c][1].replace(' ', '_')}"
    
    run_pipeline(ticker, outdir=outdir, do_walk_forward=do_wf)