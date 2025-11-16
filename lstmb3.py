# lstm_bancos.py
# =========================================================
# 🧠 LSTM (Zanotto) + Backtesting: Bancos do Brasil - CORRIGIDO
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

# ----------------------------
# Reprodutibilidade
# ----------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 1) Preparação de dados (estilo Zanotto)
#    - yfinance (auto_adjust=False)
#    - Ajuste OHLC via fator AdjClose/Close
#    - Interpola preços (linear) e Volume (ffill/bfill)
#    - EMA(60) como feature adicional
#    - Normalização Z-score (fit só no treino)
# =========================================================
def fetch_hist(ticker: str, years: int = 10) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=365*years + 5)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Nenhum dado para {ticker}.")
    df = df.rename(columns={'Adj Close': 'AdjClose'})
    # mantém ordem consistente
    df = df[['Open','High','Low','Close','AdjClose','Volume']]
    return df

def zanotto_preprocess(df: pd.DataFrame, ema_span: int = 60) -> pd.DataFrame:
    # ------------------------------------------
    # 1) Remove MultiIndex nas colunas (Yahoo bug)
    # ------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # ------------------------------------------
    # 2) Ajuste OHLC com AdjClose/Close
    # ------------------------------------------
    factor = df['AdjClose'] / df['Close']
    out = df.copy()
    out['Open'] *= factor
    out['High'] *= factor
    out['Low']  *= factor
    out['Close'] = out['AdjClose']
    out = out.drop(columns=['AdjClose'])

    # ------------------------------------------
    # 3) Interpolação de preços e volume
    # ------------------------------------------
    out[['Open','High','Low','Close']] = out[['Open','High','Low','Close']].interpolate(
        method='linear', limit_direction='both'
    )
    out['Volume'] = out['Volume'].replace(0, np.nan).ffill().bfill()

    # ------------------------------------------
    # 4) EMA(60)
    # ------------------------------------------
    span = max(5, min(ema_span, int(len(out)*0.2)))
    ema_series = out['Close'].ewm(span=span, adjust=False).mean()

    # ⚠️ Garante que 'Close' é Series
    close_series = out['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:,0]

    # ⚠️ Garante que EMA é Series
    if isinstance(ema_series, pd.DataFrame):
        ema_series = ema_series.iloc[:,0]

    out[f'EMA_{span}'] = ema_series.fillna(close_series)

    return out, span

def train_test_split_ordered(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    n_train = int((1 - test_ratio) * n)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()

def make_windows(arr: np.ndarray, target_idx: int, window: int):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i, :])
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)

def rmse(y_true, y_pred): return math.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true)+eps))) * 100

# =========================================================
# 2) Modelo LSTM 2×500 + Dropout 0.3 (Zanotto)
# =========================================================
def build_lstm(n_steps: int, n_features: int, units: int = 500, dropout: float = 0.3):
    model = models.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================================================
# 3) Geração de sinais + StopLoss/TakeProfit - CORRIGIDAS
# =========================================================
def generate_signals_trend(pred, real, ema):
    """
    Trend Filter CORRIGIDO - sem vazamento de dados
    Usa apenas informação disponível no momento da decisão (i-1)
    """
    signals = ["HOLD"]
    
    for i in range(1, len(pred)):
        # ⚠️ CORREÇÃO: Usar apenas dados disponíveis no momento da decisão (i-1)
        previous_price = real[i-1]    # Preço conhecido no momento da decisão
        current_pred = pred[i]        # Previsão para o próximo período (isso é OK)
        previous_ema = ema[i-1]       # EMA conhecido no momento da decisão
        
        # BUY: previsão de alta + preço anterior acima da EMA
        if current_pred > previous_price and previous_price > previous_ema:
            signals.append("BUY")
        
        # SELL: previsão de baixa + preço anterior abaixo da EMA  
        elif current_pred < previous_price and previous_price < previous_ema:
            signals.append("SELL")
        
        else:
            signals.append("HOLD")
    
    return signals

def backtest_discrete(prices: np.ndarray,
                      signals: list,
                      initial_capital: float = 5000.0,
                      fee_rate: float = 0.0003,
                      stop_loss: float = 0.02,
                      take_profit: float = 0.05):
    """
    Backtest CORRIGIDO - evita trades consecutivos irrealistas
    """
    cash = initial_capital
    shares = 0
    entry_price = None
    equity = [cash]
    trades = []
    last_trade_action = None  # ⚠️ CORREÇÃO: controla ação anterior
    
    for i, sig in enumerate(signals):
        price = float(prices[i])
        
        # Se há posição, checa SL/TP antes de tudo
        if shares > 0 and entry_price is not None:
            var = (price - entry_price) / entry_price
            if var <= -stop_loss:
                revenue = shares * price
                fee = revenue * fee_rate
                cash += revenue - fee
                trades.append((i, 'STOP_LOSS', shares, price, equity[-1]))
                shares = 0
                entry_price = None
                last_trade_action = 'SELL'
            elif var >= take_profit:
                revenue = shares * price
                fee = revenue * fee_rate
                cash += revenue - fee
                trades.append((i, 'TAKE_PROFIT', shares, price, equity[-1]))
                shares = 0
                entry_price = None
                last_trade_action = 'SELL'

        # ⚠️ CORREÇÃO: Evita trades consecutivos do mesmo tipo
        if sig == 'BUY' and shares == 0 and cash >= price and last_trade_action != 'BUY':
            qty = int(cash // price)
            if qty > 0:
                cost = qty * price
                fee = cost * fee_rate
                cash -= cost + fee
                shares = qty
                entry_price = price
                trades.append((i, 'BUY', qty, price, equity[-1]))
                last_trade_action = 'BUY'

        elif sig == 'SELL' and shares > 0 and last_trade_action != 'SELL':
            revenue = shares * price
            fee = revenue * fee_rate
            cash += revenue - fee
            trades.append((i, 'SELL', shares, price, equity[-1]))
            shares = 0
            entry_price = None
            last_trade_action = 'SELL'

        equity.append(cash + shares * price)

    # Força liquidação no fim
    if shares > 0:
        price = float(prices[-1])
        revenue = shares * price
        fee = revenue * fee_rate
        cash += revenue - fee
        trades.append((len(signals)-1, 'LIQ_FINAL', shares, price, equity[-1]))
        shares = 0
        entry_price = None
        equity[-1] = cash

    return np.array(equity), pd.DataFrame(trades, columns=['idx','action','shares','price','equity_before'])

def validate_data_alignment(train_df, test_df, y_true, y_pred, ema_test, window):
    """
    Valida se não há vazamento de dados e se todos os arrays estão alinhados
    """
    print("\n🔍 VALIDAÇÃO DE ALINHAMENTO:")
    
    # 1. Verifica sobreposição temporal
    train_end = train_df.index[-1]
    test_start = test_df.index[0]
    
    if train_end >= test_start:
        print("🚨 CRÍTICO: Dados de treino e teste se sobrepõem!")
        return False
    else:
        print(f"✅ Sem sobreposição: treino até {train_end}, teste desde {test_start}")
    
    # 2. Verifica tamanhos dos arrays
    print(f"✅ Tamanho y_true: {len(y_true)}")
    print(f"✅ Tamanho y_pred: {len(y_pred)}")
    print(f"✅ Tamanho ema_test: {len(ema_test)}")
    
    # 3. Verifica se EMA está alinhado com y_true
    if len(ema_test) != len(y_true):
        print(f"🚨 EMA não alinhado! EMA: {len(ema_test)} vs y_true: {len(y_true)}")
        return False
    else:
        print("✅ EMA alinhado com y_true")
    
    # 4. Verifica janela temporal
    expected_test_size = len(test_df) - window
    if len(y_true) != expected_test_size:
        print(f"🚨 Tamanho incorreto! Esperado: {expected_test_size}, Obtido: {len(y_true)}")
        return False
    else:
        print(f"✅ Tamanho do teste correto: {len(y_true)}")
    
    return True

# =========================================================
# 4) Execução completa para 1 ticker - CORRIGIDA
# =========================================================
def run_pipeline(ticker: str,
                 window: int = 50,
                 test_ratio: float = 0.2,
                 price_gap: float = 1.0,
                 outdir: str = 'outputs'):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {ticker} ===")

    # 1. Dados
    raw = fetch_hist(ticker, years=10)
    df, ema_span = zanotto_preprocess(raw)
    features = ['Open','High','Low','Close','Volume', f'EMA_{ema_span}']
    target_col = 'Close'
    target_idx = features.index(target_col)

    # 2. Split + Scale (fit só no treino)
    train_df, test_df = train_test_split_ordered(df[features], test_ratio=test_ratio)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled  = scaler.transform(test_df.values)

    # 3. Janelas
    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test,  y_test  = make_windows(test_scaled,  target_idx, window)

    # 4. Modelo
    model = build_lstm(window, len(features))
    ckpt_path = os.path.join(outdir, f'best_{ticker.replace(".","_")}.keras')
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    ]
    hist = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=64,
                     shuffle=False, callbacks=cbs, verbose=1)

    # 5. Salva modelo treinado
    model.save(ckpt_path)

    # 6. Previsão no teste + inversão
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)

    def invert_scaling(col_scaled: np.ndarray):
        zeros = np.zeros((len(col_scaled), len(features)))
        zeros[:, target_idx] = col_scaled.ravel()
        inv = scaler.inverse_transform(zeros)
        return inv[:, target_idx]

    y_true = invert_scaling(y_test.reshape(-1,1))
    y_pred = invert_scaling(y_pred_scaled)

    # 7. Métricas de regressão (RMSE, MAE, MAPE, R²)
    m_rmse = rmse(y_true, y_pred)
    m_mae  = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2   = r2_score(y_true, y_pred)
    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    # 8. SINAIS COM ALINHAMENTO CORRIGIDO
    test_index = df.iloc[len(train_df):].index
    plot_index = test_index[window:]
    
    # EMA alinhado corretamente - CORREÇÃO CRÍTICA
    ema_full = df[f'EMA_{ema_span}'].values
    # ⚠️ CORREÇÃO: -1 para alinhar temporalmente
    ema_test = ema_full[len(train_df)+window-1 : len(train_df)+window+len(y_pred)-1]
    
    print(f"📊 Período de teste: {plot_index[0]} até {plot_index[-1]}")
    
    # Valida alinhamento antes de gerar sinais
    if not validate_data_alignment(train_df, test_df, y_true, y_pred, ema_test, window):
        print("🚨 CORRIJA O ALINHAMENTO ANTES DE CONTINUAR!")
        return
    
    signals = generate_signals_trend(y_pred, y_true, ema_test)

    # 9. Backtesting (com stop/take) e Buy&Hold
    initial_capital = 5000.0
    fee = 0.0003
    equity, trades = backtest_discrete(y_true, signals, initial_capital=initial_capital,
                                       fee_rate=fee, stop_loss=0.02, take_profit=0.05)

    # Buy&Hold
    bh_shares = int(initial_capital // y_true[0])
    bh_cash   = initial_capital - bh_shares * y_true[0] - bh_shares * y_true[0] * fee
    bh_equity = bh_cash + bh_shares * y_true
    # venda final
    bh_final  = bh_cash + bh_shares * y_true[-1] - bh_shares * y_true[-1] * fee

    roi_lstm = (equity[-1] / initial_capital - 1) * 100
    roi_bh   = (bh_final   / initial_capital - 1) * 100
    print(f"ROI LSTM: {roi_lstm:.2f}% | ROI Buy&Hold: {roi_bh:.2f}%")
    print(f"📊 Total de trades: {len(trades)}")

    # 10. Gráficos
    plt.figure(figsize=(16, 12))

    # (1) Preço real x previsto (+ sinais)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(plot_index, y_true, label="Real", linewidth=1.5)
    ax1.plot(plot_index, y_pred, label="Previsto", linestyle="--", linewidth=1.2)
    buy_idx  = [i for i, s in enumerate(signals) if s == 'BUY']
    sell_idx = [i for i, s in enumerate(signals) if s == 'SELL']
    ax1.scatter(plot_index[buy_idx],  y_true[buy_idx],  marker='^', color='green', s=60, label='Compra')
    ax1.scatter(plot_index[sell_idx], y_true[sell_idx], marker='v', color='red',   s=60, label='Venda')
    ax1.set_title(f"{ticker} – Preço Real vs Previsto")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # (2) Evolução do patrimônio (LSTM x Buy&Hold)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(plot_index, equity[1:], label="LSTM", linewidth=1.5)
    ax2.plot(plot_index, bh_equity, label="Buy&Hold (ref.)", linestyle='--')
    ax2.axhline(initial_capital, color='gray', linestyle=':', label='Capital Inicial')
    ax2.set_title("Evolução do Patrimônio")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # (3) Evolução da Loss
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(hist.history['loss'], label='Treino')
    ax3.plot(hist.history['val_loss'], label='Validação')
    ax3.set_title('Evolução da Loss'); ax3.legend(); ax3.grid(True, alpha=0.3)

    # (4) Erros de previsão (y_true - y_pred)
    ax4 = plt.subplot(3, 2, 4)
    errors = y_true - y_pred
    ax4.plot(plot_index, errors, color='crimson')
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax4.set_title('Erros de Previsão'); ax4.grid(True, alpha=0.3)

    # (5) Distribuição dos erros (hist)
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(errors, bins=40, alpha=0.8)
    ax5.set_title('Distribuição dos Erros'); ax5.grid(True, alpha=0.3)

    # (6) Retorno em % das estratégias
    ax6 = plt.subplot(3, 2, 6)
    ax6.bar(['LSTM','Buy & Hold'], [roi_lstm, roi_bh])
    ax6.set_title('Retorno Final (%)'); ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(outdir, f"report_{ticker.replace('.','_')}.png")
    plt.savefig(png_path, dpi=140)
    print(f"📈 Relatório salvo em: {png_path}")

    # 11. Salva métricas e trades
    summary_df = pd.DataFrame([{
        'ticker': ticker,
        'rmse': m_rmse,
        'mae': m_mae,
        'mape(%)': m_mape,
        'r2': m_r2,
        'roi_lstm(%)': roi_lstm,
        'roi_buyhold(%)': roi_bh,
        'n_trades': len(trades),
        'valido': 'SIM' if m_r2 < 0.9 else 'VERIFICAR'  # ⚠️ Alerta se R² muito alto
    }])
    summary_df.to_csv(os.path.join(outdir, f"summary_{ticker.replace('.','_')}.csv"), index=False)

    trades.to_csv(os.path.join(outdir, f"trades_{ticker.replace('.','_')}.csv"), index=False)
    print("💾 Métricas e trades salvos em CSV.")

# =========================================================
# 5) Função Test Only Corrigida
# =========================================================
def run_test_only(ticker, window=50, test_ratio=0.2, price_gap=1.0, outdir="outputs"):
    print(f"\n=== TEST ONLY — {ticker} ===")

    # 1. Dados
    raw = fetch_hist(ticker, years=10)
    df, ema_span = zanotto_preprocess(raw)
    features = ['Open','High','Low','Close','Volume', f'EMA_{ema_span}']
    target_col = 'Close'
    target_idx = features.index(target_col)

    # 2. Split + Scale
    train_df, test_df = train_test_split_ordered(df[features], test_ratio=test_ratio)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled  = scaler.transform(test_df.values)

    # 3. Janelas
    X_train, y_train = make_windows(train_scaled, target_idx, window)
    X_test, y_test   = make_windows(test_scaled,  target_idx, window)

    # 4. Carrega modelo treinado
    model_path = os.path.join(outdir, f'best_{ticker.replace(".","_")}.keras')
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado. Treine primeiro!")
        return

    print(f"📦 Carregando modelo salvo em {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 5. Previsões
    y_pred_scaled = model.predict(X_test).reshape(-1,1)

    def invert_scaling(col_scaled):
        zeros = np.zeros((len(col_scaled), len(features)))
        zeros[:, target_idx] = col_scaled.ravel()
        return scaler.inverse_transform(zeros)[:, target_idx]

    y_true = invert_scaling(y_test.reshape(-1,1))
    y_pred = invert_scaling(y_pred_scaled)

    # 6. Métricas
    m_rmse = rmse(y_true, y_pred)
    m_mae  = mean_absolute_error(y_true, y_pred)
    m_mape = mape(y_true, y_pred)
    m_r2   = r2_score(y_true, y_pred)

    print(f"RMSE: {m_rmse:.4f} | MAE: {m_mae:.4f} | MAPE: {m_mape:.2f}% | R²: {m_r2:.4f}")

    # 7. Sinais e Backtesting CORRIGIDOS
    test_index = df.iloc[len(train_df):].index
    plot_index = test_index[window:]
    
    # EMA alinhado corretamente
    ema_full = df[f'EMA_{ema_span}'].values
    ema_test = ema_full[len(train_df)+window-1 : len(train_df)+window+len(y_pred)-1]
    
    signals = generate_signals_trend(y_pred, y_true, ema_test)
    equity, trades = backtest_discrete(y_true, signals)

    roi_lstm = (equity[-1]/5000 - 1)*100
    print(f"ROI LSTM (test-only): {roi_lstm:.2f}%")

    # 8. Salva gráfico
    plt.figure(figsize=(12,5))
    plt.plot(plot_index, y_true, label="Real")
    plt.plot(plot_index, y_pred, label="Prev")
    plt.title(f"TEST ONLY — {ticker}")
    plt.legend()
    plt.grid(True)
    path = f"{outdir}/testonly_{ticker.replace('.','_')}.png"
    plt.savefig(path, dpi=140)
    print(f"📈 Test-only salvo em: {path}")

# =========================================================
# 6) MENU FINAL
# =========================================================
if __name__ == '__main__':
    bancos = {
        '1': ('ITUB4.SA', 'Itaú Unibanco PN'),
        '2': ('BBDC4.SA', 'Bradesco PN'),
        '3': ('BBAS3.SA', 'Banco do Brasil ON'),
        '4': ('SANB11.SA','Santander Units'),
        '5': ('BPAC11.SA','BTG Pactual Units')
    }

    print("\n=== MENU LSTM BANCOS (CORRIGIDO) ===")
    print("1) Treinar + Testar")
    print("2) Somente Testar (modelo salvo)")
    print("3) Re-testar com outro price_gap")
    print("4) Ver últimos trades")
    print("5) Ver métricas salvas")
    print("6) Sair")

    opt = input("\nEscolha uma opção: ").strip()

    if opt == "6":
        exit()

    # Escolher ticker
    print("\nSelecione o ticker:")
    for k,(t,n) in bancos.items():
        print(f"{k}) {t} — {n}")
    c = input("Opção [1-5]: ").strip()
    if c not in bancos: c = "1"
    ticker = bancos[c][0]

    if opt == "1":
        run_pipeline(ticker)

    elif opt == "2":
        run_test_only(ticker)

    elif opt == "3":
        pg = float(input("Novo price_gap em R$: ").replace(",", "."))
        run_test_only(ticker, price_gap=pg)

    elif opt == "4":
        path = f"outputs/trades_{ticker.replace('.','_')}.csv"
        if os.path.exists(path):
            print(pd.read_csv(path).tail(10))
        else:
            print("❌ Nenhum trade encontrado. Treine primeiro.")

    elif opt == "5":
        path = f"outputs/summary_{ticker.replace('.','_')}.csv"
        if os.path.exists(path):
            print(pd.read_csv(path))
        else:
            print("❌ Nenhum resumo encontrado. Treine primeiro.")

    else:
        print("Opção inválida.")