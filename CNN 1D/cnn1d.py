# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, mean_absolute_percentage_error)
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURAÇÕES DE PLOT =====================
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

# ===================== PARÂMETROS GERAIS =====================
TICKER        = "MGLU3.SA"
TRAIN_START   = "2020-01-01"
TRAIN_END     = "2022-12-31"
TEST_START    = "2023-01-01"
TEST_END      = "2023-08-31"
INITIAL_CAPITAL = 1000.00

LOOKBACK   = 50
EPOCHS     = 80
BATCH_SIZE = 16
N_FEATURES = 10

# Parâmetros de trading
STOP_LOSS_PCT   = 0.03   # 3%
TAKE_PROFIT_PCT = 0.05   # 5%
RSI_OVERBOUGHT  = 65
RSI_OVERSOLD    = 35

# O que rodar: "LSTM", "CNN" ou "BOTH"
MODEL_TO_RUN = "BOTH"

# ===================== UTIL =====================
def ensure_dirs():
    os.makedirs("./img", exist_ok=True)
    os.makedirs("./csv/trades", exist_ok=True)

def download_data(ticker, start_date, end_date):
    print(f"Baixando dados do {ticker}...")
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        return raw if not raw.empty else None
    except Exception as e:
        print(f"Erro no download: {e}")
        return None

# ===================== FEATURES =====================
def calculate_enhanced_indicators(df):
    print("Calculando indicadores técnicos avançados...")

    # Preços e volume
    df['returns']      = df['close'].pct_change()
    df['log_volume']   = np.log1p(df['volume'])
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # Tendência
    df['sma_10'] = ta.sma(df['close'], length=10)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)

    # Momentum
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_7']  = ta.rsi(df['close'], length=7)

    macd = ta.macd(df['close'])
    if macd is not None and not macd.empty:
        df['macd']        = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_hist']   = macd.iloc[:, 2]

    # Volatilidade / Bandas
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    bb = ta.bbands(df['close'], length=20)
    if bb is not None and not bb.empty:
        df['bb_upper'] = bb.iloc[:, 0]
        df['bb_lower'] = bb.iloc[:, 2]
        df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume / VWAP
    df['obv']  = ta.obv(df['close'], df['volume'])
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Indicadores calculados: {len(df.columns)} colunas")
    return df

def select_optimized_features(train_df, n_features=N_FEATURES):
    candidate_features = [
        'open', 'high', 'low', 'log_volume', 'volume_ratio', 'returns',
        'sma_10', 'sma_20', 'ema_12', 'ema_26',
        'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
        'atr_14', 'bb_pct', 'obv', 'vwap'
    ]
    available = [f for f in candidate_features if f in train_df.columns]

    X = train_df[available].fillna(0)
    y = train_df['close']

    selector = SelectKBest(score_func=f_regression, k=min(n_features, len(available)))
    selector.fit(X, y)

    feature_scores = pd.DataFrame({
        'feature': available,
        'score': selector.scores_[:len(available)]
    }).sort_values('score', ascending=False)

    selected = feature_scores['feature'].head(n_features).tolist()

    print("Melhores features selecionadas:")
    for i, feat in enumerate(selected, 1):
        score = feature_scores.loc[feature_scores['feature'] == feat, 'score'].values[0]
        print(f"  {i:02d}. {feat:<12} | score: {score:8.2f}")
    return selected

def prepare_optimized_data(train_df, test_df, features, lookback=LOOKBACK):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(train_df[features])
    y_train_scaled = scaler_y.fit_transform(train_df[['close']])

    X_test_scaled  = scaler_x.transform(test_df[features])
    y_test_scaled  = scaler_y.transform(test_df[['close']])

    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test,  y_test  = create_sequences(X_test_scaled,  y_test_scaled,  lookback)

    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler_x, scaler_y

# ===================== MODELOS =====================
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(48, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.35),
        LSTM(24, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.25),
        Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0008), loss='mse', metrics=['mae'])
    return model

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same',
               input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.30),

        Conv1D(32, 2, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.25),

        GlobalAveragePooling1D(),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.20),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ===================== SINAIS & BACKTEST =====================
def generate_signals(df, predictions, dates):
    """
    Mesmo conjunto de regras para LSTM e CNN, para comparação justa.
    """
    signals = ['HOLD']
    positions = [0]
    current_position = 0
    entry_price = 0.0

    for i in range(1, len(predictions)):
        d = dates[i]
        price = df.loc[d, 'close']
        pred  = predictions[i]

        rsi  = df.loc[d, 'rsi_14']
        macd = df.loc[d, 'macd']
        macs = df.loc[d, 'macd_signal']
        bbp  = df.loc[d, 'bb_pct']

        signal = 'HOLD'

        if current_position == 0:
            buy_conds = [
                pred > price * 1.015,     # previsão > +1.5%
                rsi < RSI_OVERSOLD,       # RSI sobrevendido
                macd > macs,              # cruzamento positivo
                bbp < 0.8                 # não no topo das bandas
            ]
            if sum(buy_conds) >= 2:
                signal = 'BUY'
                current_position = 1
                entry_price = price
        else:
            sell_conds = [
                pred < price * 0.995,                  # previsão < -0.5%
                rsi > RSI_OVERBOUGHT,                  # RSI sobrecomprado
                macd < macs,                           # cruzamento negativo
                bbp > 0.8,                             # topo das bandas
                price >= entry_price*(1+TAKE_PROFIT_PCT),
                price <= entry_price*(1-STOP_LOSS_PCT)
            ]
            if sum(sell_conds) >= 2:
                signal = 'SELL'
                current_position = 0
                entry_price = 0.0

        signals.append(signal)
        positions.append(current_position)

    return signals, positions

def backtest(df, dates, signals, initial_capital=INITIAL_CAPITAL):
    capital = initial_capital
    shares = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [capital]
    trade_active = False

    for i, signal in enumerate(signals):
        if i >= len(dates):
            continue
        d = dates[i]
        price = df.loc[d, 'close']

        # SL/TP ativos
        if trade_active and shares > 0:
            if price <= entry_price*(1-STOP_LOSS_PCT):
                capital = shares * price
                trades.append({'date': d, 'action': 'SELL (SL)', 'price': price,
                               'return_pct': (price/entry_price - 1)*100})
                shares = 0.0
                trade_active = False
            elif price >= entry_price*(1+TAKE_PROFIT_PCT):
                capital = shares * price
                trades.append({'date': d, 'action': 'SELL (TP)', 'price': price,
                               'return_pct': (price/entry_price - 1)*100})
                shares = 0.0
                trade_active = False

        # Execução sinais
        if signal == 'BUY' and capital > 0 and not trade_active:
            shares = capital / price
            trades.append({'date': d, 'action': 'BUY', 'price': price, 'shares': shares})
            capital = 0.0
            entry_price = price
            trade_active = True
        elif signal == 'SELL' and shares > 0 and trade_active:
            capital = shares * price
            trades.append({'date': d, 'action': 'SELL', 'price': price,
                           'return_pct': (price/entry_price - 1)*100})
            shares = 0.0
            trade_active = False

        portfolio_value = capital + (shares * price if shares > 0 else 0.0)
        equity_curve.append(portfolio_value)

    # Fecha posição no final (se aberta)
    if shares > 0:
        last_price = df['close'].iloc[-1]
        capital = shares * last_price
        trades.append({'date': dates[-1], 'action': 'SELL (FINAL)', 'price': last_price,
                       'return_pct': (last_price/entry_price - 1)*100})
        shares = 0.0
        equity_curve[-1] = capital

    return equity_curve, trades, capital

# ===================== MÉTRICAS =====================
def calculate_regression_metrics(y_true, y_pred):
    return {
        'MSE' : mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE' : mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R²'  : r2_score(y_true, y_pred)
    }

def calculate_detailed_metrics(equity_curve, initial_capital, trades, y_true, y_pred):
    final_value  = float(equity_curve[-1])
    total_return = (final_value/initial_capital - 1) * 100.0

    # Buy & Hold no MESMO intervalo das previsões (test window pós-lookback)
    buy_hold_return = (y_true[-1] / y_true[0] - 1) * 100.0

    # Drawdown
    ec = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    drawdown = (peak - ec) / peak * 100.0
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0

    # Sharpe (aprox. diária)
    daily_returns = np.diff(ec) / ec[:-1] if len(ec) > 1 else np.array([0.0])
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0.0)

    # Trades
    ret_trades = [t['return_pct'] for t in trades if 'return_pct' in t]
    wins  = [r for r in ret_trades if r > 0]
    loss  = [r for r in ret_trades if r <= 0]
    win_rate = (len(wins) / len(ret_trades)) if ret_trades else 0.0
    avg_win  = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(loss) if loss else 0.0
    profit_factor = (abs(avg_win*len(wins) / (avg_loss*len(loss)))
                     if loss else float('inf')) if ret_trades else 0.0

    # Regressão
    reg = calculate_regression_metrics(y_true, y_pred)

    return {
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(ret_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        **reg
    }

# ===================== PIPELINE GENÉRICO =====================
def run_pipeline(model_name, build_model_fn):
    """
    model_name: "LSTM" | "CNN"
    build_model_fn: função que recebe input_shape e devolve um modelo compilado
    """
    print("="*70)
    print(f"TCC - SISTEMA DE TRADING ({model_name})")
    print("="*70)

    ensure_dirs()

    # 1) Dados
    full = download_data(TICKER, "2019-01-01", TEST_END)
    if full is None or full.empty:
        raise RuntimeError("Sem dados.")
    df = full[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = df.ffill().dropna()

    # 2) Indicadores
    df = calculate_enhanced_indicators(df)

    # 3) Split
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    test_mask  = (df.index >= TEST_START)  & (df.index <= TEST_END)
    train_df, test_df = df[train_mask].copy(), df[test_mask].copy()

    print(f"Período de treino: {len(train_df)} dias")
    print(f"Período de teste:  {len(test_df)} dias")

    # 4) Features
    feats = select_optimized_features(train_df, N_FEATURES)

    # 5) Prep
    X_train, X_test, y_train, y_test, sx, sy = prepare_optimized_data(train_df, test_df, feats, LOOKBACK)

    # 6) Modelo
    print(f"Treinando {model_name} ...")
    model = build_model_fn((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1, shuffle=False,
        callbacks=[EarlyStopping(patience=12, restore_best_weights=True)]
    )

    # 7) Previsões
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred   = sy.inverse_transform(y_pred_scaled).flatten()
    y_actual = sy.inverse_transform(y_test).flatten()

    # 8) Sinais e Backtest (alinhando datas às previsões)
    test_dates = test_df.index[LOOKBACK:LOOKBACK + len(y_pred)]
    signals, positions = generate_signals(test_df, y_pred, test_dates)
    equity_curve, trades, final_portfolio = backtest(test_df, test_dates, signals, INITIAL_CAPITAL)

    # 9) Métricas detalhadas com comparação Buy & Hold
    metrics = calculate_detailed_metrics(equity_curve, INITIAL_CAPITAL, trades, y_actual, y_pred)

    # 10) Impressão
    print("\n" + "="*70)
    print(f"RESULTADOS DO BACKTEST  - {model_name}")
    print("="*70)
    print(f"$ Capital inicial: R$ {INITIAL_CAPITAL:,.2f}")
    print(f"$ Capital final:   R$ {metrics['final_value']:,.2f}")
    print(f"= Retorno total:   {metrics['total_return']:+.2f}%")
    print(f"= Buy & Hold:      {metrics['buy_hold_return']:+.2f}%")
    print(f"< Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    print(f"> Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"+- Nº trades:      {metrics['num_trades']}")
    print(f"OK Taxa de acerto: {metrics['win_rate']:.2%}")
    print(f"=/ Ret. méd. trade:{metrics['avg_win']:+.2f}%")
    print(f"-/ Prej. méd. tr.: {metrics['avg_loss']:.2f}%")
    print(f"Fator de ganho:    {metrics['profit_factor']:.2f}")

    print("\n" + "="*70)
    print("MÉTRICAS DE REGRESSÃO")
    print("="*70)
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"MSE : {metrics['MSE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R²  : {metrics['R²']:.4f}")

    # 11) Gráficos
    plt.figure(figsize=(15, 15))

    # (1) Preços e Previsões com marcações
    plt.subplot(4,1,1)
    plt.plot(test_dates, y_actual, label='Preço Real', linewidth=2)
    plt.plot(test_dates, y_pred,   label='Preço Previsto', linewidth=2, linestyle='--')
    # marcações de BUY/SELL
    buy_idx  = [i for i, s in enumerate(signals) if s == 'BUY'  and i < len(test_dates)]
    sell_idx = [i for i, s in enumerate(signals) if s == 'SELL' and i < len(test_dates)]
    if buy_idx:
        plt.scatter(test_dates[buy_idx], test_df.loc[test_dates[buy_idx], 'close'],
                    marker='^', s=100, label='Compra', zorder=5)
    if sell_idx:
        plt.scatter(test_dates[sell_idx], test_df.loc[test_dates[sell_idx], 'close'],
                    marker='v', s=100, label='Venda', zorder=5)
    plt.title(f'{TICKER} - {model_name}: Preço vs Previsão & Sinais')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (2) Equity Curve
    plt.subplot(4,1,2)
    plt.plot(range(len(equity_curve)), equity_curve, label=f'Estratégia {model_name}', linewidth=2)
    plt.axhline(y=INITIAL_CAPITAL, linestyle='--', label='Capital Inicial')
    plt.title('Evolução do Capital')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (3) Drawdown
    plt.subplot(4,1,3)
    ec = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    drawdown = (peak - ec) / peak * 100.0
    plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3)
    plt.plot(drawdown, linewidth=1)
    plt.title('Drawdown (%)')
    plt.ylabel('DD (%)')
    plt.grid(True, alpha=0.3)

    # (4) Erros
    plt.subplot(4,1,4)
    errors = y_actual - y_pred
    plt.plot(test_dates, errors, linewidth=1)
    plt.axhline(y=0, linestyle='--')
    plt.title('Erros de Previsão (Real - Previsto)')
    plt.xlabel('Data')
    plt.ylabel('Erro (R$)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_img = f'./img/resultado_{model_name.lower()}_{TICKER}.png'
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Figura salva em: {out_img}")

    # 12) CSVs
    results_df = pd.DataFrame({
        'Data': test_dates[:len(signals)],
        'Preco_Real': y_actual[:len(signals)],
        'Preco_Previsto': y_pred[:len(signals)],
        'Erro_Abs': np.abs(y_actual[:len(signals)] - y_pred[:len(signals)]),
        'Erro_%': (np.abs(y_actual[:len(signals)] - y_pred[:len(signals)]) / y_actual[:len(signals)]) * 100,
        'Sinal': signals,
        'Posicao': positions[:len(signals)],
        'Capital_Acumulado': equity_curve[1:len(signals)+1]
    })
    trades_df = pd.DataFrame(trades)

    res_csv   = f'./csv/trades/resultados_{model_name.lower()}_{TICKER}.csv'
    trades_csv= f'./csv/trades/trades_{model_name.lower()}_{TICKER}.csv'
    results_df.to_csv(res_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)
    print(f"Resultados salvos em: {res_csv}")
    print(f"Trades salvos em:     {trades_csv}")

    return {
        'model': model_name,
        'metrics': metrics,
        'equity_curve': equity_curve,
        'trades': trades,
        'results_df': results_df,
        'trades_df': trades_df
    }

# ===================== MAIN =====================
if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    out = []
    if MODEL_TO_RUN in ("LSTM", "BOTH"):
        out.append(run_pipeline("LSTM", create_lstm_model))
    if MODEL_TO_RUN in ("CNN", "BOTH"):
        out.append(run_pipeline("CNN", create_cnn_model))

    # Comparação (se houver 2 resultados)
    if len(out) == 2:
        print("\n" + "="*70)
        print("COMPARAÇÃO LADO A LADO (LSTM vs CNN)".center(70))
        print("="*70)
        rows = []
        keys = [
            'final_value','total_return','buy_hold_return',
            'max_drawdown','sharpe_ratio','num_trades','win_rate',
            'avg_win','avg_loss','profit_factor','MAE','MAPE','RMSE','R²'
        ]
        for key in keys:
            v_lstm = out[0]['metrics'][key]
            v_cnn  = out[1]['metrics'][key]
            rows.append([key, v_lstm, v_cnn])

        comp_df = pd.DataFrame(rows, columns=['Métrica', 'LSTM', 'CNN'])
        # Impressão enxuta
        with pd.option_context('display.float_format', '{:,.4f}'.format):
            print(comp_df.to_string(index=False))
