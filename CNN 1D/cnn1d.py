import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
                                   Input, concatenate, BatchNormalization, LSTM)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

# ===================== PARÂMETROS CNN 1D =====================
TICKER = "PETR4.SA"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-08-31"
INITIAL_CAPITAL = 1000.00
LOOKBACK = 20  # Janela temporal para a CNN
EPOCHS = 100
BATCH_SIZE = 32
N_FEATURES = 12

# Parâmetros de trading
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.05
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35

# ===================== FUNÇÕES DE PREPARAÇÃO =====================
def download_data(ticker, start_date, end_date):
    """Baixa dados do Yahoo Finance"""
    print(f"Baixando dados do {ticker}...")
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        return raw if not raw.empty else None
    except Exception as e:
        print(f"Erro: {e}")
        return None

def calculate_cnn_indicators(df):
    """Calcula indicadores otimizados para CNN"""
    print("Calculando indicadores técnicos para CNN...")
    
    # Preços e volume
    df['returns'] = df['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Tendência
    df['sma_10'] = ta.sma(df['close'], length=10)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)
    
    # Momentum
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close']).iloc[:, 0]
    df['stoch_d'] = ta.stoch(df['high'], df['low'], df['close']).iloc[:, 1]
    
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_hist'] = macd.iloc[:, 2]
    
    # Volatilidade
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    bb = ta.bbands(df['close'], length=20)
    if bb is not None:
        df['bb_upper'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 2]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Volume
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # Novos indicadores para CNN
    df['price_vwap_ratio'] = df['close'] / df['vwap']
    df['momentum_5'] = df['close'].pct_change(5)
    df['volatility_10'] = df['close'].rolling(10).std()
    df['price_sma_ratio'] = df['close'] / df['sma_20']
    
    df = df.dropna()
    print(f"Indicadores calculados: {len(df.columns)} colunas")
    return df

def select_cnn_features(train_df, n_features=N_FEATURES):
    """Seleção de features para CNN"""
    candidate_features = [
        'open', 'high', 'low', 'log_volume', 'volume_ratio', 'returns',
        'sma_10', 'sma_20', 'ema_12', 'ema_26', 
        'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d',
        'macd', 'macd_signal', 'macd_hist',
        'atr_14', 'bb_width', 'bb_upper', 'bb_lower',
        'obv', 'vwap', 'price_vwap_ratio', 'momentum_5', 
        'volatility_10', 'price_sma_ratio'
    ]
    
    available_features = [f for f in candidate_features if f in train_df.columns]
    
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['close']
    
    selector = SelectKBest(score_func=f_regression, k=min(n_features, len(available_features)))
    selector.fit(X_train, y_train)
    
    # Selecionar features com maior importância
    feature_scores = pd.DataFrame({
        'feature': available_features,
        'score': selector.scores_[:len(available_features)]
    })
    feature_scores = feature_scores.sort_values('score', ascending=False)
    
    selected_features = feature_scores['feature'].head(n_features).tolist()
    
    print("📊 Melhores features para CNN:")
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i}. {feat} (score: {feature_scores[feature_scores['feature'] == feat]['score'].values[0]:.2f})")
    
    return selected_features

def prepare_cnn_data(train_df, test_df, features, lookback=LOOKBACK):
    """Prepara dados para CNN 1D"""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Treino
    X_train_scaled = scaler_x.fit_transform(train_df[features])
    y_train_scaled = scaler_y.fit_transform(train_df[['close']])
    
    # Teste
    X_test_scaled = scaler_x.transform(test_df[features])
    y_test_scaled = scaler_y.transform(test_df[['close']])
    
    # Criar sequências para CNN (formato: [samples, timesteps, features])
    def create_cnn_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train, y_train = create_cnn_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test = create_cnn_sequences(X_test_scaled, y_test_scaled, lookback)
    
    print(f"Shape dos dados para CNN:")
    print(f"X_train: {X_train.shape} (amostras, timesteps, features)")
    print(f"X_test: {X_test.shape} (amostras, timesteps, features)")
    
    return X_train, X_test, y_train, y_test, scaler_x, scaler_y

# ===================== MODELO CNN 1D =====================
def create_cnn_1d_model(input_shape):
    """Cria modelo CNN 1D para séries temporais"""
    print("Criando modelo CNN 1D...")
    
    model = Sequential([
        # Primeira camada convolucional
        Conv1D(filters=64, kernel_size=3, activation='relu', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Segunda camada convolucional
        Conv1D(filters=32, kernel_size=2, activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Terceira camada convolucional
        Conv1D(filters=16, kernel_size=2, activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Camadas fully connected
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def create_hybrid_cnn_lstm_model(input_shape):
    """Modelo híbrido CNN-LSTM"""
    print("Criando modelo híbrido CNN-LSTM...")
    
    inputs = Input(shape=input_shape)
    
    # CNN Branch
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # LSTM Branch
    y = LSTM(32, return_sequences=True)(inputs)
    y = Dropout(0.3)(y)
    y = LSTM(16)(y)
    y = Dropout(0.2)(y)
    
    # Combine branches
    combined = concatenate([x, y])
    combined = Flatten()(combined)
    
    # Fully connected layers
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.2)(z)
    z = Dense(16, activation='relu')(z)
    outputs = Dense(1)(z)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.0005), loss='mse', metrics=['mae'])
    
    return model

# ===================== ESTRATÉGIA DE TRADING =====================
def generate_cnn_signals(df, predictions, dates):
    """Gera sinais de trading baseados nas previsões da CNN"""
    signals = ['HOLD']
    positions = [0]
    current_position = 0
    entry_price = 0
    
    for i in range(1, len(predictions)):
        current_date = dates[i]
        price_today = df.loc[current_date, 'close']
        prediction_tomorrow = predictions[i]
        
        # Indicadores para decisão
        rsi = df.loc[current_date, 'rsi_14']
        macd = df.loc[current_date, 'macd']
        macd_signal = df.loc[current_date, 'macd_signal']
        bb_width = df.loc[current_date, 'bb_width'] if 'bb_width' in df.columns else 0
        volume_ratio = df.loc[current_date, 'volume_ratio']
        
        signal = 'HOLD'
        
        if current_position == 0:  # Fora do mercado
            # Condições de COMPRA para CNN
            buy_conditions = [
                prediction_tomorrow > price_today * 1.018,  # Previsão > 1.8%
                rsi < RSI_OVERSOLD,
                macd > macd_signal,
                bb_width > 0.1,  # Mercado volátil
                volume_ratio > 1.0  # Volume acima da média
            ]
            
            if sum(buy_conditions) >= 3:
                signal = 'BUY'
                current_position = 1
                entry_price = price_today
                
        else:  # Comprado
            # Condições de VENDA para CNN
            sell_conditions = [
                prediction_tomorrow < price_today * 0.992,
                rsi > RSI_OVERBOUGHT,
                macd < macd_signal,
                price_today >= entry_price * (1 + TAKE_PROFIT_PCT),
                price_today <= entry_price * (1 - STOP_LOSS_PCT)
            ]
            
            if sum(sell_conditions) >= 2:
                signal = 'SELL'
                current_position = 0
                entry_price = 0
        
        signals.append(signal)
        positions.append(current_position)
    
    return signals, positions

def backtest_cnn_strategy(df, dates, signals, initial_capital=INITIAL_CAPITAL):
    """Backtest para estratégia CNN"""
    capital = initial_capital
    shares = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    trade_active = False
    
    for i, signal in enumerate(signals):
        if i >= len(dates):
            continue
            
        current_date = dates[i]
        current_price = df.loc[current_date, 'close']
        
        # Gestão de risco para trades ativos
        if trade_active and shares > 0:
            if current_price <= entry_price * (1 - STOP_LOSS_PCT):
                capital = shares * current_price
                shares = 0
                trade_active = False
                trades.append({'date': current_date, 'action': 'SELL (SL)', 'price': current_price})
            elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                capital = shares * current_price
                shares = 0
                trade_active = False
                trades.append({'date': current_date, 'action': 'SELL (TP)', 'price': current_price})
        
        # Executar sinais
        if signal == 'BUY' and capital > 0 and not trade_active:
            shares = capital / current_price
            capital = 0
            entry_price = current_price
            trade_active = True
            trades.append({'date': current_date, 'action': 'BUY', 'price': current_price})
            
        elif signal == 'SELL' and shares > 0 and trade_active:
            capital = shares * current_price
            shares = 0
            trade_active = False
            trades.append({'date': current_date, 'action': 'SELL', 'price': current_price})
        
        # Valor do portfólio
        portfolio_value = capital + (shares * current_price if shares > 0 else 0)
        equity_curve.append(portfolio_value)
    
    # Fechar posição no final
    if shares > 0:
        final_price = df['close'].iloc[-1]
        capital = shares * final_price
        trades.append({'date': dates[-1], 'action': 'SELL (FINAL)', 'price': final_price})
    
    return equity_curve, trades, capital

def calculate_performance(equity_curve, initial_capital, trades):
    """Calcula métricas de performance"""
    final_value = equity_curve[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Buy & Hold
    buy_hold_return = (equity_curve[-1] / initial_capital - 1) * 100
    
    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Sharpe ratio
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    winning_trades = len([t for t in trades if t['action'].startswith('SELL') and 'price' in t])
    win_rate = winning_trades / len(trades) if trades else 0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(trades),
        'win_rate': win_rate
    }

# ===================== EXECUÇÃO PRINCIPAL CNN =====================
# ===================== EXECUÇÃO PRINCIPAL CNN =====================
def main():
    print("="*70)
    print("TCC - SISTEMA DE TRADING COM CNN 1D PARA PETR4.SA")
    print("="*70)
    
    # 1. Baixar e processar dados
    print("1. Coletando e processando dados...")
    full_data = download_data(TICKER, "2019-01-01", TEST_END)
    if full_data is None:
        return
    
    df = full_data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.ffill().dropna()
    
    # 2. Calcular indicadores
    df = calculate_cnn_indicators(df)
    
    # 3. Dividir dados
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"2. Período de treino: {len(train_df)} dias")
    print(f"3. Período de teste: {len(test_df)} dias (8 meses)")
    print(f"4. Capital inicial: R$ {INITIAL_CAPITAL:,.2f}")
    
    # 4. Selecionar features
    features = select_cnn_features(train_df)
    
    # 5. Preparar dados para CNN
    X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_cnn_data(train_df, test_df, features)
    
    # 6. Criar e treinar modelo CNN
    print("5. Treinando modelo CNN 1D...")
    model = create_cnn_1d_model((X_train.shape[1], X_train.shape[2]))
    print("Resumo do modelo CNN 1D:")
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=False,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
    )
    
    # 7. Fazer previsões
    print("6. Gerando previsões...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler_y.inverse_transform(y_test).flatten()
    
    # 8. Gerar sinais de trading
    test_dates = test_df.index[LOOKBACK:LOOKBACK + len(y_pred)]
    signals, positions = generate_cnn_signals(test_df, y_pred, test_dates)
    
    # 9. Backtest da estratégia
    print("7. Executando backtest...")
    equity_curve, trades, final_portfolio = backtest_cnn_strategy(test_df, test_dates, signals, INITIAL_CAPITAL)
    
    # 10. Calcular métricas
    metrics = calculate_performance(equity_curve, INITIAL_CAPITAL, trades)
    
    # ===================== RESULTADOS =====================
    print("\n" + "="*70)
    print("RESULTADOS DO BACKTEST - CNN 1D")
    print("="*70)
    print(f"💰 Capital inicial: R$ {INITIAL_CAPITAL:,.2f}")
    print(f"💰 Capital final: R$ {metrics['final_value']:,.2f}")
    print(f"📈 Retorno total: {metrics['total_return']:+.2f}%")
    print(f"📊 Buy & Hold: {metrics['buy_hold_return']:+.2f}%")
    print(f"🔻 Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"🎯 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"🔄 Número de trades: {metrics['num_trades']}")
    print(f"✅ Taxa de acerto: {metrics['win_rate']:.2%}")
    
    # 11. Plotar resultados
    plt.figure(figsize=(15, 10))
    
    # Preços e previsões
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, y_actual, label='Preço Real', linewidth=2, color='blue')
    plt.plot(test_dates, y_pred, label='Preço Previsto CNN', linewidth=2, color='red', linestyle='--')
    
    # Marcar pontos de compra/venda
    buy_dates = [test_dates[i] for i, signal in enumerate(signals) if signal == 'BUY' and i < len(test_dates)]
    sell_dates = [test_dates[i] for i, signal in enumerate(signals) if signal == 'SELL' and i < len(test_dates)]
    
    if buy_dates:
        buy_prices = [test_df.loc[date, 'close'] for date in buy_dates]
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Compra', zorder=5)
    
    if sell_dates:
        sell_prices = [test_df.loc[date, 'close'] for date in sell_dates]
        plt.scatter(sell_dates, sell_prices, color='orange', marker='v', s=100, label='Venda', zorder=5)
    
    plt.title('PETR4.SA - CNN 1D: Preços e Sinais de Trading')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Equity curve
    plt.subplot(2, 1, 2)
    plt.plot(range(len(equity_curve)), equity_curve, label='Estratégia CNN 1D', linewidth=2, color='green')
    plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', label='Capital Inicial')
    plt.title('Evolução do Capital - CNN 1D')
    plt.xlabel('Dias')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultado_cnn_1d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 12. Salvar resultados
    results_df = pd.DataFrame({
        'Data': test_dates[:len(signals)],
        'Preço_Real': y_actual[:len(signals)],
        'Preço_Previsto_CNN': y_pred[:len(signals)],
        'Sinal': signals,
        'Posição': positions[:len(signals)],
        'Capital_Acumulado': equity_curve[1:len(signals)+1]
    })
    
    results_df.to_csv('resultados_cnn_1d.csv', index=False)
    print(f"\n📊 Resultados salvos em 'resultados_cnn_1d.csv'")
    
    # Análise final
    print("\n" + "="*70)
    print("ANÁLISE FINAL - CNN 1D")
    print("="*70)
    
    if metrics['total_return'] > metrics['buy_hold_return']:
        print("🎯 SUCESSO: CNN 1D superou o Buy & Hold!")
        print(f"   📈 Vantagem: {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
    else:
        print("⚠️  CNN 1D não superou o Buy & Hold")
        print(f"   📉 Diferença: {metrics['total_return'] - metrics['buy_hold_return']:.2f}%")
    
    print(f"🔒 Drawdown controlado: {metrics['max_drawdown']:.2f}%")
    print(f"🎯 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    return metrics, results_df

if __name__ == "__main__":
    # Configurar para usar CPU (já é default sem GPU)
    tf.config.set_visible_devices([], 'GPU')
    
    try:
        metrics, results = main()
        print("\n🎓 TCC COM CNN 1D EXECUTADO COM SUCESSO!")
        
        # Comparativo com resultado anterior
        print(f"\n🔍 COMPARATIVO COM LSTM ANTERIOR:")
        print(f"   CNN 1D: {metrics['total_return']:+.2f}%")
        print(f"   Buy & Hold: {metrics['buy_hold_return']:+.2f}%")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()