import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

# ===================== PARÂMETROS OTIMIZADOS =====================
TICKER = "MGLU3.SA"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-08-31"
INITIAL_CAPITAL = 1000.00
#considerar valor da taxas
#energia, pretoleo, realimentar o agente.
LOOKBACK = 10  # 
EPOCHS = 80
BATCH_SIZE = 16  # 
N_FEATURES = 10  # 

# Parâmetros de trading otimizados
STOP_LOSS_PCT = 0.03  # 3% de stop loss
TAKE_PROFIT_PCT = 0.05  # 5% de take profit
RSI_OVERBOUGHT = 65  # Mais sensível para venda
RSI_OVERSOLD = 35    # Mais sensível para compra

# ===================== FUNÇÕES OTIMIZADAS =====================
def download_data(ticker, start_date, end_date):
    print(f"Baixando dados do {ticker}...")
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        return raw if not raw.empty else None
    except Exception as e:
        print(f"Erro: {e}")
        return None

def calculate_enhanced_indicators(df):
    print("Calculando indicadores técnicos avançados...")
    
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
    df['rsi_7'] = ta.rsi(df['close'], length=7)  # RSI mais sensível
    
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
        df['bb_lower'] = bb.iloc[:, 2]
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # # Novos indicadores
    # df['price_vwap_ratio'] = df['close'] / df['vwap']
    # df['momentum_5'] = df['close'].pct_change(5)
    
    df = df.dropna()
    print(f"Indicadores calculados: {len(df.columns)} colunas")
    return df

def select_optimized_features(train_df, n_features=N_FEATURES):
    """Seleção mais inteligente de features"""
    candidate_features = [
        'open', 'high', 'low', 'log_volume', 'volume_ratio', 'returns',
        'sma_10', 'sma_20', 'ema_12', 'ema_26', 
        'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
        'atr_14', 'bb_pct', 'obv', 'vwap', 'price_vwap_ratio', 'momentum_5'
    ]
    
    available_features = [f for f in candidate_features if f in train_df.columns]
    
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['close']
    
    selector = SelectKBest(score_func=f_regression, k=min(n_features, len(available_features)))
    selector.fit(X_train, y_train)
    
    # Pegar features com maior importância
    feature_scores = pd.DataFrame({
        'feature': available_features,
        'score': selector.scores_[:len(available_features)]
    })
    feature_scores = feature_scores.sort_values('score', ascending=False)
    
    selected_features = feature_scores['feature'].head(n_features).tolist()
    
    print("Melhor feature")
    for i, feat in enumerate(selected_features, 1):
        print(f"   {i}. {feat} (score: {feature_scores[feature_scores['feature'] == feat]['score'].values[0]:.2f})")
    
    return selected_features

def prepare_optimized_data(train_df, test_df, features, lookback=LOOKBACK):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Treino
    X_train_scaled = scaler_x.fit_transform(train_df[features])
    y_train_scaled = scaler_y.fit_transform(train_df[['close']])
    
    # Teste
    X_test_scaled = scaler_x.transform(test_df[features])
    y_test_scaled = scaler_y.transform(test_df[['close']])
    
    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, lookback)
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler_x, scaler_y

def create_enhanced_model(input_shape):
    """Modelo LSTM otimizado"""
    model = Sequential([
        LSTM(48, return_sequences=True, input_shape=input_shape, 
             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.35),
        LSTM(24, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.25),
        Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0008),  
        loss='mse',
        metrics=['mae']
    )
    return model

# ===================== ESTRATÉGIA DE TRADING OTIMIZADA =====================
def generate_enhanced_signals(df, predictions, dates):
    """Geração de sinais com múltiplos filtros"""
    signals = ['HOLD']  # Primeiro dia sempre hold
    positions = [0]
    current_position = 0
    entry_price = 0
    
    for i in range(1, len(predictions)):
        current_date = dates[i]
        price_today = df.loc[current_date, 'close']
        prediction_tomorrow = predictions[i]
        
        # Indicadores para filtragem
        rsi = df.loc[current_date, 'rsi_14']
        macd = df.loc[current_date, 'macd']
        macd_signal = df.loc[current_date, 'macd_signal']
        bb_pct = df.loc[current_date, 'bb_pct'] if 'bb_pct' in df.columns else 0.5
        volume_ratio = df.loc[current_date, 'volume_ratio']
        
        signal = 'HOLD'
        
        if current_position == 0: 
            # Condições de COMPRA 
            buy_conditions = [
                prediction_tomorrow > price_today * 1.015,  
                rsi < RSI_OVERSOLD,                        
                macd > macd_signal,                         
                bb_pct < 0.8,                               
                volume_ratio > 0.8                         
            ]
            # Previsão > 1.5%
             # RSI não sobrevendido
             # MACD positivo
             # Não no topo da Bollinger
              # Volume acima da média
              
            if sum(buy_conditions) >= 3:  # Pelo menos 3 condições
                signal = 'BUY'
                current_position = 1
                entry_price = price_today
                
        else:  # Comprado
            # Condições de VENDA 
            sell_conditions = [
                prediction_tomorrow < price_today * 0.995, 
                rsi > RSI_OVERBOUGHT,                       
                macd < macd_signal,                        
                bb_pct > 0.8,                               
                price_today >= entry_price * (1 + TAKE_PROFIT_PCT), 
                price_today <= entry_price * (1 - STOP_LOSS_PCT)    
            ]
            # Previsão < -0.5%
            # RSI sobrecomprado
            # MACD negativo
            # Topo da Bollinger
            # Take profit
            # Stop loss
            if sum(sell_conditions) >= 2:  # Pelo menos 2 condições
                signal = 'SELL'
                current_position = 0
                entry_price = 0
        
        signals.append(signal)
        positions.append(current_position)
    
    return signals, positions

def backtest_enhanced_strategy(df, dates, signals, initial_capital=INITIAL_CAPITAL):
    """Backtest com gestão de risco completa"""
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
        
        #-----------STOP LOSS-----------------
        if trade_active and shares > 0:
            # Stop Loss
            if current_price <= entry_price * (1 - STOP_LOSS_PCT):
                capital = shares * current_price
                shares = 0
                trade_active = False
                trades.append({
                    'date': current_date, 
                    'action': 'SELL (SL)', 
                    'price': current_price, 
                    'capital': capital,
                    'return_pct': (current_price / entry_price - 1) * 100
                })
            
            # ------------------Take Profit  -----------------
            elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                capital = shares * current_price
                shares = 0
                trade_active = False
                trades.append({
                    'date': current_date, 
                    'action': 'SELL (TP)', 
                    'price': current_price, 
                    'capital': capital,
                    'return_pct': (current_price / entry_price - 1) * 100
                })
        
        # ----------------- SINAL DE COMPRA -------------------------
        if signal == 'BUY' and capital > 0 and not trade_active:
            shares = capital / current_price
            capital = 0
            entry_price = current_price
            trade_active = True
            trades.append({
                'date': current_date, 
                'action': 'BUY', 
                'price': current_price, 
                'shares': shares
            })
        # ----------------- SINAL DE VENDA -------------------------  
        elif signal == 'SELL' and shares > 0 and trade_active:
            capital = shares * current_price
            shares = 0
            trade_active = False
            trades.append({
                'date': current_date, 
                'action': 'SELL', 
                'price': current_price, 
                'capital': capital,
                'return_pct': (current_price / entry_price - 1) * 100
            })
        
        portfolio_value = capital + (shares * current_price if shares > 0 else 0)
        equity_curve.append(portfolio_value)
    
    # Fechar posição aberta no final
    if shares > 0:
        final_price = df['close'].iloc[-1]
        capital = shares * final_price
        trades.append({
            'date': dates[-1], 
            'action': 'SELL (FINAL)', 
            'price': final_price, 
            'capital': capital,
            'return_pct': (final_price / entry_price - 1) * 100
        })
    
    return equity_curve, trades, capital

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convertendo para porcentagem
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

def calculate_detailed_metrics(equity_curve, initial_capital, trades, test_dates, y_true, y_pred):
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
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    # Estatísticas dos trades
    winning_trades = [t for t in trades if 'return_pct' in t and t['return_pct'] > 0]
    losing_trades = [t for t in trades if 'return_pct' in t and t['return_pct'] <= 0]
    
    avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    # Métricas de regressão
    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf'),
        **regression_metrics  # Inclui todas as métricas de regressão
    }

# ===================== EXECUÇÃO PRINCIPAL OTIMIZADA =====================
def main():
    print("="*70)
    print("TCC - SISTEMA DE TRADING AVANÇADO COM LSTM")
    print("="*70)
    
    # 1. Baixar e processar dados
    print("1. Coletando e processando dados...")
    full_data = download_data(TICKER, "2019-01-01", TEST_END)
    if full_data is None:
        return
    
    df = full_data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.ffill().dropna()
    
    # 2. Calcular indicadores avançados
    df = calculate_enhanced_indicators(df)
    
    # 3. Dividir dados
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"2. Período de treino: {len(train_df)} dias")
    print(f"3. Período de teste: {len(test_df)} dias (8 meses)")
    print(f"4. Capital inicial: R$ {INITIAL_CAPITAL:,.2f}")
    
    # 4. Selecionar features otimizadas
    features = select_optimized_features(train_df)
    
    # 5. Preparar dados
    X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_optimized_data(train_df, test_df, features)
    
    # 6. Criar e treinar modelo
    print("5. Treinando modelo LSTM ...")
    model = create_enhanced_model((X_train.shape[1], X_train.shape[2]))
    
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test), 
        verbose=1, 
        shuffle=False,
        callbacks=[EarlyStopping(patience=12, restore_best_weights=True)]
    )
    
    # 7. Fazer previsões
    print("6. Gerando previsões...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler_y.inverse_transform(y_test).flatten()
    
    # 8. Gerar sinais de trading otimizados
    test_dates = test_df.index[LOOKBACK:LOOKBACK + len(y_pred)]
    signals, positions = generate_enhanced_signals(test_df, y_pred, test_dates)
    
    # 9. Backtest com gestão de risco
    print("7. Executando backtest ...")
    equity_curve, trades, final_portfolio = backtest_enhanced_strategy(test_df, test_dates, signals, INITIAL_CAPITAL)
    
    # 10. Calcular métricas detalhadas
    metrics = calculate_detailed_metrics(equity_curve, INITIAL_CAPITAL, trades, test_dates, y_actual, y_pred)
    
    # ===================== RESULTADOS DETALHADOS =====================
    print("\n" + "="*70)
    print("RESULTADOS DO BACKTEST  - 8 MESES")
    print("="*70)
    print(f"$ Capital inicial: R$ {INITIAL_CAPITAL:,.2f}")
    print(f"$ Capital final: R$ {metrics['final_value']:,.2f}")
    print(f"= Retorno total: {metrics['total_return']:+.2f}%")
    # print(f"= Buy & Hold: {metrics['buy_hold_return']:+.2f}%")
    print(f"< Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"> Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"+- Número de trades: {metrics['num_trades']}")
    print(f"OK Taxa de acerto: {metrics['win_rate']:.2%}")
    print(f"=/ Retorno médio por trade: {metrics['avg_win']:+.2f}%")
    print(f"-/ Prejuízo médio por trade: {metrics['avg_loss']:.2f}%")
    print(f"fator de ganho: {metrics['profit_factor']:.2f}")
    
    # Métricas de regressão
    print("\n" + "="*70)
    print("MÉTRICAS DE REGRESSÃO - QUALIDADE DAS PREVISÕES")
    print("="*70)
    print(f"MAE (Erro Absoluto Médio): R$ {metrics['MAE']:.2f}")
    print(f"MAPE (Erro Percentual Absoluto Médio): {metrics['MAPE']:.2f}%")
    print(f"MSE (Erro Quadrático Médio): {metrics['MSE']:.2f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): R$ {metrics['RMSE']:.2f}")
    
    # 12. Plotar resultados
    plt.figure(figsize=(15, 15))
    
    # Preços e previsões
    plt.subplot(4, 1, 1)
    plt.plot(test_dates, y_actual, label='Preço Real', linewidth=2, color='blue')
    plt.plot(test_dates, y_pred, label='Preço Previsto', linewidth=2, color='red', linestyle='--')
    
    # Marcar pontos de compra/venda
    buy_dates = [test_dates[i] for i, signal in enumerate(signals) if signal == 'BUY' and i < len(test_dates)]
    sell_dates = [test_dates[i] for i, signal in enumerate(signals) if signal == 'SELL' and i < len(test_dates)]
    
    if buy_dates:
        buy_prices = [test_df.loc[date, 'close'] for date in buy_dates]
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Compra', zorder=5)
    
    if sell_dates:
        sell_prices = [test_df.loc[date, 'close'] for date in sell_dates]
        plt.scatter(sell_dates, sell_prices, color='orange', marker='v', s=100, label='Venda', zorder=5)
    
    plt.title(f'{TICKER} - Preços e Sinais de Trading')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Equity curve
    plt.subplot(4, 1, 2)
    plt.plot(range(len(equity_curve)), equity_curve, label='Estratégia LSTM', linewidth=2, color='green')
    plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', label='Capital Inicial')
    plt.title('Evolução do Capital')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdown
    plt.subplot(4, 1, 3)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    plt.plot(drawdown, color='red', linewidth=1)
    plt.title('Drawdown Máximo')
    plt.xlabel('Dias')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Erros de previsão
    plt.subplot(4, 1, 4)
    errors = y_actual - y_pred
    plt.plot(test_dates, errors, color='purple', linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.fill_between(test_dates, errors, 0, alpha=0.3, color='purple')
    plt.title('Erros de Previsão (Real - Previsto)')
    plt.xlabel('Data')
    plt.ylabel('Erro (R$)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./img/resultado_tcc_otimizado_{TICKER}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 13. Salvar resultados detalhados
    results_df = pd.DataFrame({
        'Data': test_dates[:len(signals)],
        'Preço_Real': y_actual[:len(signals)],
        'Preço_Previsto': y_pred[:len(signals)],
        'Erro_Absoluto': np.abs(y_actual[:len(signals)] - y_pred[:len(signals)]),
        'Erro_Percentual': (np.abs(y_actual[:len(signals)] - y_pred[:len(signals)]) / y_actual[:len(signals)]) * 100,
        'Sinal': signals,
        'Posição': positions[:len(signals)],
        'Capital_Acumulado': equity_curve[1:len(signals)+1]
    })
    
    trades_df = pd.DataFrame(trades)
    results_df.to_csv(f'./csv/trades/resultados_tcc_detalhados{TICKER}.csv', index=False)
    trades_df.to_csv(f'./csv/trades/trades_detalhados_{TICKER}.csv', index=False)
    
    print(f"\n salvo em 'resultados_tcc_detalhados.csv'")
    print(f" Trades salvo em 'trades_detalhados.csv'")
    
    # Análise final
    print("\n" + "="*70)
    print("ANÁLISE DAS MÉTRICAS DE REGRESSÃO")
    print("="*70)
    
    # Interpretação das métricas de regressão
    print("MAE (Mean Absolute Error):")
    print("  - Erro médio absoluto em reais")
    print(f"  - O modelo erra em média R$ {metrics['MAE']:.2f} por previsão")
    
    print("\nMAPE (Mean Absolute Percentage Error):")
    print("  - Erro percentual médio absoluto")
    print(f"  - O modelo erra em média {metrics['MAPE']:.2f}% por previsão")
    
    # print("\nR² (Coeficiente de Determinação):")
    # print("  - Proporção da variância explicada pelo modelo")
    # print(f"  - {metrics['R²']:.1%} da variância dos preços é explicada pelo modelo")
    
    if metrics['total_return'] > metrics['buy_hold_return']:
        print("\n✅ Estratégia superou o Buy & Hold!")
        print(f"   Vantagem: {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
    else:
        print("\n❌ Estratégia não superou Buy & Hold")
    
    return metrics, results_df, trades_df

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    try:
        metrics, results, trades = main()
    except Exception as e:
        print(f"*** Erro: {e}***")
        import traceback
        traceback.print_exc()