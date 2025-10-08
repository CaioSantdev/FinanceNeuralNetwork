import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURAÇÕES COM TAXAS =====================
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

TICKER = "PETR4.SA"
TRAIN_START = "2020-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-08-31"
INITIAL_CAPITAL = 1000.00
LOOKBACK = 30
EPOCHS = 100
BATCH_SIZE = 16
N_FEATURES = 10

STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.10

# TAXAS DE TRANSAÇÃO REALISTAS PARA AÇÕES BRASILEIRAS
TAXA_CORRETAGEM = 0.005  # 0.5% por operação (compra ou venda)
TAXA_EMOLUMENTOS = 0.0005  # 0.05% 
TAXA_LIQUIDAÇÃO = 0.0002  # 0.02%
IMPOSTO_DAY_TRADE = 0.20  # 20% para day trade (operaciones no mesmo dia)

# Taxa total por operação (compra + venda = 2 operações)
TAXA_TOTAL_POR_OPERACAO = TAXA_CORRETAGEM * 2 + TAXA_EMOLUMENTOS * 2 + TAXA_LIQUIDAÇÃO * 2

print(f"Taxa total por trade round-trip: {TAXA_TOTAL_POR_OPERACAO*100:.2f}%")

# ===================== DOWNLOAD =====================
def download_data(ticker, start_date, end_date):
    print(f"Baixando dados de {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    return data if not data.empty else None

# ===================== INDICADORES =====================
def calculate_indicators(df):
    df = df.copy()
    
    df['returns'] = df['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    df['sma_10'] = ta.sma(df['close'], length=10)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)
    
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_hist'] = macd.iloc[:, 2]
    
    bb = ta.bbands(df['close'], length=20)
    if bb is not None and len(bb.columns) >= 3:
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_pct'] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
    
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    df = df.dropna()
    return df

# ===================== SELEÇÃO DE FEATURES =====================
def select_features(train_df, n_features=N_FEATURES):
    correlation_with_target = train_df.corr()['close'].abs().sort_values(ascending=False)
    
    features_to_consider = [col for col in correlation_with_target.index 
                          if col not in ['close', 'open', 'high', 'low']]
    
    selected_features = []
    for feature in features_to_consider:
        if len(selected_features) < n_features:
            add_feature = True
            for selected in selected_features:
                if abs(train_df[feature].corr(train_df[selected])) > 0.8:
                    add_feature = False
                    break
            if add_feature:
                selected_features.append(feature)
    
    print("Features selecionadas:", selected_features)
    return selected_features

# ===================== PREPARAÇÃO =====================
def prepare_data(train_df, test_df, features, lookback=LOOKBACK):
    scaler_x_train = StandardScaler()
    scaler_y_train = StandardScaler()
    
    X_train_scaled = scaler_x_train.fit_transform(train_df[features])
    y_train_scaled = scaler_y_train.fit_transform(train_df[['close']])
    
    X_test_scaled = scaler_x_train.transform(test_df[features])
    y_test_scaled = scaler_y_train.transform(test_df[['close']])

    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, lookback)

    return X_train, X_test, y_train, y_test, scaler_y_train

# ===================== MODELO LSTM =====================
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(16, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

# ===================== BACKTEST COM TAXAS =====================
def aplicar_taxas_compra(valor_compra):
    """Aplica taxas na operação de compra"""
    taxas_compra = TAXA_CORRETAGEM + TAXA_EMOLUMENTOS + TAXA_LIQUIDAÇÃO
    return valor_compra * (1 + taxas_compra)

def aplicar_taxas_venda(valor_venda, lucro_operacao, eh_day_trade):
    """Aplica taxas na operação de venda e impostos"""
    taxas_venda = TAXA_CORRETAGEM + TAXA_EMOLUMENTOS + TAXA_LIQUIDAÇÃO
    
    # Aplica taxas de venda
    valor_liquido = valor_venda * (1 - taxas_venda)
    
    # Aplica impostos se houver lucro
    if lucro_operacao > 0:
        if eh_day_trade:
            # Day trade: 20% sobre o lucro
            imposto = lucro_operacao * IMPOSTO_DAY_TRADE
        else:
            # Swing trade: isento até R$20k/mês (vamos considerar isento para simplificar)
            imposto = 0
        
        valor_liquido -= imposto
    
    return max(valor_liquido, 0)  # Não pode ser negativo

def backtest_strategy_com_taxas(df, dates, y_pred, initial_capital=INITIAL_CAPITAL):
    capital = initial_capital
    shares = 0
    entry_price = 0
    entry_date = None
    trades = []
    equity_curve = [capital]
    trade_active = False
    in_position_days = 0
    max_position_days = 10
    
    total_taxas_pagas = 0
    total_impostos_pagos = 0

    for i, current_date in enumerate(dates):
        if current_date not in df.index:
            continue
            
        current_price = df.loc[current_date, 'close']
        predicted_price = y_pred[i]

        signal = "HOLD"
        
        if not trade_active:
            # AJUSTE: Condição de compra mais sensível
            if predicted_price > current_price * 1.01:  # Reduzido de 1.02 para 1.01
                signal = "BUY"
        else:
            in_position_days += 1
            stop_loss_condition = current_price <= entry_price * (1 - STOP_LOSS_PCT)
            take_profit_condition = current_price >= entry_price * (1 + TAKE_PROFIT_PCT)
            time_exit_condition = in_position_days >= max_position_days
            prediction_exit_condition = predicted_price < current_price * 0.99  # Ajustado
            
            if stop_loss_condition or take_profit_condition or time_exit_condition or prediction_exit_condition:
                signal = "SELL"

        if signal == "BUY" and capital > 0:
            # COMPRA COM TAXAS
            valor_compra = capital
            valor_pos_taxas = aplicar_taxas_compra(valor_compra)
            taxas_compra = valor_compra - valor_pos_taxas
            
            if valor_pos_taxas > 0:
                shares = valor_pos_taxas / current_price
                capital = 0
                entry_price = current_price
                entry_date = current_date
                trade_active = True
                in_position_days = 0
                
                trades.append({
                    "date": current_date, 
                    "action": "BUY", 
                    "price": current_price,
                    "shares": shares,
                    "taxas": taxas_compra,
                    "valor_total": valor_compra
                })
                total_taxas_pagas += taxas_compra
                
        elif signal == "SELL" and shares > 0:
            # VENDA COM TAXAS E IMPOSTOS
            valor_venda_bruto = shares * current_price
            lucro_operacao = valor_venda_bruto - (shares * entry_price)
            
            # Verifica se é day trade
            eh_day_trade = (entry_date == current_date)
            
            valor_liquido = aplicar_taxas_venda(
                valor_venda_bruto, 
                lucro_operacao, 
                eh_day_trade
            )
            
            taxas_venda = valor_venda_bruto - valor_liquido
            capital = valor_liquido
            
            trades.append({
                "date": current_date, 
                "action": "SELL", 
                "price": current_price,
                "shares": shares,
                "return_pct": (current_price / entry_price - 1) * 100,
                "entry_price": entry_price,
                "taxas": taxas_venda,
                "eh_day_trade": eh_day_trade,
                "lucro_bruto": lucro_operacao,
                "valor_liquido": valor_liquido
            })
            
            total_taxas_pagas += taxas_venda
            if lucro_operacao > 0 and eh_day_trade:
                total_impostos_pagos += lucro_operacao * IMPOSTO_DAY_TRADE
            
            shares = 0
            trade_active = False
            in_position_days = 0

        portfolio_value = capital + (shares * current_price if shares > 0 else 0)
        equity_curve.append(portfolio_value)

    # Fechar posição aberta no final com taxas
    if trade_active and len(dates) > 0:
        last_date = dates[-1]
        if last_date in df.index:
            last_price = df.loc[last_date, 'close']
            valor_venda_bruto = shares * last_price
            lucro_operacao = valor_venda_bruto - (shares * entry_price)
            eh_day_trade = (entry_date == last_date)
            
            valor_liquido = aplicar_taxas_venda(
                valor_venda_bruto, 
                lucro_operacao, 
                eh_day_trade
            )
            
            taxas_venda = valor_venda_bruto - valor_liquido
            capital = valor_liquido
            
            trades.append({
                "date": last_date, 
                "action": "SELL", 
                "price": last_price,
                "return_pct": (last_price / entry_price - 1) * 100,
                "entry_price": entry_price,
                "taxas": taxas_venda,
                "eh_day_trade": eh_day_trade,
                "lucro_bruto": lucro_operacao,
                "valor_liquido": valor_liquido
            })
            
            total_taxas_pagas += taxas_venda
            if lucro_operacao > 0 and eh_day_trade:
                total_impostos_pagos += lucro_operacao * IMPOSTO_DAY_TRADE

    return equity_curve, trades, capital, total_taxas_pagas, total_impostos_pagos

# ===================== ANÁLISE DE TRADES CORRIGIDA =====================
def analyze_trades(trades):
    if len(trades) < 2:
        return {
            "total_trades": 0, 
            "win_rate": 0, 
            "avg_profit_bruto": 0,
            "avg_profit_liquido": 0,
            "total_taxas_pagas": 0,
            "trades": []
        }
    
    trade_results = []
    winning_trades = 0
    total_taxas = 0
    
    for i in range(0, len(trades)-1, 2):
        if trades[i]['action'] == 'BUY' and trades[i+1]['action'] == 'SELL':
            buy_trade = trades[i]
            sell_trade = trades[i+1]
            
            ret_liquido = sell_trade['return_pct']
            taxas_operacao = buy_trade.get('taxas', 0) + sell_trade.get('taxas', 0)
            
            trade_results.append({
                'retorno_bruto': sell_trade['return_pct'],
                'retorno_liquido': ret_liquido,
                'taxas': taxas_operacao,
                'day_trade': sell_trade.get('eh_day_trade', False),
                'lucro_bruto': sell_trade.get('lucro_bruto', 0)
            })
            
            total_taxas += taxas_operacao
            if ret_liquido > 0:
                winning_trades += 1
    
    if trade_results:
        win_rate = (winning_trades / len(trade_results)) * 100
        avg_profit_bruto = np.mean([t['retorno_bruto'] for t in trade_results])
        avg_profit_liquido = np.mean([t['retorno_liquido'] for t in trade_results])
        
        return {
            "total_trades": len(trade_results),
            "win_rate": win_rate,
            "avg_profit_bruto": avg_profit_bruto,
            "avg_profit_liquido": avg_profit_liquido,
            "total_taxas_pagas": total_taxas,
            "trades": trade_results
        }
    
    return {
        "total_trades": 0, 
        "win_rate": 0, 
        "avg_profit_bruto": 0,
        "avg_profit_liquido": 0,
        "total_taxas_pagas": 0,
        "trades": []
    }

# ===================== MAIN =====================
def main():
    data = download_data(TICKER, "2019-01-01", TEST_END)
    if data is None:
        print("Erro ao baixar dados")
        return
    
    df = data[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = calculate_indicators(df)

    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
    test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Dados de treino ou teste vazios")
        return

    features = select_features(train_df)
    X_train, X_test, y_train, y_test, scaler_y = prepare_data(train_df, test_df, features)

    print(f"Shape dos dados de treino: {X_train.shape}")
    print(f"Shape dos dados de teste: {X_test.shape}")

    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    print("Treinando modelo LSTM...")
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test), 
        verbose=1,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
    )

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler_y.inverse_transform(y_test).flatten()
    test_dates = test_df.index[LOOKBACK:LOOKBACK + len(y_pred)]

    # Métricas de Regressão
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    r2 = r2_score(y_actual, y_pred)

    print("\n========== MÉTRICAS DE REGRESSÃO (LSTM) ==========")
    print(f"MAE : R$ {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: R$ {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

    # Backtest COM TAXAS
    equity_curve, trades, final_capital, total_taxas, total_impostos = backtest_strategy_com_taxas(
        test_df, test_dates, y_pred, INITIAL_CAPITAL
    )
    
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    buy_hold_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1) * 100

    trade_analysis = analyze_trades(trades)

    print("\n========== RESULTADOS DO BACKTEST COM TAXAS ==========")
    print(f"Capital inicial: R$ {INITIAL_CAPITAL:.2f}")
    print(f"Capital final:   R$ {final_capital:.2f}")
    print(f"Retorno líquido: {total_return:+.2f}%")
    print(f"Buy & Hold:      {buy_hold_return:+.2f}%")
    print(f"\n--- ESTATÍSTICAS DE TRADING ---")
    print(f"Nº Trades:           {trade_analysis['total_trades']}")
    
    # VERIFICAÇÃO SEGURA DAS CHAVES
    if trade_analysis['total_trades'] > 0:
        print(f"Win Rate:            {trade_analysis['win_rate']:.1f}%")
        print(f"Lucro médio bruto:   {trade_analysis['avg_profit_bruto']:.2f}%")
        print(f"Lucro médio líquido: {trade_analysis['avg_profit_liquido']:.2f}%")
    else:
        print(f"Win Rate:            {trade_analysis['win_rate']:.1f}%")
        print(f"Lucro médio bruto:   {trade_analysis['avg_profit_bruto']:.2f}%")
        print(f"Lucro médio líquido: {trade_analysis['avg_profit_liquido']:.2f}%")
    
    print(f"Total em taxas:      R$ {total_taxas:.2f}")
    print(f"Total em impostos:   R$ {total_impostos:.2f}")
    print(f"Taxa total/% capital: {(total_taxas/INITIAL_CAPITAL)*100:.2f}%")

    # DEBUG: Mostrar primeiras previsões vs reais
    print("\n=== DEBUG: Primeiras previsões ===")
    for i in range(min(5, len(y_pred))):
        print(f"Data: {test_dates[i]}, Real: R$ {y_actual[i]:.2f}, Previsto: R$ {y_pred[i]:.2f}, Dif: {((y_pred[i]/y_actual[i])-1)*100:.2f}%")

    # Gráficos
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.plot(test_dates, y_actual, label="Preço Real", color='blue', linewidth=1)
    plt.plot(test_dates, y_pred, label="Preço Previsto (LSTM)", color='red', linestyle='--', linewidth=1)
    
    if trades:
        buy_dates = [t['date'] for t in trades if t['action'] == "BUY"]
        sell_dates = [t['date'] for t in trades if t['action'] == "SELL"]
        
        if buy_dates:
            buy_prices = [test_df.loc[d, 'close'] for d in buy_dates if d in test_df.index]
            plt.scatter(buy_dates[:len(buy_prices)], buy_prices, color='green', marker='^', s=80, label="BUY", zorder=5)
        
        if sell_dates:
            sell_prices = [test_df.loc[d, 'close'] for d in sell_dates if d in test_df.index]
            plt.scatter(sell_dates[:len(sell_prices)], sell_prices, color='orange', marker='v', s=80, label="SELL", zorder=5)
    
    plt.title(f"{TICKER} - Preço Real vs Previsto LSTM + Sinais (COM TAXAS)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(equity_curve, label="Estratégia LSTM (com taxas)", color='green', linewidth=2)
    plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', label="Capital Inicial")
    
    bh_curve = [INITIAL_CAPITAL * (test_df['close'].iloc[i] / test_df['close'].iloc[0]) 
                for i in range(len(equity_curve)-1)]
    plt.plot(bh_curve, label="Buy & Hold", color='blue', linewidth=2)
    
    plt.title("Evolução do Capital - LSTM (COM TAXAS)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    plt.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3)
    plt.plot(drawdown, color='red', linewidth=1)
    plt.title("Drawdown (%) - LSTM (COM TAXAS)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'resultado_lstm_com_taxas_{TICKER}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Gráfico de perda durante o treinamento
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda Treino')
    plt.plot(history.history['val_loss'], label='Perda Validação')
    plt.title('Evolução da Perda durante o Treinamento LSTM')
    plt.xlabel('Época')
    plt.ylabel('Perda (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    main()