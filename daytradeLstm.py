import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURAÇÃO DA ESTRATÉGIA =====================
TICKER = "PETR4.SA"
START = "2018-01-01"
END = "2023-12-31"

# ESCOLHA SUA ESTRATÉGIA:
STRATEGY = "DAY_TRADING"  # "DAY_TRADING", "SWING_TRADING", "POSITION_TRADING"

if STRATEGY == "DAY_TRADING":
    INITIAL_TRAIN_DAYS = 504    # 2 anos
    STEPS_AHEAD = 1             # Prever 1 dia
    LOOKBACK = 60               # 3 meses
    RETRAIN_EVERY = 5           # Retreinar a cada 5 dias
elif STRATEGY == "SWING_TRADING":
    INITIAL_TRAIN_DAYS = 504    # 2 anos  
    STEPS_AHEAD = 5             # Prever 5 dias
    LOOKBACK = 90               # 4-5 meses
    RETRAIN_EVERY = 10          # Retreinar a cada 10 dias
else:  # POSITION_TRADING
    INITIAL_TRAIN_DAYS = 756    # 3 anos
    STEPS_AHEAD = 20            # Prever 20 dias
    LOOKBACK = 180              # 8-9 meses
    RETRAIN_EVERY = 20          # Retreinar a cada 20 dias

# ===================== FUNÇÕES PRINCIPAIS =====================
def prepare_sequences(scaled_data, lookback, steps_ahead, feature_index):
    """Prepara sequências para previsão multi-step"""
    X, y = [], []
    for i in range(lookback, len(scaled_data) - steps_ahead):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i+steps_ahead-1, feature_index])  # Prever steps_ahead à frente
    return np.array(X), np.array(y)

def create_trading_model(input_shape):
    """Cria modelo LSTM para trading"""
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def walk_forward_validation(data, features, initial_train_days, retrain_every, steps_ahead, lookback):
    """Validação em janela móvel para trading realista"""
    
    all_predictions = []
    all_actuals = []
    all_dates = []
    
    scaler = MinMaxScaler()
    close_index = features.index('close')
    
    # Primeira janela de treino
    train_data = data.iloc[:initial_train_days][features]
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    
    # Preparar sequências iniciais
    X_train, y_train = prepare_sequences(scaled_train, lookback, steps_ahead, close_index)
    
    # Criar e treinar modelo inicial
    model = create_trading_model((lookback, len(features)))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    # Walk-forward validation
    for i in range(initial_train_days, len(data) - steps_ahead, retrain_every):
        
        # Dados atuais para treino
        current_train_data = data.iloc[i-initial_train_days:i][features]
        scaled_current = scaler.transform(current_train_data)
        
        # Preparar sequências
        X_current, y_current = prepare_sequences(scaled_current, lookback, steps_ahead, close_index)
        
        # Retreinar modelo periodicamente
        if i > initial_train_days:
            model = clone_model(model)
            model.compile(optimizer=Adam(0.001), loss='mse')
            model.fit(X_current, y_current, epochs=30, batch_size=32, verbose=0)
        
        # Dados para previsão
        prediction_data = data.iloc[i-lookback:i+steps_ahead][features]
        scaled_pred = scaler.transform(prediction_data)
        
        # Fazer previsão
        X_pred = scaled_pred[-lookback-steps_ahead:-steps_ahead].reshape(1, lookback, len(features))
        prediction_scaled = model.predict(X_pred, verbose=0)[0, 0]
        
        # Converter para valor real
        dummy = np.zeros((1, len(features)))
        dummy[:, close_index] = prediction_scaled
        prediction = scaler.inverse_transform(dummy)[0, close_index]
        
        # Armazenar resultados
        actual = data['close'].iloc[i + steps_ahead - 1]
        all_predictions.append(prediction)
        all_actuals.append(actual)
        all_dates.append(data.index[i + steps_ahead - 1])
        
        print(f"{all_dates[-1].strftime('%Y-%m-%d')} | Prev: R$ {prediction:.2f} | Real: R$ {actual:.2f}")
    
    return all_predictions, all_actuals, all_dates

def analyze_trading_performance(predictions, actuals, dates):
    """Analisa performance de trading"""
    
    # Calcular direções
    actual_directions = [1 if actuals[i] > actuals[i-1] else 0 for i in range(1, len(actuals))]
    pred_directions = [1 if predictions[i] > actuals[i-1] else 0 for i in range(1, len(predictions))]
    
    # Métricas
    direction_accuracy = accuracy_score(actual_directions, pred_directions) * 100
    mae = np.mean(np.abs(np.array(predictions[1:]) - np.array(actuals[1:])))
    
    # Simular trades
    returns = []
    for i in range(1, len(predictions)):
        if pred_directions[i-1] == 1:  # Previu alta → Compra
            trade_return = (actuals[i] - actuals[i-1]) / actuals[i-1]
            returns.append(trade_return)
        else:  # Previu baixa → Venda a descoberto ou fica fora
            trade_return = (actuals[i-1] - actuals[i]) / actuals[i-1]  # Short selling
            returns.append(trade_return)
    
    total_return = np.prod([1 + r for r in returns]) - 1 if returns else 0
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if returns else 0
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
    
    print(f"\n🎯 PERFORMANCE - {STRATEGY}")
    print("=" * 50)
    print(f"Acurácia de Direção: {direction_accuracy:.1f}%")
    print(f"Retorno Total: {total_return:.1%}")
    print(f"Retorno Anualizado: {annual_return:.1%}") 
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"MAE: R$ {mae:.2f}")
    print(f"Total de Trades: {len(returns)}")
    
    return direction_accuracy, annual_return, sharpe_ratio

# ===================== EXECUÇÃO PRINCIPAL =====================
def main():
    print(f"🚀 INICIANDO BACKTEST - {STRATEGY}")
    print(f"Ticket: {TICKER} | Período: {START} até {END}")
    print(f"Lookback: {LOOKBACK} dias | Previsão: {STEPS_AHEAD} dias à frente")
    print(f"Retreino a cada: {RETRAIN_EVERY} dias")
    
    # Baixar dados
    data = yf.download(TICKER, START, END, auto_adjust=True, progress=False)
    data = data[['Close', 'Volume']]
    data.columns = ['close', 'volume']
    
    # Calcular features técnicas
    data['returns'] = data['close'].pct_change()
    data['momentum_5'] = data['close'].pct_change(5)
    data['rsi'] = ta.rsi(data['close'])
    data = data.dropna()
    
    features = ['close', 'volume', 'returns', 'momentum_5', 'rsi']
    
    # Executar walk-forward validation
    predictions, actuals, dates = walk_forward_validation(
        data, features, INITIAL_TRAIN_DAYS, RETRAIN_EVERY, STEPS_AHEAD, LOOKBACK
    )
    
    # Analisar performance
    direction_acc, annual_ret, sharpe = analyze_trading_performance(predictions, actuals, dates)
    
    return predictions, actuals, dates, direction_acc, annual_ret, sharpe

if __name__ == "__main__":
    predictions, actuals, dates, acc, ret, sharpe = main()