# Instalar bibliotecas necessárias

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
np.random.seed(42)

# 1. OBTER DADOS DO YAHOO FINANCE
def get_stock_data(ticker, start_date, end_date):
    """
    Obtém dados históricos do Yahoo Finance
    """
    print(f"Baixando dados para {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

# 2. CALCULAR INDICADORES TÉCNICOS
def calculate_technical_indicators(df):
    """
    Adiciona indicadores técnicos ao DataFrame
    """
    # Converter para Series para a biblioteca ta
    close_series = df['Close']
    high_series = df['High']
    low_series = df['Low']
    volume_series = df['Volume']
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=close_series)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Médias Móveis
    df['sma_20'] = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(close=close_series, window=50).sma_indicator()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=close_series)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    
    # Estocástico
    stoch = ta.momentum.StochasticOscillator(high=high_series, low=low_series, close=close_series)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Volume
    df['volume_sma'] = ta.volume.VolumeWeightedAveragePrice(
        high=high_series, low=low_series, close=close_series, volume=volume_series
    ).volume_weighted_average_price()
    
    # Remove valores NaN
    df = df.dropna()
    
    return df

# 3. PREPARAR DADOS PARA LSTM
def prepare_data(df, lookback=60, test_size=0.2):
    """
    Prepara dados para treinamento da LSTM
    """
    # Selecionar features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 
                'macd_signal', 'sma_20', 'sma_50', 'bb_high', 'bb_low', 
                'bb_middle', 'stoch_k', 'stoch_d', 'volume_sma']
    
    target = 'Close'
    
    # Verificar se todas as features existem no DataFrame
    available_features = [f for f in features if f in df.columns]
    print(f"Features disponíveis: {available_features}")
    
    # Normalizar dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[available_features])
    
    # Criar sequências para LSTM
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, available_features.index(target)])
    
    X, y = np.array(X), np.array(y)
    
    # Dividir em treino e teste
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, available_features

# 4. CRIAR MODELO LSTM
def create_lstm_model(input_shape):
    """
    Cria e compila o modelo LSTM
    """
    model = Sequential()
    
    # Primeira camada LSTM
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Segunda camada LSTM
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Terceira camada LSTM
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Camada de saída
    model.add(Dense(units=1))
    
    # Compilar modelo
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='mean_squared_error',
                 metrics=['mae'])
    
    return model

# 5. FUNÇÃO PRINCIPAL
def main():
    # Configurações
    TICKER = 'PETR4.SA'  # Exemplo: Petrobras
    START_DATE = '2018-01-01'
    END_DATE = '2021-12-31'
    LOOKBACK = 60  # Janela temporal
    TEST_SIZE = 0.2
    EPOCHS = 50  # Reduzido para teste
    BATCH_SIZE = 32
    
    # Obter dados
    df = get_stock_data(TICKER, START_DATE, END_DATE)
    print(f"Dados baixados: {df.shape}")
    
    # Calcular indicadores técnicos
    df = calculate_technical_indicators(df)
    print(f"Dados com indicadores: {df.shape}")
    
    # Preparar dados
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(
        df, LOOKBACK, TEST_SIZE
    )
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Features usadas: {features}")
    
    # Criar modelo
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Resumo do modelo
    model.summary()
    
    # Treinar modelo
    print("Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=False
    )
    
    # 6. FAZER PREVISÕES
    # Prever no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Reverter a normalização para obter preços reais
    # Criar array dummy para inversão
    dummy_array = np.zeros((len(y_pred), len(features)))
    dummy_array[:, features.index('Close')] = y_pred.flatten()
    y_pred_actual = scaler.inverse_transform(dummy_array)[:, features.index('Close')]
    
    dummy_array[:, features.index('Close')] = y_test
    y_test_actual = scaler.inverse_transform(dummy_array)[:, features.index('Close')]
    
    # 7. AVALIAR MODELO
    # Calcular métricas
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    
    print(f"\nMétricas de avaliação:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Calcular acurácia percentual
    accuracy = 100 * (1 - np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)))
    print(f"Acurácia: {accuracy:.2f}%")
    
    # 8. VISUALIZAR RESULTADOS
    plt.figure(figsize=(15, 10))
    
    # Plotar histórico de treinamento
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE do Modelo')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plotar previsões vs valores reais
    plt.subplot(2, 1, 2)
    plt.plot(y_test_actual, label='Preço Real', alpha=0.7, linewidth=2)
    plt.plot(y_pred_actual, label='Preço Previsto', alpha=0.7, linewidth=2)
    plt.title(f'Previsões LSTM para {TICKER} (Período de Teste)')
    plt.xlabel('Dias')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 9. ANALISAR PERÍODO DA PANDEMIA (2020)
    pandemic_start = '2020-01-01'
    pandemic_end = '2020-12-31'
    
    pandemic_data = df.loc[pandemic_start:pandemic_end]
    print(f"\nDados da pandemia (2020): {pandemic_data.shape}")
    print(f"Variação de preço durante a pandemia:")
    print(f"Início: R$ {pandemic_data['Close'].iloc[0]:.2f}")
    print(f"Fim: R$ {pandemic_data['Close'].iloc[-1]:.2f}")
    print(f"Variação: {((pandemic_data['Close'].iloc[-1] / pandemic_data['Close'].iloc[0] - 1) * 100):.2f}%")
    
    return model, df, y_test_actual, y_pred_actual, features

# Executar o script
if __name__ == "__main__":
    try:
        model, df, y_test, y_pred, features = main()
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        print("Verificando se há dados suficientes...")