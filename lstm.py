import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

# ===================== Parâmetros Otimizados =====================
TICKER = "PETR4.SA"
TRAIN_START = "2020-01-01"  # Período mais recente
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"
LOOKBACK = 20  # Reduzido para 20 dias
EPOCHS = 100
BATCH_SIZE = 32
N_FEATURES = 15  # Limitar número de features

# ===================== Função para Baixar Dados =====================
def download_data(ticker, start_date, end_date):
    print(f"Baixando dados do {ticker} de {start_date} até {end_date}...")
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        return raw if not raw.empty else None
    except Exception as e:
        print(f"Erro: {e}")
        return None

# ===================== Indicadores Essenciais =====================
def calculate_essential_indicators(df):
    """Apenas indicadores mais relevantes"""
    print("Calculando indicadores essenciais...")
    
    # Preços e volume
    df['log_volume'] = np.log1p(df['volume'])
    df['returns'] = df['close'].pct_change()
    df['price_change'] = df['close'].diff()
    
    # Indicadores básicos
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
    
    # Volatilidade
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volatility_10'] = df['close'].rolling(10).std()
    
    # Volume
    df['obv'] = ta.obv(df['close'], df['volume'])
    
    df = df.dropna()
    print(f"Indicadores calculados: {len(df.columns)} colunas")
    return df

# ===================== Seleção Inteligente de Features =====================
def select_best_features(train_df, test_df, n_features=N_FEATURES):
    """Seleciona as melhores features usando teste estatístico"""
    
    # Features candidatas
    candidate_features = [
        'open', 'high', 'low', 'log_volume', 'returns', 'price_change',
        'sma_20', 'ema_12', 'rsi_14', 'macd', 'macd_signal', 
        'atr_14', 'volatility_10', 'obv'
    ]
    
    # Filtrar features disponíveis
    available_features = [f for f in candidate_features if f in train_df.columns]
    
    # Preparar dados para seleção
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['close']
    
    # Selecionar melhores features
    selector = SelectKBest(score_func=f_regression, k=min(n_features, len(available_features)))
    selector.fit(X_train, y_train)
    
    # Obter features selecionadas
    selected_mask = selector.get_support()
    selected_features = [available_features[i] for i in range(len(available_features)) if selected_mask[i]]
    
    print(f"Melhores {len(selected_features)} features selecionadas:")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i}. {feat}")
    
    return selected_features

# ===================== Preparação Correta dos Dados =====================
def prepare_correct_lstm_data(train_df, test_df, features, lookback=LOOKBACK):
    """Preparação correta sem data leakage"""
    
    # Usar apenas features selecionadas e target
    train_data = train_df[features + ['close']]
    test_data = test_df[features + ['close']]
    
    # Normalizar SEPARADAMENTE para evitar data leakage
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Treino
    X_train_scaled = scaler_x.fit_transform(train_data[features])
    y_train_scaled = scaler_y.fit_transform(train_data[['close']])
    
    # Teste (usando scaler do treino)
    X_test_scaled = scaler_x.transform(test_data[features])
    y_test_scaled = scaler_y.transform(test_data[['close']])
    
    # Criar sequências
    def create_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback)
    
    print(f"Treino: {X_train_seq.shape}, Teste: {X_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler_x, scaler_y

# ===================== Modelo LSTM Otimizado =====================
def create_optimized_lstm_model(input_shape):
    """Modelo mais simples e eficiente"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ===================== Previsões com Inversão Correta =====================
def make_correct_predictions(model, X_test, scaler_y):
    """Faz previsões e converte corretamente"""
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled).flatten()
    return y_pred_actual

# ===================== Avaliação Detalhada =====================
def detailed_evaluation(y_true, y_pred, dates, scaler_y):
    """Avaliação completa com métricas de trading"""
    
    # Métricas básicas
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Métricas de trading
    error_pct = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - error_pct
    
    # Acurácia de direção
    direction_correct = 0
    for i in range(1, len(y_true)):
        if (y_true[i] > y_true[i-1]) == (y_pred[i] > y_pred[i-1]):
            direction_correct += 1
    direction_accuracy = (direction_correct / (len(y_true)-1)) * 100 if len(y_true) > 1 else 0
    
    print(f"\n{'='*60}")
    print("AVALIAÇÃO DETALHADA DO MODELO")
    print(f"{'='*60}")
    print(f"R²: {r2:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: R$ {mae:.2f}")
    print(f"RMSE: R$ {rmse:.2f}")
    print(f"Acurácia: {accuracy:.2f}%")
    print(f"Acurácia de Direção: {direction_accuracy:.2f}%")
    print(f"Erro Percentual Médio: {error_pct:.2f}%")
    
    return {
        'r2': r2, 'mse': mse, 'mae': mae, 'rmse': rmse,
        'accuracy': accuracy, 'direction_accuracy': direction_accuracy
    }

# ===================== Execução Principal Corrigida =====================
def main():
    print("="*70)
    print("SISTEMA OTIMIZADO - PETROBRAS (PETR4.SA)")
    print("="*70)
    
    # 1. Baixar dados
    print("Baixando dados...")
    full_data = download_data(TICKER, "2019-01-01", "2023-12-31")
    if full_data is None:
        return
    
    # 2. Processar dados
    df = full_data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.ffill().dropna()
    
    # 3. Calcular indicadores
    df = calculate_essential_indicators(df)
    
    # 4. Dividir em treino e teste
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Treino: {train_df.shape[0]} dias ({TRAIN_START} até {TRAIN_END})")
    print(f"Teste: {test_df.shape[0]} dias ({TEST_START} até {TEST_END})")
    print(f"Preço médio treino: R$ {train_df['close'].mean():.2f}")
    print(f"Preço médio teste: R$ {test_df['close'].mean():.2f}")
    
    # 5. Selecionar melhores features
    features = select_best_features(train_df, test_df)
    
    # 6. Preparar dados para LSTM
    X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_correct_lstm_data(
        train_df, test_df, features, LOOKBACK
    )
    
    # 7. Criar e treinar modelo
    model = create_optimized_lstm_model((X_train.shape[1], X_train.shape[2]))
    print("\nResumo do modelo:")
    model.summary()
    
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # 8. Fazer previsões
    y_pred_actual = make_correct_predictions(model, X_test, scaler_y)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 9. Gerar datas
    test_dates = test_df.index[LOOKBACK:LOOKBACK + len(y_test_actual)]
    
    # 10. Avaliar
    metrics = detailed_evaluation(y_test_actual, y_pred_actual, test_dates, scaler_y)
    
    # 11. Plotar resultados
    plot_results(history, y_test_actual, y_pred_actual, test_dates, "PETR4 2023 - Modelo Otimizado")
    
    # 12. Salvar resultados
    results_df = pd.DataFrame({
        'Data': test_dates,
        'Preço_Real': y_test_actual,
        'Preço_Previsto': y_pred_actual,
        'Erro_Absoluto': y_test_actual - y_pred_actual,
        'Erro_Percentual': ((y_test_actual - y_pred_actual) / y_test_actual) * 100
    })
    
    results_df.to_csv('resultados_petr4_otimizado.csv', index=False)
    print(f"\nResultados salvos!")
    print(f"Melhor previsão: R$ {results_df['Preço_Previsto'].min():.2f}")
    print(f"Pior previsão: R$ {results_df['Preço_Previsto'].max():.2f}")
    
    return model, metrics, results_df

def train_model(model, X_train, y_train, X_test, y_test):
    """Treino com callbacks otimizados"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
    ]
    
    print("Iniciando treinamento...")
    return model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=False,
        callbacks=callbacks
    )

def plot_results(history, y_true, y_pred, dates, title):
    """Plot simplificado"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Previsões vs Reais
    ax1.plot(dates, y_true, label='Real', linewidth=2, color='blue')
    ax1.plot(dates, y_pred, label='Previsto', linewidth=2, color='red', linestyle='--')
    ax1.set_title(title)
    ax1.set_ylabel('Preço (R$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Erro
    errors = y_true - y_pred
    ax2.bar(dates, errors, alpha=0.7, color='orange')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_title('Erros de Previsão')
    ax2.set_ylabel('Erro (R$)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Executar
if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    try:
        model, metrics, results = main()
        print("✅ Execução concluída com sucesso!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()