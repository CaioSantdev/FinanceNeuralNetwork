import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para exibir no VSCode
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

# ===================== Parâmetros =====================
TICKER = "PETR4.SA"
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"
LOOKBACK = 5  # Reduzido para 5 dias
EPOCHS = 100
BATCH_SIZE = 16

# ===================== Função para Baixar Dados =====================
def download_data(ticker, start_date, end_date):
    """Baixa dados do Yahoo Finance"""
    print(f"Baixando dados do {ticker} de {start_date} até {end_date}...")
    
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"Nenhum dado retornado para {ticker}.")
        return raw
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        return None

# ===================== Preparar e Processar Dados =====================
def prepare_and_process_data(raw_data):
    """Prepara e processa os dados"""
    df = raw_data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.ffill().dropna().copy()
    
    # Transformações
    df["volume"] = np.log1p(df["volume"])
    df['returns'] = df['close'].pct_change()
    df['price_change'] = df['close'].diff()
    
    # Remover primeira linha com NaN
    df = df.dropna()
    
    print(f"Período: {df.index.min()} até {df.index.max()}")
    print(f"Preço - Mín: R$ {df['close'].min():.2f}, Máx: R$ {df['close'].max():.2f}")
    
    return df

# ===================== Preparação para LSTM =====================
def prepare_lstm_data(train_df, test_df, lookback=LOOKBACK):
    """Prepara os dados para LSTM com treino e teste separados"""
    # Features simplificadas
    features = ['close', 'volume', 'returns', 'price_change']
    available_features = [f for f in features if f in train_df.columns]
    
    print(f"Features utilizadas: {available_features}")

    # Separar dados
    train_data = train_df[available_features]
    test_data = test_df[available_features]
    
    # Normalizar separadamente
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Ajustar scaler apenas com dados de treino
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    
    # Criar sequências para treino
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_train)):
        X_train.append(scaled_train[i-lookback:i])
        y_train.append(scaled_train[i, available_features.index('close')])
    
    # Criar sequências para teste
    X_test, y_test = [], []
    for i in range(lookback, len(scaled_test)):
        X_test.append(scaled_test[i-lookback:i])
        y_test.append(scaled_test[i, available_features.index('close')])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"Treino: {X_train.shape}")
    print(f"Teste: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, available_features

# ===================== Modelo LSTM Simplificado =====================
def create_lstm_model(input_shape):
    """Cria modelo LSTM simplificado"""
    model = Sequential([
        LSTM(32, input_shape=input_shape),
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

# ===================== Treinamento =====================
def train_model(model, X_train, y_train, X_test, y_test):
    """Treina o modelo LSTM"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.0001)
    ]
    
    print("Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=False,
        callbacks=callbacks
    )
    
    return history

# ===================== Previsões =====================
def make_predictions(model, X_test, y_test, scaler, features):
    """Faz previsões e converte para valores reais"""
    y_pred = model.predict(X_test, verbose=0)
    
    # Criar arrays dummy para inversão correta
    dummy_test = np.zeros((len(y_test), len(features)))
    dummy_pred = np.zeros((len(y_pred), len(features)))
    
    dummy_test[:, features.index('close')] = y_test
    dummy_pred[:, features.index('close')] = y_pred.flatten()
    
    # Inverter a normalização
    y_test_actual = scaler.inverse_transform(dummy_test)[:, features.index('close')]
    y_pred_actual = scaler.inverse_transform(dummy_pred)[:, features.index('close')]
    
    return y_test_actual, y_pred_actual

# ===================== Avaliação =====================
def evaluate_model(y_true, y_pred, period_name="TESTE"):
    """Avalia o desempenho do modelo"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }
    
    accuracy = 100 * (1 - np.mean(np.abs((y_true - y_pred) / y_true)))
    metrics['Accuracy'] = accuracy
    
    # Calcular acurácia de direção
    correct_direction = 0
    for i in range(1, len(y_true)):
        actual_dir = 1 if y_true[i] > y_true[i-1] else 0
        pred_dir = 1 if y_pred[i] > y_pred[i-1] else 0
        if actual_dir == pred_dir:
            correct_direction += 1
    
    direction_accuracy = (correct_direction / (len(y_true) - 1)) * 100 if len(y_true) > 1 else 0
    metrics['Direction_Accuracy'] = direction_accuracy
    
    print(f"\n" + "="*60)
    print(f"METRICAS DE AVALIAÇÃO - {period_name}")
    print("="*60)
    for metric, value in metrics.items():
        if 'Accuracy' in metric:
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.6f}")
    
    return metrics

# ===================== Visualizações =====================
def plot_results(history, y_true, y_pred, dates, title):
    """Plota os resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Evolução da Loss durante Treinamento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[0, 1].set_title('Evolução do MAE durante Treinamento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Previsões vs Reais
    axes[1, 0].plot(dates, y_true, label='Preço Real', linewidth=2, color='blue', marker='o', markersize=3)
    axes[1, 0].plot(dates, y_pred, label='Preço Previsto', linewidth=2, color='red', linestyle='--', marker='x', markersize=3)
    axes[1, 0].set_title(f'{title} - Previsões vs Valores Reais')
    axes[1, 0].set_xlabel('Data')
    axes[1, 0].set_ylabel('Preço (R$)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Scatter plot
    axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_xlabel('Valores Reais (R$)')
    axes[1, 1].set_ylabel('Valores Previstos (R$)')
    axes[1, 1].set_title('Dispersão: Reais vs Previstos')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ===================== Diagnóstico =====================
def diagnostic_analysis(y_true, y_pred, dates, period_name):
    """Análise diagnóstica dos resultados"""
    print(f"\n" + "="*60)
    print(f"DIAGNÓSTICO - {period_name}")
    print("="*60)
    
    print(f"Período: {dates[0].strftime('%d/%m/%Y')} até {dates[-1].strftime('%d/%m/%Y')}")
    print(f"Dias de trading: {len(y_true)}")
    print(f"Preço real médio: R$ {y_true.mean():.2f}")
    print(f"Preço previsto médio: R$ {y_pred.mean():.2f}")
    print(f"Variação real: R$ {y_true.min():.2f} - {y_true.max():.2f}")
    print(f"Variação prevista: R$ {y_pred.min():.2f} - {y_pred.max():.2f}")
    print(f"Erro médio absoluto: R$ {np.mean(np.abs(y_true - y_pred)):.2f}")
    print(f"Erro percentual médio: {np.mean(np.abs((y_true - y_pred) / y_true)) * 100:.2f}%")

# ===================== Execução Principal =====================
def main():
    print("="*70)
    print("SISTEMA DE PREVISÃO - TREINO: 2020-2022 | TESTE: 2023")
    print("="*70)
    
    # 1. Baixar dados de treino (2020-2022)
    train_raw = download_data(TICKER, TRAIN_START, TRAIN_END)
    if train_raw is None:
        return
    
    # 2. Baixar dados de teste (2023)
    test_raw = download_data(TICKER, TEST_START, TEST_END)
    if test_raw is None:
        return
    
    # 3. Processar dados
    print("\nProcessando dados de treino (2020-2022)...")
    train_df = prepare_and_process_data(train_raw)
    
    print("\nProcessando dados de teste (2023)...")
    test_df = prepare_and_process_data(test_raw)
    
    # 4. Preparar dados para LSTM
    X_train, X_test, y_train, y_test, scaler, features = prepare_lstm_data(train_df, test_df)
    
    # 5. Criar e treinar modelo
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    print("\nResumo do modelo:")
    model.summary()
    
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # 6. Fazer previsões
    y_test_actual, y_pred_actual = make_predictions(model, X_test, y_test, scaler, features)
    
    # 7. Gerar datas para o período de teste
    test_dates = test_df.index[LOOKBACK:LOOKBACK + len(y_test_actual)]
    
    # 8. Avaliar modelo
    metrics = evaluate_model(y_test_actual, y_pred_actual, "TESTE 2023")
    
    # 9. Diagnóstico
    diagnostic_analysis(y_test_actual, y_pred_actual, test_dates, "TESTE 2023")
    
    # 10. Plotar resultados
    plot_results(history, y_test_actual, y_pred_actual, test_dates, "PETR4 2023")
    
    # 11. Salvar resultados
    results_df = pd.DataFrame({
        'Data': test_dates,
        'Preço_Real': y_test_actual,
        'Preço_Previsto': y_pred_actual,
        'Erro_Absoluto': y_test_actual - y_pred_actual,
        'Erro_Percentual': ((y_test_actual - y_pred_actual) / y_test_actual) * 100
    })
    
    results_df.to_csv('resultados_previsao_2023.csv', index=False)
    print(f"\nResultados salvos em 'resultados_previsao_2023.csv'")
    print(f"Primeiras previsões:\n{results_df.head(10)}")
    
    return model, train_df, test_df, metrics, results_df

# ===================== Executar =====================
if __name__ == "__main__":
    # Configurar para usar CPU
    tf.config.set_visible_devices([], 'GPU')
    
    try:
        model, train_df, test_df, metrics, results = main()
        print("\n✅ Execução concluída com sucesso!")
        
        # Informações adicionais
        print(f"\n📊 RESUMO FINAL:")
        print(f"Período de treino: {train_df.index.min().strftime('%d/%m/%Y')} - {train_df.index.max().strftime('%d/%m/%Y')}")
        print(f"Período de teste: {test_df.index.min().strftime('%d/%m/%Y')} - {test_df.index.max().strftime('%d/%m/%Y')}")
        print(f"Preço médio treino: R$ {train_df['close'].mean():.2f}")
        print(f"Preço médio teste: R$ {test_df['close'].mean():.2f}")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()