import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class LSTMModel:
    def __init__(self, ticker, start_date, end_date, window_size=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = None
        
    def download_and_adjust_data(self):
        """Baixa e ajusta os dados históricos conforme metodologia descrita"""
        print(f"📥 Baixando dados de {self.ticker} de {self.start_date} até {self.end_date}...")

        # Download dos dados
        stock = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        # ✅ Corrige colunas em caso de MultiIndex (ex: ('Open','PETR4.SA'))
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = [col[0] for col in stock.columns]

        # ✅ Corrige o nome da coluna, se necessário
        if 'Adj Close' not in stock.columns:
            if 'adjclose' in stock.columns:
                stock.rename(columns={'adjclose': 'Adj Close'}, inplace=True)
            else:
                print("⚠️  Coluna 'Adj Close' não encontrada — usando 'Close' como ajustado.")
                stock['Adj Close'] = stock['Close']

        # Cálculo do fator de ajuste
        adjustment_factor = stock['Adj Close'] / stock['Close']

        # Aplicação do fator de ajuste nos outros preços
        stock['Open_adj'] = stock['Open'] * adjustment_factor
        stock['High_adj'] = stock['High'] * adjustment_factor
        stock['Low_adj'] = stock['Low'] * adjustment_factor
        stock['Close_adj'] = stock['Adj Close']

        # Tratamento de dados faltantes
        price_columns = ['Open_adj', 'High_adj', 'Low_adj', 'Close_adj']
        stock[price_columns] = stock[price_columns].interpolate(method='linear')

        # Forward fill para volume
        stock['Volume'] = stock['Volume'].replace(0, np.nan)
        stock['Volume'] = stock['Volume'].fillna(method='ffill')

        # Cálculo da EMA
        stock['EMA_60'] = stock['Close_adj'].ewm(span=60, adjust=False).mean()

        # Seleção das features finais
        features = ['Open_adj', 'High_adj', 'Low_adj', 'Close_adj', 'Volume', 'EMA_60']
        self.data = stock[features].dropna()

        print(f"✅ Dados ajustados com sucesso — {len(self.data)} registros válidos.")
        return self.data

    def create_sequences(self, data):
        """Cria sequências para o modelo LSTM"""
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data)
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i, 3])  # Close_adj é o target
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Constrói a arquitetura LSTM"""
        self.model = Sequential([
            LSTM(500, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(500, return_sequences=False),
            Dropout(0.3),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model
    
    def train_test_split(self, test_size=0.2):
        """Divide os dados em treino e teste"""
        sequences, targets = self.create_sequences(self.data)
        split_idx = int(len(sequences) * (1 - test_size))
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        y_train = targets[:split_idx]
        y_test = targets[split_idx:]
        return X_train, X_test, y_train, y_test
    
    def train(self, epochs=100, batch_size=32):
        """Treina o modelo"""
        X_train, X_test, y_train, y_test = self.train_test_split()
        self.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            shuffle=False
        )
        # Armazena histórico para gráficos
        self.history = history
        return history
    
    def predict(self):
        """Faz previsões e retorna resultados invertidos"""
        X_train, X_test, y_train, y_test = self.train_test_split()
        train_predict = self.model.predict(X_train)
        test_predict = self.model.predict(X_test)

        # Inverter a normalização
        def inverse_transform(predictions):
            dummy = np.zeros((len(predictions), self.data.shape[1]))
            dummy[:, 3] = predictions.flatten()
            return self.scaler.inverse_transform(dummy)[:, 3]

        train_predict_inv = inverse_transform(train_predict)
        test_predict_inv = inverse_transform(test_predict)
        y_train_inv = inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = inverse_transform(y_test.reshape(-1, 1))

        return train_predict_inv, test_predict_inv, y_train_inv, y_test_inv

    def evaluate_and_plot(self):
        """Avalia e plota resultados"""
        train_predict, test_predict, y_train, y_test = self.predict()

        # Métricas
        metrics = {
            'Train MAE': mean_absolute_error(y_train, train_predict),
            'Test MAE': mean_absolute_error(y_test, test_predict),
            'Train RMSE': np.sqrt(mean_squared_error(y_train, train_predict)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, test_predict))
        }

        print("\n📊 Métricas do Modelo:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # === GRÁFICOS ===
        plt.figure(figsize=(14, 8))

        # 1️⃣ Preço real vs previsto - treino
        plt.subplot(2, 1, 1)
        plt.plot(y_train, label='Real (Treino)', color='blue')
        plt.plot(train_predict, label='Previsto (Treino)', color='red', alpha=0.7)
        plt.title('Preço Real vs Previsto - Conjunto de Treino')
        plt.xlabel('Amostras')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2️⃣ Preço real vs previsto - teste
        plt.subplot(2, 1, 2)
        plt.plot(y_test, label='Real (Teste)', color='blue')
        plt.plot(test_predict, label='Previsto (Teste)', color='orange', alpha=0.7)
        plt.title('Preço Real vs Previsto - Conjunto de Teste')
        plt.xlabel('Amostras')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 3️⃣ Histórico de perda (loss)
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Loss Treino')
        plt.plot(self.history.history['val_loss'], label='Loss Validação')
        plt.title('Evolução da Função de Perda')
        plt.xlabel('Época')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return metrics


# === Exemplo de uso ===
if __name__ == "__main__":
    model = LSTMModel(
        ticker='PETR4.SA', 
        start_date='2020-01-01', 
        end_date='2024-01-01',
        window_size=60
    )

    data = model.download_and_adjust_data()
    print("Dados processados:")
    print(data.head())

    history = model.train(epochs=50, batch_size=32)
    metrics = model.evaluate_and_plot()
