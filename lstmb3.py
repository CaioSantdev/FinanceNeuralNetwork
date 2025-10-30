# =========================================================
# 🧠 MODELO LSTM DE PREVISÃO E BACKTESTING (ZANOTTO + TCC)
# =========================================================
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
import os, random

# =============================
# 🔒 Reprodutibilidade
# =============================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# =============================
# 📈 Classe Principal
# =============================
class TradingLSTMModel:
    def __init__(self, ticker='ITUB4.SA', start_date='2014-01-01', end_date='2024-12-31', window_size=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
        self.initial_capital = 10000
        self.taxa_corretagem = 0.005

    # =========================================================
    # 🧩 DOWNLOAD E FEATURE ENGINEERING
    # =========================================================
    def download_data(self):
        print(f"📥 Baixando dados de {self.ticker} ({self.start_date} a {self.end_date})...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("❌ Nenhum dado encontrado!")

        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # Indicadores técnicos
        df['EMA_10'] = df[close_col].ewm(span=10, adjust=False).mean()
        df['EMA_30'] = df[close_col].ewm(span=30, adjust=False).mean()
        df['EMA_60'] = df[close_col].ewm(span=60, adjust=False).mean()

        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df[close_col].ewm(span=12, adjust=False).mean()
        exp2 = df[close_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df.dropna(inplace=True)
        self.data = df[[close_col, 'Volume', 'EMA_10', 'EMA_30', 'EMA_60', 'RSI_14', 'MACD', 'MACD_Signal']]
        self.data.rename(columns={close_col: 'Close'}, inplace=True)
        print(f"✅ Dados carregados: {len(self.data)} registros válidos.")
        return self.data

    # =========================================================
    # 📊 PREPARAÇÃO DOS DADOS
    # =========================================================
    def create_sequences(self):
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)-1):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i+1, 0])  # Prever o preço do próximo dia
        return np.array(X), np.array(y)

    def train_test_split(self, X, y, test_size=0.2):
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self.train_dates = self.data.index[self.window_size:split_idx + self.window_size]
        self.test_dates = self.data.index[split_idx + self.window_size + 1:]
        return X_train, X_test, y_train, y_test

    # =========================================================
    # 🧠 CONSTRUÇÃO DO MODELO
    # =========================================================
    def build_model(self, input_shape):
        model = Sequential([
           LSTM(500, return_sequences=True, input_shape=input_shape),
           Dropout(0.3),
           LSTM(500, return_sequences=False),
           Dropout(0.3),
           Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model

    # =========================================================
    # 🚀 TREINAMENTO
    # =========================================================
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        print("🚀 Treinando modelo...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=False
        )
        self.history = history
        return history

    # =========================================================
    # 📈 PREVISÕES E INVERSÃO
    # =========================================================
    def predict(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.model.predict(X_train, verbose=0)
        y_test_pred = self.model.predict(X_test, verbose=0)

        def inverse(preds):
            dummy = np.zeros((len(preds), self.data.shape[1]))
            dummy[:, 0] = preds.flatten()
            return self.scaler.inverse_transform(dummy)[:, 0]

        y_train_inv = inverse(y_train)
        y_test_inv = inverse(y_test)
        y_train_pred_inv = inverse(y_train_pred)
        y_test_pred_inv = inverse(y_test_pred)

        return y_train_pred_inv, y_test_pred_inv, y_train_inv, y_test_inv

    # =========================================================
    # 💹 GERAÇÃO DE SINAIS
    # =========================================================
    def generate_signals(self, preds, actuals):
        signals = []
        position = 0
        for i in range(2, len(preds)):
            change = (preds[i] - preds[i-1]) / preds[i-1]
            if position == 0 and change > 0.003:
                signals.append('BUY'); position = 1
            elif position == 1 and change < -0.003:
                signals.append('SELL'); position = 0
            else:
                signals.append('HOLD')
        return ['HOLD', 'HOLD'] + signals

    # =========================================================
    # 📉 BACKTESTING
    # =========================================================
    def backtest(self, prices, signals):
        cash, shares = self.initial_capital, 0
        equity, trades = [cash], []
        for i, sig in enumerate(signals):
            price = prices[i]
            if sig == 'BUY' and cash > 0:
                shares = cash / price
                cash = 0
                trades.append((i, 'BUY', price))
            elif sig == 'SELL' and shares > 0:
                cash = shares * price * (1 - self.taxa_corretagem)
                shares = 0
                trades.append((i, 'SELL', price))
            equity.append(cash + shares * price)
        return equity, trades

    # =========================================================
    # 📊 AVALIAÇÃO E GRÁFICOS
    # =========================================================
    def evaluate_and_plot(self, X_train, X_test, y_train, y_test):
        y_train_pred, y_test_pred, y_train_inv, y_test_inv = self.predict(X_train, X_test, y_train, y_test)

        # Métricas
        mae = mean_absolute_error(y_test_inv, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_test_pred))
        print(f"\n📊 MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        signals = self.generate_signals(y_test_pred, y_test_inv)
        equity, trades = self.backtest(y_test_inv, signals)

        # Retornos
        initial_price = y_test_inv[0]
        final_price = y_test_inv[-1]
        bh_return = ((final_price - initial_price) / initial_price) * 100
        lstm_return = ((equity[-1] - self.initial_capital) / self.initial_capital) * 100
        print(f"💰 Retorno LSTM: {lstm_return:.2f}% | Buy & Hold: {bh_return:.2f}%")

        # Gráficos
        plt.figure(figsize=(15, 12))

        # 1️⃣ Preços reais vs previstos
        plt.subplot(3, 2, 1)
        plt.plot(self.test_dates, y_test_inv, label='Real', color='blue')
        plt.plot(self.test_dates, y_test_pred, label='Previsto', color='orange', linestyle='--')
        buy_idx = [i for i, s in enumerate(signals) if s == 'BUY']
        sell_idx = [i for i, s in enumerate(signals) if s == 'SELL']
        plt.scatter(np.array(self.test_dates)[buy_idx], np.array(y_test_inv)[buy_idx], marker='^', color='green', s=80, label='Compra')
        plt.scatter(np.array(self.test_dates)[sell_idx], np.array(y_test_inv)[sell_idx], marker='v', color='red', s=80, label='Venda')
        plt.title(f'{self.ticker} - Preços Reais vs Previstos (2019–2024)')
        plt.legend(); plt.grid(True, alpha=0.3)

        # 2️⃣ Patrimônio
        plt.subplot(3, 2, 2)
        plt.plot(equity, label='LSTM', color='green')
        plt.axhline(self.initial_capital, color='red', linestyle='--', label='Capital Inicial')
        plt.title('Evolução do Patrimônio'); plt.legend(); plt.grid(True, alpha=0.3)

        # 3️⃣ Loss
        plt.subplot(3, 2, 3)
        plt.plot(self.history.history['loss'], label='Treino')
        plt.plot(self.history.history['val_loss'], label='Validação')
        plt.title('Evolução da Loss'); plt.legend(); plt.grid(True, alpha=0.3)

        # 4️⃣ Erros
        plt.subplot(3, 2, 4)
        errors = y_test_inv - y_test_pred
        plt.plot(errors, color='red'); plt.axhline(0, color='black', linestyle='--')
        plt.title('Erros de Previsão'); plt.grid(True, alpha=0.3)

        # 5️⃣ Distribuição dos erros
        plt.subplot(3, 2, 5)
        plt.hist(errors, bins=40, color='purple', alpha=0.7)
        plt.title('Distribuição dos Erros'); plt.grid(True, alpha=0.3)

        # 6️⃣ Retorno comparativo
        plt.subplot(3, 2, 6)
        plt.bar(['LSTM', 'Buy & Hold'], [lstm_return, bh_return], color=['green', 'blue'])
        plt.title('Retorno Final (%)'); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# =========================================================
# 🚀 EXECUÇÃO
# =========================================================
if __name__ == "__main__":
    model = TradingLSTMModel('ITUB4.SA')
    data = model.download_data()
    X, y = model.create_sequences()
    X_train, X_test, y_train, y_test = model.train_test_split(X, y)
    model.build_model((X_train.shape[1], X_train.shape[2]))
    model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    model.evaluate_and_plot(X_train, X_test, y_train, y_test)
