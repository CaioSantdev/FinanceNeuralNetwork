# =========================================================
# 🧠 MODELO LSTM DE PREVISÃO DE PREÇOS (ESTILO ZANOTTO, 2024)
# =========================================================
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, random

# =============================
# 🔒 Reprodutibilidade Total
# =============================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =============================
# 📈 Classe Principal
# =============================
class LSTMPricePredictor:
    def __init__(self, ticker='ITUB4.SA', start_date='2019-01-01', end_date='2024-12-31', window_size=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
        self.initial_capital = 10000
        self.taxa_corretagem = 0.005

    def download_data(self):
        """Baixa e processa os dados históricos."""
        print(f"📥 Baixando dados de {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
        
        if df.empty:
            raise ValueError("❌ Nenhum dado encontrado!")

        # 🔧 AJUSTE DE PREÇOS CONFORME TCC [Santana 2022]
        print("🔧 Ajustando preços para eventos corporativos...")
        df = self._adjust_prices(df)

        # 🔍 TRATAMENTO DE DADOS FALTANTES
        print("🔍 Tratando dados faltantes...")
        df = self._handle_missing_data(df)

        # Verifica qual coluna usar
        close_col = 'Close_adj'

        # 📈 INDICADORES TÉCNICOS
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

        # Volume médio
        df['Volume_MA'] = df['Volume_adj'].rolling(window=20).mean()

        # Retornos
        df['Retorno_1d'] = df[close_col].pct_change()
        df['Retorno_5d'] = df[close_col].pct_change(5)

        df.dropna(inplace=True)
        
        # Seleção de features conforme TCC
        features = ['Open_adj', 'High_adj', 'Low_adj', 'Close_adj', 'Volume_adj', 
                   'EMA_60', 'RSI_14', 'MACD', 'MACD_Signal']
        
        self.data = df[features]
        self.data.rename(columns={'Close_adj': 'Close'}, inplace=True)

        print(f"✅ Dados carregados: {len(self.data)} registros válidos.")
        return self.data

    def _adjust_prices(self, df):
        """Ajusta preços para eventos corporativos conforme TCC [Santana 2022]"""
        # Fator de ajuste: Adj Close / Close
        adjustment_factor = df['Adj Close'] / df['Close']
        
        # Aplicar ajuste para Open, High, Low
        df['Open_adj'] = df['Open'] * adjustment_factor
        df['High_adj'] = df['High'] * adjustment_factor  
        df['Low_adj'] = df['Low'] * adjustment_factor
        df['Close_adj'] = df['Adj Close']  # Já é ajustado
        df['Volume_adj'] = df['Volume']
        
        return df

    def _handle_missing_data(self, df):
        """Trata dados faltantes conforme metodologia do TCC"""
        # Interpolação linear para preços (conforme TCC)
        price_columns = ['Open_adj', 'High_adj', 'Low_adj', 'Close_adj']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
        
        # Forward fill para volume (conforme TCC)
        if 'Volume_adj' in df.columns:
            df['Volume_adj'] = df['Volume_adj'].fillna(method='ffill')
            df['Volume_adj'] = df['Volume_adj'].fillna(0)
        
        return df

    def create_sequences(self):
        """Cria janelas de 60 dias para prever o preço do próximo dia."""
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)-1):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i+1, 3])  # Previsão de Close do próximo dia (posição 3)
        return np.array(X), np.array(y)

    def train_test_split(self, X, y, test_size=0.2):
        """Divide em treino e teste mantendo ordem temporal."""
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self.train_dates = self.data.index[self.window_size:split_idx + self.window_size]
        self.test_dates = self.data.index[split_idx + self.window_size + 1:]
        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        """Modelo LSTM (estrutura idêntica à de Zanotto)."""
        model = Sequential([
            LSTM(500, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(500, return_sequences=False),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        print("✅ Modelo LSTM construído conforme arquitetura do TCC")
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Treina o modelo com callbacks."""
        print("🚀 Treinando modelo...")
        
        # Callbacks conforme TCC
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            f'best_model_{self.ticker}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,
            shuffle=False
        )
        self.history = history
        print("✅ Treinamento concluído")
        return history

    def calculate_metrics(self, y_true, y_pred):
        """Calcula métricas completas conforme TCC"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }

    def generate_signals(self, predictions, actual_prices, dates):
        """Gera sinais de compra/venda baseados nas previsões"""
        signals = []
        positions = []
        current_position = 0

        for i in range(1, len(predictions)):
            pred_price = predictions[i]
            price_today = actual_prices[i - 1]

            # Variação percentual prevista
            pred_diff = ((pred_price - price_today) / price_today) * 100

            # Usar EMA como indicador adicional
            current_date = dates[i]
            if current_date in self.data.index:
                ema_today = self.data.loc[current_date, 'EMA_60']
                price_vs_ema = ((price_today - ema_today) / ema_today) * 100
                rsi = self.data.loc[current_date, 'RSI_14']
            else:
                price_vs_ema = 0
                rsi = 50

            if current_position == 0:
                # Condição de compra: previsão de alta + condições técnicas
                if pred_diff > 0.3 and price_vs_ema > -1 and rsi < 70:
                    signals.append('BUY')
                    current_position = 1
                else:
                    signals.append('HOLD')
            else:
                # Condição de venda: previsão de baixa ou condições de proteção
                if pred_diff < -0.3 or price_vs_ema < -2 or rsi > 75:
                    signals.append('SELL')
                    current_position = 0
                else:
                    signals.append('HOLD')

            positions.append(current_position)

        # Ajusta o tamanho da lista
        signals.insert(0, 'HOLD')
        positions.insert(0, 0)

        return signals, positions

    def backtest_strategy(self, test_predict, y_test):
        """Executa backtesting da estratégia"""
        predictions = test_predict.flatten()
        actual_prices = y_test.flatten()

        # Gerar sinais
        signals, positions = self.generate_signals(predictions, actual_prices, self.test_dates)

        # Backtesting
        cash = self.initial_capital
        shares = 0
        trades = []
        equity_curve = [cash]
        entry_price = 0

        for i, signal in enumerate(signals):
            if i >= len(self.test_dates):
                continue

            current_date = self.test_dates[i]
            current_price = actual_prices[i]

            if signal == 'BUY' and cash > 0:
                shares_to_buy = cash / current_price
                cost = shares_to_buy * current_price
                taxa = cost * self.taxa_corretagem

                shares += shares_to_buy
                cash = 0
                entry_price = current_price

                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'taxa': taxa
                })

            elif signal == 'SELL' and shares > 0:
                revenue = shares * current_price
                taxa = revenue * self.taxa_corretagem
                cash = revenue - taxa
                profit_pct = ((current_price - entry_price) / entry_price) * 100

                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'taxa': taxa,
                    'profit_pct': profit_pct
                })
                shares = 0

            # Atualiza o patrimônio
            portfolio_value = cash + (shares * current_price if shares > 0 else 0)
            equity_curve.append(portfolio_value)

        # Fechar posição final
        if shares > 0:
            final_price = actual_prices[-1]
            revenue = shares * final_price
            taxa = revenue * self.taxa_corretagem
            cash = revenue - taxa
            profit_pct = ((final_price - entry_price) / entry_price) * 100
            trades.append({
                'date': self.test_dates[-1],
                'action': 'SELL_FINAL',
                'price': final_price,
                'taxa': taxa,
                'profit_pct': profit_pct
            })
            equity_curve[-1] = cash

        return equity_curve, trades, signals

    def calculate_buy_hold(self, y_test):
        """Calcula estratégia Buy & Hold"""
        initial_price = y_test[0]
        shares_bought = self.initial_capital / initial_price
        
        final_price = y_test[-1]
        final_value = shares_bought * final_price
        
        # Taxas
        taxa_compra = self.initial_capital * self.taxa_corretagem
        taxa_venda = final_value * self.taxa_corretagem
        
        final_value_net = final_value - (taxa_compra + taxa_venda)
        
        # Curva de equity do buy & hold
        bh_curve = [self.initial_capital]
        for price in y_test:
            bh_curve.append(shares_bought * price - taxa_compra)
        
        return bh_curve, final_value_net

    def predict_and_plot(self, X_train, X_test, y_train, y_test):
        """Faz previsões e gera gráficos completos."""
        y_train_pred = self.model.predict(X_train, verbose=0)
        y_test_pred = self.model.predict(X_test, verbose=0)

        # Reverte o escalonamento
        def inverse(preds):
            dummy = np.zeros((len(preds), self.data.shape[1]))
            dummy[:, 3] = preds.flatten()  # Close na posição 3
            return self.scaler.inverse_transform(dummy)[:, 3]

        y_train_inv = inverse(y_train.reshape(-1, 1))
        y_test_inv = inverse(y_test.reshape(-1, 1))
        y_train_pred_inv = inverse(y_train_pred)
        y_test_pred_inv = inverse(y_test_pred)

        # Métricas completas
        train_metrics = self.calculate_metrics(y_train_inv, y_train_pred_inv)
        test_metrics = self.calculate_metrics(y_test_inv, y_test_pred_inv)

        print(f"\n📊 MÉTRICAS DO MODELO - {self.ticker}:")
        print("Conjunto de Treino:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print("\nConjunto de Teste:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Backtesting
        strategy_equity, trades, signals = self.backtest_strategy(y_test_pred_inv, y_test_inv)
        bh_equity, bh_final = self.calculate_buy_hold(y_test_inv)
        
        strategy_final = strategy_equity[-1]
        strategy_return = ((strategy_final - self.initial_capital) / self.initial_capital) * 100
        bh_return = ((bh_final - self.initial_capital) / self.initial_capital) * 100
        
        print(f"\n💵 RESULTADOS DO BACKTESTING:")
        print(f"Capital Inicial: R$ {self.initial_capital:,.2f}")
        print(f"Estratégia LSTM: R$ {strategy_final:,.2f} ({strategy_return:+.2f}%)")
        print(f"Buy & Hold: R$ {bh_final:,.2f} ({bh_return:+.2f}%)")
        print(f"Número de trades: {len([t for t in trades if t['action'] in ['BUY', 'SELL']])}")
        
        if len(trades) > 0:
            winning_trades = [t for t in trades if 'profit_pct' in t and t['profit_pct'] > 0]
            total_trades = len([t for t in trades if 'profit_pct' in t])
            if total_trades > 0:
                win_rate = len(winning_trades) / total_trades * 100
                print(f"Taxa de acerto: {win_rate:.1f}%")

        # Previsão para o próximo dia
        last_seq = self.scaler.transform(self.data)[-self.window_size:]
        next_pred = self.model.predict(last_seq.reshape(1, self.window_size, self.data.shape[1]), verbose=0)
        next_price = inverse(next_pred)[0]
        next_date = self.data.index[-1] + timedelta(days=1)
        
        print(f"📅 Previsão para {next_date.date()}: R$ {next_price:.2f}")

        # Geração de gráficos
        self._plot_comprehensive_results(y_train_inv, y_test_inv, y_train_pred_inv, y_test_pred_inv, 
                                       strategy_equity, bh_equity, trades, signals, test_metrics,
                                       strategy_return, bh_return, next_date, next_price)

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'strategy_final': strategy_final,
            'bh_final': bh_final,
            'strategy_return': strategy_return,
            'bh_return': bh_return,
            'trades': trades,
            'next_prediction': next_price
        }

    def _plot_comprehensive_results(self, y_train, y_test, y_train_pred, y_test_pred,
                                  strategy_equity, bh_equity, trades, signals, metrics,
                                  strategy_return, bh_return, next_date, next_price):
        """Gera gráficos completos no estilo TCC."""
        
        plt.figure(figsize=(18, 12))
        
        # 1️⃣ Preços Reais vs Previstos (Teste)
        plt.subplot(3, 3, 1)
        plt.plot(self.test_dates, y_test, label='Real', color='blue', linewidth=1.5)
        plt.plot(self.test_dates, y_test_pred, label='Previsto', color='red', linewidth=1, alpha=0.8)

        # Plotar sinais de compra e venda
        buy_dates = [self.test_dates[i] for i, s in enumerate(signals) if s == 'BUY']
        sell_dates = [self.test_dates[i] for i, s in enumerate(signals) if s == 'SELL']
        buy_prices = [y_test[i] for i, s in enumerate(signals) if s == 'BUY']
        sell_prices = [y_test[i] for i, s in enumerate(signals) if s == 'SELL']
        
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', s=80, label='Compra', zorder=5)
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', s=80, label='Venda', zorder=5)

        # Previsão próximo dia
        plt.scatter(next_date, next_price, color='gray', s=150, label=f'Próximo: R$ {next_price:.2f}', zorder=6)

        plt.title(f'{self.ticker} - Preços Reais vs Previstos\nRMSE: {metrics["RMSE"]:.4f}, R²: {metrics["R²"]:.4f}')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 2️⃣ Equity Curve
        plt.subplot(3, 3, 2)
        plt.plot(strategy_equity, label=f'LSTM ({strategy_return:+.1f}%)', color='green', linewidth=2)
        plt.plot(bh_equity, label=f'Buy & Hold ({bh_return:+.1f}%)', color='blue', linewidth=2, alpha=0.7)
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Capital Inicial')

        # Marcações de trades
        for trade in trades:
            if trade['action'] == 'BUY':
                idx = strategy_equity.index(trade['price'] * trade['shares']) if trade['price'] * trade['shares'] in strategy_equity else -1
                if idx != -1:
                    plt.scatter(idx, strategy_equity[idx], color='lime', marker='^', s=100, zorder=5)
            elif 'SELL' in trade['action']:
                idx = strategy_equity.index(trade['price'] * trade.get('shares', 0)) if trade['price'] * trade.get('shares', 0) in strategy_equity else -1
                if idx != -1:
                    plt.scatter(idx, strategy_equity[idx], color='red', marker='v', s=100, zorder=5)

        plt.title('Evolução do Patrimônio')
        plt.ylabel('Patrimônio (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3️⃣ Histórico de Loss
        plt.subplot(3, 3, 3)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Evolução da Loss durante Treinamento')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4️⃣ Erros de Previsão
        plt.subplot(3, 3, 4)
        errors = y_test - y_test_pred
        plt.plot(self.test_dates, errors, color='red', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Erros de Previsão (Teste)')
        plt.xlabel('Data')
        plt.ylabel('Erro (R$)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 5️⃣ Distribuição dos Erros
        plt.subplot(3, 3, 5)
        plt.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Distribuição dos Erros')
        plt.xlabel('Erro (R$)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)

        # 6️⃣ Retornos Comparativos
        plt.subplot(3, 3, 6)
        strategies = ['LSTM', 'Buy & Hold']
        returns = [strategy_return, bh_return]
        colors = ['green' if x > 0 else 'red' for x in returns]
        
        bars = plt.bar(strategies, returns, color=colors, alpha=0.7)
        plt.title('Retorno Final das Estratégias')
        plt.ylabel('Retorno (%)')
        
        for bar, ret in zip(bars, returns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if ret > 0 else -1), 
                    f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top')
        
        plt.grid(True, alpha=0.3)

        # 7️⃣ Série Temporal Completa
        plt.subplot(3, 3, 7)
        all_dates = np.concatenate([self.train_dates, self.test_dates])
        all_real = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        
        plt.plot(all_dates, all_real, label='Real', color='blue', linewidth=1)
        plt.plot(all_dates, all_pred, label='Previsto', color='orange', linewidth=1, alpha=0.8)
        plt.axvline(x=self.test_dates[0], color='red', linestyle='--', alpha=0.5, label='Início Teste')
        plt.title('Série Temporal Completa')
        plt.xlabel('Data')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 8️⃣ Métricas de Desempenho
        plt.subplot(3, 3, 8)
        metric_names = ['RMSE', 'MAE', 'MAPE', 'R²']
        metric_values = [metrics['RMSE'], metrics['MAE'], metrics['MAPE'], metrics['R²'] * 100]
        colors = ['red', 'orange', 'yellow', 'green']
        
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
        plt.title('Métricas de Desempenho (Teste)')
        plt.ylabel('Valor')
        
        for bar, val in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)

        # 9️⃣ Resumo de Trades
        plt.subplot(3, 3, 9)
        if len(trades) > 0:
            trade_actions = ['BUY', 'SELL', 'HOLD']
            trade_counts = [signals.count('BUY'), signals.count('SELL'), signals.count('HOLD')]
            plt.pie(trade_counts, labels=trade_actions, autopct='%1.1f%%', startangle=90)
            plt.title('Distribuição de Sinais')
        else:
            plt.text(0.5, 0.5, 'Sem trades\nexecutados', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Distribuição de Sinais')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        output_file = f"{self.ticker.replace('.SA', '').replace('.', '_')}_resultados_completos.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico salvo como {output_file}")
        plt.show()

# =============================
# 🚀 Execução
# =============================
if __name__ == "__main__":
    # Lista de tickers para teste
    tickers = ['ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 'PETR4.SA', 'VALE3.SA']
    
    for ticker in tickers:
        try:
            print(f"\n{'='*60}")
            print(f"🎯 ANALISANDO {ticker}")
            print(f"{'='*60}")
            
            model = LSTMPricePredictor(ticker=ticker, start_date='2019-01-01', end_date='2024-12-31')
            data = model.download_data()
            X, y = model.create_sequences()
            X_train, X_test, y_train, y_test = model.train_test_split(X, y)
            model.build_model((X_train.shape[1], X_train.shape[2]))
            model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
            results = model.predict_and_plot(X_train, X_test, y_train, y_test)
            
            print(f"\n✅ Análise de {ticker} concluída!")
            
        except Exception as e:
            print(f"❌ Erro ao processar {ticker}: {e}")
            continue

    print(f"\n🎉 Todas as análises foram concluídas!")