import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TradingLSTMModel:
    def __init__(self, ticker, start_date='2019-01-01', end_date='2024-31-12', window_size=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.initial_capital = 10000  # R$ 10.000 inicial
        self.taxa_corretagem = 0.005  # 0.5% por trade (B3 + corretora)
        
    def download_and_adjust_data(self):
        """Baixa e ajusta os dados históricos"""
        print(f"📥 Baixando dados de {self.ticker} de {self.start_date} até {self.end_date}...")

        try:
            # Download dos dados
            stock = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
            
            if stock.empty:
                print(f"❌ Nenhum dado encontrado para {self.ticker}")
                return None

            # ✅ Corrige colunas em caso de MultiIndex
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = [col[0] for col in stock.columns]

            # ✅ Garante que temos a coluna Close
            if 'Close' not in stock.columns:
                print("❌ Coluna 'Close' não encontrada")
                return None

            # Cálculo de indicadores técnicos
            stock['EMA_20'] = stock['Close'].ewm(span=20, adjust=False).mean()
            stock['EMA_60'] = stock['Close'].ewm(span=60, adjust=False).mean()
            
            # RSI
            delta = stock['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = stock['Close'].ewm(span=12, adjust=False).mean()
            exp2 = stock['Close'].ewm(span=26, adjust=False).mean()
            stock['MACD'] = exp1 - exp2
            stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()
            
            # Volume médio
            stock['Volume_MA'] = stock['Volume'].rolling(window=20).mean()
            
            # Retornos
            stock['Retorno_1d'] = stock['Close'].pct_change()
            stock['Retorno_5d'] = stock['Close'].pct_change(5)
            
            # Seleção das features finais
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'EMA_20', 'EMA_60', 'RSI', 'MACD', 'MACD_Signal',
                       'Volume_MA', 'Retorno_1d', 'Retorno_5d']
            
            self.data = stock[features].dropna()
            
            print(f"✅ Dados processados com sucesso — {len(self.data)} registros válidos.")
            return self.data
            
        except Exception as e:
            print(f"❌ Erro ao baixar dados: {e}")
            return None

    def create_sequences(self, data):
        """Cria sequências para o modelo LSTM"""
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data)
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i, 3])  # Close é o target
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Constrói a arquitetura LSTM otimizada"""
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape, dropout=0.2),
            LSTM(50, return_sequences=False, dropout=0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        # self.model = Sequential([
        #   LSTM(500, return_sequences=True, input_shape=input_shape),
        #   Dropout(0.3),
        #   LSTM(500, return_sequences=False),
        #   Dropout(0.3),
        #   Dense(1)
        # ])
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
        
        # Guardar datas de teste para o backtesting
        self.test_dates = self.data.index[split_idx + self.window_size:]
        
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
        self.history = history
        return history
    
    def predict(self):
        """Faz previsões e retorna resultados invertidos"""
        X_train, X_test, y_train, y_test = self.train_test_split()
        train_predict = self.model.predict(X_train, verbose=0)
        test_predict = self.model.predict(X_test, verbose=0)

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

    def generate_signals(self, predictions, actual_prices, dates):
        """Gera sinais de compra, venda e venda a descoberto (short selling)."""
        signals = []
        positions = []
        current_position = 0  # 0 = fora, 1 = comprado, -1 = vendido

        for i in range(1, len(predictions)):
            pred_price = predictions[i]
            price_today = actual_prices[i - 1]
            pred_diff = ((pred_price - price_today) / price_today) * 100  # variação %

            # Indicadores auxiliares
            rsi = self.data.loc[dates[i], 'RSI'] if dates[i] in self.data.index else 50
            macd = self.data.loc[dates[i], 'MACD'] if dates[i] in self.data.index else 0
            macd_signal = self.data.loc[dates[i], 'MACD_Signal'] if dates[i] in self.data.index else 0
            ema20 = self.data.loc[dates[i], 'EMA_20'] if dates[i] in self.data.index else 0
            ema60 = self.data.loc[dates[i], 'EMA_60'] if dates[i] in self.data.index else 0

            # ===============================
            # 📈 COMPRADO (LONG)
            # ===============================
            if current_position == 0:
                # Compra normal (tendência de alta)
                if pred_diff > 0.2 and rsi < 70 and macd > macd_signal and ema20 > ema60:
                    signals.append('BUY')
                    current_position = 1

                # Venda a descoberto (short) — tendência de baixa
                elif pred_diff < -0.2 and rsi > 30 and macd < macd_signal and ema20 < ema60:
                    signals.append('SELL_SHORT')
                    current_position = -1
                else:
                    signals.append('HOLD')

            # ===============================
            # 🔼 JÁ COMPRADO
            # ===============================
            elif current_position == 1:
                # Fecha posição comprada se o preço virar
                if pred_diff < -0.15 or rsi > 75 or macd < macd_signal or ema20 < ema60:
                    signals.append('SELL')
                    current_position = 0
                else:
                    signals.append('HOLD')

            # ===============================
            # 🔽 JÁ VENDIDO (SHORT)
            # ===============================
            elif current_position == -1:
                # Fecha short se tendência inverter
                if pred_diff > 0.15 or rsi < 25 or macd > macd_signal or ema20 > ema60:
                    signals.append('CLOSE_SHORT')
                    current_position = 0
                else:
                    signals.append('HOLD')

            positions.append(current_position)

        # Ajuste inicial
        signals.insert(0, 'HOLD')
        positions.insert(0, 0)

        return signals, positions



    def backtest_strategy(self):
        """Executa backtesting da estratégia (com suporte a posições short)."""
        _, test_predict, _, y_test = self.predict()

        predictions = test_predict.flatten()
        actual_prices = y_test.flatten()

        # Gerar sinais
        signals, positions = self.generate_signals(predictions, actual_prices, self.test_dates)

        # Inicialização
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

            # ===============================
            # 🟢 COMPRA NORMAL (LONG)
            # ===============================
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

            # ===============================
            # 🔴 VENDA NORMAL (fechar posição comprada)
            # ===============================
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

            # ===============================
            # 🔽 VENDA A DESCOBERTO (SHORT)
            # ===============================
            elif signal == 'SELL_SHORT' and cash > 0:
                shares = - (cash / current_price)
                cash += abs(shares) * current_price
                entry_price = current_price
                trades.append({
                    'date': current_date,
                    'action': 'SELL_SHORT',
                    'price': current_price,
                    'shares': shares
                })

            # ===============================
            # 🔁 FECHAR POSIÇÃO SHORT
            # ===============================
            elif signal == 'CLOSE_SHORT' and shares < 0:
                # Lucro: diferença entre preço de venda (entry_price) e recompra (current_price)
                profit = abs(shares) * (entry_price - current_price)
                taxa = abs(shares) * current_price * self.taxa_corretagem
                cash += profit - taxa
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                trades.append({
                    'date': current_date,
                    'action': 'CLOSE_SHORT',
                    'price': current_price,
                    'profit_pct': profit_pct
                })
                shares = 0

            # ===============================
            # 💰 Atualiza patrimônio total
            # ===============================
            portfolio_value = cash + (shares * current_price if shares != 0 else 0)
            equity_curve.append(portfolio_value)

        # Fecha qualquer posição aberta no fim
        if shares > 0:
            final_price = actual_prices[-1]
            revenue = shares * final_price
            taxa = revenue * self.taxa_corretagem
            cash = revenue - taxa
            trades.append({
                'date': self.test_dates[-1],
                'action': 'SELL_FINAL',
                'price': final_price
            })
            shares = 0
        elif shares < 0:
            final_price = actual_prices[-1]
            profit = abs(shares) * (entry_price - final_price)
            taxa = abs(shares) * final_price * self.taxa_corretagem
            cash += profit - taxa
            trades.append({
                'date': self.test_dates[-1],
                'action': 'CLOSE_SHORT_FINAL',
                'price': final_price
            })
            shares = 0

        return equity_curve, trades


    def calculate_buy_hold(self):
        """Calcula estratégia Buy & Hold"""
        _, _, _, y_test = self.predict()
        
        # Compra no primeiro dia de teste
        initial_price = y_test[0]
        shares_bought = self.initial_capital / initial_price
        
        # Venda no último dia
        final_price = y_test[-1]
        final_value = shares_bought * final_price
        
        # Taxa de compra e venda
        taxa_compra = self.initial_capital * self.taxa_corretagem
        taxa_venda = final_value * self.taxa_corretagem
        taxa_retirada = final_value * 0.01  # 1% de retirada final
        
        final_value_net = final_value - (taxa_compra + taxa_venda + taxa_retirada)
        
        # Curva de equity do buy & hold
        bh_curve = [self.initial_capital]
        for price in y_test:
            bh_curve.append(shares_bought * price - taxa_compra)
        
        return bh_curve, final_value_net

    def evaluate_and_plot(self):
        """Avalia e plota resultados completos"""
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

        # Backtesting
        strategy_equity, trades = self.backtest_strategy()
        bh_equity, bh_final = self.calculate_buy_hold()
        
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
            win_rate = len(winning_trades) / len([t for t in trades if 'profit_pct' in t]) * 100
            print(f"Taxa de acerto: {win_rate:.1f}%")

        # === GRÁFICOS COMPLETOS ===
        plt.figure(figsize=(15, 12))
        
      # 1️⃣ Previsões vs Real com pontos de trade
        plt.subplot(3, 2, 1)
        plt.plot(y_test, label='Real', color='blue', linewidth=1)
        plt.plot(test_predict, label='Previsto', color='red', linewidth=1, alpha=0.7)

        # Adicionar setas de compra e venda (com base nas datas)
        for trade in trades:
            if trade['date'] in self.test_dates:
                idx = np.where(self.test_dates == trade['date'])[0]
                if len(idx) > 0:
                    idx = idx[0]
                    price = y_test[idx]
                    if trade['action'] == 'BUY':
                        plt.scatter(idx, price, color='lime', marker='^', s=120, label='Compra' if 'Compra' not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif 'SELL' in trade['action']:
                        plt.scatter(idx, price, color='red', marker='v', s=120, label='Venda' if 'Venda' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title(f'{self.ticker} - Preços Reais vs Previstos (Teste)')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2️⃣ Equity Curve - Comparação + marcações
        plt.subplot(3, 2, 2)
        plt.plot(strategy_equity, label='Estratégia LSTM', color='green', linewidth=2)
        plt.plot(bh_equity, label='Buy & Hold', color='blue', linewidth=2, alpha=0.7)
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Capital Inicial')

        # 🔽 Marcações de compra e venda
        buy_points = [i for i, t in enumerate(trades) if t['action'] == 'BUY']
        sell_points = [i for i, t in enumerate(trades) if 'SELL' in t['action']]

        for trade in trades:
            idx = np.where(self.test_dates == trade['date'])[0]
            if len(idx) > 0:
                idx = idx[0]
                if trade['action'] == 'BUY':
                    plt.scatter(idx, strategy_equity[idx], color='lime', marker='^', s=100, label='Compra' if 'Compra' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif 'SELL' in trade['action']:
                    plt.scatter(idx, strategy_equity[idx], color='red', marker='v', s=100, label='Venda' if 'Venda' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('Evolução do Patrimônio com Pontos de Trade')
        plt.ylabel('Patrimônio (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3️⃣ Histórico de Loss
        plt.subplot(3, 2, 3)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Evolução da Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4️⃣ Erros de Previsão
        plt.subplot(3, 2, 4)
        errors = y_test - test_predict
        plt.plot(errors, color='red', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Erros de Previsão')
        plt.xlabel('Amostras')
        plt.ylabel('Erro (R$)')
        plt.grid(True, alpha=0.3)

        # 5️⃣ Distribuição dos Erros
        plt.subplot(3, 2, 5)
        plt.hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Distribuição dos Erros')
        plt.xlabel('Erro (R$)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)

        # 6️⃣ Retornos Comparativos
        plt.subplot(3, 2, 6)
        strategies = ['LSTM', 'Buy & Hold']
        returns = [strategy_return, bh_return]
        colors = ['green' if x > 0 else 'red' for x in returns]
        
        bars = plt.bar(strategies, returns, color=colors, alpha=0.7)
        plt.title('Retorno Final das Estratégias')
        plt.ylabel('Retorno (%)')
        
        # Adicionar valores nas barras
        for bar, ret in zip(bars, returns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if ret > 0 else -1), 
                    f'{ret:.1f}%', ha='center', va='bottom' if ret > 0 else 'top')
        
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        output_path = f"{self.ticker.replace('.SA', '')}_resultados.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico salvo como {output_path}")
        plt.show()

        return {
            'metrics': metrics,
            'strategy_final': strategy_final,
            'bh_final': bh_final,
            'strategy_return': strategy_return,
            'bh_return': bh_return,
            'trades': trades
        }


# Lista dos maiores tickers de bancos da B3
MAIORES_BANCOS = {
    '1': 'ITUB4.SA',  # Itaú Unibanco
    '2': 'BBDC4.SA',  # Bradesco
    '3': 'BBAS3.SA',  # Banco do Brasil
    '4': 'SANB11.SA', # Santander
    '5': 'B3SA3.SA',  # B3 (Bolsa)
    '6': 'BPAC11.SA'  # BTG Pactual
}

def main():
    """Função principal com menu interativo"""
    print("🏦 SISTEMA DE TRADING COM LSTM - TCC")
    print("="*50)
    print("📊 Maiores Bancos da B3:")
    for key, value in MAIORES_BANCOS.items():
        print(f"   {key}. {value}")
    print("   -1. Sair")
    print("="*50)
    
    while True:
        escolha = input("\n🎯 Escolha o número do ticker para análise: ")
        
        if escolha == '-1':
            print("👋 Encerrando programa...")
            break
            
        if escolha not in MAIORES_BANCOS:
            print("❌ Opção inválida! Tente novamente.")
            continue
            
        ticker = MAIORES_BANCOS[escolha]
        print(f"\n🔍 Analisando {ticker}...")
        
        # Criar e executar modelo
        model = TradingLSTMModel(
            ticker=ticker,
            start_date='2019-01-01',
            end_date='2024-01-01',
            window_size=60
        )
        
        data = model.download_and_adjust_data()
        if data is None:
            print(f"❌ Falha ao processar dados de {ticker}")
            continue
            
        print(f"📈 Treinando modelo para {ticker}...")
        history = model.train(epochs=50, batch_size=32)
        
        print(f"📊 Avaliando resultados para {ticker}...")
        results = model.evaluate_and_plot()
        
        # Resumo final
        print(f"\n{'='*50}")
        print(f"🎯 RESUMO FINAL - {ticker}")
        print(f"{'='*50}")
        print(f"Retorno LSTM: {results['strategy_return']:+.2f}%")
        print(f"Retorno Buy & Hold: {results['bh_return']:+.2f}%")
        
        if results['strategy_return'] > results['bh_return']:
            print("✅ Estratégia LSTM superou Buy & Hold!")
        else:
            print("❌ Buy & Hold foi melhor")
        
        continuar = input("\nDeseja analisar outro ticker? (s/n): ").lower()
        if continuar != 's':
            print("👋 Encerrando programa...")
            break

if __name__ == "__main__":
    main()

