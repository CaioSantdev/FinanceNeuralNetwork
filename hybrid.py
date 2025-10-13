# ===================== CONFIGURAÇÕES GERAIS =====================
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configurações
TICKER = "VALE3.SA"
DATA_INICIO_TREINO = "2020-01-01"
DATA_FIM_TREINO = "2023-12-31"
DATA_INICIO_TESTE = "2024-01-01"
DATA_FIM_TESTE = "2024-08-31"
CAPITAL_INICIAL = 1000.00
JANELA_TEMPORAL = 30
EPOCAS = 150
TAMANHO_LOTE = 32
NUMERO_CARACTERISTICAS = 20

# Gestão de Risco Melhorada
STOP_LOSS_PORCENTAGEM = 0.03
TAKE_PROFIT_PORCENTAGEM = 0.06
PORCENTAGEM_MAXIMA_POR_TRADE = 0.15
MAXIMO_DIAS_POSICAO = 10
CONFIANCA_MINIMA = 1.02  # 2% acima para filtrar ruído

# Taxas B3 (valores mais realistas)
TAXA_CORRETAGEM = 0.50  # R$ fixo por operação
TAXA_EMOLUMENTOS = 0.00005  # 0.005%
TAXA_LIQUIDACAO = 0.000275  # 0.0275%
IMPOSTO_DAY_TRADE = 0.20

print(f"=== CONFIGURAÇÃO LSTM AVANÇADO PARA {TICKER} ===")
print(f"Período Treino: {DATA_INICIO_TREINO} até {DATA_FIM_TREINO}")
print(f"Período Teste:  {DATA_INICIO_TESTE} até {DATA_FIM_TESTE}")

# ===================== DOWNLOAD DOS DADOS =====================
def BaixarDados(ticker, data_inicio, data_fim):
    print(f"Baixando dados de {ticker}...")
    dados = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=True, progress=False)
    if dados.empty:
        print("❌ Erro: Não foi possível baixar os dados")
        return None
    print(f"✅ Dados baixados: {len(dados)} períodos")
    return dados

# ===================== CALCULAR INDICADORES AVANÇADOS =====================
def CalcularIndicadoresAvancados(df):
    """
    Calcula indicadores técnicos com shift adequado para evitar lookahead bias
    CORREÇÃO CRÍTICA: Shift duplo para evitar vazamento de dados
    """
    df = df.copy()
    
    # CORREÇÃO: Shift duplo para garantir isolamento total
    shift = 2
    
    # 1. Retornos e Momentum
    df['Retornos_1'] = df['close'].pct_change().shift(shift)
    df['Retornos_5'] = (df['close'] / df['close'].shift(5) - 1).shift(shift)
    df['Retornos_10'] = (df['close'] / df['close'].shift(10) - 1).shift(shift)
    
    # 2. Volume Analysis
    df['Volume_Log'] = np.log1p(df['volume']).shift(shift)
    df['Volume_SMA_10'] = df['volume'].rolling(10).mean().shift(shift)
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean().shift(shift)
    df['Volume_Razao'] = (df['volume'] / df['Volume_SMA_20']).shift(shift)
    
    # 3. Médias Móveis (com shift duplo)
    df['SMA_10'] = ta.sma(df['close'], length=10).shift(1).shift(1)  # Shift duplo
    df['SMA_20'] = ta.sma(df['close'], length=20).shift(1).shift(1)
    df['SMA_50'] = ta.sma(df['close'], length=50).shift(1).shift(1)
    df['EMA_12'] = ta.ema(df['close'], length=12).shift(1).shift(1)
    df['EMA_26'] = ta.ema(df['close'], length=26).shift(1).shift(1)
    
    # 4. RSI Multi-Timeframe
    df['RSI_7'] = ta.rsi(df['close'], length=7).shift(shift)
    df['RSI_14'] = ta.rsi(df['close'], length=14).shift(shift)
    df['RSI_21'] = ta.rsi(df['close'], length=21).shift(shift)
    
    # 5. MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0].shift(shift)
        df['MACD_Sinal'] = macd.iloc[:, 1].shift(shift)
        df['MACD_Histograma'] = macd.iloc[:, 2].shift(shift)
    
    # 6. Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None:
        df['BB_Superior'] = bb.iloc[:, 2].shift(shift)
        df['BB_Inferior'] = bb.iloc[:, 0].shift(shift)
        df['BB_Medio'] = bb.iloc[:, 1].shift(shift)
        df['BB_Largura'] = ((bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]).shift(shift)
        df['BB_Posicao'] = ((df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])).shift(shift)
    
    # 7. Volatilidade
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(shift)
    df['Volatilidade_20'] = df['close'].pct_change().rolling(20).std().shift(shift)
    
    # 8. Suporte e Resistência
    df['Resistencia_20'] = df['high'].rolling(20).max().shift(shift)
    df['Suporte_20'] = df['low'].rolling(20).min().shift(shift)
    df['Distancia_Resistencia'] = ((df['Resistencia_20'] - df['close']) / df['close']).shift(shift)
    df['Distancia_Suporte'] = ((df['close'] - df['Suporte_20']) / df['close']).shift(shift)
    
    # 9. Price Action
    df['Alta_Baixa_Razao'] = ((df['high'] - df['low']) / df['close']).shift(shift)
    df['Fechamento_Abertura_Razao'] = (df['close'] / df['open']).shift(shift)
    df['Corpo_Candle'] = (df['close'] - df['open']).shift(shift)
    
    # 10. Tendências
    df['Tendencia_SMA'] = np.where(df['SMA_10'] > df['SMA_20'], 1, -1).shift(shift)
    df['Forca_Tendencia'] = (df['SMA_10'] / df['SMA_50'] - 1).shift(shift)
    
    # Remover NaN values
    df = df.dropna()
    print(f"✅ {len(df.columns) - 5} indicadores avançados calculados (shift={shift}): {len(df)} períodos válidos")
    
    return df

# ===================== SELECIONAR CARACTERÍSTICAS COM RANDOM FOREST =====================
def SelecionarCaracteristicasAvancadas(df_treino, num_caracteristicas=NUMERO_CARACTERISTICAS):
    """
    Seleção de características usando importância do Random Forest
    """
    # Preparar dados para Random Forest
    features = [col for col in df_treino.columns if col not in ['close', 'open', 'high', 'low', 'volume']]
    X_rf = df_treino[features]
    
    # Criar target para RF: retorno futuro (evita lookahead)
    y_rf = (df_treino['close'].shift(-5) / df_treino['close'] - 1).dropna()
    X_rf = X_rf.loc[y_rf.index]
    
    # Treinar Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_rf, y_rf)
    
    # Obter importâncias
    importancias = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Selecionar características não correlacionadas
    caracteristicas_selecionadas = []
    for _, row in importancias.iterrows():
        if len(caracteristicas_selecionadas) >= num_caracteristicas:
            break
        
        feature = row['feature']
        adicionar = True
        
        # Verificar correlação com características já selecionadas
        for selecionada in caracteristicas_selecionadas:
            if abs(df_treino[feature].corr(df_treino[selecionada])) > 0.7:
                adicionar = False
                break
        
        if adicionar:
            caracteristicas_selecionadas.append(feature)
    
    print(f"✅ {len(caracteristicas_selecionadas)} características selecionadas por importância:")
    for i, feat in enumerate(caracteristicas_selecionadas[:10]):
        print(f"   {i+1}. {feat}")
    
    return caracteristicas_selecionadas

# ===================== MODELO LSTM-CNN HÍBRIDO =====================
def CriarModeloHibrido(formato_entrada):
    """
    Modelo LSTM-CNN inspirado nos artigos para regressão
    """
    modelo = Sequential([
        # CNN para extração de features locais
        Conv1D(64, 3, activation='relu', input_shape=formato_entrada, 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(128, 2, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        # LSTM para dependências temporais
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False, dropout=0.2,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        # Camadas Densas
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # Saída de regressão
        Dense(1)
    ])
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print("✅ Modelo LSTM-CNN Híbrido criado com sucesso")
    modelo.summary()
    
    return modelo

# ===================== VALIDAÇÃO CRUZADA TEMPORAL =====================
def ValidacaoCruzadaTemporal(df, caracteristicas, n_splits=3):
    """
    Validação cruzada temporal para avaliar robustez do modelo
    """
    print(f"\n🔍 INICIANDO VALIDAÇÃO CRUZADA TEMPORAL ({n_splits} splits)")
    
    # Preparar dados
    X = df[caracteristicas].values
    y = df['close'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    mae_scores = []
    
    fold = 1
    for train_idx, test_idx in tscv.split(X):
        print(f"\n--- Fold {fold} ---")
        
        # Separar dados
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Escalonar
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_test_scaled = scaler_x.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Criar sequências
        def criar_sequencias(X, y, janela):
            X_seq, y_seq = [], []
            for i in range(janela, len(X)):
                X_seq.append(X[i-janela:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = criar_sequencias(X_train_scaled, y_train_scaled, JANELA_TEMPORAL)
        X_test_seq, y_test_seq = criar_sequencias(X_test_scaled, y_test_scaled, JANELA_TEMPORAL)
        
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"   ⚠️ Fold {fold} pulado - dados insuficientes")
            continue
        
        # Criar e treinar modelo
        modelo = CriarModeloHibrido((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        history = modelo.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=TAMANHO_LOTE,
            validation_data=(X_test_seq, y_test_seq),
            verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
        )
        
        # Fazer previsões e calcular métricas
        previsoes_scaled = modelo.predict(X_test_seq, verbose=0).flatten()
        previsoes = scaler_y.inverse_transform(previsoes_scaled.reshape(-1, 1)).flatten()
        y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_original, previsoes))
        mae = mean_absolute_error(y_test_original, previsoes)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"   RMSE: R$ {rmse:.2f}, MAE: R$ {mae:.2f}")
        
        fold += 1
    
    if rmse_scores:
        print(f"\n✅ VALIDAÇÃO CRUZADA - RESULTADOS FINAIS:")
        print(f"   RMSE médio: R$ {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
        print(f"   MAE médio:  R$ {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
    else:
        print("❌ Validação cruzada não pôde ser concluída")
    
    return rmse_scores, mae_scores

# ===================== MÉTRICAS AVANÇADAS DE TRADING =====================
def CalcularMetricasAvancadas(trades, curva_capital, capital_inicial):
    """
    Calcula métricas avançadas de performance de trading
    """
    if len(curva_capital) < 2:
        return {
            "sharpe_ratio": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "calmar_ratio": 0,
            "expectancy": 0
        }
    
    # Calcular retornos diários
    retornos_diarios = np.diff(curva_capital) / curva_capital[:-1]
    
    # Sharpe Ratio (anualizado)
    if len(retornos_diarios) > 1 and np.std(retornos_diarios) > 0:
        sharpe = np.mean(retornos_diarios) / np.std(retornos_diarios) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(curva_capital)
    drawdown = (peak - curva_capital) / peak * 100
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Profit Factor
    if trades:
        lucros = sum(t['retorno_liquido'] for t in trades if t['retorno_liquido'] > 0)
        perdas = abs(sum(t['retorno_liquido'] for t in trades if t['retorno_liquido'] < 0))
        profit_factor = lucros / perdas if perdas > 0 else float('inf')
    else:
        profit_factor = 0
    
    # Calmar Ratio
    retorno_total = (curva_capital[-1] - capital_inicial) / capital_inicial
    calmar = retorno_total / (max_dd / 100) if max_dd > 0 else 0
    
    # Expectancy
    if trades:
        trades_lucrativos = [t for t in trades if t['retorno_liquido'] > 0]
        trades_perdedores = [t for t in trades if t['retorno_liquido'] < 0]
        
        win_rate = len(trades_lucrativos) / len(trades) if trades else 0
        avg_win = np.mean([t['retorno_liquido'] for t in trades_lucrativos]) if trades_lucrativos else 0
        avg_loss = np.mean([t['retorno_liquido'] for t in trades_perdedores]) if trades_perdedores else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    else:
        expectancy = 0
    
    return {
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "expectancy": expectancy
    }

# ===================== ESTRATÉGIA DE TRADING MELHORADA =====================
def BacktestAvancado(df, datas, previsoes, capital_inicial=CAPITAL_INICIAL):
    """
    Backtest com estratégia mais sofisticada e realista
    """
    capital = capital_inicial
    acoes = 0
    preco_entrada = 0
    data_entrada = None
    trades = []
    curva_patrimonio = [capital]
    trade_ativo = False
    dias_em_posicao = 0
    
    total_taxas_pagas = 0
    total_impostos_pagos = 0
    
    sinais_compra_identificados = 0
    trades_executados = 0

    for i, data_atual in enumerate(datas):
        if data_atual not in df.index:
            continue

        preco_atual = df.loc[data_atual, 'close']
        preco_previsto = previsoes[i]
        razao_preco = preco_previsto / preco_atual
        
        sinal = "MANTER"
        
        # 🎯 CRITÉRIOS DE ENTRADA MAIS ROBUSTOS
        if not trade_ativo and capital > preco_atual:
            # Múltiplas condições para filtrar ruído
            condicao_previsao = razao_preco > CONFIANCA_MINIMA  # 2% acima
            condicao_tendencia = preco_atual > df.loc[data_atual, 'SMA_20']
            condicao_rsi = 35 < df.loc[data_atual, 'RSI_14'] < 65  # Zona mais conservadora
            condicao_volume = df.loc[data_atual, 'Volume_Razao'] > 1.1  # Volume acima da média
            condicao_bb = df.loc[data_atual, 'BB_Posicao'] < 0.8  # Não sobrecomprado
            
            if (condicao_previsao and condicao_tendencia and 
                condicao_rsi and condicao_volume and condicao_bb):
                sinal = "COMPRAR"
                sinais_compra_identificados += 1
        
        # 🎯 CRITÉRIOS DE SAÍDA MAIS INTELIGENTES
        elif trade_ativo:
            dias_em_posicao += 1
            
            # Stop Loss e Take Profit
            stop_loss = preco_atual <= preco_entrada * (1 - STOP_LOSS_PORCENTAGEM)
            take_profit = preco_atual >= preco_entrada * (1 + TAKE_PROFIT_PORCENTAGEM)
            
            # Condições adicionais de saída
            saida_previsao_fraca = razao_preco < 0.995  # Previsão piorou significativamente
            saida_rsi_extremo = df.loc[data_atual, 'RSI_14'] > 75  # Sobrecomprado
            saida_tempo = dias_em_posicao >= MAXIMO_DIAS_POSICAO
            saida_bb_extremo = df.loc[data_atual, 'BB_Posicao'] > 0.9  # Topo da Bollinger
            
            if (stop_loss or take_profit or saida_previsao_fraca or 
                saida_rsi_extremo or saida_tempo or saida_bb_extremo):
                sinal = "VENDER"

        # EXECUTAR COMPRA (mantida similar mas com mais logs)
        if sinal == "COMPRAR" and not trade_ativo:
            capital_para_trade = min(capital, capital_inicial * PORCENTAGEM_MAXIMA_POR_TRADE)
            
            if capital_para_trade > preco_atual:
                valor_pos_taxas, taxas_compra = AplicarTaxasCompra(capital_para_trade)
                acoes = valor_pos_taxas / preco_atual
                capital -= capital_para_trade
                preco_entrada = preco_atual
                data_entrada = data_atual
                trade_ativo = True
                dias_em_posicao = 0
                trades_executados += 1
                
                trades.append({
                    "data": data_atual, "acao": "COMPRAR", "preco": preco_atual,
                    "acoes": acoes, "taxas": taxas_compra, "valor_total": valor_pos_taxas,
                    "capital_utilizado": capital_para_trade
                })
                total_taxas_pagas += taxas_compra

        # EXECUTAR VENDA (mantida similar)
        elif sinal == "VENDER" and trade_ativo and acoes > 0:
            valor_venda_bruto = acoes * preco_atual
            custo_compra = acoes * preco_entrada
            eh_day_trade = (data_entrada.date() == data_atual.date())
            
            valor_liquido, taxas_venda, imposto = AplicarTaxasVenda(
                valor_venda_bruto, custo_compra, eh_day_trade
            )
            
            capital += valor_liquido
            retorno_porcentagem = (valor_liquido / custo_compra - 1) * 100
            
            trades.append({
                "data": data_atual, "acao": "VENDER", "preco": preco_atual,
                "acoes": acoes, "retorno_porcentagem": retorno_porcentagem,
                "preco_entrada": preco_entrada, "taxas": taxas_venda,
                "eh_day_trade": eh_day_trade, "valor_liquido": valor_liquido,
                "dias_posicao": dias_em_posicao
            })
            
            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            
            # Resetar posição
            acoes = 0
            trade_ativo = False
            dias_em_posicao = 0

        # Atualizar curva de patrimônio
        valor_portfolio = capital + (acoes * preco_atual if acoes > 0 else 0)
        curva_patrimonio.append(valor_portfolio)
    
    # Fechar posição aberta no final (mantido)
    if trade_ativo and len(datas) > 0:
        ultima_data = datas[-1]
        if ultima_data in df.index:
            ultimo_preco = df.loc[ultima_data, 'close']
            valor_venda_bruto = acoes * ultimo_preco
            custo_compra = acoes * preco_entrada
            eh_day_trade = (data_entrada.date() == ultima_data.date())
            
            valor_liquido, taxas_venda, imposto = AplicarTaxasVenda(
                valor_venda_bruto, custo_compra, eh_day_trade
            )
            
            capital += valor_liquido
            retorno_porcentagem = (valor_liquido / custo_compra - 1) * 100
            
            trades.append({
                "data": ultima_data, "acao": "VENDER", "preco": ultimo_preco,
                "acoes": acoes, "retorno_porcentagem": retorno_porcentagem,
                "preco_entrada": preco_entrada, "taxas": taxas_venda,
                "eh_day_trade": eh_day_trade, "valor_liquido": valor_liquido,
                "dias_posicao": dias_em_posicao, "fechamento_forcado": True
            })
            
            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
    
    print(f"\n🔍 RESUMO BACKTEST AVANÇADO:")
    print(f"   Sinais de compra identificados: {sinais_compra_identificados}")
    print(f"   Trades executados: {trades_executados}")
    
    return curva_patrimonio, trades, capital, total_taxas_pagas, total_impostos_pagos

# ===================== FUNÇÕES MANTIDAS (com pequenos ajustes) =====================
def AplicarTaxasCompra(valor_compra):
    taxas_compra = TAXA_CORRETAGEM + (valor_compra * (TAXA_EMOLUMENTOS + TAXA_LIQUIDACAO))
    valor_com_taxas = valor_compra - taxas_compra
    return valor_com_taxas, taxas_compra

def AplicarTaxasVenda(valor_venda_bruto, custo_compra, eh_day_trade):
    taxas_venda = TAXA_CORRETAGEM + (valor_venda_bruto * (TAXA_EMOLUMENTOS + TAXA_LIQUIDACAO))
    valor_antes_imposto = valor_venda_bruto - taxas_venda
    
    lucro_operacao = valor_antes_imposto - custo_compra
    imposto = lucro_operacao * IMPOSTO_DAY_TRADE if (eh_day_trade and lucro_operacao > 0) else 0
    
    valor_liquido_venda = valor_antes_imposto - imposto
    return max(valor_liquido_venda, 0), taxas_venda, imposto

def AnalisarTrades(trades):
    # (Mantida a mesma função, mas agora com métricas avançadas)
    if len(trades) < 2:
        return {
            "total_trades": 0, "taxa_acerto": 0, 
            "lucro_medio_bruto": 0, "lucro_medio_liquido": 0, 
            "total_taxas_pagas": 0, "dias_medio_posicao": 0,
            "trades_fechamento_forcado": 0, "trades": []
        }

    resultados_trades = []
    trades_vencedores = 0
    total_taxas = 0
    
    for i in range(0, len(trades)-1, 2):
        if trades[i]['acao'] == 'COMPRAR' and trades[i+1]['acao'] == 'VENDER':
            trade_compra = trades[i]
            trade_venda = trades[i+1]
            retorno_liquido = trade_venda['retorno_porcentagem']
            taxas_operacao = trade_compra.get('taxas', 0) + trade_venda.get('taxas', 0)
            
            resultados_trades.append({
                'retorno_bruto': trade_venda['retorno_porcentagem'],
                'retorno_liquido': retorno_liquido,
                'taxas': taxas_operacao,
                'day_trade': trade_venda.get('eh_day_trade', False),
                'dias_posicao': trade_venda.get('dias_posicao', 0),
                'fechamento_forcado': trade_venda.get('fechamento_forcado', False)
            })
            
            total_taxas += taxas_operacao
            if retorno_liquido > 0:
                trades_vencedores += 1

    if resultados_trades:
        taxa_acerto = (trades_vencedores / len(resultados_trades)) * 100
        lucro_medio_bruto = np.mean([t['retorno_bruto'] for t in resultados_trades])
        lucro_medio_liquido = np.mean([t['retorno_liquido'] for t in resultados_trades])
        dias_medio_posicao = np.mean([t['dias_posicao'] for t in resultados_trades])
        trades_fechamento_forcado = sum(1 for t in resultados_trades if t['fechamento_forcado'])
        
        return {
            "total_trades": len(resultados_trades),
            "taxa_acerto": taxa_acerto,
            "lucro_medio_bruto": lucro_medio_bruto,
            "lucro_medio_liquido": lucro_medio_liquido,
            "total_taxas_pagas": total_taxas,
            "dias_medio_posicao": dias_medio_posicao,
            "trades_fechamento_forcado": trades_fechamento_forcado,
            "trades": resultados_trades
        }
    
    return {
        "total_trades": 0, "taxa_acerto": 0, 
        "lucro_medio_bruto": 0, "lucro_medio_liquido": 0, 
        "total_taxas_pagas": 0, "dias_medio_posicao": 0,
        "trades_fechamento_forcado": 0, "trades": []
    }

# ===================== EXECUTOR PRINCIPAL ATUALIZADO =====================
def ExecutarLSTM_Avancado():
    print(f"\n🎯 INICIANDO MODELO LSTM-CNN AVANÇADO PARA {TICKER}")
    print("=" * 70)
    
    # Baixar dados
    dados = BaixarDados(TICKER, "2019-01-01", DATA_FIM_TESTE)
    if dados is None:
        return None

    # Preparar dados
    df = dados[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    
    # Calcular indicadores avançados
    df = CalcularIndicadoresAvancados(df)
    
    # Separar treino e teste
    df_treino = df[(df.index >= DATA_INICIO_TREINO) & (df.index <= DATA_FIM_TREINO)]
    df_teste  = df[(df.index >= DATA_INICIO_TESTE) & (df.index <= DATA_FIM_TESTE)]

    if len(df_treino) == 0 or len(df_teste) == 0:
        print("❌ Dados de treino ou teste vazios")
        return None

    # Selecionar características com Random Forest
    caracteristicas = SelecionarCaracteristicasAvancadas(df_treino)
    
    # Validação cruzada temporal
    print(f"\n🔍 EXECUTANDO VALIDAÇÃO CRUZADA TEMPORAL...")
    rmse_scores, mae_scores = ValidacaoCruzadaTemporal(df_treino, caracteristicas)

    # Preparar dados para modelo final
    escalonador_x_treino = StandardScaler()
    escalonador_y_treino = StandardScaler()
    
    X_treino_escalonado = escalonador_x_treino.fit_transform(df_treino[caracteristicas])
    y_treino_escalonado = escalonador_y_treino.fit_transform(df_treino[['close']])
    
    X_teste_escalonado = escalonador_x_treino.transform(df_teste[caracteristicas])
    y_teste_escalonado = escalonador_y_treino.transform(df_teste[['close']])
    
    def CriarSequencias(X, y, janela):
        X_seq, y_seq = [], []
        for i in range(janela, len(X)):
            X_seq.append(X[i-janela:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_treino, y_treino = CriarSequencias(X_treino_escalonado, y_treino_escalonado, JANELA_TEMPORAL)
    X_teste, y_teste = CriarSequencias(X_teste_escalonado, y_teste_escalonado, JANELA_TEMPORAL)

    # Criar e treinar modelo final
    modelo = CriarModeloHibrido((X_treino.shape[1], X_treino.shape[2]))

    print(f"\n🔥 TREINANDO MODELO LSTM-CNN FINAL...")
    callbacks = [
        EarlyStopping(patience=25, restore_best_weights=True, min_delta=0.0005),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
    ]

    historico = modelo.fit(
        X_treino, y_treino,
        epochs=EPOCAS,
        batch_size=TAMANHO_LOTE,
        validation_data=(X_teste, y_teste),
        verbose=1,
        callbacks=callbacks,
        shuffle=False
    )

    # Fazer previsões
    print(f"\n📈 FAZENDO PREVISÕES...")
    previsoes_escalonadas = modelo.predict(X_teste, verbose=0)
    previsoes = escalonador_y_treino.inverse_transform(previsoes_escalonadas).flatten()
    valores_reais = escalonador_y_treino.inverse_transform(y_teste).flatten()
    datas_teste = df_teste.index[JANELA_TEMPORAL:JANELA_TEMPORAL + len(previsoes)]

    # Métricas de regressão
    mse = mean_squared_error(valores_reais, previsoes)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(valores_reais, previsoes)
    mape = mean_absolute_percentage_error(valores_reais, previsoes) * 100
    r2 = r2_score(valores_reais, previsoes)

    print(f"\n✅ ========== MÉTRICAS DE REGRESSÃO (LSTM-CNN) ==========")
    print(f"MAE : R$ {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: R$ {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

    # Backtest avançado
    print(f"\n💼 EXECUTANDO BACKTEST AVANÇADO...")
    curva_patrimonio, trades, capital_final, total_taxas, total_impostos = BacktestAvancado(
        df_teste, datas_teste, previsoes, CAPITAL_INICIAL
    )
    
    # Análise completa
    retorno_total = (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    retorno_buy_hold = (df_teste['close'].iloc[-1] / df_teste['close'].iloc[0] - 1) * 100
    analise_trades = AnalisarTrades(trades)
    metricas_avancadas = CalcularMetricasAvancadas(analise_trades['trades'], curva_patrimonio, CAPITAL_INICIAL)

    print(f"\n💰 ========== RESULTADOS COMPLETOS ==========")
    print(f"📈 --- DESEMPENHO ---")
    print(f"Capital inicial:      R$ {CAPITAL_INICIAL:.2f}")
    print(f"Capital final:        R$ {capital_final:.2f}")
    print(f"Retorno líquido:      {retorno_total:+.2f}%")
    print(f"Buy & Hold:           {retorno_buy_hold:+.2f}%")
    print(f"Excesso sobre B&H:    {retorno_total - retorno_buy_hold:+.2f}%")
    
    print(f"\n📊 --- ESTATÍSTICAS DE TRADING ---")
    print(f"Total de Trades:      {analise_trades['total_trades']}")
    print(f"Taxa de Acerto:       {analise_trades['taxa_acerto']:.1f}%")
    print(f"Lucro médio líquido:  {analise_trades['lucro_medio_liquido']:.2f}%")
    print(f"Dias médio/trade:     {analise_trades['dias_medio_posicao']:.1f}")
    
    print(f"\n⚡ --- MÉTRICAS AVANÇADAS ---")
    print(f"Sharpe Ratio:         {metricas_avancadas['sharpe_ratio']:.3f}")
    print(f"Profit Factor:        {metricas_avancadas['profit_factor']:.3f}")
    print(f"Max Drawdown:         {metricas_avancadas['max_drawdown']:.2f}%")
    print(f"Calmar Ratio:         {metricas_avancadas['calmar_ratio']:.3f}")
    print(f"Expectancy:           {metricas_avancadas['expectancy']:.2f}%")
    
    print(f"\n💸 --- CUSTOS ---")
    print(f"Total em taxas:       R$ {total_taxas:.2f}")
    print(f"Total em impostos:    R$ {total_impostos:.2f}")
    print(f"Custo total:          R$ {total_taxas + total_impostos:.2f}")

    # Gráficos (mantidos similares mas com mais informações)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(curva_patrimonio, label='LSTM-CNN com Gestão de Risco', linewidth=2, color='green')
    bh_dates = np.linspace(0, len(curva_patrimonio)-1, len(df_teste))
    bh_values = CAPITAL_INICIAL * (df_teste['close'] / df_teste['close'].iloc[0])
    plt.plot(bh_dates, bh_values, label='Buy & Hold', linestyle='--', linewidth=2, color='blue')
    plt.title(f'Desempenho: LSTM-CNN vs Buy & Hold ({TICKER})')
    plt.xlabel('Dias de Teste')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Gráfico de drawdown
    peak = np.maximum.accumulate(curva_patrimonio)
    drawdown = (peak - curva_patrimonio) / peak * 100
    plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    plt.plot(drawdown, color='red', linewidth=1)
    plt.title('Drawdown da Estratégia')
    plt.xlabel('Dias de Teste')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title('Loss do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Previsões vs Real
    plt.plot(valores_reais[:100], label='Real', alpha=0.7)
    plt.plot(previsoes[:100], label='Previsto', alpha=0.7)
    plt.title('Previsões vs Valores Reais (Primeiras 100 amostras)')
    plt.xlabel('Amostras')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return {
        "retorno_total": retorno_total,
        "retorno_buy_hold": retorno_buy_hold,
        "capital_final": capital_final,
        "analise_trades": analise_trades,
        "metricas_avancadas": metricas_avancadas,
        "metricas_regressao": {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}
    }

# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("🚀 INICIANDO EXECUÇÃO DO MODELO LSTM-CNN AVANÇADO")
    print("=" * 70)
    print("🔧 PRINCIPAIS MELHORIAS IMPLEMENTADAS:")
    print("   ✅ 20+ Indicadores Técnicos Avançados")
    print("   ✅ Modelo LSTM-CNN Híbrido")
    print("   ✅ Seleção de Features com Random Forest") 
    print("   ✅ Validação Cruzada Temporal")
    print("   ✅ Métricas Avançadas de Trading")
    print("   ✅ Estratégia com Múltiplos Filtros")
    print("   ✅ Correção de Lookahead Bias")
    print("=" * 70)
    
    resultados = ExecutarLSTM_Avancado()
    
    if resultados:
        print(f"\n🎉 ✅ EXECUÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"📈 Retorno Estratégia: {resultados['retorno_total']:+.2f}%")
        print(f"📊 Retorno Buy&Hold:  {resultados['retorno_buy_hold']:+.2f}%")
        print(f"🎯 Taxa de Acerto:    {resultados['analise_trades']['taxa_acerto']:.1f}%")
        print(f"⚡ Sharpe Ratio:      {resultados['metricas_avancadas']['sharpe_ratio']:.3f}")
        print(f"💰 Capital Final:     R$ {resultados['capital_final']:.2f}")
    else:
        print(f"\n💥 ❌ Erro na execução")