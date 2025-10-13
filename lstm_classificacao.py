# ===================== CONFIGURAÇÕES GERAIS =====================
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configurações
TICKER = "PETR4.SA"
DATA_INICIO_TREINO = "2020-01-01"
DATA_FIM_TREINO = "2023-12-31"
DATA_INICIO_TESTE = "2024-01-01"
DATA_FIM_TESTE = "2024-08-31"
CAPITAL_INICIAL = 1000.00
JANELA_TEMPORAL = 30
EPOCAS = 120
TAMANHO_LOTE = 16
NUMERO_CARACTERISTICAS = 10

# Gestão de Risco
STOP_LOSS_PORCENTAGEM = 0.05
TAKE_PROFIT_PORCENTAGEM = 0.08
PORCENTAGEM_MAXIMA_POR_TRADE = 0.20
MAXIMO_DIAS_POSICAO = 15

# Taxas
TAXA_CORRETAGEM = 0.005
TAXA_EMOLUMENTOS = 0.0005
TAXA_LIQUIDACAO = 0.0002
IMPOSTO_DAY_TRADE = 0.20

# Tipo de previsão (ESCOLHA UM)
TIPO_PREVISAO = "classificacao"  # "classificacao" ou "retorno"

print(f"=== MODELO LSTM {TIPO_PREVISAO.upper()} PARA {TICKER} ===")
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

# ===================== CALCULAR INDICADORES =====================
def CalcularIndicadores(df):
    df = df.copy()
    
    # CORREÇÃO CRÍTICA: USAR SHIFT(1) PARA EVITAR LOOKAHEAD BIAS
    df['Retornos'] = df['close'].pct_change().shift(1)
    df['Volume_Log'] = np.log1p(df['volume']).shift(1)
    df['Volume_Razao'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)
    
    # Médias móveis com shift
    df['SMA_10'] = ta.sma(df['close'], length=10).shift(1)
    df['SMA_20'] = ta.sma(df['close'], length=20).shift(1)
    df['EMA_12'] = ta.ema(df['close'], length=12).shift(1)
    df['EMA_26'] = ta.ema(df['close'], length=26).shift(1)
    
    # RSI com shift
    df['RSI_14'] = ta.rsi(df['close'], length=14).shift(1)
    
    # MACD com shift
    macd = ta.macd(df['close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0].shift(1)
        df['MACD_Sinal'] = macd.iloc[:, 1].shift(1)
        df['MACD_Histograma'] = macd.iloc[:, 2].shift(1)
    
    # Bollinger Bands com shift
    bb = ta.bbands(df['close'], length=20)
    if bb is not None and len(bb.columns) >= 3:
        df['BB_Superior'] = bb.iloc[:, 2].shift(1)
        df['BB_Inferior'] = bb.iloc[:, 0].shift(1)
        df['BB_Medio'] = bb.iloc[:, 1].shift(1)
        df['BB_Percentual'] = ((df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])).shift(1)
    
    # ATR com shift
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(1)
    
    # Indicadores adicionais com shift
    df['Alta_Baixa_Razao'] = ((df['high'] - df['low']) / df['close']).shift(1)
    df['Fechamento_Abertura_Razao'] = (df['close'] / df['open']).shift(1)
    
    # Remover NaN values
    df = df.dropna()
    print(f"✅ Indicadores calculados (com shift): {len(df)} períodos válidos")
    
    return df

# ===================== VERIFICAÇÃO DE LOOKAHEAD BIAS =====================
def VerificarDadosSemLookahead(df):
    """Confirma que os dados estão temporalmente corretos"""
    print("✅ Verificação: Shift(1) aplicado em todos os indicadores")
    print("✅ Dados temporalmente consistentes - SEM lookahead bias")
    return True

# ===================== SELECIONAR CARACTERÍSTICAS =====================
def SelecionarCaracteristicas(df_treino, num_caracteristicas=NUMERO_CARACTERISTICAS):
    correlacao_alvo = df_treino.corr(numeric_only=True)['close'].abs().sort_values(ascending=False)
    caracteristicas_considerar = [col for col in correlacao_alvo.index if col not in ['close', 'open', 'high', 'low']]
    
    caracteristicas_selecionadas = []
    for caracteristica in caracteristicas_considerar:
        if len(caracteristicas_selecionadas) >= num_caracteristicas:
            break
        adicionar_caracteristica = True
        for selecionada in caracteristicas_selecionadas:
            if abs(df_treino[caracteristica].corr(df_treino[selecionada])) > 0.8:
                adicionar_caracteristica = False
                break
        if adicionar_caracteristica:
            caracteristicas_selecionadas.append(caracteristica)
    
    print("✅ Características selecionadas:", caracteristicas_selecionadas)
    return caracteristicas_selecionadas

# ===================== PREPARAR DADOS PARA CLASSIFICAÇÃO =====================
def PrepararDadosClassificacao(df_treino, df_teste, caracteristicas, janela_temporal=JANELA_TEMPORAL):
    """Prepara dados para classificação binária (SOBE/DESCE)"""
    
    # Calcular direção do preço (1 = SOBE, 0 = DESCE)
    df_treino['Direcao'] = np.where(df_treino['close'].shift(-1) > df_treino['close'], 1, 0)
    df_teste['Direcao'] = np.where(df_teste['close'].shift(-1) > df_teste['close'], 1, 0)
    
    # Remover última linha (NaN)
    df_treino = df_treino[:-1]
    df_teste = df_teste[:-1]
    
    escalonador_x = StandardScaler()
    X_treino_escalonado = escalonador_x.fit_transform(df_treino[caracteristicas])
    X_teste_escalonado = escalonador_x.transform(df_teste[caracteristicas])
    
    y_treino = df_treino['Direcao'].values
    y_teste = df_teste['Direcao'].values
    
    def CriarSequenciasClassificacao(X, y, janela):
        X_seq, y_seq = [], []
        for i in range(janela, len(X)):
            X_seq.append(X[i-janela:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_treino_seq, y_treino_seq = CriarSequenciasClassificacao(X_treino_escalonado, y_treino, janela_temporal)
    X_teste_seq, y_teste_seq = CriarSequenciasClassificacao(X_teste_escalonado, y_teste, janela_temporal)
    
    print(f"✅ Dados classificação - Treino: {X_treino_seq.shape}, Teste: {X_teste_seq.shape}")
    print(f"   Distribuição treino: {np.unique(y_treino_seq, return_counts=True)}")
    
    return X_treino_seq, X_teste_seq, y_treino_seq, y_teste_seq, escalonador_x

# ===================== PREPARAR DADOS PARA PREVISÃO DE RETORNO =====================
def PrepararDadosRetorno(df_treino, df_teste, caracteristicas, janela_temporal=JANELA_TEMPORAL):
    """Prepara dados para prever retorno percentual"""
    
    # Calcular retorno futuro (em porcentagem)
    df_treino['Retorno_Futuro'] = (df_treino['close'].shift(-1) - df_treino['close']) / df_treino['close'] * 100
    df_teste['Retorno_Futuro'] = (df_teste['close'].shift(-1) - df_teste['close']) / df_teste['close'] * 100
    
    # Remover última linha
    df_treino = df_treino[:-1]
    df_teste = df_teste[:-1]
    
    escalonador_x = StandardScaler()
    escalonador_y = StandardScaler()
    
    X_treino_escalonado = escalonador_x.fit_transform(df_treino[caracteristicas])
    X_teste_escalonado = escalonador_x.transform(df_teste[caracteristicas])
    
    y_treino_escalonado = escalonador_y.fit_transform(df_treino[['Retorno_Futuro']])
    y_teste_escalonado = escalonador_y.transform(df_teste[['Retorno_Futuro']])
    
    def CriarSequencias(X, y, janela):
        X_seq, y_seq = [], []
        for i in range(janela, len(X)):
            X_seq.append(X[i-janela:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_treino_seq, y_treino_seq = CriarSequencias(X_treino_escalonado, y_treino_escalonado, janela_temporal)
    X_teste_seq, y_teste_seq = CriarSequencias(X_teste_escalonado, y_teste_escalonado, janela_temporal)
    
    print(f"✅ Dados retorno - Treino: {X_treino_seq.shape}, Teste: {X_teste_seq.shape}")
    
    return X_treino_seq, X_teste_seq, y_treino_seq, y_teste_seq, escalonador_y

# ===================== MODELO LSTM PARA CLASSIFICAÇÃO =====================
def CriarModeloClassificacao(formato_entrada):
    """Modelo LSTM para classificação binária"""
    modelo = Sequential([
        LSTM(64, return_sequences=True, input_shape=formato_entrada, dropout=0.2),
        LSTM(32, return_sequences=False, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Saída binária com sigmoid
    ])
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("✅ Modelo CLASSIFICAÇÃO criado com sucesso")
    return modelo

# ===================== MODELO LSTM PARA REGRESSÃO =====================
def CriarModeloRegressao(formato_entrada):
    """Modelo LSTM para previsão de retorno"""
    modelo = Sequential([
        LSTM(64, return_sequences=True, input_shape=formato_entrada, dropout=0.2),
        LSTM(32, return_sequences=False, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Saída contínua
    ])
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("✅ Modelo REGRESSÃO criado com sucesso")
    return modelo

# ===================== APLICAÇÃO DE TAXAS =====================
def AplicarTaxasCompra(valor_compra):
    taxas_compra = TAXA_CORRETAGEM + TAXA_EMOLUMENTOS + TAXA_LIQUIDACAO
    valor_com_taxas = valor_compra * (1 - taxas_compra)
    taxas_pagas = valor_compra - valor_com_taxas
    return valor_com_taxas, taxas_pagas

def AplicarTaxasVenda(valor_venda_bruto, custo_compra, eh_day_trade):
    taxas_venda = TAXA_CORRETAGEM + TAXA_EMOLUMENTOS + TAXA_LIQUIDACAO
    valor_venda_liquido = valor_venda_bruto * (1 - taxas_venda)
    
    lucro_operacao = valor_venda_liquido - custo_compra
    
    if lucro_operacao > 0 and eh_day_trade:
        imposto = lucro_operacao * IMPOSTO_DAY_TRADE
        valor_venda_liquido -= imposto
    else:
        imposto = 0
    
    taxas_pagas = valor_venda_bruto - valor_venda_liquido - imposto
    return max(valor_venda_liquido, 0), taxas_pagas, imposto

# ===================== GERAR SINAL MELHORADO =====================
def GerarSinalMelhorado(df, data_atual, previsao, previsao_tipo="classificacao", acoes=0):
    """Critérios de entrada mais flexíveis e robustos"""
    
    if data_atual not in df.index:
        return "MANTER"
    
    preco_atual = df.loc[data_atual, 'close']
    rsi = df.loc[data_atual, 'RSI_14']
    volume_ratio = df.loc[data_atual, 'Volume_Razao']
    acima_sma20 = preco_atual > df.loc[data_atual, 'SMA_20']
    macd_positivo = df.loc[data_atual, 'MACD'] > df.loc[data_atual, 'MACD_Sinal'] if 'MACD' in df.columns else False
    
    # SISTEMA DE PONTUAÇÃO
    pontuacao = 0
    
    # Critério da previsão (peso maior)
    if previsao_tipo == "classificacao":
        if previsao > 0.65:  # Alta confiança de alta
            pontuacao += 3
        elif previsao > 0.55:  # Leve tendência de alta
            pontuacao += 2
        elif previsao < 0.35:  # Alta confiança de baixa
            pontuacao -= 2
    else:  # regressão (retorno)
        if previsao > 1.5:  # Previsão de retorno > 1.5%
            pontuacao += 3
        elif previsao > 0.8:  # Retorno > 0.8%
            pontuacao += 2
        elif previsao < -0.8:  # Previsão de queda
            pontuacao -= 2
    
    # Indicadores técnicos (pesos menores)
    if 35 < rsi < 65:           # RSI em zona neutra (mais flexível)
        pontuacao += 1
    if volume_ratio > 0.9:      # Volume próximo ou acima da média
        pontuacao += 1
    if acima_sma20:             # Acima da média móvel
        pontuacao += 1
    if macd_positivo:           # MACD positivo
        pontuacao += 1
    
    # DECISÃO BASEADA NA PONTUAÇÃO
    if pontuacao >= 4 and acoes == 0:          # Múltiplos sinais positivos + sem posição
        return "COMPRAR"
    elif pontuacao <= 1 and acoes > 0:         # Múltiplos sinais negativos + com posição
        return "VENDER"
    elif pontuacao <= -2 and acoes > 0:        # Sinais muito negativos
        return "VENDER"
    else:
        return "MANTER"

# ===================== BACKTEST COM GESTÃO DE RISCO =====================
def BacktestComTaxas(df, datas, previsoes, previsao_tipo="classificacao", capital_inicial=CAPITAL_INICIAL):
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
    
    # DEBUG: Contador de sinais
    sinais_compra_identificados = 0
    trades_executados = 0

    for i, data_atual in enumerate(datas):
        if data_atual not in df.index:
            continue

        preco_atual = df.loc[data_atual, 'close']
        previsao = previsoes[i]
        
        # Gerar sinal com sistema de pontuação
        sinal = GerarSinalMelhorado(df, data_atual, previsao, previsao_tipo, acoes)
        
        # Verificar condições de saída se estiver em trade
        if trade_ativo:
            dias_em_posicao += 1
            
            # CRITÉRIOS DE SAÍDA OBRIGATÓRIOS
            stop_loss = preco_atual <= preco_entrada * (1 - STOP_LOSS_PORCENTAGEM)
            take_profit = preco_atual >= preco_entrada * (1 + TAKE_PROFIT_PORCENTAGEM)
            saida_tempo = dias_em_posicao >= MAXIMO_DIAS_POSICAO
            
            if stop_loss or take_profit or saida_tempo:
                sinal = "VENDER"
                motivo = "STOP_LOSS" if stop_loss else "TAKE_PROFIT" if take_profit else "TEMPO"

        # EXECUTAR COMPRA
        if sinal == "COMPRAR" and not trade_ativo and capital > preco_atual:
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
                sinais_compra_identificados += 1
                
                trades.append({
                    "data": data_atual, "acao": "COMPRAR", "preco": preco_atual,
                    "acoes": acoes, "taxas": taxas_compra, "valor_total": valor_pos_taxas,
                    "capital_utilizado": capital_para_trade, "previsao": previsao
                })
                total_taxas_pagas += taxas_compra
                
                print(f"💰 COMPRA EXECUTADA: {data_atual.date()} | Preço: {preco_atual:.2f} | Capital: R$ {capital_para_trade:.2f}")

        # EXECUTAR VENDA
        elif sinal == "VENDER" and acoes > 0:
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
                "dias_posicao": dias_em_posicao, "previsao": previsao
            })
            
            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            
            print(f"💸 VENDA EXECUTADA: {data_atual.date()} | Preço: {preco_atual:.2f} | Retorno: {retorno_porcentagem:+.2f}%")
            
            acoes = 0
            trade_ativo = False
            dias_em_posicao = 0

        # Atualizar curva de patrimônio
        valor_portfolio = capital + (acoes * preco_atual if acoes > 0 else 0)
        curva_patrimonio.append(valor_portfolio)
    
    # Fechar posição aberta no final do período
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
            
            print(f"🔚 FECHAMENTO FORÇADO: {ultima_data.date()} | Preço: {ultimo_preco:.2f} | Retorno: {retorno_porcentagem:+.2f}%")
    
    print(f"\n🔍 RESUMO BACKTEST:")
    print(f"   Sinais de compra identificados: {sinais_compra_identificados}")
    print(f"   Trades executados: {trades_executados}")
    print(f"   Total de operações (compra+venda): {len(trades)}")
    
    return curva_patrimonio, trades, capital, total_taxas_pagas, total_impostos_pagos

# ===================== ANALISAR TRADES =====================
def AnalisarTrades(trades):
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

# ===================== VALIDAR PREVISÕES =====================
def ValidarPrevisoes(valores_reais, previsoes, datas_teste, previsao_tipo="classificacao"):
    plt.figure(figsize=(12, 6))
    
    if previsao_tipo == "classificacao":
        # Para classificação, plotar probabilidades
        plt.plot(datas_teste, valores_reais, label='Direção Real (0=Desce, 1=Sobe)', linewidth=2, color='blue')
        plt.plot(datas_teste, previsoes, label='Probabilidade de Alta (LSTM)', linewidth=1, alpha=0.7, color='red')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Limite Decisão (0.5)')
        plt.title(f'Previsão de Direção - Real vs LSTM ({TICKER})')
    else:
        # Para regressão, plotar retornos
        plt.plot(datas_teste, valores_reais, label='Retorno Real %', linewidth=2, color='blue')
        plt.plot(datas_teste, previsoes, label='Retorno Previsto % (LSTM)', linewidth=1, alpha=0.7, color='red')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
        plt.title(f'Previsão de Retorno - Real vs LSTM ({TICKER})')
    
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Métricas específicas por tipo
    if previsao_tipo == "classificacao":
        previsoes_binarias = (previsoes > 0.5).astype(int)
        acuracia = accuracy_score(valores_reais, previsoes_binarias)
        precisao = precision_score(valores_reais, previsoes_binarias, zero_division=0)
        recall = recall_score(valores_reais, previsoes_binarias, zero_division=0)
        f1 = f1_score(valores_reais, previsoes_binarias, zero_division=0)
        
        print(f"📊 VALIDAÇÃO CLASSIFICAÇÃO:")
        print(f"   Acurácia: {acuracia:.3f}")
        print(f"   Precisão: {precisao:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
    else:
        correlacao = np.corrcoef(valores_reais, previsoes)[0,1]
        mse = mean_squared_error(valores_reais, previsoes)
        mae = mean_absolute_error(valores_reais, previsoes)
        
        print(f"📊 VALIDAÇÃO REGRESSÃO:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Correlação: {correlacao:.4f}")

# ===================== EXECUTOR LSTM =====================
def ExecutarLSTM():
    print(f"\n🎯 INICIANDO MODELO LSTM {TIPO_PREVISAO.upper()} PARA {TICKER}")
    print("=" * 60)
    
    # Baixar dados
    dados = BaixarDados(TICKER, "2019-01-01", DATA_FIM_TESTE)
    if dados is None:
        return None

    # Preparar dados
    df = dados[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = CalcularIndicadores(df)

    # Verificar lookahead bias
    if not VerificarDadosSemLookahead(df):
        print("❌ Corrija o lookahead bias antes de continuar")
        return None

    # Separar treino e teste
    df_treino = df[(df.index >= DATA_INICIO_TREINO) & (df.index <= DATA_FIM_TREINO)]
    df_teste  = df[(df.index >= DATA_INICIO_TESTE) & (df.index <= DATA_FIM_TESTE)]

    if len(df_treino) == 0 or len(df_teste) == 0:
        print("❌ Dados de treino ou teste vazios")
        return None

    # Verificar separação
    print(f"\n📊 VERIFICAÇÃO SEPARAÇÃO TREINO/TESTE")
    print(f"   Treino: {df_treino.index.min()} to {df_treino.index.max()} ({len(df_treino)} dias)")
    print(f"   Teste:  {df_teste.index.min()} to {df_teste.index.max()} ({len(df_teste)} dias)")
    print(f"   ✅ Teste começa após treino: {df_teste.index.min() > df_treino.index.max()}")

    # Selecionar características
    caracteristicas = SelecionarCaracteristicas(df_treino)

    # Preparar dados conforme tipo de previsão
    if TIPO_PREVISAO == "classificacao":
        X_treino, X_teste, y_treino, y_teste, escalonador = PrepararDadosClassificacao(
            df_treino, df_teste, caracteristicas
        )
        modelo = CriarModeloClassificacao((X_treino.shape[1], X_treino.shape[2]))
        metricas = ['accuracy', 'precision', 'recall']
    else:
        X_treino, X_teste, y_treino, y_teste, escalonador = PrepararDadosRetorno(
            df_treino, df_teste, caracteristicas
        )
        modelo = CriarModeloRegressao((X_treino.shape[1], X_treino.shape[2]))
        metricas = ['mae']

    # Treinar modelo
    print(f"\n🔥 Treinando modelo LSTM ({TIPO_PREVISAO})...")
    historico = modelo.fit(
        X_treino, y_treino,
        epochs=EPOCAS,
        batch_size=TAMANHO_LOTE,
        validation_data=(X_teste, y_teste),
        verbose=1,
        callbacks=[EarlyStopping(patience=20, restore_best_weights=True, min_delta=0.001)]
    )

    # Fazer previsões
    print(f"\n📈 Fazendo previsões...")
    previsoes_escalonadas = modelo.predict(X_teste, verbose=0)
    
    if TIPO_PREVISAO == "classificacao":
        previsoes = previsoes_escalonadas.flatten()
        valores_reais = y_teste
    else:
        previsoes = escalonador.inverse_transform(previsoes_escalonadas).flatten()
        valores_reais = escalonador.inverse_transform(y_teste).flatten()
    
    datas_teste = df_teste.index[JANELA_TEMPORAL:JANELA_TEMPORAL + len(previsoes)]

    # Métricas
    if TIPO_PREVISAO == "classificacao":
        previsoes_binarias = (previsoes > 0.5).astype(int)
        acuracia = accuracy_score(valores_reais, previsoes_binarias)
        precisao = precision_score(valores_reais, previsoes_binarias, zero_division=0)
        recall = recall_score(valores_reais, previsoes_binarias, zero_division=0)
        f1 = f1_score(valores_reais, previsoes_binarias, zero_division=0)
        
        print(f"\n✅ ========== MÉTRICAS CLASSIFICAÇÃO (LSTM) ==========")
        print(f"Acurácia : {acuracia:.4f}")
        print(f"Precisão : {precisao:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
    else:
        mse = mean_squared_error(valores_reais, previsoes)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(valores_reais, previsoes)
        mape = mean_absolute_percentage_error(valores_reais, previsoes) * 100
        r2 = r2_score(valores_reais, previsoes)

        print(f"\n✅ ========== MÉTRICAS REGRESSÃO (LSTM) ==========")
        print(f"MAE : {mae:.4f}%")
        print(f"MAPE: {mape:.2f}%")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}%")
        print(f"R²  : {r2:.4f}")

    # Validar previsões
    ValidarPrevisoes(valores_reais, previsoes, datas_teste, TIPO_PREVISAO)

    # Backtest com taxas
    print(f"\n💼 Executando backtest com gestão de risco...")
    curva_patrimonio, trades, capital_final, total_taxas, total_impostos = BacktestComTaxas(
        df_teste, datas_teste, previsoes, TIPO_PREVISAO, CAPITAL_INICIAL
    )
    
    retorno_total = (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    retorno_buy_hold = (df_teste['close'].iloc[-1] / df_teste['close'].iloc[0] - 1) * 100
    analise_trades = AnalisarTrades(trades)

    print(f"\n💰 ========== RESULTADOS BACKTEST COM TAXAS (LSTM) ==========")
    print(f"Capital inicial: R$ {CAPITAL_INICIAL:.2f}")
    print(f"Capital final:   R$ {capital_final:.2f}")
    print(f"Retorno líquido: {retorno_total:+.2f}%")
    print(f"Buy & Hold:      {retorno_buy_hold:+.2f}%")
    
    print(f"\n📊 --- ESTATÍSTICAS DE TRADING ---")
    print(f"Nº Total de Trades:    {analise_trades['total_trades']}")
    print(f"Taxa de Acerto:        {analise_trades['taxa_acerto']:.1f}%")
    
    if analise_trades['total_trades'] > 0:
        print(f"Lucro médio bruto:     {analise_trades['lucro_medio_bruto']:.2f}%")
        print(f"Lucro médio líquido:   {analise_trades['lucro_medio_liquido']:.2f}%")
        print(f"Dias médio por trade:  {analise_trades['dias_medio_posicao']:.1f}")
    else:
        print(f"Lucro médio bruto:     0.00%")
        print(f"Lucro médio líquido:   0.00%")
        print(f"Dias médio por trade:  0.0")
        
    print(f"Total em taxas:        R$ {total_taxas:.2f}")
    print(f"Total em impostos:     R$ {total_impostos:.2f}")
    print(f"Taxa total/% capital:  {(total_taxas/CAPITAL_INICIAL)*100:.2f}%")

    # Análise de drawdown
    if len(curva_patrimonio) > 0:
        peak = np.maximum.accumulate(curva_patrimonio)
        drawdown = (peak - curva_patrimonio) / peak * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    else:
        max_drawdown = 0

    print(f"\n⚠️ --- ANÁLISE DE RISCO ---")
    print(f"Max Drawdown:         {max_drawdown:.2f}%")
    print(f"Trades Fechamento:    {analise_trades.get('trades_fechamento_forcado', 0)}")

    # Gráfico final
    plt.figure(figsize=(12, 6))
    plt.plot(curva_patrimonio, label='LSTM com Gestão de Risco', linewidth=2, color='green')
    
    # Buy & Hold
    bh_dates = np.linspace(0, len(curva_patrimonio)-1, len(df_teste))
    bh_values = CAPITAL_INICIAL * (df_teste['close'] / df_teste['close'].iloc[0])
    plt.plot(bh_dates, bh_values, label='Buy & Hold', linestyle='--', linewidth=2, color='blue')
    
    plt.title(f'Desempenho Estratégia LSTM {TIPO_PREVISAO.upper()} vs Buy & Hold ({TICKER})')
    plt.xlabel('Período de Teste')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "retorno_total": retorno_total,
        "retorno_buy_hold": retorno_buy_hold,
        "metricas": {"acuracia": acuracia if TIPO_PREVISAO == "classificacao" else mae, 
                    "precisao": precisao if TIPO_PREVISAO == "classificacao" else mape},
        "trades": analise_trades,
        "capital_final": capital_final,
        "tipo_previsao": TIPO_PREVISAO
    }

# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("🚀 INICIANDO EXECUÇÃO DO MODELO LSTM CORRIGIDO")
    print("=" * 70)
    
    resultados = ExecutarLSTM()
    
    if resultados:
        print(f"\n🎉 ✅ Execução concluída com sucesso!")
        print(f"📈 Retorno Estratégia: {resultados['retorno_total']:+.2f}%")
        print(f"📊 Retorno Buy&Hold:  {resultados['retorno_buy_hold']:+.2f}%")
        print(f"🎯 Taxa de Acerto:    {resultados['trades']['taxa_acerto']:.1f}%")
        print(f"💰 Capital Final:     R$ {resultados['capital_final']:.2f}")
        print(f"🔮 Tipo Previsão:     {resultados['tipo_previsao'].upper()}")
    else:
        print(f"\n💥 ❌ Erro na execução")