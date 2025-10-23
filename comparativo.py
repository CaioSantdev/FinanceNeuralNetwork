# ===================== CONFIGURAÇÕES GERAIS UNIFICADAS =====================
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configurações

TICKER = "BBAS3.SA"
DATA_INICIO_TREINO = "2020-01-01"
DATA_FIM_TREINO = "2023-12-31"
DATA_INICIO_TESTE = "2024-01-01"
DATA_FIM_TESTE = "2024-08-31"
CAPITAL_INICIAL = 5000.00  
JANELA_TEMPORAL = 30  # Reduzindo para 30 para mais oportunidades
EPOCAS = 100
TAMANHO_LOTE = 16
NUMERO_CARACTERISTICAS = 10

# Gestão de Risco (MAIS AGRESSIVA)
STOP_LOSS_PORCENTAGEM = 0.03  # Reduzindo stop loss
TAKE_PROFIT_PORCENTAGEM = 0.05  # Reduzindo take profit
PORCENTAGEM_MAXIMA_POR_TRADE = 0.30  # Aumentando para 30%
MAXIMO_DIAS_POSICAO = 10  # Reduzindo tempo máximo

# Taxas
TAXA_CORRETAGEM = 0.005
TAXA_EMOLUMENTOS = 0.0005
TAXA_LIQUIDACAO = 0.0002
IMPOSTO_DAY_TRADE = 0.20

print(f"=== COMPARAÇÃO LSTM vs CNN1D PARA {TICKER} ===")
print(f"Período Treino: {DATA_INICIO_TREINO} até {DATA_FIM_TREINO}")
print(f"Período Teste:  {DATA_INICIO_TESTE} até {DATA_FIM_TESTE}")

# ===================== FUNÇÕES COMPARTILHADAS =====================

def BaixarDados(ticker, data_inicio, data_fim):
    print(f"Baixando dados de {ticker}...")
    dados = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=True, progress=False)
    if dados.empty:
        print("❌ Erro: Não foi possível baixar os dados")
        return None
    print(f"✅ Dados baixados: {len(dados)} períodos")
    return dados

def CalcularIndicadores(df):
    """Versão melhorada mas compatível com o código existente"""
    df = df.copy()
    
    # ✅ CORREÇÃO: Shift para evitar lookahead
    shift_period = 2
    
    # --- INDICADORES ORIGINAIS (compatibilidade) ---
    df['Retornos'] = df['close'].pct_change().shift(shift_period)
    df['Volume_Log'] = np.log1p(df['volume']).shift(shift_period)
    df['SMA_10'] = ta.sma(df['close'], length=10).shift(shift_period)
    df['SMA_20'] = ta.sma(df['close'], length=20).shift(shift_period)
    df['RSI_14'] = ta.rsi(df['close'], length=14).shift(shift_period)
    
    # --- NOVAS FEATURES ADICIONAIS ---
    # EMA para tendência curta
    df['EMA_12'] = ta.ema(df['close'], length=12).shift(shift_period)
    df['EMA_5'] = ta.ema(df['close'], length=5).shift(shift_period)  # EMA mais curta
    
    # Bandas de Bollinger
    bb = ta.bbands(df['close'], length=20)
    if bb is not None:
        df['BB_upper'] = bb.iloc[:, 0].shift(shift_period)
        df['BB_lower'] = bb.iloc[:, 2].shift(shift_period)
        df['BB_middle'] = bb.iloc[:, 1].shift(shift_period)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_posicao'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # ATR para volatilidade
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(shift_period)
    
    # Volume relativo
    df['Volume_SMA'] = ta.sma(df['volume'], length=20).shift(shift_period)
    df['Volume_Ratio'] = (df['volume'] / df['Volume_SMA']).shift(shift_period)
    
    # MACD (mantido do original)
    macd = ta.macd(df['close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0].shift(shift_period)
        df['MACD_Sinal'] = macd.iloc[:, 1].shift(shift_period)
        df['MACD_Histograma'] = macd.iloc[:, 2].shift(shift_period)
    
    # Retornos adicionais
    df['Retorno_1d'] = df['close'].pct_change(1).shift(shift_period)
    df['Retorno_3d'] = df['close'].pct_change(3).shift(shift_period)
    df['Retorno_5d'] = df['close'].pct_change(5).shift(shift_period)
    
    # Gap de abertura
    df['Open_Gap_Pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).shift(shift_period)
    
    # Momento
    df['Momentum_5d'] = (df['close'] / df['close'].shift(5) - 1).shift(shift_period)
    
    return df.dropna()

def SelecionarCaracteristicas(df_treino, num_caracteristicas=NUMERO_CARACTERISTICAS):
    """Versão melhorada mantendo compatibilidade"""
    
    # Lista de features para considerar (excluir preços)
    excluir = ['close', 'open', 'high', 'low', 'volume']
    features_candidatas = [col for col in df_treino.columns if col not in excluir]
    
    # ✅ Método mais robusto de seleção
    correlacao_alvo = df_treino.corr(numeric_only=True)['close'].abs().sort_values(ascending=False)
    
    # Priorizar features com alta correlação com target
    features_priorizadas = []
    for feat in correlacao_alvo.index:
        if feat not in excluir and feat in features_candidatas:
            features_priorizadas.append(feat)
    
    # Selecionar features com baixa correlação entre si
    caracteristicas_selecionadas = []
    correlation_matrix = df_treino[features_priorizadas].corr().abs()
    
    for caracteristica in features_priorizadas:
        if len(caracteristicas_selecionadas) >= num_caracteristicas:
            break
            
        # Verificar correlação com features já selecionadas
        alta_correlacao = False
        for selecionada in caracteristicas_selecionadas:
            if correlation_matrix.loc[caracteristica, selecionada] > 0.85:
                alta_correlacao = True
                break
                
        if not alta_correlacao:
            caracteristicas_selecionadas.append(caracteristica)
    
    print(f"✅ Características selecionadas ({len(caracteristicas_selecionadas)}):")
    for i, feat in enumerate(caracteristicas_selecionadas, 1):
        corr = correlacao_alvo[feat]
        print(f"   {i:2d}. {feat} (corr: {corr:.4f})")
    
    return caracteristicas_selecionadas

def PrepararDados(df_treino, df_teste, caracteristicas, janela_temporal=JANELA_TEMPORAL):
    """Versão com tratamento de outliers"""
    
    from sklearn.preprocessing import RobustScaler
    
    # ✅ Usar RobustScaler para reduzir impacto de outliers
    escalonador_x_treino = RobustScaler()
    escalonador_y_treino = RobustScaler()
    
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
    
    X_treino, y_treino = CriarSequencias(X_treino_escalonado, y_treino_escalonado, janela_temporal)
    X_teste, y_teste = CriarSequencias(X_teste_escalonado, y_teste_escalonado, janela_temporal)
    
    print(f"✅ Dados preparados - Treino: {X_treino.shape}, Teste: {X_teste.shape}")
    print(f"   Escalonador: RobustScaler (resistente a outliers)")
    
    return X_treino, X_teste, y_treino, y_teste, escalonador_y_treino

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

def BacktestComTaxas(df, datas, previsoes, capital_inicial=CAPITAL_INICIAL):
    """
    Backtest mais dinâmico — não tão restrito, permitindo trades leves.
    Estratégia híbrida com múltiplos sinais e pontuação adaptativa.
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

        # =====================================================
        # 🟢 CONDIÇÕES DE ENTRADA (compra)
        # =====================================================
        if not trade_ativo and capital > preco_atual:
            cond_previsao = razao_preco > 1.002  # previsão apenas 0.2% acima
            cond_tendencia = df.loc[data_atual, 'SMA_10'] > df.loc[data_atual, 'SMA_20']
            cond_macd = df.loc[data_atual, 'MACD'] > df.loc[data_atual, 'MACD_Sinal']
            cond_rsi = 35 < df.loc[data_atual, 'RSI_14'] < 70
            cond_volume = df.loc[data_atual, 'Volume_Ratio'] > 0.6
            cond_bb = df.loc[data_atual, 'BB_posicao'] < 0.85

            # sistema de pontuação mais permissivo
            pontuacao = 0
            if cond_previsao: pontuacao += 2  # previsão positiva vale mais
            if cond_tendencia: pontuacao += 1
            if cond_macd: pontuacao += 1
            if cond_rsi: pontuacao += 1
            if cond_volume: pontuacao += 1
            if cond_bb: pontuacao += 1

            # exige apenas 3 pontos totais
            if pontuacao >= 3:
                sinal = "COMPRAR"
                sinais_compra_identificados += 1
                print(f"🟢 COMPRA SINAL: {data_atual.date()} | Pontuação {pontuacao}/7 | "
                      f"Prev: {preco_previsto:.2f} | Atual: {preco_atual:.2f} | Razão: {razao_preco:.4f}")

        # =====================================================
        # 🔴 CONDIÇÕES DE SAÍDA (venda)
        # =====================================================
        elif trade_ativo:
            dias_em_posicao += 1

            stop_loss = preco_atual <= preco_entrada * (1 - STOP_LOSS_PORCENTAGEM)
            take_profit = preco_atual >= preco_entrada * (1 + TAKE_PROFIT_PORCENTAGEM)
            saida_previsao = razao_preco < 0.999  # previsão 0.1% pior já é sinal de saída
            saida_tempo = dias_em_posicao >= MAXIMO_DIAS_POSICAO
            saida_rsi_alto = df.loc[data_atual, 'RSI_14'] > 75
            saida_bb_alto = df.loc[data_atual, 'BB_posicao'] > 0.9
            saida_macd_negativo = df.loc[data_atual, 'MACD'] < df.loc[data_atual, 'MACD_Sinal']

            if stop_loss or take_profit or saida_previsao or saida_tempo or saida_rsi_alto or saida_bb_alto or saida_macd_negativo:
                sinal = "VENDER"
                motivo = (
                    "STOP_LOSS" if stop_loss else
                    "TAKE_PROFIT" if take_profit else
                    "PREVISAO" if saida_previsao else
                    "TEMPO" if saida_tempo else
                    "RSI_ALTO" if saida_rsi_alto else
                    "BB_ALTO" if saida_bb_alto else
                    "MACD_NEG" if saida_macd_negativo else
                    "DESCONHECIDO"
                )

        # =====================================================
        # 💰 EXECUTAR COMPRA
        # =====================================================
        if sinal == "COMPRAR":
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
                    "data": data_atual,
                    "acao": "COMPRAR",
                    "preco": preco_atual,
                    "acoes": acoes,
                    "taxas": taxas_compra,
                    "valor_total": valor_pos_taxas,
                    "razao_preco": razao_preco,
                    "preco_previsto": preco_previsto
                })
                total_taxas_pagas += taxas_compra

                print(f"💵 COMPRA EXECUTADA: {data_atual.date()} | Preço: {preco_atual:.2f} | "
                      f"Prev: {preco_previsto:.2f} | Razão: {razao_preco:.4f}")

        # =====================================================
        # 💸 EXECUTAR VENDA
        # =====================================================
        elif sinal == "VENDER" and acoes > 0:
            valor_venda_bruto = acoes * preco_atual
            custo_compra = acoes * preco_entrada
            eh_day_trade = (data_entrada.date() == data_atual.date())

            valor_liquido, taxas_venda, imposto = AplicarTaxasVenda(valor_venda_bruto, custo_compra, eh_day_trade)
            capital += valor_liquido
            retorno_porcentagem = (valor_liquido / custo_compra - 1) * 100

            trades.append({
                "data": data_atual,
                "acao": "VENDER",
                "preco": preco_atual,
                "acoes": acoes,
                "retorno_porcentagem": retorno_porcentagem,
                "preco_entrada": preco_entrada,
                "taxas": taxas_venda,
                "valor_liquido": valor_liquido,
                "dias_posicao": dias_em_posicao,
                "motivo_saida": motivo
            })

            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            print(f"💣 VENDA: {data_atual.date()} | Motivo: {motivo} | "
                  f"Retorno: {retorno_porcentagem:+.2f}% | Dias: {dias_em_posicao}")

            acoes = 0
            trade_ativo = False
            dias_em_posicao = 0

        # =====================================================
        # Atualizar curva de patrimônio
        # =====================================================
        valor_portfolio = capital + (acoes * preco_atual if acoes > 0 else 0)
        curva_patrimonio.append(valor_portfolio)

    # =====================================================
    # Fechar posição aberta no fim do teste
    # =====================================================
    if trade_ativo and len(datas) > 0:
        ultima_data = datas[-1]
        if ultima_data in df.index:
            ultimo_preco = df.loc[ultima_data, 'close']
            valor_venda_bruto = acoes * ultimo_preco
            custo_compra = acoes * preco_entrada
            eh_day_trade = (data_entrada.date() == ultima_data.date())

            valor_liquido, taxas_venda, imposto = AplicarTaxasVenda(valor_venda_bruto, custo_compra, eh_day_trade)
            capital += valor_liquido
            retorno_porcentagem = (valor_liquido / custo_compra - 1) * 100

            trades.append({
                "data": ultima_data,
                "acao": "VENDER",
                "preco": ultimo_preco,
                "acoes": acoes,
                "retorno_porcentagem": retorno_porcentagem,
                "preco_entrada": preco_entrada,
                "taxas": taxas_venda,
                "valor_liquido": valor_liquido,
                "dias_posicao": dias_em_posicao,
                "fechamento_forcado": True
            })

            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            print(f"🔚 FECHAMENTO FORÇADO: {ultima_data.date()} | "
                  f"Retorno: {retorno_porcentagem:+.2f}%")

    # =====================================================
    # 🔍 Resumo do backtest
    # =====================================================
    print("\n📊 RESUMO BACKTEST:")
    print(f"   Sinais de compra identificados: {sinais_compra_identificados}")
    print(f"   Trades executados: {trades_executados}")
    print(f"   Total de operações: {len(trades)}")
    print(f"   Taxas pagas totais: R$ {total_taxas_pagas:.2f}")

    return curva_patrimonio, trades, capital, total_taxas_pagas, total_impostos_pagos
def AnalisarTrades(trades):
    if len(trades) < 2:
        return {
            "total_trades": 0, "taxa_acerto": 0, 
            "lucro_medio_bruto": 0, "lucro_medio_liquido": 0, 
            "total_taxas_pagas": 0, "dias_medio_posicao": 0,
            "trades_fechamento_forcado": 0, "motivos_saida": {},
            "razao_preco_media": 0, "trades": []
        }

    resultados_trades = []
    trades_vencedores = 0
    total_taxas = 0
    razoes_preco = []

    for i in range(0, len(trades)-1, 2):
        if trades[i]['acao'] == 'COMPRAR' and trades[i+1]['acao'] == 'VENDER':
            trade_compra = trades[i]
            trade_venda = trades[i+1]
            retorno_liquido = trade_venda['retorno_porcentagem']
            taxas_operacao = trade_compra.get('taxas', 0) + trade_venda.get('taxas', 0)
            
            # Coletar razão preço previsto/real na compra
            razao = trade_compra.get('razao_preco', 0)
            razoes_preco.append(razao)
            
            resultados_trades.append({
                'retorno_bruto': trade_venda['retorno_porcentagem'],
                'retorno_liquido': retorno_liquido,
                'taxas': taxas_operacao,
                'day_trade': trade_venda.get('eh_day_trade', False),
                'dias_posicao': trade_venda.get('dias_posicao', 0),
                'fechamento_forcado': trade_venda.get('fechamento_forcado', False),
                'motivo_saida': trade_venda.get('motivo_saida', 'DESCONHECIDO'),
                'razao_preco': razao,
                'preco_previsto_compra': trade_compra.get('preco_previsto', 0),
                'preco_real_compra': trade_compra.get('preco', 0)
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
        razao_preco_media = np.mean(razoes_preco)
        
        # Análise dos motivos de saída
        motivos_saida = {}
        for trade in resultados_trades:
            motivo = trade.get('motivo_saida', 'DESCONHECIDO')
            motivos_saida[motivo] = motivos_saida.get(motivo, 0) + 1
        
        return {
            "total_trades": len(resultados_trades),
            "taxa_acerto": taxa_acerto,
            "lucro_medio_bruto": lucro_medio_bruto,
            "lucro_medio_liquido": lucro_medio_liquido,
            "total_taxas_pagas": total_taxas,
            "dias_medio_posicao": dias_medio_posicao,
            "trades_fechamento_forcado": trades_fechamento_forcado,
            "motivos_saida": motivos_saida,
            "razao_preco_media": razao_preco_media,
            "trades": resultados_trades
        }
    
    return {
        "total_trades": 0, "taxa_acerto": 0, 
        "lucro_medio_bruto": 0, "lucro_medio_liquido": 0, 
        "total_taxas_pagas": 0, "dias_medio_posicao": 0,
        "trades_fechamento_forcado": 0, "motivos_saida": {},
        "razao_preco_media": 0, "trades": []
    }

# ===================== MODELOS =====================

def CriarModeloLSTM(formato_entrada):
    modelo = Sequential([
        LSTM(100, return_sequences=True, input_shape=formato_entrada, dropout=0.2),
        LSTM(50, return_sequences=False, dropout=0.2),
        Dense(25, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return modelo

def CriarModeloCNN1D(formato_entrada):
    modelo = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
               input_shape=formato_entrada),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.1),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return modelo

# ===================== EXECUTORES DOS MODELOS =====================

def ExecutarLSTM(df_treino, df_teste, caracteristicas):
    print(f"\n🎯 INICIANDO MODELO LSTM")
    
    X_treino, X_teste, y_treino, y_teste, escalonador_y = PrepararDados(df_treino, df_teste, caracteristicas)

    modelo = CriarModeloLSTM((X_treino.shape[1], X_treino.shape[2]))
    print("✅ Modelo LSTM criado")

    print(f"🔥 Treinando LSTM...")
    historico = modelo.fit(
        X_treino, y_treino,
        epochs=EPOCAS,
        batch_size=TAMANHO_LOTE,
        validation_data=(X_teste, y_teste),
        verbose=0,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True, min_delta=0.001)]
    )

    previsoes_escalonadas = modelo.predict(X_teste, verbose=0)
    previsoes = escalonador_y.inverse_transform(previsoes_escalonadas).flatten()
    valores_reais = escalonador_y.inverse_transform(y_teste).flatten()
    datas_teste = df_teste.index[JANELA_TEMPORAL:JANELA_TEMPORAL + len(previsoes)]

    # Métricas
    mse = mean_squared_error(valores_reais, previsoes)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(valores_reais, previsoes)
    mape = mean_absolute_percentage_error(valores_reais, previsoes) * 100
    r2 = r2_score(valores_reais, previsoes)

    # Backtest
    curva_patrimonio, trades, capital_final, total_taxas, total_impostos = BacktestComTaxas(
        df_teste, datas_teste, previsoes, CAPITAL_INICIAL
    )
    
    retorno_total = (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    analise_trades = AnalisarTrades(trades)

    return {
        "nome": "LSTM",
        "previsoes": previsoes,
        "valores_reais": valores_reais,
        "datas_teste": datas_teste,
        "curva_patrimonio": curva_patrimonio,
        "metricas": {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2},
        "trades": analise_trades,
        "capital_final": capital_final,
        "retorno_total": retorno_total,
        "historico": historico.history
    }

def ExecutarCNN1D(df_treino, df_teste, caracteristicas):
    print(f"\n🎯 INICIANDO MODELO CNN1D")
    
    X_treino, X_teste, y_treino, y_teste, escalonador_y = PrepararDados(df_treino, df_teste, caracteristicas)

    modelo = CriarModeloCNN1D((X_treino.shape[1], X_treino.shape[2]))
    print("✅ Modelo CNN1D criado")

    print(f"🔥 Treinando CNN1D...")
    historico = modelo.fit(
        X_treino, y_treino,
        epochs=EPOCAS,
        batch_size=TAMANHO_LOTE,
        validation_data=(X_teste, y_teste),
        verbose=0,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True, min_delta=0.001)]
    )

    previsoes_escalonadas = modelo.predict(X_teste, verbose=0)
    previsoes = escalonador_y.inverse_transform(previsoes_escalonadas).flatten()
    valores_reais = escalonador_y.inverse_transform(y_teste).flatten()
    datas_teste = df_teste.index[JANELA_TEMPORAL:JANELA_TEMPORAL + len(previsoes)]

    # Métricas
    mse = mean_squared_error(valores_reais, previsoes)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(valores_reais, previsoes)
    mape = mean_absolute_percentage_error(valores_reais, previsoes) * 100
    r2 = r2_score(valores_reais, previsoes)

    # Backtest
    curva_patrimonio, trades, capital_final, total_taxas, total_impostos = BacktestComTaxas(
        df_teste, datas_teste, previsoes, CAPITAL_INICIAL
    )
    
    retorno_total = (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    analise_trades = AnalisarTrades(trades)

    return {
        "nome": "CNN1D",
        "previsoes": previsoes,
        "valores_reais": valores_reais,
        "datas_teste": datas_teste,
        "curva_patrimonio": curva_patrimonio,
        "metricas": {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2},
        "trades": analise_trades,
        "capital_final": capital_final,
        "retorno_total": retorno_total,
        "historico": historico.history
    }

# ===================== GRÁFICOS COMPARATIVOS =====================

def GerarGraficosComparativos(resultados_lstm, resultados_cnn, df_teste):
    print(f"\n📊 GERANDO GRÁFICOS COMPARATIVOS...")
    
    # Dados Buy & Hold
    retorno_buy_hold = (df_teste['close'].iloc[-1] / df_teste['close'].iloc[0] - 1) * 100
    bh_dates = np.linspace(0, len(resultados_lstm['curva_patrimonio'])-1, len(df_teste))
    bh_values = CAPITAL_INICIAL * (df_teste['close'] / df_teste['close'].iloc[0])
    
    # === 1️⃣ COMPARAÇÃO DE RETORNOS ===
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 3, 1)
    modelos = ['LSTM', 'CNN1D', 'Buy & Hold']
    retornos = [resultados_lstm['retorno_total'], resultados_cnn['retorno_total'], retorno_buy_hold]
    cores = ['blue', 'red', 'green']
    bars = plt.bar(modelos, retornos, color=cores, alpha=0.7)
    plt.title('Comparação de Retornos Totais')
    plt.ylabel('Retorno (%)')
    plt.grid(alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, retorno in zip(bars, retornos):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if retorno >= 0 else -3),
                f'{retorno:.2f}%', ha='center', va='bottom' if retorno >= 0 else 'top', fontweight='bold')

    # === 2️⃣ CURVAS DE PATRIMÔNIO ===
    plt.subplot(2, 3, 2)
    plt.plot(resultados_lstm['curva_patrimonio'], label='LSTM', linewidth=2, color='blue')
    plt.plot(resultados_cnn['curva_patrimonio'], label='CNN1D', linewidth=2, color='red')
    plt.plot(bh_dates, bh_values, label='Buy & Hold', linestyle='--', linewidth=2, color='green')
    plt.title('Curvas de Patrimônio')
    plt.xlabel('Período')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(alpha=0.3)

    # === 3️⃣ MÉTRICAS DE REGRESSÃO ===
    plt.subplot(2, 3, 3)
    metricas = ['MAE', 'RMSE', 'R²']
    lstm_vals = [resultados_lstm['metricas']['MAE'], resultados_lstm['metricas']['RMSE'], resultados_lstm['metricas']['R2']]
    cnn_vals = [resultados_cnn['metricas']['MAE'], resultados_cnn['metricas']['RMSE'], resultados_cnn['metricas']['R2']]
    
    x = np.arange(len(metricas))
    width = 0.35
    plt.bar(x - width/2, lstm_vals, width, label='LSTM', alpha=0.7, color='blue')
    plt.bar(x + width/2, cnn_vals, width, label='CNN1D', alpha=0.7, color='red')
    plt.title('Métricas de Regressão')
    plt.xticks(x, metricas)
    plt.legend()
    plt.grid(alpha=0.3)

    # === 4️⃣ TAXA DE ACERTO E TRADES ===
    plt.subplot(2, 3, 4)
    categorias = ['Taxa Acerto', 'Total Trades', 'Lucro Médio']
    lstm_trade = [resultados_lstm['trades']['taxa_acerto'], resultados_lstm['trades']['total_trades'], resultados_lstm['trades']['lucro_medio_liquido']]
    cnn_trade = [resultados_cnn['trades']['taxa_acerto'], resultados_cnn['trades']['total_trades'], resultados_cnn['trades']['lucro_medio_liquido']]
    
    x = np.arange(len(categorias))
    plt.bar(x - width/2, lstm_trade, width, label='LSTM', alpha=0.7, color='blue')
    plt.bar(x + width/2, cnn_trade, width, label='CNN1D', alpha=0.7, color='red')
    plt.title('Estatísticas de Trading')
    plt.xticks(x, categorias)
    plt.legend()
    plt.grid(alpha=0.3)

    # === 5️⃣ PREVISÕES vs REAL ===
    plt.subplot(2, 3, 5)
    plt.plot(resultados_lstm['datas_teste'], resultados_lstm['valores_reais'], label='Real', linewidth=2, color='black')
    plt.plot(resultados_lstm['datas_teste'], resultados_lstm['previsoes'], label='LSTM', linewidth=1, alpha=0.8, color='blue')
    plt.plot(resultados_cnn['datas_teste'], resultados_cnn['previsoes'], label='CNN1D', linewidth=1, alpha=0.8, color='red')
    plt.title('Previsões vs Valor Real')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    # === 6️⃣ EVOLUÇÃO DO TREINAMENTO ===
    plt.subplot(2, 3, 6)
    plt.plot(resultados_lstm['historico']['loss'], label='LSTM Treino', alpha=0.7, color='blue')
    plt.plot(resultados_lstm['historico']['val_loss'], label='LSTM Validação', linestyle='--', alpha=0.7, color='lightblue')
    plt.plot(resultados_cnn['historico']['loss'], label='CNN1D Treino', alpha=0.7, color='red')
    plt.plot(resultados_cnn['historico']['val_loss'], label='CNN1D Validação', linestyle='--', alpha=0.7, color='pink')
    plt.title('Evolução da Loss no Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f"COMPARACAO_LSTM_vs_CNN1D_{TICKER}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # === GRÁFICO ADICIONAL: DRAWDOWN COMPARATIVO ===
    plt.figure(figsize=(12, 6))
    
    # Calcular drawdown para LSTM
    peak_lstm = np.maximum.accumulate(resultados_lstm['curva_patrimonio'])
    drawdown_lstm = (peak_lstm - resultados_lstm['curva_patrimonio']) / peak_lstm * 100
    
    # Calcular drawdown para CNN1D
    peak_cnn = np.maximum.accumulate(resultados_cnn['curva_patrimonio'])
    drawdown_cnn = (peak_cnn - resultados_cnn['curva_patrimonio']) / peak_cnn * 100
    
    plt.plot(drawdown_lstm, label='LSTM Drawdown', linewidth=2, color='blue', alpha=0.7)
    plt.plot(drawdown_cnn, label='CNN1D Drawdown', linewidth=2, color='red', alpha=0.7)
    plt.title('Comparação de Drawdown (Risco)')
    plt.xlabel('Período')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"COMPARACAO_DRAWDOWN_{TICKER}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # === GRÁFICO ADICIONAL: DISTRIBUIÇÃO DOS RETORNOS POR TRADE ===
    if resultados_lstm['trades']['total_trades'] > 0 and resultados_cnn['trades']['total_trades'] > 0:
        plt.figure(figsize=(12, 5))
        
        retornos_lstm = [t['retorno_liquido'] for t in resultados_lstm['trades']['trades']]
        retornos_cnn = [t['retorno_liquido'] for t in resultados_cnn['trades']['trades']]
        
        plt.subplot(1, 2, 1)
        plt.hist(retornos_lstm, bins=20, alpha=0.7, color='blue', label='LSTM')
        plt.axvline(np.mean(retornos_lstm), color='darkblue', linestyle='--', linewidth=2, label=f'Média: {np.mean(retornos_lstm):.2f}%')
        plt.title('Distribuição Retornos - LSTM')
        plt.xlabel('Retorno por Trade (%)')
        plt.ylabel('Frequência')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(retornos_cnn, bins=20, alpha=0.7, color='red', label='CNN1D')
        plt.axvline(np.mean(retornos_cnn), color='darkred', linestyle='--', linewidth=2, label=f'Média: {np.mean(retornos_cnn):.2f}%')
        plt.title('Distribuição Retornos - CNN1D')
        plt.xlabel('Retorno por Trade (%)')
        plt.ylabel('Frequência')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"COMPARACAO_DISTRIBUICAO_RETORNOS_{TICKER}.png", dpi=300, bbox_inches='tight')
        plt.show()

# ===================== EXECUÇÃO PRINCIPAL =====================

def ExecutarComparacaoCompleta():
    print(f"\n🚀 INICIANDO COMPARAÇÃO COMPLETA LSTM vs CNN1D")
    print("=" * 70)
    
    # Baixar e preparar dados
    dados = BaixarDados(TICKER, "2019-01-01", DATA_FIM_TESTE)
    if dados is None:
        return None

    df = dados[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = CalcularIndicadores(df)

    # Separar treino e teste
    df_treino = df[(df.index >= DATA_INICIO_TREINO) & (df.index <= DATA_FIM_TREINO)]
    df_teste  = df[(df.index >= DATA_INICIO_TESTE) & (df.index <= DATA_FIM_TESTE)]

    if len(df_treino) == 0 or len(df_teste) == 0:
        print("❌ Dados de treino ou teste vazios")
        return None

    print(f"\n📊 SEPARAÇÃO TREINO/TESTE:")
    print(f"   Treino: {len(df_treino)} dias | Teste: {len(df_teste)} dias")

    # Selecionar características
    caracteristicas = SelecionarCaracteristicas(df_treino)

    # Executar ambos os modelos
    resultados_lstm = ExecutarLSTM(df_treino, df_teste, caracteristicas)
    resultados_cnn = ExecutarCNN1D(df_treino, df_teste, caracteristicas)

    # Calcular Buy & Hold
    retorno_buy_hold = (df_teste['close'].iloc[-1] / df_teste['close'].iloc[0] - 1) * 100

    # Gerar gráficos comparativos
    GerarGraficosComparativos(resultados_lstm, resultados_cnn, df_teste)

    # Relatório final
    print(f"\n🎯 ========== RELATÓRIO FINAL DE COMPARAÇÃO ==========")
    print(f"📈 RETORNOS:")
    print(f"   LSTM:       {resultados_lstm['retorno_total']:+.2f}%")
    print(f"   CNN1D:      {resultados_cnn['retorno_total']:+.2f}%")
    print(f"   Buy & Hold: {retorno_buy_hold:+.2f}%")
    
    print(f"\n📊 MÉTRICAS DE PREVISÃO:")
    print(f"   MAE  - LSTM: R$ {resultados_lstm['metricas']['MAE']:.2f} | CNN1D: R$ {resultados_cnn['metricas']['MAE']:.2f}")
    print(f"   RMSE - LSTM: R$ {resultados_lstm['metricas']['RMSE']:.2f} | CNN1D: R$ {resultados_cnn['metricas']['RMSE']:.2f}")
    print(f"   R²   - LSTM: {resultados_lstm['metricas']['R2']:.4f} | CNN1D: {resultados_cnn['metricas']['R2']:.4f}")
    
    print(f"\n💼 ESTATÍSTICAS DE TRADING:")
    print(f"   Taxa Acerto - LSTM: {resultados_lstm['trades']['taxa_acerto']:.1f}% | CNN1D: {resultados_cnn['trades']['taxa_acerto']:.1f}%")
    print(f"   Total Trades - LSTM: {resultados_lstm['trades']['total_trades']} | CNN1D: {resultados_cnn['trades']['total_trades']}")
    print(f"   Lucro Médio - LSTM: {resultados_lstm['trades']['lucro_medio_liquido']:.2f}% | CNN1D: {resultados_cnn['trades']['lucro_medio_liquido']:.2f}%")
    
    # Motivos de saída
    if resultados_lstm['trades']['total_trades'] > 0:
        print(f"\n📋 MOTIVOS DE SAÍDA - LSTM:")
        for motivo, count in resultados_lstm['trades']['motivos_saida'].items():
            print(f"   {motivo}: {count} trades")
    
    if resultados_cnn['trades']['total_trades'] > 0:
        print(f"📋 MOTIVOS DE SAÍDA - CNN1D:")
        for motivo, count in resultados_cnn['trades']['motivos_saida'].items():
            print(f"   {motivo}: {count} trades")
    
    print(f"\n💰 CAPITAL FINAL:")
    print(f"   LSTM:  R$ {resultados_lstm['capital_final']:.2f}")
    print(f"   CNN1D: R$ {resultados_cnn['capital_final']:.2f}")

    # Determinar vencedor
    if resultados_lstm['retorno_total'] > resultados_cnn['retorno_total']:
        vencedor = "LSTM"
    elif resultados_cnn['retorno_total'] > resultados_lstm['retorno_total']:
        vencedor = "CNN1D"
    else:
        vencedor = "EMPATE"
    
    print(f"\n🏆 VENCEDOR: {vencedor}")
    
    return {
        "lstm": resultados_lstm,
        "cnn": resultados_cnn,
        "buy_hold": retorno_buy_hold,
        "vencedor": vencedor
    }

# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    resultados = ExecutarComparacaoCompleta()
    
    if resultados:
        print(f"\n🎉 ✅ COMPARAÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"🏆 Modelo vencedor: {resultados['vencedor']}")
        print(f"📈 LSTM: {resultados['lstm']['retorno_total']:+.2f}%")
        print(f"📊 CNN1D: {resultados['cnn']['retorno_total']:+.2f}%")
        print(f"📉 Buy & Hold: {resultados['buy_hold']:+.2f}%")
    else:
        print(f"\n💥 ❌ Erro na execução da comparação")