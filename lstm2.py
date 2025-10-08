# ===================== CONFIGURAÇÕES GERAIS LSTM =====================
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
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
EPOCAS = 100
TAMANHO_LOTE = 16

# Gestão de Risco
STOP_LOSS_PORCENTAGEM = 0.05
TAKE_PROFIT_PORCENTAGEM = 0.08
PORCENTAGEM_MAXIMA_POR_TRADE = 0.20
MAXIMO_DIAS_POSICAO = 10

# Taxas
TAXA_CORRETAGEM = 0.005
TAXA_EMOLUMENTOS = 0.0005
TAXA_LIQUIDACAO = 0.0002
IMPOSTO_DAY_TRADE = 0.20

print(f"=== CONFIGURAÇÃO LSTM MELHORADA PARA {TICKER} ===")

# ===================== DOWNLOAD DOS DADOS =====================
def BaixarDados(ticker, data_inicio, data_fim):
    print(f"Baixando dados de {ticker}...")
    dados = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=True, progress=False)
    if dados.empty:
        print("❌ Erro: Não foi possível baixar os dados")
        return None
    print(f"✅ Dados baixados: {len(dados)} períodos")
    return dados

# ===================== CALCULAR INDICADORES MELHORADOS =====================
def CalcularIndicadoresMelhorados(df):
    df = df.copy()
    
    # RETORNOS HISTÓRICOS
    df['Retorno_1'] = df['close'].pct_change(1)
    df['Retorno_5'] = df['close'].pct_change(5)
    df['Retorno_10'] = df['close'].pct_change(10)
    
    # VOLUME FEATURES (CRÍTICAS)
    df['Volume_SMA_5'] = df['volume'].rolling(5).mean()
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    df['Volume_Alto'] = (df['Volume_Ratio'] > 1.5).astype(int)
    
    # PRICE FEATURES
    df['High_Low_Ratio'] = (df['high'] / df['low']) - 1
    df['Close_Open_Ratio'] = (df['close'] / df['open']) - 1
    
    # INDICADORES TÉCNICOS COM SHIFT (evitar lookahead)
    df['SMA_10'] = ta.sma(df['close'], length=10).shift(1)
    df['SMA_20'] = ta.sma(df['close'], length=20).shift(1)
    df['SMA_50'] = ta.sma(df['close'], length=50).shift(1)
    
    df['EMA_12'] = ta.ema(df['close'], length=12).shift(1)
    df['EMA_26'] = ta.ema(df['close'], length=26).shift(1)
    
    # RSI
    df['RSI_14'] = ta.rsi(df['close'], length=14).shift(1)
    
    # MACD
    macd = ta.macd(df['close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0].shift(1)
        df['MACD_Sinal'] = macd.iloc[:, 1].shift(1)
        df['MACD_Histograma'] = macd.iloc[:, 2].shift(1)
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20)
    if bb is not None:
        df['BB_Superior'] = bb.iloc[:, 2].shift(1)
        df['BB_Inferior'] = bb.iloc[:, 0].shift(1)
        df['BB_Medio'] = bb.iloc[:, 1].shift(1)
        df['BB_Largura'] = (df['BB_Superior'] - df['BB_Inferior']) / df['BB_Medio']
        df['BB_Posicao'] = (df['close'] - df['BB_Inferior']) / (df['BB_Superior'] - df['BB_Inferior'])
    
    # ATR
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(1)
    
    # TARGET: RETORNO FUTURO (5 dias) - ISSO É FUNDAMENTAL!
    df['Target_Retorno_5d'] = (df['close'].shift(-5) / df['close'] - 1)
    
    # FEATURES DE TENDÊNCIA
    df['Tendencia_SMA_10_20'] = (df['SMA_10'] / df['SMA_20'] - 1)
    df['Tendencia_EMA_12_26'] = (df['EMA_12'] / df['EMA_26'] - 1)
    
    # VOLATILIDADE
    df['Volatilidade_5d'] = df['Retorno_1'].rolling(5).std()
    df['Volatilidade_20d'] = df['Retorno_1'].rolling(20).std()
    
    # MOMENTUM
    df['Momentum_5d'] = (df['close'] / df['close'].shift(5) - 1)
    df['Momentum_10d'] = (df['close'] / df['close'].shift(10) - 1)
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"✅ Indicadores melhorados calculados: {len(df)} períodos")
    print(f"✅ Target: Retorno 5 dias futuros")
    
    return df

# ===================== SELECIONAR CARACTERÍSTICAS RELEVANTES =====================
def SelecionarCaracteristicasRelevantes(df_treino):
    # Features prioritárias baseadas em importância
    caracteristicas_prioritarias = [
        'Retorno_1', 'Retorno_5', 'Volume_Ratio', 'Volume_Alto',
        'RSI_14', 'MACD', 'MACD_Histograma', 'BB_Posicao',
        'Momentum_5d', 'Tendencia_SMA_10_20', 'Volatilidade_5d',
        'Close_Open_Ratio', 'High_Low_Ratio'
    ]
    
    # Verificar quais existem no DataFrame
    caracteristicas_disponiveis = [col for col in caracteristicas_prioritarias if col in df_treino.columns]
    
    print("✅ Características selecionadas:", caracteristicas_disponiveis)
    return caracteristicas_disponiveis

# ===================== PREPARAR DADOS PARA RETORNO FUTURO =====================
def PrepararDadosParaRetorno(df_treino, df_teste, caracteristicas):
    escalonador_x = StandardScaler()
    escalonador_y = StandardScaler()
    
    # USAR RETORNO FUTURO COMO TARGET (não preço absoluto!)
    y_treino = df_treino['Target_Retorno_5d'].values.reshape(-1, 1)
    y_teste = df_teste['Target_Retorno_5d'].values.reshape(-1, 1)
    
    X_treino_escalonado = escalonador_x.fit_transform(df_treino[caracteristicas])
    X_teste_escalonado = escalonador_x.transform(df_teste[caracteristicas])
    
    y_treino_escalonado = escalonador_y.fit_transform(y_treino)
    y_teste_escalonado = escalonador_y.transform(y_teste)
    
    def CriarSequencias(X, y, janela):
        X_seq, y_seq = [], []
        for i in range(janela, len(X)):
            X_seq.append(X[i-janela:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_treino, y_treino_seq = CriarSequencias(X_treino_escalonado, y_treino_escalonado, JANELA_TEMPORAL)
    X_teste, y_teste_seq = CriarSequencias(X_teste_escalonado, y_teste_escalonado, JANELA_TEMPORAL)
    
    print(f"✅ Dados preparados - Treino: {X_treino.shape}, Teste: {X_teste.shape}")
    print(f"✅ Target: Retorno futuro escalonado")
    
    return X_treino, X_teste, y_treino_seq, y_teste_seq, escalonador_y

# ===================== MODELO LSTM MELHORADO =====================
def CriarModeloLSTMMelhorado(formato_entrada):
    modelo = Sequential([
        # Primeira camada LSTM com return_sequences
        LSTM(128, return_sequences=True, input_shape=formato_entrada, 
             dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Segunda camada LSTM
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Terceira camada LSTM
        LSTM(32, return_sequences=False, dropout=0.2),
        BatchNormalization(),
        
        # Camadas densas
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Saída: retorno previsto
    ])
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("✅ Modelo LSTM melhorado criado")
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

# ===================== ESTRATÉGIA DE TRADING INTELIGENTE =====================
def GerarSinalInteligente(df, data_atual, retorno_previsto):
    """Gera sinal baseado em MÚLTIPLOS fatores (não apenas preço)"""
    
    if data_atual not in df.index:
        return "MANTER"
    
    try:
        preco_atual = df.loc[data_atual, 'close']
        rsi = df.loc[data_atual, 'RSI_14']
        volume_ratio = df.loc[data_atual, 'Volume_Ratio']
        macd = df.loc[data_atual, 'MACD']
        macd_sinal = df.loc[data_atual, 'MACD_Sinal']
        bb_posicao = df.loc[data_atual, 'BB_Posicao']
        acima_media_20 = preco_atual > df.loc[data_atual, 'SMA_20']
        
        # CRITÉRIOS MÚLTIPLOS
        criterio_retorno = retorno_previsto > 0.015  # 1.5% de retorno esperado
        criterio_rsi = 35 < rsi < 65  # Zona neutra (nem sobrecomprado nem sobrevendido)
        criterio_volume = volume_ratio > 1.2  # Volume 20% acima da média
        criterio_macd = macd > macd_sinal  # MACD acima da linha de sinal
        criterio_bb = 0.2 < bb_posicao < 0.8  # Dentro das Bollinger Bands (não extremo)
        criterio_tendencia = acima_media_20  # Acima da média de 20 períodos
        
        # PONTUAÇÃO (sistema de pontos)
        pontuacao = 0
        if criterio_retorno: pontuacao += 3
        if criterio_rsi: pontuacao += 2
        if criterio_volume: pontuacao += 2  # Volume tem peso alto!
        if criterio_macd: pontuacao += 1
        if criterio_bb: pontuacao += 1
        if criterio_tendencia: pontuacao += 1
        
        # DECISÃO BASEADA NA PONTUAÇÃO
        if pontuacao >= 6:  # Limiar alto para evitar trades ruins
            return "COMPRAR"
        elif pontuacao <= 2:
            return "VENDER"
        else:
            return "MANTER"
            
    except KeyError as e:
        print(f"⚠️ Erro nos dados para {data_atual}: {e}")
        return "MANTER"

# ===================== BACKTEST COM ESTRATÉGIA INTELIGENTE =====================
def BacktestEstrategiaInteligente(df, datas, previsoes_retorno, capital_inicial=CAPITAL_INICIAL):
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
    
    print(f"\n🔍 INICIANDO BACKTEST LSTM - ESTRATÉGIA INTELIGENTE")
    sinais_compra = 0
    trades_executados = 0

    for i, data_atual in enumerate(datas):
        if data_atual not in df.index:
            continue

        preco_atual = df.loc[data_atual, 'close']
        retorno_previsto = previsoes_retorno[i]
        
        # USAR ESTRATÉGIA MULTIVARIADA
        sinal = GerarSinalInteligente(df, data_atual, retorno_previsto)
        
        if trade_ativo:
            dias_em_posicao += 1
            
            # CALCULAR RETORNO ATUAL
            retorno_atual = (preco_atual - preco_entrada) / preco_entrada
            
            # CONDIÇÕES DE SAÍDA
            stop_loss = retorno_atual <= -STOP_LOSS_PORCENTAGEM
            take_profit = retorno_atual >= TAKE_PROFIT_PORCENTAGEM
            saida_tempo = dias_em_posicao >= MAXIMO_DIAS_POSICAO
            sinal_venda = sinal == "VENDER"
            
            if stop_loss or take_profit or saida_tempo or sinal_venda:
                sinal = "VENDER"
                motivo = "STOP_LOSS" if stop_loss else "TAKE_PROFIT" if take_profit else "TEMPO" if saida_tempo else "SINAL"

        # EXECUTAR COMPRA
        if sinal == "COMPRAR" and not trade_ativo and capital > preco_atual:
            capital_para_trade = min(capital, capital_inicial * PORCENTAGEM_MAXIMA_POR_TRADE)
            
            if capital_para_trade > preco_atual:
                valor_pos_taxas, taxas_compra = AplicarTaxasCompra(capital_para_trade)
                acoes_compra = valor_pos_taxas / preco_atual
                acoes = acoes_compra
                capital -= capital_para_trade
                preco_entrada = preco_atual
                data_entrada = data_atual
                trade_ativo = True
                dias_em_posicao = 0
                trades_executados += 1
                sinais_compra += 1
                
                trades.append({
                    "data": data_atual, "acao": "COMPRAR", "preco": preco_atual,
                    "acoes": acoes_compra, "taxas": taxas_compra, 
                    "valor_total": valor_pos_taxas, "retorno_previsto": retorno_previsto
                })
                total_taxas_pagas += taxas_compra
                
                print(f"💰 COMPRA LSTM: {data_atual} | Preço: {preco_atual:.2f} | Retorno Previsto: {retorno_previsto:.2%}")

        # EXECUTAR VENDA
        elif sinal == "VENDER" and trade_ativo and acoes > 0:
            valor_venda_bruto = acoes * preco_atual
            custo_compra = acoes * preco_entrada
            eh_day_trade = (data_entrada.date() == data_atual.date())
            
            valor_liquido, taxas_venda, imposto = AplicarTaxasVenda(
                valor_venda_bruto, custo_compra, eh_day_trade
            )
            
            capital += valor_liquido
            retorno_real = (valor_liquido / custo_compra - 1) * 100
            
            trades.append({
                "data": data_atual, "acao": "VENDER", "preco": preco_atual,
                "acoes": acoes, "retorno_real": retorno_real,
                "preco_entrada": preco_entrada, "taxas": taxas_venda,
                "eh_day_trade": eh_day_trade, "valor_liquido": valor_liquido,
                "dias_posicao": dias_em_posicao, "motivo": motivo
            })
            
            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            
            print(f"💸 VENDA LSTM ({motivo}): {data_atual} | Preço: {preco_atual:.2f} | Retorno Real: {retorno_real:+.2f}%")
            
            acoes = 0
            trade_ativo = False
            dias_em_posicao = 0

        # Atualizar curva de patrimônio
        valor_portfolio = capital + (acoes * preco_atual if acoes > 0 else 0)
        curva_patrimonio.append(valor_portfolio)
    
    # Fechar posição aberta no final
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
            retorno_real = (valor_liquido / custo_compra - 1) * 100
            
            trades.append({
                "data": ultima_data, "acao": "VENDER", "preco": ultimo_preco,
                "retorno_real": retorno_real, "preco_entrada": preco_entrada,
                "taxas": taxas_venda, "valor_liquido": valor_liquido,
                "fechamento_forcado": True
            })
            
            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            
            print(f"🔚 FECHAMENTO FORÇADO LSTM: {ultima_data} | Retorno: {retorno_real:+.2f}%")
    
    print(f"\n📊 RESUMO ESTRATÉGIA LSTM:")
    print(f"   Sinais de compra gerados: {sinais_compra}")
    print(f"   Trades executados: {trades_executados}")
    print(f"   Total de operações: {len(trades)}")
    
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
            retorno_real = trade_venda['retorno_real']
            taxas_operacao = trade_compra.get('taxas', 0) + trade_venda.get('taxas', 0)
            
            resultados_trades.append({
                'retorno_real': retorno_real,
                'taxas': taxas_operacao,
                'dias_posicao': trade_venda.get('dias_posicao', 0),
                'fechamento_forcado': trade_venda.get('fechamento_forcado', False),
                'retorno_previsto': trade_compra.get('retorno_previsto', 0)
            })
            
            total_taxas += taxas_operacao
            if retorno_real > 0:
                trades_vencedores += 1

    if resultados_trades:
        taxa_acerto = (trades_vencedores / len(resultados_trades)) * 100
        lucro_medio = np.mean([t['retorno_real'] for t in resultados_trades])
        
        return {
            "total_trades": len(resultados_trades),
            "taxa_acerto": taxa_acerto,
            "lucro_medio": lucro_medio,
            "total_taxas_pagas": total_taxas,
            "trades": resultados_trades
        }
    
    return {
        "total_trades": 0, "taxa_acerto": 0, 
        "lucro_medio": 0, "total_taxas_pagas": 0,
        "trades": []
    }

# ===================== VALIDAR PREVISÕES =====================
def ValidarPrevisoesRetorno(valores_reais, previsoes, datas_teste, nome_modelo="LSTM"):
    plt.figure(figsize=(12, 6))
    plt.plot(datas_teste, valores_reais, label='Retorno Real 5d', linewidth=2, color='blue')
    plt.plot(datas_teste, previsoes, label=f'Retorno Previsto {nome_modelo}', linewidth=1, alpha=0.7, color='red')
    plt.title(f'Comparação: Retorno Real vs Previsto {nome_modelo} ({TICKER})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calcular métricas de retorno
    correlacao = np.corrcoef(valores_reais, previsoes)[0,1]
    acerto_direcional = np.mean(np.sign(valores_reais) == np.sign(previsoes)) * 100
    
    print(f"📊 VALIDAÇÃO RETORNOS {nome_modelo}:")
    print(f"   Correlação: {correlacao:.4f}")
    print(f"   Acerto direcional: {acerto_direcional:.1f}%")
    print(f"   Retorno real médio: {np.mean(valores_reais):.4f}")
    print(f"   Retorno previsto médio: {np.mean(previsoes):.4f}")

# ===================== EXECUTOR LSTM MELHORADO =====================
def ExecutarLSTMMelhorado():
    print(f"\n🎯 INICIANDO MODELO LSTM MELHORADO PARA {TICKER}")
    print("=" * 70)
    
    # Baixar dados
    dados = BaixarDados(TICKER, "2019-01-01", DATA_FIM_TESTE)
    if dados is None:
        return None

    # Preparar dados com indicadores melhorados
    df = dados[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = CalcularIndicadoresMelhorados(df)

    # Separar treino e teste
    df_treino = df[(df.index >= DATA_INICIO_TREINO) & (df.index <= DATA_FIM_TREINO)]
    df_teste  = df[(df.index >= DATA_INICIO_TESTE) & (df.index <= DATA_FIM_TESTE)]

    if len(df_treino) == 0 or len(df_teste) == 0:
        print("❌ Dados de treino ou teste vazios")
        return None

    print(f"\n📊 SEPARAÇÃO TREINO/TESTE:")
    print(f"   Treino: {len(df_treino)} dias | Teste: {len(df_teste)} dias")

    # Selecionar características relevantes
    caracteristicas = SelecionarCaracteristicasRelevantes(df_treino)
    
    # Preparar dados para prever RETORNO (não preço)
    X_treino, X_teste, y_treino, y_teste, escalonador_y = PrepararDadosParaRetorno(
        df_treino, df_teste, caracteristicas
    )

    # Criar e treinar modelo LSTM
    modelo = CriarModeloLSTMMelhorado((X_treino.shape[1], X_treino.shape[2]))

    print(f"\n🔥 Treinando modelo LSTM para prever retornos...")
    historico = modelo.fit(
        X_treino, y_treino,
        epochs=EPOCAS,
        batch_size=TAMANHO_LOTE,
        validation_data=(X_teste, y_teste),
        verbose=1,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
    )

    # Fazer previsões de RETORNO
    print(f"\n📈 Prevendo retornos futuros com LSTM...")
    previsoes_escalonadas = modelo.predict(X_teste, verbose=0)
    previsoes_retorno = escalonador_y.inverse_transform(previsoes_escalonadas).flatten()
    
    # Obter retornos reais para comparação
    retornos_reais = escalonador_y.inverse_transform(y_teste).flatten()
    datas_teste = df_teste.index[JANELA_TEMPORAL:JANELA_TEMPORAL + len(previsoes_retorno)]

    # Métricas de regressão para RETORNO
    mse = mean_squared_error(retornos_reais, previsoes_retorno)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(retornos_reais, previsoes_retorno)
    r2 = r2_score(retornos_reais, previsoes_retorno)

    print(f"\n✅ ========== MÉTRICAS DE PREVISÃO DE RETORNO (LSTM) ==========")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    # Validar previsões
    ValidarPrevisoesRetorno(retornos_reais, previsoes_retorno, datas_teste, "LSTM")

    # Backtest com estratégia inteligente
    print(f"\n💼 Executando estratégia de trading inteligente com LSTM...")
    curva_patrimonio, trades, capital_final, total_taxas, total_impostos = BacktestEstrategiaInteligente(
        df_teste, datas_teste, previsoes_retorno, CAPITAL_INICIAL
    )
    
    retorno_total = (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    retorno_buy_hold = (df_teste['close'].iloc[-1] / df_teste['close'].iloc[0] - 1) * 100
    analise_trades = AnalisarTrades(trades)

    print(f"\n💰 ========== RESULTADOS ESTRATÉGIA LSTM INTELIGENTE ==========")
    print(f"Capital inicial: R$ {CAPITAL_INICIAL:.2f}")
    print(f"Capital final:   R$ {capital_final:.2f}")
    print(f"Retorno líquido: {retorno_total:+.2f}%")
    print(f"Buy & Hold:      {retorno_buy_hold:+.2f}%")
    
    print(f"\n📊 --- ESTATÍSTICAS DE TRADING ---")
    print(f"Nº Total de Trades:    {analise_trades['total_trades']}")
    print(f"Taxa de Acerto:        {analise_trades['taxa_acerto']:.1f}%")
    print(f"Lucro médio por trade: {analise_trades['lucro_medio']:.2f}%")
    print(f"Total em taxas:        R$ {total_taxas:.2f}")

    # Gráfico final
    plt.figure(figsize=(12, 6))
    plt.plot(curva_patrimonio, label='LSTM - Estratégia Inteligente', linewidth=2, color='purple')
    
    # Buy & Hold
    bh_dates = np.linspace(0, len(curva_patrimonio)-1, len(df_teste))
    bh_values = CAPITAL_INICIAL * (df_teste['close'] / df_teste['close'].iloc[0])
    plt.plot(bh_dates, bh_values, label='Buy & Hold', linestyle='--', linewidth=2, color='blue')
    
    plt.title(f'LSTM Melhorado vs Buy & Hold ({TICKER})')
    plt.xlabel('Período de Teste')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "retorno_total": retorno_total,
        "retorno_buy_hold": retorno_buy_hold,
        "metricas": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "trades": analise_trades,
        "capital_final": capital_final,
        "nome_modelo": "LSTM"
    }

# ===================== COMPARAÇÃO ENTRE MODELOS =====================
def CompararModelos(resultados_cnn, resultados_lstm):
    if not resultados_cnn or not resultados_lstm:
        print("❌ Não é possível comparar - resultados faltando")
        return
    
    print(f"\n🏆 ========== COMPARAÇÃO CNN1D vs LSTM ==========")
    print(f"{'Métrica':<25} {'CNN1D':<12} {'LSTM':<12} {'Vencedor':<10}")
    print("-" * 60)
    
    # Retorno Total
    retorno_cnn = resultados_cnn['retorno_total']
    retorno_lstm = resultados_lstm['retorno_total']
    vencedor_retorno = "CNN1D" if retorno_cnn > retorno_lstm else "LSTM" if retorno_lstm > retorno_cnn else "EMPATE"
    print(f"{'Retorno Líquido (%)':<25} {retorno_cnn:<12.2f} {retorno_lstm:<12.2f} {vencedor_retorno:<10}")
    
    # R²
    r2_cnn = resultados_cnn['metricas']['R2']
    r2_lstm = resultados_lstm['metricas']['R2']
    vencedor_r2 = "CNN1D" if r2_cnn > r2_lstm else "LSTM" if r2_lstm > r2_cnn else "EMPATE"
    print(f"{'R²':<25} {r2_cnn:<12.4f} {r2_lstm:<12.4f} {vencedor_r2:<10}")
    
    # MAE
    mae_cnn = resultados_cnn['metricas']['MAE']
    mae_lstm = resultados_lstm['metricas']['MAE']
    vencedor_mae = "CNN1D" if mae_cnn < mae_lstm else "LSTM" if mae_lstm < mae_cnn else "EMPATE"
    print(f"{'MAE':<25} {mae_cnn:<12.4f} {mae_lstm:<12.4f} {vencedor_mae:<10}")
    
    # Taxa de Acerto
    acerto_cnn = resultados_cnn['trades']['taxa_acerto']
    acerto_lstm = resultados_lstm['trades']['taxa_acerto']
    vencedor_acerto = "CNN1D" if acerto_cnn > acerto_lstm else "LSTM" if acerto_lstm > acerto_cnn else "EMPATE"
    print(f"{'Taxa Acerto (%)':<25} {acerto_cnn:<12.1f} {acerto_lstm:<12.1f} {vencedor_acerto:<10}")
    
    # Número de Trades
    trades_cnn = resultados_cnn['trades']['total_trades']
    trades_lstm = resultados_lstm['trades']['total_trades']
    vencedor_trades = "CNN1D" if trades_cnn > trades_lstm else "LSTM" if trades_lstm > trades_cnn else "EMPATE"
    print(f"{'Nº Trades':<25} {trades_cnn:<12} {trades_lstm:<12} {vencedor_trades:<10}")

# ===================== EXECUÇÃO =====================
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("🚀 INICIANDO LSTM MELHORADO - ESTRATÉGIA INTELIGENTE")
    print("=" * 70)
    
    resultados_lstm = ExecutarLSTMMelhorado()
    
    if resultados_lstm:
        print(f"\n🎉 ✅ Execução LSTM concluída com sucesso!")
        print(f"📈 Retorno LSTM: {resultados_lstm['retorno_total']:+.2f}%")
        print(f"📊 Retorno Buy&Hold: {resultados_lstm['retorno_buy_hold']:+.2f}%")
        print(f"🎯 Taxa de Acerto: {resultados_lstm['trades']['taxa_acerto']:.1f}%")
        print(f"💰 Trades Executados: {resultados_lstm['trades']['total_trades']}")
        
        # Perguntar se quer comparar com CNN1D
        comparar = input("\n🔍 Deseja executar o modelo CNN1D para comparação? (s/n): ")
        if comparar.lower() == 's':
            try:
                # Importar e executar CNN1D
                from cnn1d_melhorado import ExecutarCNN1DMelhorado
                resultados_cnn = ExecutarCNN1DMelhorado()
                if resultados_cnn:
                    CompararModelos(resultados_cnn, resultados_lstm)
            except ImportError:
                print("❌ Módulo CNN1D não encontrado. Execute o código CNN1D separadamente.")
    else:
        print(f"\n💥 ❌ Erro na execução LSTM")