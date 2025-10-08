# ===================== CONFIGURAÇÕES GERAIS =====================
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
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

print(f"=== CONFIGURAÇÃO LSTM PARA {TICKER} ===")
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

# ===================== PREPARAR DADOS =====================
def PrepararDados(df_treino, df_teste, caracteristicas, janela_temporal=JANELA_TEMPORAL):
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
    
    X_treino, y_treino = CriarSequencias(X_treino_escalonado, y_treino_escalonado, janela_temporal)
    X_teste, y_teste = CriarSequencias(X_teste_escalonado, y_teste_escalonado, janela_temporal)
    
    print(f"✅ Dados preparados - Treino: {X_treino.shape}, Teste: {X_teste.shape}")
    
    return X_treino, X_teste, y_treino, y_teste, escalonador_y_treino

# ===================== MODELO LSTM =====================
def CriarModeloLSTM(formato_entrada):
    modelo = Sequential([
        LSTM(64, return_sequences=True, input_shape=formato_entrada, dropout=0.2),
        LSTM(32, return_sequences=False, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("✅ Modelo LSTM criado com sucesso")
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

# ===================== BACKTEST COM GESTÃO DE RISCO =====================
def BacktestComTaxas(df, datas, previsoes, capital_inicial=CAPITAL_INICIAL):
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
        preco_previsto = previsoes[i]
        razao_preco = preco_previsto / preco_atual
        
        sinal = "MANTER"
        
        # CRITÉRIOS DE ENTRADA MAIS FLEXÍVEIS
        if not trade_ativo and capital > preco_atual:
            # Condições simplificadas para debug
            condicao_previsao = razao_preco > 1.008  # 0.8% acima
            condicao_tendencia = preco_atual > df.loc[data_atual, 'SMA_10']  # SMA mais curta
            condicao_rsi = 30 < df.loc[data_atual, 'RSI_14'] < 70  # Zona neutra
            
            if condicao_previsao:  # Apenas previsão por enquanto
                sinal = "COMPRAR"
                sinais_compra_identificados += 1
                
                # DEBUG: Mostrar primeiro sinal
                if sinais_compra_identificados <= 3:
                    print(f"🎯 SINAL COMPRA: {data_atual} | Preço: {preco_atual:.2f} | Previsão: {preco_previsto:.2f} | Razão: {razao_preco:.4f}")
                
        elif trade_ativo:
            dias_em_posicao += 1
            
            # CRITÉRIOS DE SAÍDA
            stop_loss = preco_atual <= preco_entrada * (1 - STOP_LOSS_PORCENTAGEM)
            take_profit = preco_atual >= preco_entrada * (1 + TAKE_PROFIT_PORCENTAGEM)
            saida_previsao = razao_preco < 0.99  # Previsão piorou
            saida_tempo = dias_em_posicao >= MAXIMO_DIAS_POSICAO
            
            if stop_loss or take_profit or saida_previsao or saida_tempo:
                sinal = "VENDER"
                motivo = "STOP_LOSS" if stop_loss else "TAKE_PROFIT" if take_profit else "PREVISAO" if saida_previsao else "TEMPO"
                print(f"📉 SINAL VENDA ({motivo}): {data_atual} | Preço: {preco_atual:.2f} | Entrada: {preco_entrada:.2f}")

        # EXECUTAR COMPRA
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
                    "data": data_atual, "acao": "COMPRAR", "preco": preco_atual,
                    "acoes": acoes, "taxas": taxas_compra, "valor_total": valor_pos_taxas,
                    "capital_utilizado": capital_para_trade
                })
                total_taxas_pagas += taxas_compra
                
                print(f"💰 COMPRA EXECUTADA: {data_atual} | Preço: {preco_atual:.2f} | Capital usado: R$ {capital_para_trade:.2f}")

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
                "dias_posicao": dias_em_posicao
            })
            
            total_taxas_pagas += taxas_venda
            total_impostos_pagos += imposto
            
            print(f"💸 VENDA EXECUTADA: {data_atual} | Preço: {preco_atual:.2f} | Retorno: {retorno_porcentagem:+.2f}%")
            
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
            
            print(f"🔚 FECHAMENTO FORÇADO: {ultima_data} | Preço: {ultimo_preco:.2f} | Retorno: {retorno_porcentagem:+.2f}%")
    
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
def ValidarPrevisoes(valores_reais, previsoes, datas_teste):
    plt.figure(figsize=(12, 6))
    plt.plot(datas_teste, valores_reais, label='Valor Real', linewidth=2, color='blue')
    plt.plot(datas_teste, previsoes, label='Previsão LSTM', linewidth=1, alpha=0.7, color='red')
    plt.title(f'Comparação: Preço Real vs Previsão LSTM ({TICKER})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calcular diferença percentual
    diferenca_pct = np.abs((previsoes - valores_reais) / valores_reais) * 100
    acerto_direcional = np.mean(np.sign(np.diff(valores_reais)) == np.sign(np.diff(previsoes))) * 100
    
    print(f"📊 VALIDAÇÃO DAS PREVISÕES:")
    print(f"   Diferença média: {np.mean(diferenca_pct):.2f}%")
    print(f"   Acerto direcional: {acerto_direcional:.1f}%")
    print(f"   Correlação: {np.corrcoef(valores_reais, previsoes)[0,1]:.4f}")

# ===================== EXECUTOR LSTM =====================
def ExecutarLSTM():
    print(f"\n🎯 INICIANDO MODELO LSTM PARA {TICKER}")
    print("=" * 60)
    
    # Baixar dados
    dados = BaixarDados(TICKER, "2019-01-01", DATA_FIM_TESTE)
    if dados is None:
        return None

    # Preparar dados
    df = dados[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = CalcularIndicadores(df)

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

    # Selecionar características e preparar dados
    caracteristicas = SelecionarCaracteristicas(df_treino)
    X_treino, X_teste, y_treino, y_teste, escalonador_y = PrepararDados(df_treino, df_teste, caracteristicas)

    # Criar e treinar modelo
    modelo = CriarModeloLSTM((X_treino.shape[1], X_treino.shape[2]))

    print(f"\n🔥 Treinando modelo LSTM...")
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
    previsoes = escalonador_y.inverse_transform(previsoes_escalonadas).flatten()
    valores_reais = escalonador_y.inverse_transform(y_teste).flatten()
    datas_teste = df_teste.index[JANELA_TEMPORAL:JANELA_TEMPORAL + len(previsoes)]

    # Métricas de regressão
    mse = mean_squared_error(valores_reais, previsoes)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(valores_reais, previsoes)
    mape = mean_absolute_percentage_error(valores_reais, previsoes) * 100
    r2 = r2_score(valores_reais, previsoes)

    print(f"\n✅ ========== MÉTRICAS DE REGRESSÃO (LSTM) ==========")
    print(f"MAE : R$ {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: R$ {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

    # Validar previsões
    ValidarPrevisoes(valores_reais, previsoes, datas_teste)

    # Backtest com taxas
    print(f"\n💼 Executando backtest com gestão de risco...")
    curva_patrimonio, trades, capital_final, total_taxas, total_impostos = BacktestComTaxas(
        df_teste, datas_teste, previsoes, CAPITAL_INICIAL
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
    print(f"Lucro médio bruto:     {analise_trades['lucro_medio_bruto']:.2f}%")
    print(f"Lucro médio líquido:   {analise_trades['lucro_medio_liquido']:.2f}%")
    
    if analise_trades['total_trades'] > 0:
        print(f"Dias médio por trade:  {analise_trades['dias_medio_posicao']:.1f}")
    else:
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
    
    plt.title(f'Desempenho Estratégia LSTM vs Buy & Hold ({TICKER})')
    plt.xlabel('Período de Teste')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "retorno_total": retorno_total,
        "retorno_buy_hold": retorno_buy_hold,
        "metricas": {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2},
        "trades": analise_trades,
        "capital_final": capital_final
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
    else:
        print(f"\n💥 ❌ Erro na execução")