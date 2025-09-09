import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

# ===================== Parâmetros =====================
TICKER = "PETR4.SA"
START  = "2019-01-01"
END    = "2019-12-31"

# ===================== Extração e Preparação dos Dados =====================
print(f"Baixando dados do ticker: {TICKER} para o ano de 2019...")
try:
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"Nenhum dado retornado para {TICKER}.")
except Exception as e:
    print(f"Erro ao baixar dados: {e}")
    exit()

# CORREÇÃO: Remove MultiIndex das colunas
df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.columns = ['open', 'high', 'low', 'close', 'volume']

print("Limpando e preparando os dados...")
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Preenche valores ausentes
df = df.ffill().dropna().copy()

# Estabiliza a escala do volume com log (evita log(0))
df["volume"] = np.log1p(df["volume"])

# Verifica se há dados suficientes
print(f"Total de registros em 2019: {len(df)}")
print(f"Período: {df.index.min()} até {df.index.max()}")

# ===================== Calcula os Indicadores =====================
print("Calculando indicadores técnicos...")

# RSI (Relative Strength Index)
df['rsi'] = ta.rsi(df['close'], length=14)

# MACD (Moving Average Convergence Divergence)
macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
if macd_data is not None:
    df['macd'] = macd_data['MACD_12_26_9']
    df['macd_signal'] = macd_data['MACDs_12_26_9']
    df['macd_hist'] = macd_data['MACDh_12_26_9']

# ADX (Average Directional Index)
adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
if adx_data is not None:
    df['adx'] = adx_data['ADX_14']

# ATR (Average True Range)
atr_data = ta.atr(df['high'], df['low'], df['close'], length=14)
if atr_data is not None:
    df['atr'] = atr_data

# Remove apenas as primeiras linhas que ficaram com NaN dos indicadores
df = df.dropna()

print(f"\nDataFrame final tem {df.shape[0]} linhas e {df.shape[1]} colunas.")

# ===================== Exibe Resultados =====================
if not df.empty:
    print("\n--- Primeiras 5 linhas ---")
    print(df.head())
    print("\n--- Últimas 5 linhas ---")
    print(df.tail())
    
    print(f"\nValores NaN por coluna:")
    print(df.isna().sum())
    
    print(f"\nEstatísticas descritivas dos preços:")
    print(df[['open', 'high', 'low', 'close']].describe())
    
    print(f"\nEstatísticas descritivas dos indicadores:")
    if 'rsi' in df.columns:
        print(df[['rsi', 'macd', 'adx', 'atr']].describe())

# ===================== Informações Adicionais =====================
print(f"\n=== RESUMO DO ANO DE 2019 ===")
print(f"Dias de trading: {len(df)}")
print(f"Preço de abertura do ano: R$ {df['open'].iloc[0]:.2f}")
print(f"Preço de fechamento do ano: R$ {df['close'].iloc[-1]:.2f}")
print(f"Variação anual: {((df['close'].iloc[-1] / df['open'].iloc[0] - 1) * 100):.2f}%")
print(f"Preço máximo do ano: R$ {df['high'].max():.2f}")
print(f"Preço mínimo do ano: R$ {df['low'].min():.2f}")

# Verifica se os indicadores foram calculados
print(f"\nVerificação dos indicadores:")
print(f"RSI calculado: {not df['rsi'].isna().all()}")
print(f"MACD calculado: {not df['macd'].isna().all() if 'macd' in df.columns else 'Não'}")
print(f"ADX calculado: {not df['adx'].isna().all() if 'adx' in df.columns else 'Não'}")
print(f"ATR calculado: {not df['atr'].isna().all() if 'atr' in df.columns else 'Não'}")

# ===================== Salva em CSV (opcional) =====================
try:
    df.to_csv('petr4_2019_com_indicadores.csv')
    print("\nDados salvos em 'petr4_2019_com_indicadores.csv'")
except Exception as e:
    print(f"\nErro ao salvar CSV: {e}")