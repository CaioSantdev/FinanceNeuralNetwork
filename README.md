# README — `lstmB3_comparativos.py`

## Visão geral

O arquivo `lstmB3_comparativos.py` implementa um pipeline completo de previsão de preços com **LSTM** para ações bancárias da B3, seguido por **backtests de múltiplas estratégias** e uma **comparação entre experimentos com 5 anos e 10 anos de histórico**.

O fluxo geral do script é:

1. Baixar dados históricos do Yahoo Finance.
2. Fazer o pré-processamento dos preços.
3. Gerar uma feature adicional de EMA sem vazamento temporal.
4. Separar treino e teste preservando a ordem temporal.
5. Normalizar os dados com `StandardScaler`.
6. Montar janelas temporais para a LSTM.
7. Treinar a rede neural.
8. Gerar previsões no conjunto de teste.
9. Avaliar métricas de regressão.
10. Testar estratégias de trading sobre os preços previstos.
11. Comparar os resultados dos cenários de 5 anos e 10 anos.
12. Salvar gráficos e um CSV-resumo.

---

## Bibliotecas e requisitos

### Bibliotecas principais usadas pelo script

- `numpy`
- `pandas`
- `yfinance`
- `tensorflow`
- `scikit-learn`
- `matplotlib`

### Imports do código

O script usa especificamente:

- `numpy` e `pandas` para manipulação numérica e tabular
- `yfinance` para download dos dados históricos
- `tensorflow.keras` para a arquitetura LSTM
- `StandardScaler` do `scikit-learn` para normalização
- `mean_absolute_error` e `r2_score` para avaliação
- `matplotlib` para geração dos gráficos

### Requisitos do ambiente

O projeto já possui um `requirements.txt` com as dependências fixadas. As bibliotecas mais relevantes para este script são:

```txt
numpy==1.23.5
pandas==2.3.3
yfinance==0.2.66
tensorflow==2.10.1
keras==2.10.0
scikit-learn==1.6.1
matplotlib==3.9.4
```

### Instalação

Se quiser instalar tudo pelo arquivo existente:

```bash
pip install -r requirements.txt
```

Se quiser instalar apenas o necessário para este script:

```bash
pip install numpy pandas yfinance tensorflow scikit-learn matplotlib
```

---

## Ativos suportados

O menu principal oferece os seguintes tickers:

- `ITUB4.SA` — Itaú
- `BBDC4.SA` — Bradesco
- `BBAS3.SA` — Banco do Brasil
- `SANB11.SA` — Santander
- `BPAC11.SA` — BTG Pactual

---

## Estrutura lógica do código

## 1. Download dos dados

A função `fetch_hist()` baixa os dados históricos com `yfinance` até a data fixa **2020-12-31**.

Campos utilizados:

- `Open`
- `High`
- `Low`
- `Close`
- `AdjClose`
- `Volume`

Isso garante que os experimentos sejam reproduzíveis no mesmo período histórico.

---

## 2. Pré-processamento

A função `zanotto_preprocess()` faz o tratamento principal dos dados.

### Ajuste de preços

Ela calcula um fator:

```python
factor = AdjClose / Close
```

Depois aplica esse fator em:

- `Open`
- `High`
- `Low`
- `Close`

Na prática, o `Close` passa a ser o `AdjClose`, e os demais preços são ajustados na mesma proporção.

### Tratamento de dados faltantes

- `Open`, `High`, `Low` e `Close` passam por interpolação linear
- `Volume` com zero é convertido em `NaN` e preenchido com `ffill()` e `bfill()`

### Objetivo desse pré-processamento

Esse passo tenta produzir uma série historicamente consistente para treino, reduzindo distorções causadas por eventos corporativos e por falhas nos dados.

---

## 3. Controle de vazamento temporal

O script toma alguns cuidados importantes para evitar leakage:

### Split temporal ordenado

A função `train_test_split_ordered()` divide os dados em treino e teste sem embaralhar.

- treino: primeiros 80%
- teste: últimos 20%

### EMA sem vazamento

A função `compute_ema_no_leakage()` calcula a EMA do treino normalmente e depois propaga o valor para o teste usando apenas informação passada.

Isso evita que a EMA do teste seja contaminada por valores futuros.

### Normalização correta

O `StandardScaler` é ajustado **somente no treino**:

```python
scaler.fit_transform(train_df)
scaler.transform(test_df)
```

Isso também evita vazamento entre treino e teste.

---

## 4. Features utilizadas

Cada experimento usa as seguintes features:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `EMA_{span}`

A variável alvo é:

- `Close`

### Janela temporal

O modelo usa `window=50`, ou seja, cada amostra de entrada contém os **50 passos anteriores**.

A função `make_windows()` transforma a série escalada em:

- `X`: janelas temporais
- `y`: valor futuro do `Close`

---

## 5. Arquitetura da rede neural

A função `build_lstm()` cria uma rede com a seguinte estrutura:

1. camada de entrada com formato `(n_steps, n_features)`
2. `LSTM(500, return_sequences=True)`
3. `Dropout(0.3)`
4. `LSTM(500)`
5. `Dropout(0.3)`
6. `Dense(1)`

### Compilação

- otimizador: `adam`
- função de perda: `mse`

### Treinamento

Parâmetros usados:

- `epochs=50`
- `batch_size=64`
- `validation_split=0.1`
- `shuffle=False`
- `EarlyStopping(patience=10, restore_best_weights=True)`

O `shuffle=False` é coerente com séries temporais, preservando a ordem dos dados.

---

## 6. Métricas de regressão

Após a previsão, o script reverte a escala apenas da variável alvo e calcula:

- **RMSE** — raiz do erro quadrático médio
- **MAE** — erro absoluto médio
- **MAPE** — erro percentual absoluto médio
- **R²** — coeficiente de determinação

Essas métricas medem a qualidade da previsão do preço.

---

## 7. Estratégias implementadas

O script não para na regressão: ele transforma as previsões em sinais operacionais e faz backtest.

### 1. `gap_1%`
Compra se a previsão estiver pelo menos **1% acima** do preço real anterior e vende a descoberto se estiver pelo menos **1% abaixo**.

### 2. `gap_0.7%`
Mesma lógica da anterior, mas com limiar de **0,7%**.

### 3. `trend_filter`
Usa EMA como filtro de tendência:

- `BUY` se previsão > EMA e preço anterior > EMA
- `SELL_SHORT` se previsão < EMA e preço anterior < EMA

### 4. `mean_reversion`
Busca reversão à média:

- `BUY` se o preço anterior estiver abaixo de `98%` da EMA
- `SELL_SHORT` se estiver acima de `102%` da EMA

### 5. `always_in`
Mantém o sistema sempre posicionado:

- `BUY` se previsão > preço anterior
- `SELL_SHORT` caso contrário

---

## 8. Lógica do backtest

A função `backtest_discrete_short()` simula operações com:

- capital inicial: `R$ 5.000`
- taxa por operação: `0.03%`
- stop loss: `2%`
- take profit: `5%`
- lote mínimo: `100` ações

### Características do backtest

- suporta posição comprada e vendida
- controla entrada e saída
- fecha posição por stop ou take profit
- calcula curva de patrimônio (`equity`)
- registra trades executados

### Benchmark

Também é calculado um `buy and hold` pela função `backtest_buy_and_hold()`.

---

## 9. Comparação entre 5 anos e 10 anos

A função `run_comparison_pipeline()` roda dois experimentos independentes:

- experimento com **5 anos** de histórico
- experimento com **10 anos** de histórico

Para cada um deles, o script:

- treina uma LSTM
- calcula métricas de regressão
- testa as 5 estratégias
- gera ranking por ROI
- compara com buy and hold

Ao final, o código cria visualizações comparativas entre os dois cenários.

---

## 10. Arquivos gerados

O diretório de saída segue este padrão:

```txt
resultados/comparativo_<NomeDoBanco>/
```

Exemplos de saídas geradas:

### Gráficos individuais por experimento

- `loss_<ticker>_<anos>anos.png`
- `errors_<ticker>_<anos>anos.png`
- `error_distribution_<ticker>_<anos>anos.png`

### Gráficos comparativos

- `comparison_real_vs_pred_clean_<ticker>.png`
- `comparison_real_vs_pred_<ticker>.png`
- `comparison_strategies_<ticker>.png`
- `comparison_regression_metrics_<ticker>.png`
- `comparison_best_equity_<ticker>.png`

### CSV resumo

- `resumo_comparativo_<ticker>.csv`

---

## 11. Pipeline resumido

```text
Escolha do ticker
    ↓
Download dos dados históricos
    ↓
Ajuste dos preços e limpeza
    ↓
Split temporal treino/teste
    ↓
Cálculo da EMA sem leakage
    ↓
Padronização com StandardScaler
    ↓
Criação de janelas temporais
    ↓
Treinamento da LSTM
    ↓
Previsão no conjunto de teste
    ↓
Cálculo das métricas
    ↓
Geração de sinais operacionais
    ↓
Backtest das estratégias
    ↓
Comparação 5 anos vs 10 anos
    ↓
Salvamento de gráficos e CSV
```

---

## 12. Como executar

Na raiz do projeto:

```bash
python lstmB3_comparativos.py
```

Depois selecione uma opção de `1` a `5` no menu.

---

## 13. Principais parâmetros do script

Os principais hiperparâmetros atualmente são:

- `window=50`
- `test_ratio=0.2`
- `initial_capital=5000`
- `units=500`
- `dropout=0.3`
- `epochs=50`
- `batch_size=64`
- `patience=10`

---

## 14. Pontos fortes da implementação

- separação temporal correta entre treino e teste
- uso de scaler ajustado apenas no treino
- cálculo de EMA sem vazamento
- comparação entre diferentes horizontes históricos
- avaliação não só por erro de previsão, mas também por retorno financeiro
- benchmark com buy and hold
- geração automática de gráficos e resumo CSV

---

## 15. Limitações importantes

Apesar de ser um pipeline bem estruturado, há limitações relevantes:

- usa apenas dados até **2020-12-31**
- trabalha com um universo pequeno de ativos
- não considera slippage
- a simulação de custos é simplificada
- o backtest é feito sobre a série prevista/real do teste, não sobre execução intradiária real
- não há validação walk-forward múltipla
- a arquitetura é fixa e não faz busca de hiperparâmetros

---

## 16. Quando esse código é útil

Esse script é útil para:

- estudos acadêmicos de séries temporais financeiras
- comparação entre horizontes históricos de treinamento
- avaliação de previsões com tradução para sinais operacionais
- análise exploratória de estratégias baseadas em LSTM

---

## 17. Resumo final

O `lstmB3_comparativos.py` é um experimento de **forecasting + backtest** para ações bancárias da B3. Ele combina:

- previsão com LSTM
- pré-processamento de preços ajustados
- engenharia simples de feature com EMA
- comparação entre 5 e 10 anos de histórico
- ranking de estratégias por ROI
- comparação com buy and hold
- geração automática de gráficos e CSV

Se o objetivo for documentação técnica do pipeline, este script já está bem organizado em blocos: **dados → pré-processamento → modelagem → avaliação → estratégias → comparação**.
