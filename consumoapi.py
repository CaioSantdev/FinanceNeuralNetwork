import yfinance as yf

# Lista de tickers para ações da B3, por exemplo
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]

# Baixando os dados históricos para todos os tickers
dados = yf.download(tickers, start="2015-01-01", end="2023-12-31")

# Exibindo os primeiros dados
print(dados.head())
