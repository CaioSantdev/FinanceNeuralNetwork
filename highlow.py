import yfinance as yf
import matplotlib.pyplot as plt

# Lista de tickers
tickers = ["PETR4.SA"]

# Baixando os dados históricos para o ticker PETR4
dados = yf.download(tickers, start="2019-01-01", end="2019-12-31")

# Extraindo os dados de High e Low para o ticker PETR4
high = dados['High']
low = dados['Low']

# Plotando o gráfico com as linhas de High e Low
plt.figure(figsize=(10,6))
plt.plot(high, label='Pontos Altos (High)', color='green')
plt.plot(low, label='Pontos Baixos (Low)', color='red')

# Adicionando título e rótulos aos eixos
# plt.title('Pontos Altos e Baixos de PETR4 em 2019', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço (R$)', fontsize=12)

# Adicionando legenda
plt.legend()

# Salvando o gráfico como um arquivo PNG
plt.savefig('petr4_high_low_2019.png')

# Exibindo uma mensagem indicando que o gráfico foi salvo
print("Gráfico salvo como 'petr4_high_low_2019.png'.")

# Caso queira, pode abrir o gráfico com plt.show() (em ambientes interativos)
# plt.show()
