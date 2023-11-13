# Importação das bibliotecas necessárias para a execução do código
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Importação do arquivo Excel por meio do seu caminho
caminho_arquivo = r'C:\Users\danie\Downloads\títulos utilizados (1) (2).xlsx'

# Carregamento do arquivo Excel em um DataFrame do pandas
dados_excel = pd.read_excel(caminho_arquivo)

# Carregamento de todas as abas do Excel em um dicionário de DataFrames
dados_excel = pd.read_excel(caminho_arquivo, sheet_name=None)
def calcular_retorno_na_aba(arquivo, aba, coluna_data, coluna_pu):
    # Carregar da aba específica do arquivo Excel em um DataFrame do pandas
    dados_aba = pd.read_excel(arquivo, sheet_name=aba)

    dados_aba = dados_aba.sort_values(by=coluna_data)

    # Calcular o retorno a partir do primeiro e último valor da coluna PU
    primeiro_valor = dados_aba[coluna_pu].iloc[0]
    ultimo_valor = dados_aba[coluna_pu].iloc[-1]

    retorno = (ultimo_valor - primeiro_valor) / primeiro_valor * 100

    return retorno


nome_coluna_datas = 'Data'
nome_coluna_pu = 'PU'

# Abas (ativos)
abas = ['AALM11', 'VIXL35', 'BRSTNCLTN814', 'BRSTNCNTB054']

# Calcular e exibir o retorno para cada ativo
for aba in abas:
    retorno_calculado = calcular_retorno_na_aba(caminho_arquivo, aba, nome_coluna_datas, nome_coluna_pu)
    print(f"Retorno do ativo '{aba}': {retorno_calculado:.2f}%")



# Risco dos ativos: 
def calcular_desvio_padrao_na_aba(arquivo, aba, coluna):
    # Carregar a aba específica do arquivo Excel em um DataFrame do pandas
    dados_aba = pd.read_excel(arquivo, sheet_name=aba)

    # Calcular o desvio padrão da coluna especificada
    desvio_padrao = dados_aba[coluna].std()

    return desvio_padrao





nome_coluna_rd = 'RD'
abas = ['AALM11', 'VIXL35', 'BRSTNCLTN814', 'BRSTNCNTB054']

# Chamar a função e exibir o resultado com 6 casas decimais para cada aba
for aba in abas:
    desvio_padrao_rd = calcular_desvio_padrao_na_aba(caminho_arquivo, aba, nome_coluna_rd)
    print(f"Risco do ativo '{aba}': {desvio_padrao_rd:.6f}")



# Função para calcular o índice de Sharpe
def calcular_indice_sharpe(pesos, retornos, riscos):
    retorno_carteira = np.dot(retornos, pesos)
    risco_carteira = np.sqrt(np.dot(np.dot(pesos, riscos), pesos.T))
    indice_sharpe = retorno_carteira / risco_carteira
    return -indice_sharpe  # Negativo porque estamos maximizando


# Função para calcular o retorno da carteira
def calcular_retorno_carteira(pesos, retornos):
    return np.dot(retornos, pesos)


# Abas (ativos) - Privada
abas_privada = ['AALM11', 'VIXL35']

# Dados relevantes para cálculos - Privada
retornos_ativos_privada = []
riscos_ativos_privada = []

# Preencher retornos e riscos para cada ativo na carteira privada
for aba in abas_privada:
    retorno_calculado = calcular_retorno_na_aba(caminho_arquivo, aba, 'Data', 'PU')
    desvio_padrao_rd = calcular_desvio_padrao_na_aba(caminho_arquivo, aba, 'RD')
    retornos_ativos_privada.append(retorno_calculado)
    riscos_ativos_privada.append(desvio_padrao_rd)

# Pesos iniciais
pesos_iniciais_privada = np.array([0.5, 0.5])

# Otimização para encontrar os pesos que maximizam o índice de Sharpe
resultado_privada = minimize(calcular_indice_sharpe, pesos_iniciais_privada, args=(retornos_ativos_privada, np.diag(riscos_ativos_privada)), bounds=[(0, 1), (0, 1)])

# Retorno da carteira privada
retorno_privada = calcular_retorno_carteira(resultado_privada.x, retornos_ativos_privada)

# Exibir resultados
print(f"Pesos otimizados para a 'Carteira Privada': {resultado_privada.x}")
print(f"Retorno da 'Carteira Privada': {retorno_privada:.2f}%")


# Abas (ativos) - Pública
abas_publica = ['BRSTNCLTN814', 'BRSTNCNTB054']

# Dados relevantes para cálculos - Pública
retornos_ativos_publica = []
riscos_ativos_publica = []

# Preencher retornos e riscos para cada ativo na carteira pública
for aba in abas_publica:
    retorno_calculado = calcular_retorno_na_aba(caminho_arquivo, aba, 'Data', 'PU')
    desvio_padrao_rd = calcular_desvio_padrao_na_aba(caminho_arquivo, aba, 'RD')
    retornos_ativos_publica.append(retorno_calculado)
    riscos_ativos_publica.append(desvio_padrao_rd)



# Pesos iniciais
pesos_iniciais_publica = np.array([0.5, 0.5])

# Otimização para encontrar os pesos que maximizam o índice de Sharpe - Pública
resultado_publica = minimize(calcular_indice_sharpe, pesos_iniciais_publica, args=(retornos_ativos_publica, np.diag(riscos_ativos_publica)), bounds=[(0, 1), (0, 1)])

# Retorno da carteira pública
retorno_publica = calcular_retorno_carteira(resultado_publica.x, retornos_ativos_publica)



# Exibir resultados
print(f"Pesos otimizados para a 'Carteira Pública': {resultado_publica.x}")
print(f"Retorno da 'Carteira Pública': {retorno_publica:.2f}%")

# Calcular o retorno ponderado total
retorno_total = (retorno_privada + retorno_publica)/2

# Exibir o resultado
print(f"Retorno esperado total das carteiras: {retorno_total:.2f}%")




# Taxa livre de risco (substitua pelo valor apropriado)
taxa_livre_risco = 0.000465503460602376 #rendimento diário do CDI

# Calcular índice de Sharpe para a carteira privada
risco_privada = np.sqrt(np.dot(np.dot(resultado_privada.x, np.diag(riscos_ativos_privada)), resultado_privada.x.T))
sharpe_privada = (retorno_privada/1000 - taxa_livre_risco) / risco_privada

# Calcular índice de Sharpe para a carteira pública
risco_publica = np.sqrt(np.dot(np.dot(resultado_publica.x, np.diag(riscos_ativos_publica)), resultado_publica.x.T))
sharpe_publica = (retorno_publica/1000 - taxa_livre_risco) / risco_publica

# Exibir resultados
print(f"Índice de Sharpe da 'Carteira Privada': {sharpe_privada:.4f}")
print(f"Índice de Sharpe da'Carteira Pública': {sharpe_publica:.4f}")

# Calcular o índice de Treynor para a carteira privada
indice_treynor_privada = (retorno_privada/100 - taxa_livre_risco) / resultado_privada.hess_inv.todense()[0, 0]

# Calcular o índice de Treynor para a carteira pública
indice_treynor_publica = (retorno_publica/100 - taxa_livre_risco) / resultado_publica.hess_inv.todense()[0, 0]

# Calcular o desvio padrão da carteira privada
desvio_padrao_privada = np.sqrt(np.dot(np.dot(resultado_privada.x, riscos_ativos_privada), resultado_privada.x.T))

# Calcular o desvio padrão da carteira pública
desvio_padrao_publica = np.sqrt(np.dot(np.dot(resultado_publica.x, riscos_ativos_publica), resultado_publica.x.T))

# Exibir resultados
print(f"Índice de Treynor da 'Carteira Privada': {indice_treynor_privada:.4f}")
print(f"Índice de Treynor da 'Carteira Pública': {indice_treynor_publica:.4f}")

# Calcular o índice de M² para a carteira privada
indice_m2_privada = (retorno_privada - taxa_livre_risco) * resultado_privada.hess_inv.todense()[0, 0] / desvio_padrao_privada**2

# Calcular o índice de M² para a carteira pública
indice_m2_publica = (retorno_publica - taxa_livre_risco) * resultado_publica.hess_inv.todense()[0, 0] / desvio_padrao_publica**2

print(f"Índice de M² da 'Carteira Privada': {indice_m2_privada}")
print(f"Índice de M² da 'Carteira Pública': {indice_m2_publica}")


# Gráfico comparando o índice de treynor das duas carteiras
plt.figure(figsize=(12, 6))
plt.bar(['Carteira Privada', 'Carteira Pública'], [indice_treynor_privada,indice_treynor_publica], color=['blue', 'green'])
plt.title('Comparação do Índice de treynor das Carteiras')
plt.xlabel('Carteira')
plt.ylabel('Índice de treynor')
plt.show()


# Gráfico para o índice de M²
plt.figure(figsize=(12, 6))

# Scatter plot para a carteira privada
plt.scatter(indice_m2_privada[0], indice_m2_privada[1], marker='o', color='blue', label='Carteira Privada', s=100)

# Scatter plot para a carteira pública
plt.scatter(indice_m2_publica[0], indice_m2_publica[1], marker='o', color='red', label='Carteira Pública', s=100)

# Adicionar rótulos e título
plt.xlabel('Índice de M²')
plt.title('Índice de M² - Carteira Privada vs Carteira Pública')
plt.legend()
plt.grid(True)
plt.show()


# Gráfico comparando o índice de Sharpe das duas carteiras
plt.figure(figsize=(12, 6))
plt.bar(['Carteira Privada', 'Carteira Pública'], [sharpe_privada, sharpe_publica], color=['blue', 'green'])
plt.title('Comparação do Índice de Sharpe das Carteiras')
plt.xlabel('Carteira')
plt.ylabel('Índice de Sharpe')
plt.show()
