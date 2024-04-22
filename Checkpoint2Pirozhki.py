import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de vendas
anos = np.array([2010, 2011, 2012, 2014, 2017, 2018, 2019]).reshape(-1, 1)
vendas = np.array([723, 814, 905, 1087, 1360, 1451, 1542])

# Criar e ajustar o modelo de regressão linear
modelo_regressao = LinearRegression()
modelo_regressao.fit(anos, vendas)

# Fazer previsão para 2015
previsao_2015 = modelo_regressao.predict([[2015]])
previsao_2020 = modelo_regressao.predict([[2020]])
previsao_2021 = modelo_regressao.predict([[2021]])
previsao_2022 = modelo_regressao.predict([[2022]])
previsao_2023 = modelo_regressao.predict([[2023]])

# Imprimir a previsão para 2015
print("Previsão de vendas para 2015 (com regressão linear):", previsao_2015[0])
print("Previsão de vendas para 2020 (com regressão linear):", previsao_2020[0])
print("Previsão de vendas para 2021 (com regressão linear):", previsao_2021[0])
print("Previsão de vendas para 2022 (com regressão linear):", previsao_2022[0])
print("Previsão de vendas para 2023 (com regressão linear):", previsao_2023[0])

while True:
    anodesejado = int(input("Digite o ano desejado para a projeção do número de vendas: "))
    if anodesejado < 2010:
        print("Por favor, digite um ano maior ou igual a 2010.")
    else:
        previsao_desejada = modelo_regressao.predict([[anodesejado]])
        print(f"Previsão de vendas para {anodesejado} (com regressão linear):", previsao_desejada[0])
        break

#OBS: Nesse caso o aumento das vendas é bem linear, com aumento de 91 vendas por ano, portanto o melhor modelo é o de Regressão linear.
# Para aumento exponencial de vendas, poderiamos usar regressão exponencial ou para casos mais complexos métodos de previsão de séries temporais como Holt-Winters.

