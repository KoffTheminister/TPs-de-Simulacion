import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def generador2(seed, n):
  lista = []
  a = 2**16 + 3
  m = 2**31
  comp = []
  x = (a*seed)%m
  comp.append(x)
  comp.append(x/m)
  lista.append(comp)
  i = 1
  while(i <= n):
    comp = []
    x = (a*lista[i - 1][0])%m
    comp.append(x)
    comp.append(x/m)
    lista.append(comp)
    i += 1

  return lista
  # valores_enteros = [elem[0] for elem in lista]

  # valores_escala = [(x * 1000) // m for x in valores_enteros]

  # x_vals = valores_escala[:-1]
  # y_vals = valores_escala[1:]

  # # plt.figure(figsize=(12, 5))
  # # markerline, stemlines, baseline = plt.stem(range(len(valores_escala)), valores_escala)
  # # plt.setp(markerline, markersize=2)
  # # plt.setp(stemlines, linewidth=0.5, color='deepskyblue')
  # # plt.xlabel("series")
  # # plt.ylabel("valor")
  # # plt.title("Gráfico de Secuencia de Series (pseudoaleatorios)")
  # # plt.grid(True)
  # # plt.show()

  # # plt.figure(figsize=(10, 5))
  # # plt.scatter(x_vals, y_vals, color='deepskyblue', s=5)
  # # plt.xlabel("x(i)")
  # # plt.ylabel("x(i+1)")
  # # plt.title("Gráfico de Retardo")
  # # plt.grid(True)
  # # plt.show()
  
def midSquare(seed,n):
  resultados= []
  num = seed
  for i in range(n):
    x= str(num*num).zfill(8)
    mid = len (x) // 2
    num = int(x[mid-2:mid+2])
    resultados.append(num)
  return resultados
# print(midSquare(666667, 100))
print(generador2(666667, 1000))




def run_test(nMin,nMax,numeros):
  mayor = max(numeros)
  suma = 0
  num_combertidos=[]
  for n in numeros:
    num_combertidos.append(n/mayor)
  for n in num_combertidos:
    suma = n + suma 
  media = suma/len(num_combertidos)
  n_menores=[]
  n_mayores=[]
  for n in num_combertidos:
    if(n<nMax):
      n_mayores.append(n)
    elif(n>nMax):
      n_menores.append(n)
  n1 =len(n_menores)
  n2 =len(n_mayores)
  ## calculamos la media esperada    






# test de chi cuadradado
# k: numero de intervalos
# alpha: nivel de significancia
def chi_cuadrado(data, k, alpha):
  # se divide los datos en k intervalos
  data = np.array(data)
  min_val = 0
  max_val = 1
  bin_whidth = (max_val - min_val) / k
  bins = np.linspace(min_val, max_val, k + 1)

  # contar la frecuencia de cada intervalo
  frecuencias_observadas, _ = np.histogram(data, bins)

  # calcular la frecuencia esperada
  frecuencia_esperada = len(data) / k

  # calcular la estadística chi cuadrado
  chi_cuadrado_stat = np.sum((frecuencias_observadas - frecuencia_esperada) ** 2 / frecuencia_esperada)

  #calcular los grados de libertad -> grados de libertad = numero de categorias - 1
  grados_libertad = k - 1

  #calcular el valor crítico de chi cuadrado
  valor_critico = chi2.ppf(1 - alpha, grados_libertad)

  # comparar la estadística chi cuadrado con el valor crítico
  if chi_cuadrado_stat < valor_critico:
    result = ("No se rechaza la hipótesis nula: los datos siguen una distribución uniforme.")
  else:
    result = ("Se rechaza la hipótesis nula: los datos no siguen una distribución uniforme.")
  
  return chi_cuadrado_stat, valor_critico, result

# prueba del tes de chi-cuadrado 
valores_generados = generador2(666667, 1000)
numeros = [valor[1] for valor in valores_generados]

chi_cuadrado_stat, valor_critico, result = chi_cuadrado(numeros, 10, 0.05)
print(f"Estadística Chi Cuadrado: {chi_cuadrado_stat}")
print(f"Valor crítico: {valor_critico}")
print(result)
