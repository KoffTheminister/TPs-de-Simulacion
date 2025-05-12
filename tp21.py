import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import math
import random
import pandas as pd
from pandas.plotting import lag_plot

def generador2(seed, n):
  lista = np.array([])
  a = 2**16 + 3
  m = 2**31
  ultimo = (a*seed)%m
  lista = np.append(lista, ultimo/m)
  i = 1
  while(i <= n):
    ultimo = (a*ultimo)%m
    lista = np.append(lista, ultimo/m)
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
  resultados = np.array([])
  num = seed
  for i in range(n):
    x= str(num*num).zfill(8)
    mid = len (x) // 2
    num = int(x[mid-2:mid+2])
    resultados = np.append(resultados, num)
  return resultados

def run_test(nMin,nMax,numeros):
  ultimo = 0
  bandera1=False
  bandera2= False
  mayor = max(numeros)
  suma = 0
  num_combertidos=[]
  #se transforma el intervalo dado de tal manera que este entre 0 y 1 
  for n in numeros:
    num_combertidos.append(n/mayor)
  for n in num_combertidos:
    suma = n + suma 
  media = suma/len(num_combertidos)
  n_menores=[]
  n_mayores=[]
  for n in num_combertidos:
    if(n<nMax):
      bandera1=True
      n_mayores.append(n)
      ultimo = 1
    elif(n>nMax):
      bandera2= True
      n_menores.append(n)
      ultimo = 2
    if(bandera2== True and bandera1==True):
      c_cambio = c_cambio+1
      bandera1 = False
      bandera2 = False
      if(ultimo == 1):
        bandera1 = True
      if(ultimo == 2):
        bandera2 = True
  n1 =len(n_menores)
  n2 =len(n_mayores)
  ## calculamos la media esperada    
  media_esperada= ((2*n1*n2)/n1+n2)+1
  varianza = (2*n1*n2*(2*n1*n2-n1-n2))/((n1+n2)(n1+n2)(n1+n2-1))
  desvio_estandar= varianza ** 0.5
  # calculamos el valor de z
  z = (c_cambio-media_esperada)/desvio_estandar
  # se evalua a z con una eficacio de 0.005
  if(abs(z)<=1.96):
    print("prueba exitosa , se puede consideradr como aleatoria ")
  elif(abs(z)>1.96):
    print("prueba rechasada, la secuencia probablemente no es nula ")
  

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

# # prueba del tes de chi-cuadrado 
# valores_generados = generador2(666667, 1000)
# numeros = [valor[1] for valor in valores_generados]

# chi_cuadrado_stat, valor_critico, result = chi_cuadrado(numeros, 10, 0.05)
# print(f"Estadística Chi Cuadrado: {chi_cuadrado_stat}")
# print(f"Valor crítico: {valor_critico}")
# print(result)



def calculateAreaUnderUni(x_min, x_max, x):
    return (1 / (x_max - x_min)) * (x - x_min)

def kolmogorovSmirnovTest(samples, n_divs):
  samples = np.sort(samples)
  histo = np.zeros(n_divs)
  x_min = min(samples)
  x_max = max(samples)
  divSize = (x_max - x_min) / n_divs
  for s in samples:
    index = int((s - x_min) / divSize)
    if index >= n_divs:
      index = n_divs - 1
    histo[index] += 1

  total = len(samples)
  emp_cdf = np.cumsum(histo) / total

  maxDiff = 0
  for j in range(n_divs):
      x_val = x_min + (j + 1) * divSize 
      theor_cdf = calculateAreaUnderUni(x_min, x_max, x_val)
      diff = abs(emp_cdf[j] - theor_cdf)
      if diff > maxDiff:
          maxDiff = diff
  return maxDiff

#print(kolmogorovSmirnovTest(np.random.uniform(0, 1, size=1000), 10))
      
def andersonDarlingTest(samples, n_divs):
  samples = np.sort(samples)
  x_min = min(samples)
  x_max = max(samples)
  divSize = (x_max - x_min) / n_divs
  aSquared = 0
  n = len(samples)
  for s in range(len(samples)):
    F_i = calculateAreaUnderUni(x_min, x_max, samples[s])
    F_n_i = calculateAreaUnderUni(x_min, x_max, samples[n - s - 1])

    if F_i <= 0 or F_n_i >= 1:
      # avoid log(0) and division by zero
      continue

    term = (2 * s + 1) * (math.log(F_i) + math.log(1 - F_n_i))
    aSquared += term
  aSquared = -len(samples) -(1/len(samples))*aSquared
  return aSquared

def secDeSeries(xs):
  plt.figure(figsize=(10, 4))
  plt.plot(xs, marker='o', linestyle='-', color='steelblue')
  plt.title("Secuencia de una Serie Aleatoria")
  plt.xlabel("Índice (t)")
  plt.ylabel("Valor")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def histograma(xs, min, max):
  bins = np.linspace(min, max, 51)

  counts, bin_edges = np.histogram(xs, bins=bins)

  relative_freq = counts / len(xs)

  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

  plt.figure(figsize=(10, 4))
  plt.bar(bin_centers, relative_freq, width=bin_edges[1]-bin_edges[0],
          color='cornflowerblue', edgecolor='black')
  plt.axhline(y=0.02, color='red', linestyle='--', linewidth=1.5, label='Frecuencia esperada')
  plt.title("Histograma de Frecuencia Relativa (por bin)")
  plt.xlabel("Intervalo")
  plt.ylabel("Frecuencia relativa")
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.show()

def lagPlot(xs, m):
    # Escalar los valores para visualización discreta
    valores_escala = [(x * 1000) // m for x in xs]

    x_vals = valores_escala[:-1]
    y_vals = valores_escala[1:]
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, y_vals, color='deepskyblue', s=5)
    plt.xlabel("x(i)")
    plt.ylabel("x(i+1)")
    plt.title("Gráfico de Retardo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

lista = np.array([])
n = 50 #numero de iteraciones
m = 10000 #numero de muestras
for i in range(m):
  x = generador2(i, n) #generador
  lista = np.append(lista, x[n]) #en el generador2 poner x[n] y en el otro poner x[n - 1]
min = min(lista)
max = max(lista)
#histograma(lista, min, max)
lagPlot(lista, m)

