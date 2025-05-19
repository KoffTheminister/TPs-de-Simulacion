import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import chi2
from scipy.stats import norm

def gcl(seed, n):
  lista = []#lista = np.array([])
  a = 1664525#2**16 + 3
  m = 2**32
  ultimo = (a*seed)%m
  lista.append(ultimo/m) #lista = np.append(lista, ultimo/m)
  i = 1
  while(i <= n):
    ultimo = (a*ultimo)%m
    lista.append(ultimo/m)#lista = np.append(lista, ultimo/m)
    i += 1

  return lista

def normal_gen1(media, ds, k, n): #basico
  lista = gcl(1236543, k*n)
  nuevaLista = []
  for i in range(n):
    suma = np.sum(lista[k*i:k*(i+1)])
    nuevaLista.append(ds*((12/k)**1/2)*(suma - k/2) + media)
  return nuevaLista

def normal_gen2(media, ds, k, n): #directo
  lista = gcl(123654, k*n)
  nuevaLista = []
  for i in range(1, n, 2):
    nuevaLista.append(np.sqrt(-2*math.log(lista[i]))*math.cos(2*math.pi*lista[i+1]))
    nuevaLista.append(np.sqrt(-2*math.log(lista[i]))*math.sin(2*math.pi*lista[i+1]))
  return nuevaLista

def poisson_gen(myLambda, m):
  lista = gcl(1236543, m)
  expo = math.exp(-myLambda)
  pList = []
  for n in lista:
    sum_prob = 0
    i = 0
    while True:
      term = expo * (myLambda ** i) / math.factorial(i)
      sum_prob += term
      if n < sum_prob:
        pList.append(i)
        break
      i += 1
  return pList

def pascal_gen(k, p, m):
  lista = gcl(1236543, m)
  pList = []
  for n in lista:
    sum_prob = 0
    i = 0
    while True:
      term = math.comb(k + i - 1, i)*(p**k)*((1 - p)**i)
      sum_prob += term
      if n < sum_prob:
        pList.append(i)
        break
      i += 1
  return pList

def run_test(numbers):
  mean = np.mean(numbers)
  n1 = 0
  n2 = 0
  runs = 1
  prev = numbers[0] >= mean

  for i in range(1, len(numbers)):
    curr = numbers[i] >= mean
    if curr != prev:
        runs += 1
    prev = curr
    if curr:
      n2 += 1
    else:
      n1 += 1

  if n1 == 0 or n2 == 0:
    print("Sin varianza, por lo tanto no hay run tests aplicables")
    return

  expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
  variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
  std_dev = np.sqrt(variance)
  z = (runs - expected_runs) / std_dev

  print(f"corridas observadas: {runs}")
  print(f"corridas esperadas: {expected_runs:.4f}")
  print(f"Z-value: {z:.4f}")

  if abs(z) <= 1.96:
    print("✅ run tests pasadas, la secuencia parece ser aleatoria")
  else:
    print("❌ run tests no pasadas, la secuencia parece no ser aleatoria")

# test de chi cuadradado
# k: numero de intervalos
# alpha: nivel de significancia
def chi_cuadrado(data, k, alpha):
  data = np.array(data)
  min_val = 0
  max_val = 1
  bin_whidth = (max_val - min_val) / k
  bins = np.linspace(min_val, max_val, k + 1)

  frecuencias_observadas, _ = np.histogram(data, bins)

  frecuencia_esperada = len(data) / k

  chi_cuadrado_stat = np.sum((frecuencias_observadas - frecuencia_esperada) ** 2 / frecuencia_esperada)

  grados_libertad = k - 1

  valor_critico = chi2.ppf(1 - alpha, grados_libertad)

  if chi_cuadrado_stat < valor_critico:
    result = ("No se rechaza la hipótesis nula: los datos siguen una distribución uniforme.")
  else:
    result = ("Se rechaza la hipótesis nula: los datos no siguen una distribución uniforme.")
  
  return chi_cuadrado_stat, valor_critico, result


def calculateAreaUnderUni(x_min, x_max, x):
  return (1 / (x_max - x_min)) * (x - x_min)

def calculateAreaUnderPoi(x, myLambda):
  expo = math.exp(-myLambda)
  for i in range(x + 1):
    suma += (myLambda**i)/math.factorial(i)
  return expo*suma

def calculateAreaUnderPas(k, p, x):
  for i in range(x + 1):
    suma += math.comb(k + i - 1, i)*(p**k)*((1 - p)**i)
  return suma

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

def andersonDarlingTest(samples):
  import math
  samples = np.sort(samples)
  x_min = min(samples)
  x_max = max(samples)
  n = len(samples)
  aSquared_sum = 0
  epsilon = 1e-10

  for s in range(n):
    F_i = calculateAreaUnderUni(x_min, x_max, samples[s])
    F_n_i = calculateAreaUnderUni(x_min, x_max, samples[n - s - 1])

    F_i = max(F_i, epsilon)
    F_n_i = min(F_n_i, 1 - epsilon)

    term = (2 * s + 1) * (math.log(F_i) + math.log(1 - F_n_i))
    aSquared_sum += term

  aSquared = -n - (aSquared_sum / n)
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

def histograma(xs):
    mini = min(xs)
    maxi = max(xs)

    bins = np.linspace(mini, maxi, 101)

    counts, bin_edges = np.histogram(xs, bins=bins)
    relative_freq = counts / len(xs)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    media = 0
    desviacion = 1
    # x = np.linspace(media - 4*desviacion, media + 4*desviacion, 1000)
    # y = norm.pdf(x, media, desviacion)

    plt.figure(figsize=(10, 4))
    #plt.plot(x, y, label=f'N({media}, {desviacion}²)', color='blue')
    plt.bar(bin_centers, relative_freq, width=bin_edges[1]-bin_edges[0],
            color='cornflowerblue', edgecolor='black', alpha=0.7, label='Datos')
    
    plt.title("Histograma de Frecuencia Relativa vs Distribución Normal")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia relativa")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def lagPlot(xs):
  valores_escala = [int (x * 1000) for x in xs]

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

histograma(pascal_gen(5, 0.2, 10000))
