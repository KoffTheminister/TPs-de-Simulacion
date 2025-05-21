import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import chi2
from scipy.stats import norm, poisson

def gcl(seed, n):
  lista = []#lista = np.array([])
  a = 1664525
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
  lista = gcl(1236 + random.randint(1, 10000), k*n)#lista = gcl(123654, k*n)
  nuevaLista = []
  for i in range(1, n, 2):
    nuevaLista.append(np.sqrt(-2*math.log(lista[i]))*math.cos(2*math.pi*lista[i+1]))
    nuevaLista.append(np.sqrt(-2*math.log(lista[i]))*math.sin(2*math.pi*lista[i+1]))
  nuevaLista = np.array(nuevaLista)
  nuevaLista = nuevaLista*ds + media
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

def calculateAreaUnderNormal(m, ds, x):
    return norm.cdf(x, loc=m, scale=ds)

def calculateAreaUnderPoi(x, myLambda):
  expo = math.exp(-myLambda)
  suma = 0
  for i in range(x + 1):
    suma += (myLambda**i)/math.factorial(i)
  return expo*suma

def calculateAreaUnderPas(k, p, x):
  for i in range(x + 1):
    suma += math.comb(k + i - 1, i)*(p**k)*((1 - p)**i)
  return suma

#def kolmogorovSmirnovTest(samples, m, ds):
def kolmogorovSmirnovTest(samples, myLambda):
  samples = np.sort(samples)
  n = len(samples)
  emp_cdf = np.arange(1, n+1) / n
  #theor_cdf = norm.cdf(samples, loc=m, scale=ds)
  theor_cdf = []
  for x in samples:
    theor_cdf.append(calculateAreaUnderPoi(x, myLambda))
  theor_cdf = np.array(theor_cdf)
  max_diff = np.max(np.abs(emp_cdf - theor_cdf))
  return max_diff

def andersonDarlingTest(samples, m, ds):
  samples = np.sort(samples)
  n = len(samples)
  epsilon = 1e-10  # to avoid log(0)

  aSquared_sum = 0
  for i in range(n):
    Fi = norm.cdf(samples[i], loc=m, scale=ds)
    Fi = min(max(Fi, epsilon), 1 - epsilon)  # Clamp Fi to avoid log(0)
    aSquared_sum += (2 * i + 1) * (math.log(Fi) + math.log(1 - norm.cdf(samples[n - i - 1], loc=m, scale=ds)))

  A2 = -n - (aSquared_sum / n)
  return A2

def secDeSeries(xs):
  plt.figure(figsize=(10, 4))
  plt.plot(xs, marker='o', linestyle='-', color='steelblue')
  plt.title("Secuencia de una Serie Aleatoria")
  plt.xlabel("Índice (t)")
  plt.ylabel("Valor")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def histograma_normal(xs, realM, realDs):
    mini = min(xs)
    maxi = max(xs)
    bins = np.linspace(mini, maxi, 101)

    counts, bin_edges = np.histogram(xs, bins=bins)
    relative_freq = counts / len(xs)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    media = np.mean(xs)
    desviacion = np.std(xs)
    x = np.linspace(media - 4*desviacion, media + 4*desviacion, 1000)
    y = norm.pdf(x, media, desviacion)
    y *= (bin_edges[1] - bin_edges[0]) #
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=f'N({media}, {desviacion}²), funcion de densidad resultante', color='blue')

    xr = np.linspace(realM - 4*realDs, realM + 4*realDs, 1000)
    yr = norm.pdf(xr, realM, realDs)
    yr *= (bin_edges[1] - bin_edges[0]) #
    plt.plot(xr, yr, label=f'N({realM}, {realDs}²), verdadera distribucion', color='green')

    plt.bar(bin_centers, relative_freq, width=bin_edges[1]-bin_edges[0],
             color='cornflowerblue', edgecolor='black', alpha=0.7)
    
    plt.title("Histograma de Frecuencia Relativa vs Distribución Normal")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia relativa")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def histograma_poisson(xs, realLambda):
    mini = min(xs)
    maxi = max(xs)
    bins = np.linspace(mini, maxi, 101)

    counts, bin_edges = np.histogram(xs, bins=bins)
    relative_freq = counts / len(xs)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    x_vals = np.arange(mini, maxi + 1)
    pmf_vals = poisson.pmf(x_vals, mu=realLambda)
    plt.plot(x_vals, pmf_vals, 'o-', color='green', linewidth=2,
             label=f'Distribución Poisson (λ={realLambda})')

    plt.bar(bin_centers, relative_freq, width=bin_edges[1]-bin_edges[0],
             color='cornflowerblue', edgecolor='black', alpha=0.7)
    
    plt.title("Histograma de Frecuencia Relativa vs Distribución Poisson")
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

media = 0
ds = 0.1
k = 40
n = 10000
#histograma_normal(normal_gen1(media, ds, k, n), media, ds)
#histograma_normal(normal_gen2(media, ds, k, n), media, ds)

# suma = 0
# for i in range(1, 101):
#   suma += kolmogorovSmirnovTest(normal_gen2(media, i, k, n), media, i)
# print(suma/100)

# suma = 0
# for i in range(1, 101):
#   suma += andersonDarlingTest(normal_gen2(media, i, k, n), media, i)
# print(suma/100)

aLambda = 1
m = 10000
#histograma_poisson(poisson_gen(aLambda, m), aLambda)
suma = 0
for i in range(1, 101):
  suma += kolmogorovSmirnovTest(poisson_gen(aLambda, m), aLambda)
print(suma/100)




