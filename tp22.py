import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import chi2
from scipy.stats import norm, poisson, nbinom
import sns

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
  lista = gcl(123654, n + 2)
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

    if prev:
        n1 += 1
    else:
        n2 += 1

    for i in range(1, len(numbers)):
        curr = numbers[i] >= mean
        if curr != prev:
            runs += 1
        prev = curr
        if curr:
            n1 += 1
        else:
            n2 += 1

    if n1 == 0 or n2 == 0:
        print("Sin varianza, por lo tanto no hay run tests aplicables")
        return

    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    std_dev = np.sqrt(variance)
    z = (runs - expected_runs) / std_dev

    print(f"Corridas observadas: {runs}")
    print(f"Corridas esperadas: {expected_runs:.4f}")
    print(f"Valor Z: {z:.4f}")

    if abs(z) <= 1.96:
        print("✅ Run test pasada, la secuencia parece ser aleatoria")
    else:
        print("❌ Run test no pasada, la secuencia parece no ser aleatoria")

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

def chi_cuadrado_poisson(data, alpha=0.05, max_k=None, lambda_esperada=None):
  data = np.array(data)
  n = len(data)

  if lambda_esperada is None:
    lambda_esperada = np.mean(data)

  valores, frec_obs = np.unique(data, return_counts=True)

  if max_k is None:
    max_k = max(valores)

  frec_esp = poisson.pmf(valores, mu=lambda_esperada) * n

  valores_grup = []
  frec_obs_grup = []
  frec_esp_grup = []

  acum_obs = 0
  acum_esp = 0
  for i in range(len(valores)):
    acum_obs += frec_obs[i]
    acum_esp += frec_esp[i]
    if acum_esp >= 5:
      valores_grup.append(valores[i])
      frec_obs_grup.append(acum_obs)
      frec_esp_grup.append(acum_esp)
      acum_obs = 0
      acum_esp = 0

  if acum_esp > 0:
    frec_obs_grup[-1] += acum_obs
    frec_esp_grup[-1] += acum_esp
  chi_stat = np.sum((np.array(frec_obs_grup) - np.array(frec_esp_grup))**2 / np.array(frec_esp_grup))

  df = len(frec_esp_grup) - 1 - (1 if lambda_esperada is None else 0)

  valor_critico = chi2.ppf(1 - alpha, df)

  if chi_stat < valor_critico:
    resultado = "✅ No se rechaza H₀: los datos siguen una distribución Poisson."
  else:
    resultado = "❌ Se rechaza H₀: los datos no siguen una distribución Poisson."

  return chi_stat, valor_critico, df, resultado

import numpy as np
from scipy.stats import nbinom, chi2

def chi_cuadrado_pascal(data, k, p, alpha=0.05):
  
  data = np.array(data)
  n = len(data)
  valores, frec_obs = np.unique(data, return_counts=True)
  frec_esp = nbinom.pmf(valores, k, p) * n
  valores_grup = []
  frec_obs_grup = []
  frec_esp_grup = []
  
  acum_obs = 0
  acum_esp = 0
  for i in range(len(valores)):
    acum_obs += frec_obs[i]
    acum_esp += frec_esp[i]
    if acum_esp >= 5:
      valores_grup.append(valores[i])
      frec_obs_grup.append(acum_obs)
      frec_esp_grup.append(acum_esp)
      acum_obs = 0
      acum_esp = 0

  if acum_esp > 0:
    frec_obs_grup[-1] += acum_obs
    frec_esp_grup[-1] += acum_esp
  chi_stat = np.sum((np.array(frec_obs_grup) - np.array(frec_esp_grup))**2 / np.array(frec_esp_grup))

  df = len(frec_esp_grup) - 1 - 0

  valor_critico = chi2.ppf(1 - alpha, df)

  if chi_stat < valor_critico:
    resultado = "✅ No se rechaza H₀: los datos siguen una distribución Pascal."
  else:
    resultado = "❌ Se rechaza H₀: los datos no siguen una distribución Pascal."

  return chi_stat, valor_critico, df, resultado

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

def kolmogorovSmirnovTest(samples, m, ds):
  samples = np.sort(samples)
  n = len(samples)
  emp_cdf = np.arange(1, n+1) / n
  theor_cdf = norm.cdf(samples, loc=m, scale=ds)
  theor_cdf = np.array(theor_cdf)
  max_diff = np.max(np.abs(emp_cdf - theor_cdf))
  return max_diff

def andersonDarlingTest(samples, m, ds):
  samples = np.sort(samples)
  n = len(samples)
  epsilon = 1e-10

  aSquared_sum = 0
  for i in range(n):
    Fi = norm.cdf(samples[i], loc=m, scale=ds)
    Fi = min(max(Fi, epsilon), 1 - epsilon)
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
  y *= (bin_edges[1] - bin_edges[0]) 
  plt.figure(figsize=(10, 4))
  plt.plot(x, y, label=f'N({media}, {desviacion}²), funcion de densidad resultante', color='blue')

  xr = np.linspace(realM - 4*realDs, realM + 4*realDs, 1000)
  yr = norm.pdf(xr, realM, realDs)
  yr *= (bin_edges[1] - bin_edges[0])
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

def histograma_pascal(xs, p, k):
  mini = min(xs)
  maxi = max(xs)
  bins = np.linspace(mini, maxi, 101)

  counts, bin_edges = np.histogram(xs, bins=bins)
  relative_freq = counts / len(xs)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

  x_vals = np.arange(mini, maxi + 1)
  pmf_vals = nbinom.pmf(x_vals, k, p)

  plt.plot(x_vals, pmf_vals, 'o-', color='green', linewidth=2,
    label=f'Distribución Pascal (k={k}, p={p:.2f})')

  plt.bar(bin_centers, relative_freq, width=bin_edges[1]-bin_edges[0],
    color='cornflowerblue', edgecolor='black', alpha=0.7)

  plt.title("Histograma de Frecuencia Relativa vs Distribución Pascal")
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

# for i in range(1, 101):
#   run_test(poisson_gen(i, n))

# for i in range(1, 101):
#   print(chi_cuadrado_poisson(poisson_gen(i, n), 0.05, 10, i))

for i in range(1, 101):
  for j in np.arange(0.2, 1, 0.2):
    run_test(pascal_gen(i, j, n))

for i in range(1, 101):
  for j in np.arange(0.2, 1, 0.2):
    print(chi_cuadrado_pascal(pascal_gen(i, j, n), i, j, 0.05))


# k = 5
# p = 0.5
# m = 10000
# histograma_pascal(pascal_gen(k, p, m), p, k)
# histograma_pascal(pascal_gen(2, 0.75, m), 0.75, 2)
# histograma_pascal(pascal_gen(5, 0.3, m), 0.3, 5)
# histograma_pascal(pascal_gen(13, 0.5, m), 0.5, 13)



def generador_uniforme(seed, n, a, b):

    r_uniforme_01 = gcl(seed, n)  # usa la función generador2
    r_uniforme_ab = a + (b - a) * r_uniforme_01  

    return r_uniforme_ab





valores = generador_uniforme(seed=123, n=1000, a=5, b=15)


plt.hist(valores, bins=20, density=True, edgecolor='black')
plt.title("Distribución uniforme en [5, 15]")
plt.xlabel("Valor")
plt.ylabel("Densidad")
plt.grid(True)
plt.show()


# Nueva función objetivo: f(x) = 2x
def f(x):
    return 2 * x

def f_uniforme(x, a, b):
    if a <= x <= b:
        return 1 / (b - a)
    return 0

def metodo_rechazo_uniforme(seed, n, a, b):
    np.random.seed(seed)
    muestras = []
    f_max = 1 / (b - a)  
    c = 1  
    while len(muestras) < n:
        x = np.random.uniform(a, b)  
        u = np.random.uniform(0, f_max)
        if u <= f_uniforme(x, a, b):
            muestras.append(x)
    return np.array(muestras)

a = 5
b = 15
n = 10000
seed = 42


muestras = metodo_rechazo_uniforme(seed, n, a, b)

plt.hist(muestras, bins=30, density=True, color='steelblue', edgecolor='black')
plt.title(f'Distribución uniforme generada con el método de rechazo en [{a}, {b}]')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)
plt.show()




# Parámetros de la distribución uniforme
a, b = 0, 1  # Intervalo [a, b]
num_muestras = 1000  # Número de muestras


datos = np.random.uniform(a, b, num_muestras)


plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')


xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.ones_like(x) / (b - a)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Uniforme"
plt.title(titulo)
plt.show()



def sim_exp_beta(beta, n=1000):
    u = np.random.uniform(0, 1, n)
    return -beta * np.log(u)


sims_exp = sim_exp_beta(2)
print("Media:", np.mean(sims_exp))

plt.hist(sims_exp, bins=np.arange(0, max(sims_exp), 0.7), density=True, edgecolor='black')
plt.title("Distribución Exponencial simulada (β=2)")
plt.xlabel("Valor")
plt.ylabel("Densidad")
plt.grid(True)
plt.show()



def f_exponencial(x, beta):
    return (1 / beta) * np.exp(-x / beta)

def metodo_rechazo_exp(beta, n, b, seed=None):
  if seed is not None:
    np.random.seed(seed)

  muestras = []
  g = 1 / b  # Uniforme en [0, b]
  c = f_exponencial(0, beta) / g  # máximo de f(x) / g(x)

  while len(muestras) < n:
    x = np.random.uniform(0, b)
    u = np.random.uniform(0, 1)
    if u <= f_exponencial(x, beta) / (c * g):
      muestras.append(x)

  return np.array(muestras)

# Simulación
beta = 2
n = 1000
b = 20
muestras = metodo_rechazo_exp(beta=beta, n=n, b=b, seed=42)

# Gráfica
plt.hist(muestras, bins=40, density=True, edgecolor='black')
x_vals = np.linspace(0, b, 100)
plt.plot(x_vals, f_exponencial(x_vals, beta), color='red', label='f(x) teórica')
plt.title("Distribución Exponencial simulada (método de rechazo)")
plt.xlabel("Valor")
plt.ylabel("Densidad")
plt.legend()
plt.grid(True)
plt.show()


def gamma_rechazo(n=1000):
    sims = np.zeros(n)
    c = 1.522  

    for i in range(n):
        while True:
            # 1. Generar Y ~ g(x) = exponencial con media 3/2 → λ = 2/3
            y = np.random.exponential(scale=3/2)  # scale = 1/lambda = 3/2
            
            # 2. Generar U ~ Uniform(0,1)
            u = np.random.uniform()
            
            # 3. Calcular f(y)/cg(y)
            f = (2 / np.sqrt(np.pi)) * np.sqrt(y) * np.exp(-y)
            g = (2 / 3) * np.exp(-2 * y / 3)
            
            if u <= f / (c * g):
                sims[i] = y
                break
    return sims

# Simular y graficar
np.random.seed(123)
sims = gamma_rechazo(1000)

plt.hist(sims, bins=30, density=True, color='lightblue', alpha=0.7)
x = np.linspace(0, np.max(sims), 200)
plt.plot(x, (2 / np.sqrt(np.pi)) * np.sqrt(x) * np.exp(-x), 'r-', linewidth=2)
plt.title("Simulaciones Gamma(3/2, 1)")
plt.show()

###########


#genera el porcentaje de exitos con los parametros ingresados 
  #TM:tamaño total de la población
  #NS: número de elementos "exitosos" en la población.
  #P: tamaño de la muestra (cantidad de extracciones).
def Hipergeometrica(TM,NS,P):
  x=0 #contador de exitos
  for i in range(P):
    r=random.random()
    if(r<NS/TM):
      s=1.0
    elif(r>NS/TM):
      s=0.0
    x=s+x
    NS=NS-1
    TM=TM-1
  print("el numero de exitos en la muestra es: ",x)
  return x
#n: numero de ensayos 
#p: proababilidad de exito 
def Binomial(n,p): #usa el metodo de ensayos de bernuli para generar la variable aleatoria
  x=0 #almacena el numero de exitos
  for i in range(n):
    r=random.random()
    if(r-p<0):
      x= x+1
  return x

import random
#n: numero de ensayos 
#p: proababilidad de exito
#k: cant de numeros
def runtestBinomial(n,p,k):
  numeros = []
  for i in range(k):
    numero=Binomial(n,p)
    numeros.append(numero)
  run_test(numeros)


   
from collections import Counter
#datos_iniciales: array de valores iniciales
#cantidad: cantidad de valores que se quieren generar
def distribucion_empirica(datos_iniciales, cantidad):
    # Paso 1: Contar frecuencia de cada valor observado
    conteo = Counter(datos_iniciales)
    total = len(datos_iniciales)

    # Paso 2: Calcular probabilidades empíricas (frecuencia relativa)
    distribucion = {valor: freq / total for valor, freq in conteo.items()}

    # Paso 3: Generar valores aleatorios según la distribución
    muestra = []
    for _ in range(cantidad):
        r = random.random()
        acumulador = 0.0
        for valor, prob in distribucion.items():
            acumulador += prob
            if r < acumulador:
                muestra.append(valor)
                break
    plt.subplot(1, 2, 2)
    sns.histplot(muestra, kde=False, stat="probability", bins=len(set(muestra)))
    plt.title("Distribución de la Muestra Generada")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia Relativa")

    return distribucion, muestra

#n:numero de ensayos
#p: probabilidad de exito
#N: numero de muestras 
def rechazo_binomial(n, p, N):
    # Función de masa de probabilidad de la binomial
    def binomial_pmf(k):
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # Calcular la constante de normalización c
    k_vals = np.arange(0, n + 1)
    pmf_vals = [binomial_pmf(k) for k in k_vals]
    c = 1 / max(pmf_vals)

    # Método de rechazo
    samples = []
    while len(samples) < N:
        k_candidate = np.random.randint(0, n + 1)
        r = np.random.uniform(0, 1)
        if r <= c * binomial_pmf(k_candidate):
            samples.append(k_candidate)
    # Visualización
    k_vals = np.arange(0, n + 1)
    pmf_vals = [math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in k_vals]
    plt.plot(k_vals, pmf_vals, 'ro-', label='Binomial teórica')
    plt.xlabel("k")
    plt.ylabel("Probabilidad")
    plt.title(f"Método de rechazo - Binomial(n={n}, p={p})")
    plt.legend()
    plt.grid(True)
    plt.show()

#genera el porcentaje de exitos con los parametros ingresados 
  #TM:tamaño total de la población
  #NS: número de elementos "exitosos" en la población.
  #P: tamaño de la muestra (cantidad de extracciones).
def Hipergeometrica(TM,NS,P):
  x=0 #contador de exitos
  for i in range(P):
    r=random.random()
    if(r<NS/TM):
      s=1.0
    elif(r>NS/TM):
      s=0.0
    x=s+x
    NS=NS-1
    TM=TM-1
  print("el numero de exitos en la muestra es: ",x)
  return x
#n: numero de ensayos 
#p: proababilidad de exito 
def Binomial(n,p): #usa el metodo de ensayos de bernuli para generar la variable aleatoria
  x=0 #almacena el numero de exitos
  for i in range(n):
    r=random.random()
    if(r-p<0):
      x= x+1
  return x

import random
#n: numero de ensayos 
#p: proababilidad de exito
#k: cant de numeros
def runtestBinomial(n,p,k):
  numeros = []
  for i in range(k):
    numero=Binomial(n,p)
    numeros.append(numero)
  run_test(numeros)


   
from collections import Counter
#datos_iniciales: array de valores iniciales
#cantidad: cantidad de valores que se quieren generar
def distribucion_empirica(datos_iniciales, cantidad):
    # Paso 1: Contar frecuencia de cada valor observado
    conteo = Counter(datos_iniciales)
    total = len(datos_iniciales)

    # Paso 2: Calcular probabilidades empíricas (frecuencia relativa)
    distribucion = {valor: freq / total for valor, freq in conteo.items()}

    # Paso 3: Generar valores aleatorios según la distribución
    muestra = []
    for _ in range(cantidad):
        r = random.random()
        acumulador = 0.0
        for valor, prob in distribucion.items():
            acumulador += prob
            if r < acumulador:
                muestra.append(valor)
                break
    plt.subplot(1, 2, 2)
    sns.histplot(muestra, kde=False, stat="probability", bins=len(set(muestra)))
    plt.title("Distribución de la Muestra Generada")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia Relativa")

    return distribucion, muestra

#n:numero de ensayos
#p: probabilidad de exito
#N: numero de muestras 
def rechazo_binomial(n, p, N):
    # Función de masa de probabilidad de la binomial
    def binomial_pmf(k):
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # Calcular la constante de normalización c
    k_vals = np.arange(0, n + 1)
    pmf_vals = [binomial_pmf(k) for k in k_vals]
    c = 1 / max(pmf_vals)

    # Método de rechazo
    samples = []
    while len(samples) < N:
        k_candidate = np.random.randint(0, n + 1)
        r = np.random.uniform(0, 1)
        if r <= c * binomial_pmf(k_candidate):
            samples.append(k_candidate)
    # Visualización
    k_vals = np.arange(0, n + 1)
    pmf_vals = [math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in k_vals]
    plt.plot(k_vals, pmf_vals, 'ro-', label='Binomial teórica')
    plt.xlabel("k")
    plt.ylabel("Probabilidad")
    plt.title(f"Método de rechazo - Binomial(n={n}, p={p})")
    plt.legend()
    plt.grid(True)
    plt.show()

    return samples

#N:  tamaño total de la población
#K:  cantidad de éxitos en la población
#n:  tamaño de la muestra extraída
#num_muestras: cantidad de muestras que queremos generar
def rechazo_hipergeometrica(N, K, n, num_muestras):
    # PMF hipergeométrica
    def hypergeom_pmf(k):
        if k < max(0, n - (N - K)) or k > min(n, K):
            return 0
        return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)

    # Soporte posible de X (valores válidos)
    k_vals = np.arange(max(0, n - (N - K)), min(n, K) + 1)
    pmf_vals = [hypergeom_pmf(k) for k in k_vals]

    # Normalización: c = 1 / max(P(k))
    c = 1 / max(pmf_vals)

    # Método de rechazo
    muestras = []
    while len(muestras) < num_muestras:
        k_candidato = np.random.randint(k_vals[0], k_vals[-1] + 1)
        r = np.random.uniform(0, 1)
        if r <= c * hypergeom_pmf(k_candidato):
            muestras.append(k_candidato)

    # Visualización
    k_vals = np.arange(max(0, n - (N - K)), min(n, K) + 1)
    pmf_vals = [math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n) for k in k_vals]

    plt.hist(muestras, bins=np.arange(k_vals[0] - 0.5, k_vals[-1] + 1.5, 1), density=True,
            alpha=0.7, edgecolor='black', label='Muestras')
    plt.plot(k_vals, pmf_vals, 'ro-', label='Hipergeométrica teórica')
    plt.title(f"Método de Rechazo - Hipergeométrica(N={N}, K={K}, n={n})")
    plt.xlabel('k')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()


    return muestras
#valores:lista de valores 
#probabilidades: lista de probabilidades de cada uno de los valores
#num_muetras: numero de muestras 
def rechazo_empirica_discreta(valores, probabilidades, num_muestras):
    # Normalizar probabilidades si no suman 1
    total = sum(probabilidades)
    probabilidades = [p / total for p in probabilidades]

    # Calcular constante de normalización c
    c = 1 / max(probabilidades)

    # Método de rechazo
    muestras = []
    while len(muestras) < num_muestras:
        # Elegimos índice aleatorio del conjunto de valores
        i = np.random.randint(0, len(valores))
        x_candidato = valores[i]

        # Generamos número aleatorio uniforme
        r = np.random.uniform(0, 1)

        # Aceptamos con probabilidad proporcional a p_i
        if r <= c * probabilidades[i]:
            muestras.append(x_candidato)

    # Visualización
    plt.hist(muestras, bins=np.arange(min(valores) - 0.5, max(valores) + 1.5, 1),
            density=True, alpha=0.7, edgecolor='black', label='Muestras')
    plt.plot(valores, [p / sum(probabilidades) for p in probabilidades], 'ro-', label='Distribución empírica')
    plt.xlabel("Valor")
    plt.ylabel("Probabilidad")
    plt.title("Método de rechazo - Distribución empírica discreta")
    plt.legend()
    plt.grid(True)
    plt.show()

    return muestras

#muestras = rechazo_empirica_discreta([1,2,3,4,5], [0.1,2,0.3,0.3,0.4], 100)
#muestras = rechazo_hipergeometrica(50, 20,10, 500)
#muestras = rechazo_binomial(100, 0.4, 500)
#plt.hist(muestras, bins=np.arange(-0.5, n + 1.5, 1), density=True, alpha=0.7, edgecolor='black', label='Muestras')


#distribucion: empiria discreta 
datos_iniciales = [1, 2,3,4,5,6,7,8,9,10]
cantidad = 100
distribucion, muestra = distribucion_empirica(datos_iniciales, cantidad)
run_test(muestra)
# Preparar valores ordenados
valores_ordenados = sorted(distribucion.keys())
fda_empirica = []
acumulado = 0
for valor in valores_ordenados:
    acumulado += distribucion[valor]
    fda_empirica.append(acumulado)

# FDA de la muestra generada
conteo_muestra = Counter(muestra)
total_muestra = len(muestra)
valores_muestra_ordenados = sorted(conteo_muestra.keys())
fda_muestra = []
acumulado_muestra = 0
for valor in valores_muestra_ordenados:
    acumulado_muestra += conteo_muestra[valor] / total_muestra
    fda_muestra.append(acumulado_muestra)

# Graficar
plt.figure(figsize=(12, 5))



# FDA de la muestra
plt.subplot(1, 2, 2)
plt.step(valores_muestra_ordenados, fda_muestra, where='post', color='orange', label="FDA de la Muestra Generada")
plt.title("Función de Distribución de la Muestra")
plt.xlabel("Valor")
plt.ylabel("Probabilidad acumulada")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
#distribucion hipergeometrica
# Parámetros: población total (TM), número de éxitos en la población (NS), tamaño de la muestra (P)
TM = 50
NS = 20
P = 10

# Repetimos la simulación muchas veces para obtener la distribución
muestras = [Hipergeometrica(TM, NS, P) for _ in range(500)]

# Contamos frecuencia de cada número de éxitos
conteo = Counter(muestras)
valores = sorted(conteo.keys())
frecuencias = [conteo[v] / len(muestras) for v in valores]

# Graficamos
plt.figure(figsize=(8, 5))
sns.barplot(x=valores, y=frecuencias, color='skyblue')
plt.title("Distribución Hipergeométrica Simulada")
plt.xlabel("Número de éxitos en la muestra")
plt.ylabel("Frecuencia relativa")
plt.grid(axis='y')
plt.show()

#grafica binomial      


n = 100    # número de ensayos
p = 0.4    # probabilidad de éxito
N = 200  # cantidad de simulaciones
#runtestBinomial(n,p,N)

muestras = [Binomial(n, p) for _ in range(N)]


plt.figure(figsize=(8, 5))


plt.hist(muestras, bins=np.arange(-0.5, n + 1.5, 1), density=True, alpha=0.7, edgecolor='black', label='Muestras (simulación)')

k_vals = np.arange(0, n + 1)
pmf_vals = [math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in k_vals]
plt.plot(k_vals, pmf_vals, 'ro-', label='Binomial teórica')

plt.xlabel("Número de éxitos (k)")
plt.ylabel("Probabilidad")
plt.title(f"Simulación Binomial (n={n}, p={p}) - Método de Bernoulli")
plt.legend()
plt.grid(True)
plt.show()

