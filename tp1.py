import numpy as np
import random
import matplotlib.pyplot as plt

def corrida_parametrizada(numEl, tirs, corrs):
  frecsRels = []
  proms = []
  vars = []
  desvios = []
  for c in range(corrs):
    frecs = []
    promsitos = []
    varsitas = []
    desviitos = []
    for n in range(tirs):
      contFrec = 0
      contProm = 0
      contv = 0
      for j in range(n + 1):
        num = random.randint(0, 37)
        if(num == (numEl - 1)):
          contFrec += 1
        contProm += num
        contv += num**2
      frecs.append(contFrec/(n + 1))
      promsitos.append(contProm/(n + 1))
      var = (contv/(n + 1)) - (contProm/(n + 1))**2
      varsitas.append(var)
      desviitos.append(var**(1/2))
    frecsRels.append(frecs)
    proms.append(promsitos)
    vars.append(varsitas)
    desvios.append(desviitos)

  tiradas = list(range(1, tirs + 1))
  plt.figure(figsize=(12, 8))
  
  plt.subplot(2, 2, 1)
  for corrida in frecsRels:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.axhline(y=1/37, color='red', linestyle='--', label='Frecuencia esperada (1/37)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Frecuencia relativa')
  plt.title(f'Frecuencia relativa del número {numEl} en {tirs} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 2)
  for corrida in proms:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.axhline(y=17.5, color='red', linestyle='--', label='Promedio esperado (17,5)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Promedios')
  plt.title(f'Promedios en {tirs} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 3)
  for corrida in vars:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Varianzas')
  plt.title(f'Varianzas en {tirs} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 4)
  for corrida in desvios:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Desvios estandar')
  plt.title(f'Desvios estandar en {tirs} tiradas')
  plt.legend()
  plt.tight_layout()
  plt.show()
  plt.show()

corrida_parametrizada(5,300,5)


def corrida_parametrizada_tirada_estatica(corrs, tirs, numEl):
  frecsRels = []
  proms = []
  vars = []
  desvios = []
  for n in range(corrs):
    contFrec = 0
    contProm = 0
    contv = 0
    for j in range(tirs):
      num = random.randint(0, 37)
      if(num == (numEl - 1)):
        contFrec += 1
      contProm += num
      contv += num**2
    frecsRels.append(contFrec/tirs)
    proms.append(contProm/tirs)
    var = (contv/tirs) - (contProm/tirs)**2
    vars.append(var)
    desvios.append(var**(1/2))

  corridas = list(range(1, corrs+1))

  plt.figure(figsize=(10, 12))  # Ajusta el tamaño general de la figura
  
  plt.subplot(4, 1, 1)
  plt.plot(corridas, frecsRels, label='Frecuencia relativa')
  plt.axhline(y=1/37, color='red', linestyle='--', label='Frecuencia esperada (1/37)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Frecuencia relativa')
  plt.title(f'Frecuencia relativa del número {numEl} en {corrs} corridas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 2)
  plt.plot(corridas, proms, label='Promedios')
  plt.axhline(y=17, color='red', linestyle='--', label='Promedio esperado (17,5)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Promedios')
  plt.title(f'Promedios en {1000} corridas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 3)
  plt.plot(corridas, vars, label='Varianzas')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Varianzas')
  plt.title(f'Varianzas en {1000} corridas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 4)
  plt.plot(corridas, desvios, label='Desvios estandar')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Desvios estandar')
  plt.title(f'Desvios estandar en {corrs} corridas')
  plt.legend()
  plt.tight_layout()
  plt.show()
  plt.show()

#corrida_parametrizada(100, 10000, 5)

