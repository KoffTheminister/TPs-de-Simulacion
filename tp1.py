import numpy as np
import random
import matplotlib.pyplot as plt

def corrida_no_parametrizada(numEl):
  frecsRels = []
  proms = []
  vars = []
  desvios = []
  for n in range(1000):
    contFrec = 0
    contProm = 0
    contv = 0
    for j in range(n + 1):
      num = random.randint(0, 37)
      if(num == (numEl - 1)):
        contFrec += 1
      contProm += num
      contv += num**2
    frecsRels.append(contFrec/(n + 1))
    proms.append(contProm/(n + 1))
    var = (contv/(n + 1)) - (contProm/(n + 1))**2
    vars.append(var)
    desvios.append(var**(1/2))

  tiradas = list(range(1, 1001))
  plt.figure(figsize=(10, 12))  # Ajusta el tamaño general de la figura
  
  plt.subplot(4, 1, 1)
  plt.plot(tiradas, frecsRels, label='Frecuencia relativa')
  plt.axhline(y=1/37, color='red', linestyle='--', label='Frecuencia esperada (1/37)')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Frecuencia relativa')
  plt.title(f'Frecuencia relativa del número {numEl} en {1000} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 2)
  plt.plot(tiradas, proms, label='Promedios')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Promedios')
  plt.title(f'Promedios en {1000} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 3)
  plt.plot(tiradas, vars, label='Varianzas')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Varianzas')
  plt.title(f'Varianzas en {1000} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 4)
  plt.plot(tiradas, desvios, label='Desvios estandar')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Desvios estandar')
  plt.title(f'Desvios estandar en {1000} tiradas')
  plt.legend()
  plt.tight_layout()
  plt.show()
  plt.show()

  
#corrida_no_parametrizada(5)

def corrida_parametrizada(corrs, tirs, numEl):
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
  plt.xlabel('Número de corridas')
  plt.ylabel('Frecuencia relativa')
  plt.title(f'Frecuencia relativa del número {numEl} en {corrs} corridas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 2)
  plt.plot(corridas, proms, label='Promedios')
  plt.xlabel('Número de corridas')
  plt.ylabel('Promedios')
  plt.title(f'Promedios en {1000} corridas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 3)
  plt.plot(corridas, vars, label='Varianzas')
  plt.xlabel('Número de corridas')
  plt.ylabel('Varianzas')
  plt.title(f'Varianzas en {1000} corridas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 4)
  plt.plot(corridas, desvios, label='Desvios estandar')
  plt.xlabel('Número de corridas')
  plt.ylabel('Desvios estandar')
  plt.title(f'Desvios estandar en {corrs} corridas')
  plt.legend()
  plt.tight_layout()
  plt.show()
  plt.show()

#corrida_parametrizada(100, 10000, 5)

def generar_frecuencia_relativa_forma_leo(numero_elegido, num_tiradas):
  conteo = 0
  frn = []
  for i in range(1, num_tiradas + 1):
    tirada = random.randint(0, 37)
    if tirada == numero_elegido:
      conteo += 1
    frn.append(conteo / i)
  tiradas = list(range(1, num_tiradas+1))

  plt.plot(tiradas, frn, label='Frecuencia relativa')
  plt.axhline(y=1/37, color='red', linestyle='--', label='Frecuencia esperada (1/37)')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Frecuencia relativa')
  plt.title(f'Frecuencia relativa del número {numero_elegido} en {num_tiradas} tiradas')
  plt.legend()
  plt.grid(True)
  plt.show()

#generar_frecuencia_relativa_forma_leo(5, 100)


def corrida_no_parametrizada_forma_leo(numEl):
  frecsRels = []
  proms = []
  vars = []
  desvios = []
  for n in range(1000):
    contFrec = 0
    contProm = 0
    contv = 0
    num = random.randint(0, 37)
    if(num == (numEl - 1)):
      contFrec += 1
    contProm += num
    contv += num**2
    frecsRels.append(contFrec/(n + 1))
    proms.append(contProm/(n + 1))
    var = (contv/(n + 1)) - (contProm/(n + 1))**2
    vars.append(var)
    desvios.append(var**(1/2))

  tiradas = list(range(1, 1000+1))
  plt.figure(figsize=(10, 12))  # Ajusta el tamaño general de la figura
  
  plt.subplot(4, 1, 1)
  plt.plot(tiradas, frecsRels, label='Frecuencia relativa')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Frecuencia relativa')
  plt.title(f'Frecuencia relativa del número {numEl} en {1000} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 2)
  plt.plot(tiradas, proms, label='Promedios')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Promedios')
  plt.title(f'Promedios en {1000} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 3)
  plt.plot(tiradas, vars, label='Varianzas')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Varianzas')
  plt.title(f'Varianzas en {1000} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(4, 1, 4)
  plt.plot(tiradas, desvios, label='Desvios estandar')
  plt.xlabel('Número de tiradas')
  plt.ylabel('Desvios estandar')
  plt.title(f'Desvios estandar en {1000} tiradas')
  plt.legend()
  plt.tight_layout()
  plt.show()
  plt.show()

corrida_no_parametrizada_forma_leo(5)

