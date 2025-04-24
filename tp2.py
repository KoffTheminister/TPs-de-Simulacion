

# estrategia martingale (m): se invierte x cantidad de capital en un valor. si ese valor genera perdida, para la proxima eleccion de valor se duplica la inversion x. la esperanza de esta estrategia se situa en la confianza de que tarde o temprano va a salir un valor elegido

# estrategia D'Alembert (d): primero se debe de elegir una unidad, despues invertis x cantidad de capital en un valor. si se pierde, entonces a x se le suma una unidad y se invierte esa suma. si se gana, entonces a x se le resta una unidad y se invierte el resultado de esa resta

# estrategia Fibonacci (f): se elige una unidad y se invierte esa misma la primera vez. cada vez que se pierda, se pasa al siguiente numero en la secuencia de fibonacci. cada vez que se gana se retroceden dos numeros en la secuencia (excepto obvio en los dos primeros casos)

import random
import matplotlib.pyplot as plt

def corrida_parametrizada(numEls, tirs, corrs, cap):
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
        num = random.randint(0, 36)
        if(num in numEls):
          
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
  plt.title(f'Frecuencia relativa del número {numEls} en {tirs} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 2)
  for corrida in proms:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.axhline(y=18, color='red', linestyle='--', label='Promedio esperado (18)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Promedios')
  plt.title(f'Promedios en {tirs} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 3)
  for corrida in vars:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.axhline(y=114, color='red', linestyle='--', label='varianza poblacional esperada (114)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Varianzas')
  plt.title(f'Varianzas en {tirs} tiradas')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 4)
  for corrida in desvios:
    plt.plot(tiradas, corrida, alpha=0.5)
  plt.axhline(y=10.67, color='red', linestyle='--', label='Desvio estandar poblacional esperado (10,67)')
  plt.xlabel('Número de tiradas (n)')
  plt.ylabel('Desvios estandar')
  plt.title(f'Desvios estandar en {tirs} tiradas')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()

corrida_parametrizada(5,300,3)