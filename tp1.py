import numpy as np
import random
import matplotlib.pyplot as plt


'''
El trabajo investigar consiste en construir una programa en lenguaje Python 3.x que simule el funcionamiento del plato
de una ruleta. Para esto se debe tener en cuenta lo siguientes temas:
• Generación de valores aleatorios enteros.
• Uso de listas para el almacenamiento de datos.
• Uso de la estructura de control FOR para iterar las listas.
• Empleo de funciones estadísticas.
• Gráficas de los resultados mediante el paquete Matplotlib.
• Ingreso por consola de parámetros para la simulación (cantidad de tiradas, corridas y número elegido, Ejemplo
python -c XXX -n YYY -e ZZ).

'''
def promedio(num_tiradas, num_elegido):
  map = np.zeros((num_tiradas))

  for j in range(num_tiradas):
    n += 1
    for k in range(n):
      if(random.randint(0, 37) == (num_elegido - 1)):
        (map[i, n - 1]) = (map[i, n - 1]) + 1
      #map[i, n - 1] = map[i, n - 1]/n


  print(map)

def generar_frecuencia_relativa(numero_elegido, num_tiradas):
    conteo = 0
    frn = []
    for i in range(1, num_tiradas + 1):
        tirada = random.randint(0, 37)
        if tirada == numero_elegido:
            conteo += 1
        frn.append(conteo / i)
    return frn


def generar_promedio(num_tiradas):
    promedio_acumulado = []
    suma = 0
    for i in range(1, num_tiradas + 1):
        suma += random.randint(0, 37)
        # tot = 0
        # for j in range(i): tot += random.randint(0, 37)
        promedio_acumulado.append(suma / i)
    print(promedio_acumulado)
    #return promedio_acumulado

generar_promedio(1000)
numero = 12
tiradas = 8000

# # Generar datos
# resultado = generar_frecuencia_relativa(numero, tiradas)

# # Graficar
# plt.plot(range(1, tiradas + 1), resultado, label='Frecuencia relativa')
# plt.axhline(y=1/37, color='red', linestyle='--', label='Frecuencia esperada (1/37)')
# plt.xlabel('Número de tiradas')
# plt.ylabel('Frecuencia relativa')
# plt.title(f'Frecuencia relativa del número {numero} en {tiradas} tiradas')
# plt.legend()
# plt.grid(True)
# plt.show()

