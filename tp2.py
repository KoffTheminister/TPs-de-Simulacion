import random
import matplotlib.pyplot as plt
# estrategia martingale (m): se invierte x cantidad de capital en un valor. si ese valor genera perdida, para la proxima eleccion de valor se duplica la inversion x. la esperanza de esta estrategia se situa en la confianza de que tarde o temprano va a salir un valor elegido

# estrategia D'Alembert (d): primero se debe de elegir una unidad, despues invertis x cantidad de capital en un valor. si se pierde, entonces a x se le suma una unidad y se invierte esa suma. si se gana, entonces a x se le resta una unidad y se invierte el resultado de esa resta

# estrategia Fibonacci (f): se elige una unidad y se invierte esa misma la primera vez. cada vez que se pierda, se pasa al siguiente numero en la secuencia de fibonacci. cada vez que se gana se retroceden dos numeros en la secuencia (excepto obvio en los dos primeros casos)

negro = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]

capital = 10000
apuesta = 100
historialCapital = []


def martingala(tirs):
    tiradasPerdidas = 0
    n = 0
    global capital, apuesta, historialCapital
    while n < tirs:
        if capital < apuesta:
            print("No hay suficiente capital para continuar.")
            break
        tirada = random.randint(0, 36)
        if tirada not in negro:
            capital -= apuesta
            apuesta *= 2
            historialCapital.append(capital)
            tiradasPerdidas += 1
        else:
            capital += apuesta
            apuesta = 100
            historialCapital.append(capital)
            tiradasPerdidas = 0
        n += 1
    return capital, historialCapital, n


tiradas = 200 
capitalFinal, historial, n = martingala(tiradas)

print(f"\nCapital final después de {tiradas} tiradas: ${capitalFinal} en {n} tiradas")


plt.plot(historial)
plt.title('Evolución del capital en la estrategia Martingala')
plt.xlabel('Número de tiradas')
plt.ylabel('Capital')
plt.grid()
plt.show()
