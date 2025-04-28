import random
import matplotlib.pyplot as plt
# estrategia martingale (m): se invierte x cantidad de capital en un valor. si ese valor genera perdida, para la proxima eleccion de valor se duplica la inversion x. la esperanza de esta estrategia se situa en la confianza de que tarde o temprano va a salir un valor elegido

# estrategia D'Alembert (d): primero se debe de elegir una unidad, despues invertis x cantidad de capital en un valor. si se pierde, entonces a x se le suma una unidad y se invierte esa suma. si se gana, entonces a x se le resta una unidad y se invierte el resultado de esa resta

# estrategia Fibonacci (f): se elige una unidad y se invierte esa misma la primera vez. cada vez que se pierda, se pasa al siguiente numero en la secuencia de fibonacci. cada vez que se gana se retroceden dos numeros en la secuencia (excepto obvio en los dos primeros casos)

negro = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]

def martingala(tirs):
    capital = 10000
    historialCapital = [capital]
    apuesta = 100
    n = 0
    while n < tirs:
        if capital < apuesta:
            print("No hay suficiente capital para continuar.")
            break
        tirada = random.randint(0, 36)
        if(tirada not in negro):
            capital -= apuesta
            apuesta *= 2
            historialCapital.append(capital)
        else:
            capital += apuesta
            apuesta = apuesta
            historialCapital.append(capital)
        n += 1
    print(f"\El capital en la tirada numero {n} es de ${capital}")
    return historialCapital

def dalembert(tirs):
    n = 0
    capital = 10000
    apuesta = 100
    historialCapital = [capital]
    while n < tirs:
        if capital < apuesta:
            print("No hay suficiente capital para continuar.")
            break
        tirada = random.randint(0, 36)
        if tirada not in negro:
            capital -= apuesta
            apuesta += 100  
            historialCapital.append(capital)
        else:
            capital += apuesta
            if apuesta > 100:
                apuesta -= 100  
            historialCapital.append(capital)
        n += 1
    print(f"\ El capital en la tirada numero {n} es de ${capital}")
    return historialCapital

def generador_fibonacci(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def fibonacci_1(tirs):
    capital = 10000
    historialCapital = [capital]
    historialApuestas = []
    n = 0
    posicion_fibonacci = 0

    while n < tirs:
        apuesta = generador_fibonacci(posicion_fibonacci) * 100
        historialApuestas.append(apuesta)
        if capital < apuesta:
            print("No hay suficiente capital para continuar.")
            break   
        resultado = random.randint(0, 36)

        if resultado in negro:
            capital += apuesta
            posicion_fibonacci = max(posicion_fibonacci - 2, 0)  
        else:
            capital -= apuesta
            posicion_fibonacci += 1

        historialCapital.append(capital)
        n += 1
    print(f"\ El capital en la tirada numero {n} es de ${capital}")
    print(historialApuestas)
    return historialCapital



def simulacion_ruleta(corrs, tirs):
    martins = []
    dalamberts = []
    fibos = []
    for corr in range(corrs):
        martins.append(martingala(tirs))#, colEl, capitalMax, invIni))
        dalamberts.append(dalembert(tirs))#, colEl, capitalMax, invIni))
        fibos.append(fibonacci_1(tirs))

    #tiradas = list(range(1, tirs + 1))
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for corrida in martins:
        tiradas = list(range(1, len(corrida) + 1))
        plt.plot(tiradas, corrida, alpha=0.5)
    plt.xlabel('Número de tiradas (n)')
    plt.ylabel('Capital en n')
    plt.title(f'Evolucion del capital neto usando la estrategia Martingala en {corrs} corridas')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for corrida in dalamberts:
        tiradas = list(range(1, len(corrida) + 1))
        plt.plot(tiradas, corrida, alpha=0.5)
    plt.xlabel('Número de tiradas (n)')
    plt.ylabel('Promedios')
    plt.title(f'Evolucion del capital neto usando la estrategia DLambert en {corrs} corridas')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for corrida in fibos:
        tiradas = list(range(1, len(corrida) + 1))
        plt.plot(tiradas, corrida, alpha=0.5)
    plt.xlabel('Número de tiradas (n)')
    plt.ylabel('Capital en n')
    plt.title(f'Evolucion del capital neto usando la estrategia fibonacci en {corrs} corridas')
    plt.grid(True)
    plt.show()

    # plt.subplot(2, 2, 4)
    # for corrida in desvios:
    #     plt.plot(tiradas, corrida, alpha=0.5)
    # plt.axhline(y=10.67, color='red', linestyle='--', label='Desvio estandar poblacional esperado (10,67)')
    # plt.xlabel('Número de tiradas (n)')
    # plt.ylabel('Desvios estandar')
    # plt.title(f'Desvios estandar en {tirs} tiradas')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

simulacion_ruleta(10, 2000)





