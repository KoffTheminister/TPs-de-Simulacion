import random
import matplotlib.pyplot as plt
import sys

estrategias = ['m','d','f','o']
print(sys.argv)

if sys.argv[1]!="-c" or sys.argv[3]!="-n" or sys.argv[5]!="-s" or sys.argv[7]!="-a" or sys.argv[6] not in estrategias:
    print(f"Uso: Python SimulacionRuleta.py -c <Cantidad de tiradas> -n <numero de corridas> -s <estrategia ({estrategias})> -a <capital a usar (infinito (i) - numero de capital)>")
    sys.exit(1)

cantTiradas=int(sys.argv[2])
cantCorridas=int(sys.argv[4])
if(sys.argv[8] == 'i'):
    capital = 'i'
else:
    capital = int(sys.argv[8])
estrategia = sys.argv[6]

# estrategia martingale (m): se invierte x cantidad de capital en un valor. si ese valor genera perdida, para la proxima eleccion de valor se duplica la inversion x. la esperanza de esta estrategia se situa en la confianza de que tarde o temprano va a salir un valor elegido

# estrategia D'Alembert (d): primero se debe de elegir una unidad, despues invertis x cantidad de capital en un valor. si se pierde, entonces a x se le suma una unidad y se invierte esa suma. si se gana, entonces a x se le resta una unidad y se invierte el resultado de esa resta

# estrategia Fibonacci (f): se elige una unidad y se invierte esa misma la primera vez. cada vez que se pierda, se pasa al siguiente numero en la secuencia de fibonacci. cada vez que se gana se retroceden dos numeros en la secuencia (excepto obvio en los dos primeros casos)

negro = [2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]

def martingala(tirs, capital):
    if(capital == 'i'):
        cap = 0
        apuesta_inicial = 100
    else:
        cap = capital
        apuesta_inicial = cap * 0.01
    historialCapital = [cap]
    resultados = []
    apuesta = apuesta_inicial    
    n = 0
    cont = 0
    while n < tirs:
        if cap < apuesta and capital != 'i':
            print("No hay suficiente capital para continuar.")
            break
        tirada = random.randint(0, 36)
        if(tirada not in negro):
            cap -= apuesta
            apuesta *= 2
            historialCapital.append(cap)
            #resultados.append(0)
        else:
            cap += apuesta
            apuesta = apuesta_inicial
            historialCapital.append(cap)
            #resultados.append(1)
            cont += 1
        resultados.append(cont/(n + 1))
        n += 1

    return historialCapital, resultados

def dalembert(tirs, capital):
    if(capital == 'i'):
        cap = 0
        inv = 100
    else:
        cap = capital
        inv = cap * 0.01
    n = 0
    resultados = []
    cont = 0
    apuesta = inv
    historialCapital = [cap]
    while n < tirs:
        if cap < apuesta and capital != 'i':
            print("No hay suficiente capital para continuar.")
            break
        tirada = random.randint(0, 36)
        if tirada not in negro:
            cap -= apuesta
            apuesta += inv            
            historialCapital.append(cap)
        else:
            cap += apuesta
            if apuesta > (inv):
                apuesta -= inv
            historialCapital.append(cap)
            cont += 1
        resultados.append(cont/(n + 1))
        n += 1
    print(f"\ El capital en la tirada numero {n} es de ${cap}")
    return historialCapital, resultados

def generador_fibonacci(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def fibonacci(tirs, capital):
    if(capital == 'i'):
        cap = 0
        inv = 100
    else:
        cap = capital
        inv = cap * 0.01
    resultados = []
    cont = 0
    apuesta = inv
    historialCapital = [cap]
    historialApuestas = []
    n = 0
    posicion_fibonacci = 0
    while n < tirs:
        if cap < apuesta and capital != 'i':
            print("No hay suficiente capital para continuar.")
            break   

        apuesta = generador_fibonacci(posicion_fibonacci) * inv
        historialApuestas.append(apuesta)
        resultado = random.randint(0, 36)
        if resultado in negro:
            cap += apuesta
            posicion_fibonacci = max(posicion_fibonacci - 2, 0)  
        else:
            cap -= apuesta
            posicion_fibonacci += 1
            cont += 1
        resultados.append(cont/(n + 1))
        historialCapital.append(cap)
        n += 1
    print(f"\ El capital en la tirada numero {n} es de ${cap}")
    print(historialApuestas)
    return historialCapital, resultados

def nuevaEstrategia(tirs, capital):
    if(capital == 'i'):
        cap = 0
        inv = 100
    else:
        cap = capital
        inv = cap * 0.01
    resultados = []
    cont = 0
    apuesta = inv
    historialCapital = [cap]
    n = 0
    while n < tirs:
        if cap < apuesta and capital != 'i':
            print("No hay suficiente capital para continuar.")
            break
        tirada = random.randint(0, 36)
        if(tirada not in negro):
            cap -= apuesta
            apuesta = inv
            historialCapital.append(cap)
        else:
            cont += 1
            cap += apuesta
            apuesta *= 2
            historialCapital.append(cap)
        resultados.append(cont/(n + 1))
        n += 1
    print(f"\El capital en la tirada numero {n} es de ${cap}")
    return historialCapital, resultados

def simulacion_ruleta(corrs, tirs, cap, e):
    corridas = []
    exitosprop = []
    if(e == 'm'):
        estrategia = 'Martingala'
        for corr in range(corrs):
            c, a = martingala(tirs, cap)
            corridas.append(c)
            exitosprop.append(a)
        print(corridas)
        print(exitosprop)
    elif(e == 'd'):
        estrategia = 'DLambert'
        for corr in range(corrs):
            c, a = dalembert(tirs, cap)
            corridas.append(c)
            exitosprop.append(a)
    elif(e == 'f'):
        estrategia = 'Fibonacci'
        for corr in range(corrs):
            c, a = fibonacci(tirs, cap)
            corridas.append(c)
            exitosprop.append(a)
    else:
        estrategia = 'Estrategia Original'
        for corr in range(corrs):
            c, a = nuevaEstrategia(tirs, cap)
            corridas.append(c)
            exitosprop.append(a)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    for corrida in corridas:
        tiradas = list(range(1, len(corrida) + 1))
        plt.plot(tiradas, corrida, alpha=0.5)
    plt.xlabel('Número de tiradas (n)')
    plt.ylabel('Capital en n')
    plt.title(f'Evolucion del capital neto usando la estrategia {estrategia} en {corrs} corridas')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for exito in exitosprop:
        tiradas = list(range(1, len(exito) + 1))
        plt.bar(tiradas, exito)
    plt.title('Proporción de éxitos en cada tirada')
    plt.xlabel('Número de tirada')
    plt.ylabel('Proporción de éxitos')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


simulacion_ruleta(cantCorridas, cantTiradas, capital, estrategia)



