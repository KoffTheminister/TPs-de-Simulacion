def generador2(seed, n):
  lista = []
  a = 2**16 + 3
  m = 2**31
  comp = []
  x = (a*seed)%m
  comp.append(x)
  comp.append(x/m)
  lista.append(comp)
  i = 1
  while(i <= n):
    comp = []
    x = (a*lista[i - 1][0])%m
    comp.append(x)
    comp.append(x/m)
    lista.append(comp)
    i += 1
  print(lista)
  
def midSquare(seed,n):
  resultados= []
  num = seed
  for i in range(n):
    x= str(num*num).zfill(8)
    mid = len (x) // 2
    num = int(x[mid-2:mid+2])
    resultados.append(num)
  return resultados
print(midSquare(666667, 100))
  


