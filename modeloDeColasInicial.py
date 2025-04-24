import random

def inicialization(n, minTimeArrivalInt, maxTimeArrivalInt, minTimeServiceInt, maxTimeServiceInt):
  A = []
  D = []
  for i in range(n):
    A.append(random.randint(minTimeArrivalInt, maxTimeArrivalInt) + random.random())
    D.append(random.randint(minTimeServiceInt, maxTimeServiceInt) + random.random())
  return A, D

def main(n):
  cont1 = 0
  cont2 = 0
  for c in range(100):
    a = 0
    d = 0
    B = 0
    arrs, deps = inicialization(n, 0, 2, 0, 2)
    clk = arrs[a]
    a += 1
    momtsOfArrivals = []
    momtsOfArrivals.append(clk)
    totDelay = 0
    lastTimeEnterServer = arrs[0]
    lastArrivalTime = arrs[0]
    contUsers = 1
    while(d < n):
      if((clk + arrs[a]) < (lastTimeEnterServer + deps[d]) or len(momtsOfArrivals) == 0):
        lastArrivalTime += arrs[d]
        clk = lastArrivalTime 
        if(a < (n - 1)): a += 1
        contUsers += 1
        momtsOfArrivals.append(clk)
      else:
        B += deps[d]
        clk = lastTimeEnterServer + deps[d]
        lastTimeEnterServer = clk
        d += 1
        totDelay += (clk - momtsOfArrivals[0])
        momtsOfArrivals.pop(0)
    cont1 += totDelay/clk
    cont2 += B/clk
  print(cont1/100)
  print(cont2/100)

main(100)
