import random
import math
import numpy as np

prob_distrib_demand = [0.04,0.03,0.05,0.02,0.06,0.05,0.04,0.03,0.05,0.04,0.02,0.05,0.03,0.04,0.06,0.05,0.03,0.04,0.02,0.05,0.04,0.03,0.05,0.04,0.03,0.01]
time_next_event = [None for i in range(4)]

area_holding = 0
area_shortage = 0
num_values_demand = 0
inv_level = 0
amount = 0
next_event_type = 0
setup_cost = 32
incremental_cost = 3
holding_cost = 1 #por unidad por mes
initial_inv_level = 60
num_months = 120
mean_interdemand = 0.5
shortage_cost = 100
sim_time = 0
bigs = 100
smalls = 20
time_last_event = 0
total_ordering_cost = 0

min_order = 2
mean_order = 4
max_order = 5

inv_historial = []

minlag = 1
maxlag = 3

def inicializar():
  global sim_time, initial_inv_level, inv_level, time_last_event, total_ordering_cost, area_holding, num_months, area_shortage, min_order, mean_order, max_order, mean_interdemand, time_next_event
  sim_time = 0
  inv_level = initial_inv_level
  time_last_event = 0
  total_ordering_cost = 0.0
  area_holding = 0.0
  area_shortage = 0.0
  time_next_event[0] = np.random.triangular(min_order, mean_order, max_order, 1)[0]
  time_next_event[1] = np.random.exponential(1/mean_interdemand, 1)[0]
  time_next_event[2] = num_months
  time_next_event[3] = 0.0

def demanda():
  global inv_level, inv_historial, time_next_event, sim_time, mean_interdemand
  inv_level -= random.randint(1,4) #random_integer(prob_distrib_demand) #prob_distrib_demand[random.randint(0,25)]
  inv_historial.append(inv_level)
  time_next_event[1] = sim_time + np.random.exponential(1/mean_interdemand, 1)[0]

def evaluar():
  global inv_level, smalls, bigs, amount, total_ordering_cost, setup_cost, incremental_cost, time_next_event, sim_time, minlag, maxlag
  if(inv_level < smalls):
    amount = bigs - inv_level
    total_ordering_cost += setup_cost + incremental_cost*amount
    time_next_event[0] = sim_time + uniform(minlag, maxlag)
  time_next_event[3] = sim_time + 1

def llegada_orden():
  global inv_level, amount, inv_historial, time_next_event
  inv_level += amount
  inv_historial.append(inv_level)
  time_next_event[0] = 1.0e+30 #np.random.triangular(min_order, mean_order, max_order, 1)[0]

def reportar():
  global num_months, area_holding, area_shortage, shortage_cost, holding_cost
  avg_ordering_cost = total_ordering_cost/num_months
  avg_holding_cost = holding_cost * area_holding/num_months
  avg_shortage_cost = shortage_cost * area_shortage/num_months
  print()

def actualizar_tiempo_promedio_estadisticas():
  global time_last_event, time_since_last_event, sim_time, inv_level, area_shortage, area_holding
  time_since_last_event = sim_time - time_last_event
  time_last_event = sim_time
  if (inv_level < 0):
    area_shortage -= inv_level * time_since_last_event
  elif (inv_level > 0):
    area_holding += inv_level * time_since_last_event

def random_integer(prob_distrib):
  u = random.random()
  i = 0
  while(u >= prob_distrib[i]):
    i += 1
  return i

def uniform(a, b):
  return a + random.random()*(b - a)

def timing():
  global next_event_type, num_events, time_next_event, sim_time
  i = 0
  min_time_next_event = 1.0e+29
  next_event_type = 0
  while(i < num_events):
    if (time_next_event[i] < min_time_next_event):
      min_time_next_event = time_next_event[i]
      next_event_type = i
    i += 1

  sim_time = min_time_next_event



#main:
num_events = 4
num_policies = 100
inv_level = 0
inicializar()
while (next_event_type != 3):
  timing() # determinar el siguiente tipo de evento y para actualizar el clk de simulacion
  print(inv_level)
  actualizar_tiempo_promedio_estadisticas()
  
  if next_event_type == 1:
    llegada_orden()
  elif next_event_type == 2:
    demanda()
  elif next_event_type == 3:
    evaluar()
  else:
    reportar()


