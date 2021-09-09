import numpy as np
import math
import time
import random
import itertools
import queue
import pandas as pd
from IPython.display import display, Markdown

# Reading the data
# Dataset number. 1, 2 or 3
dataset = "10"
optimalObjectives = [695,698,728,697,713,748,728,683,761,664]
optimalSolutions = [
    [7, 3, 5, 6, 2, 9, 1, 4, 8, 10],
    [2, 4, 3, 8, 7, 10, 6, 1, 9, 5],
    [9, 4, 10, 3, 1, 7, 2, 5, 6, 8],
    [5, 4, 1, 6, 9, 10, 3, 8, 2, 7],
    [2, 9, 5, 4, 1, 6, 3, 8, 10, 7],
    [1, 5, 8, 10, 2, 6, 9, 7, 3, 4],
    [7, 1, 8, 3, 9, 6, 2, 5, 4, 10],
    [4, 10, 7, 1, 8, 5, 6, 9, 2, 3],
    [6, 1, 5, 10, 8, 2, 3, 7, 4, 9],
    [6, 2, 3, 7, 5, 9, 10, 4, 1, 8]
]


optimalObjective = optimalObjectives[int(dataset)-1]

print('DATASET : ', dataset)
filename = "ds/VFR10_5_" + dataset + "_Gap.txt"
f = open(filename, 'r')
l = f.readline().split()

# number of jobs
n = int(l[0])
print('number of jobs', n)

# number of machines
m = int(l[1])
print(m, 'number of machines')

# ith job's processing time at jth machine
cost = []

for i in range(n):
    temp = []
    for j in range(m):
        temp.append(0)
    cost.append(temp)

for i in range(n):
    line = f.readline().split()
    for j in range(int(len(line) / 2)):
        cost[i][j] = int(line[2 * j + 1])

print('cost', cost)

f.close()

# Algorithm operators and helper functions
def initialization(Npop):
    pop = []
    for i in range(Npop):
        p = list(np.random.permutation(n))
        while p in pop:
            p = list(np.random.permutation(n))
        pop.append(p)

    return pop


def calculateObjective(solution):
    qTime = queue.PriorityQueue()

    qMachines = []
    for i in range(m):
        qMachines.append(queue.Queue())

    for i in range(n):
        qMachines[0].put(solution[i])

    busyMachines = []
    for i in range(m):
        busyMachines.append(False)

    time = 0

    job = qMachines[0].get()
    qTime.put((time + cost[job][0], 0, job))
    busyMachines[0] = True

    while True:
        time, mach, job = qTime.get()
        if job == solution[n - 1] and mach == m - 1:
            break
        busyMachines[mach] = False
        if not qMachines[mach].empty():
            j = qMachines[mach].get()
            qTime.put((time + cost[j][mach], mach, j))
            busyMachines[mach] = True
        if mach < m - 1:
            if busyMachines[mach + 1] == False:
                qTime.put((time + cost[job][mach + 1], mach + 1, job))
                busyMachines[mach + 1] = True
            else:
                qMachines[mach + 1].put(job)

    return time


def selection(pop):
    popObj = []
    for i in range(len(pop)):
        popObj.append([calculateObjective(pop[i]), i])

    popObj.sort()

    distr = []
    distrInd = []

    for i in range(len(pop)):
        distrInd.append(popObj[i][1])
        prob = (2 * (i + 1)) / (len(pop) * (len(pop) + 1))
        distr.append(prob)

    parents = []
    for i in range(len(pop)):
        parents.append(list(np.random.choice(distrInd, 2, p=distr)))

    return parents


def crossover(parents):
    pos = list(np.random.permutation(np.arange(n - 1) + 1)[:2])

    if pos[0] > pos[1]:
        t = pos[0]
        pos[0] = pos[1]
        pos[1] = t

    child = list(parents[0])

    for i in range(pos[0], pos[1]):
        child[i] = -1

    p = -1
    for i in range(pos[0], pos[1]):
        while True:
            p = p + 1
            if parents[1][p] not in child:
                child[i] = parents[1][p]
                break

    return child


def mutation(sol):
    pos = list(np.random.permutation(np.arange(n))[:2])

    if pos[0] > pos[1]:
        t = pos[0]
        pos[0] = pos[1]
        pos[1] = t

    remJob = sol[pos[1]]

    for i in range(pos[1], pos[0], -1):
        sol[i] = sol[i - 1]

    sol[pos[0]] = remJob

    return sol

def elitistUpdate(oldPop, newPop):
    bestSolInd = 0
    bestSol = calculateObjective(oldPop[0])

    for i in range(1, len(oldPop)):
        tempObj = calculateObjective(oldPop[i])
        if tempObj < bestSol:
            bestSol = tempObj
            bestSolInd = i

    rndInd = random.randint(0, len(newPop) - 1)

    newPop[rndInd] = oldPop[bestSolInd]

    return newPop

# Returns best solution's index number, best solution's objective value and average objective value of the given population.
def findBestSolution(pop):
    bestObj = calculateObjective(pop[0])
    avgObj = bestObj
    bestInd = 0
    for i in range(1, len(pop)):
        tObj = calculateObjective(pop[i])
        avgObj = avgObj + tObj
        if tObj < bestObj:
            bestObj = tObj
            bestInd = i

    return bestInd, bestObj, avgObj / len(pop)

# Number of population
Npop = 3
# Probability of crossover
Pc = 1.0
# Probability of mutation
Pm = 1.0
# Stopping number for generation
stopGeneration = 100

# Start Timer
t1 = time.perf_counter()

# Creating the initial population
population = initialization(Npop)

# Run the algorithm for 'stopGeneration' times generation
for i in range(stopGeneration):
    # Selecting parents
    parents = selection(population)
    childs = []

    # Apply crossover
    for p in parents:
        r = random.random()
        if r < Pc:
            childs.append(crossover([population[p[0]], population[p[1]]]))
        else:
            if r < 0.5:
                childs.append(population[p[0]])
            else:
                childs.append(population[p[1]])

    # Apply mutation
    for c in childs:
        r = random.random()
        if r < Pm:
            c = mutation(c)

    # Update the population
    population = elitistUpdate(population, childs)

    # print(population)
    # print(findBestSolution(population))

# Stop Timer
t2 = time.perf_counter()

# Results Time

bestSol, bestObj, avgObj = findBestSolution(population)

print("Population:")
print(population)
print()

print("Solution:")
print(population[bestSol])
print()

print("Objective Value:")
print(bestObj)
print()

print("Average Objective Value of Population:")
print("%.2f" % avgObj)
print()

print("%Difference:")
G = 100 * (bestObj - optimalObjective) / optimalObjective
print("%.2f" % G)
print()

print("CPU Time (s)")
timePassed = (t2 - t1)
print("%.2f" % timePassed)
print()

print('Optimal Solution: ')
print(optimalSolutions[int(dataset)-1])
print()

print('Optimal Objective: ')
print(optimalObjectives[int(dataset)-1])