#!/usr/bin/python3

import os
import random
import numpy as np
from typing import List

# DO NOT REORDER 'import sys ...'
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
#
from optframe import *

class ExampleSol(object):
    def __init__(self):
        self.n : int = 0
        self.bag : List[int] = []
    def __str__(self):
        return f"ExampleSol(n={self.n};bag={self.bag})"

class ExampleKP(object):
    def __init__(self):
        self.engine = Engine()
        self.n : int = 0          # number of items
        self.w : List[float] = [] # item weights
        self.p : List[float] = [] # item profits
        self.Q : float = 0.0      # knapsack capacity
    def __str__(self):
        return f"ExampleKP(n={self.n};Q={self.Q};w={self.w};p={self.p})"
    @staticmethod
    def generateSolution(problem: 'ExampleKP') -> ExampleSol:
        sol = ExampleSol()
        sol.n = problem.n
        sol.bag = [random.randint(0, 1) for _ in range(sol.n)]
        return sol
    @staticmethod
    def maximize(pKP: 'ExampleKP', sol: ExampleSol) -> float:
        wsum = np.dot(sol.bag, pKP.w)
        if wsum > pKP.Q:
            return -1000.0*(wsum - pKP.Q)
        return np.dot(sol.bag, pKP.p)

assert isinstance(ExampleSol, XSolution)    # composition tests 
assert isinstance(ExampleKP, XProblem)      # composition tests 
assert isinstance(ExampleKP, XConstructive) # composition tests    
assert isinstance(ExampleKP, XMaximize)     # composition tests

class MoveBitFlip(object):
    def __init__(self, _k :int):
        self.k = _k
    @staticmethod
    def apply(problemCtx: ExampleKP, m: 'MoveBitFlip', sol: ExampleSol) -> 'MoveBitFlip':
        sol.bag[m.k] = 1 - sol.bag[m.k]
        return MoveBitFlip(m.k)
    @staticmethod
    def canBeApplied(problemCtx: ExampleKP, m: 'MoveBitFlip', sol: ExampleSol) -> bool:
        return True
    @staticmethod
    def eq(problemCtx: ExampleKP, m1: 'MoveBitFlip', m2: 'MoveBitFlip') -> bool:
        return m1.k == m2.k

class NSBitFlip(object):
    @staticmethod
    def randomMove(pKP: ExampleKP, sol: ExampleSol) -> MoveBitFlip:
        return MoveBitFlip(random.randint(0, pKP.n - 1))

assert isinstance(MoveBitFlip, XMove) # composition tests
assert isinstance(NSBitFlip, XNS)     # composition tests

#
pKP = ExampleKP()
pKP.n = 5
pKP.w = [1, 2, 3, 4, 5]
pKP.p = [5, 4, 3, 2, 1]
pKP.Q = 6.0
#

v = pKP.engine.setup(pKP)
print("v=",v)

pKP.engine.add_ns_class(pKP, NSBitFlip) 

list_idx = pKP.engine.create_component_list(
    "[ OptFrame:NS 0 ]", "OptFrame:NS[]")

# LogLevel::Info(3) for check module
pKP.engine.experimental_set_parameter("ENGINE_LOG_LEVEL", "0")

pKP.engine.check(100, 10, False)

sa = BasicSimulatedAnnealing(pKP.engine, 0, 0, list_idx, 0.99, 100, 999)
sout = sa.search(4.0)
print("Best solution: ",   sout.best_s)
print("Best evaluation: ", sout.best_e)
