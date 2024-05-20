# OptFrame Python Demo TSP - Traveling Salesman Problem

from typing import List
import random

from optframe import *
from optframe.protocols import *
from optframe.heuristics import BasicSimulatedAnnealing
from optframe.components import Move, IdInitialSearch, IdNS, IdNSSeq, IdListNS, IdGeneralEvaluator, IdEvaluator, IdConstructive

class SolutionTSP(object):
    def __init__(self):
        self.n: int = 0
        self.cities: List[int] = []

    def __str__(self):
        return f"SolutionTSP(n={self.n};cities={self.cities})"

class ProblemContextTSP(object):
    def __init__(self):
        self.engine = Engine(APILevel.API1d)
        self.n = 0
        self.vx = []
        self.vy = []
        self.dist = []

    def load(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
            self.n = int(lines[0])
            for i in range(self.n):
                id_x_y = lines[i + 1].split()
                self.vx.append(float(id_x_y[1]))
                self.vy.append(float(id_x_y[2]))
            self.dist = [[0 for _ in range(self.n)] for _ in range(self.n)]
            for i in range(self.n):
                for j in range(self.n):
                    self.dist[i][j] = round(self.euclidean(self.vx[i], self.vy[i], self.vx[j], self.vy[j]))

    def euclidean(self, x1, y1, x2, y2):
        import math
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __str__(self):
        return f"ProblemContextTSP(n={self.n};vx={self.vx};vy={self.vy};dist={self.dist})"

    @staticmethod
    def minimize(pTSP: 'ProblemContextTSP', s: SolutionTSP) -> float:
        assert s.n == pTSP.n
        assert len(s.cities) == s.n
        f = 0.0
        for i in range(pTSP.n - 1):
            f += pTSP.dist[s.cities[i]][s.cities[i + 1]]
        f += pTSP.dist[s.cities[pTSP.n - 1]][s.cities[0]]
        return f

    @staticmethod
    def generateSolution(problemCtx: 'ProblemContextTSP') -> SolutionTSP:
        sol = SolutionTSP()
        for i in range(problemCtx.n):
            sol.cities.append(i)
        random.shuffle(sol.cities)
        sol.n = problemCtx.n
        return sol

assert isinstance(SolutionTSP, XSolution)
assert isinstance(ProblemContextTSP, XProblem)
assert isinstance(ProblemContextTSP, XConstructive)
assert isinstance(ProblemContextTSP, XMinimize)

class MoveSwapClass(Move):
    def __init__(self, _i: int = 0, _j: int = 0):
        self.i = _i
        self.j = _j

    def __str__(self):
        return "MoveSwapClass(i=" + str(self.i) + ";j=" + str(self.j) + ")"

    def apply(self, problemCtx, sol: SolutionTSP) -> 'MoveSwapClass':
        aux = sol.cities[self.j]
        sol.cities[self.j] = sol.cities[self.i]
        sol.cities[self.i] = aux
        return MoveSwapClass(self.j, self.i)

    def canBeApplied(self, problemCtx, sol: SolutionTSP) -> bool:
        return True

    def eq(self, problemCtx, m2: 'MoveSwapClass') -> bool:
        return (self.i == m2.i) and (self.j == m2.j)

assert isinstance(MoveSwapClass, XMove)
assert MoveSwapClass in Move.__subclasses__()

class NSSwap(object):
    @staticmethod
    def randomMove(pTSP, sol: SolutionTSP) -> MoveSwapClass:
        n = sol.n
        i = random.randint(0, n - 1)
        j = i
        while j <= i:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
        return MoveSwapClass(i, j)

from optframe.components import NSIterator

class IteratorSwap(NSIterator):
    def __init__(self, _i: int = 0, _j: int = 0):
        self.i = _i
        self.j = _j

    def first(self, pTSP: ProblemContextTSP):
        self.i = 0
        self.j = 1

    def next(self, pTSP: ProblemContextTSP):
        if self.j < pTSP.n - 1:
            self.j += 1
        else:
            self.i += 1
            self.j = self.i + 1

    def isDone(self, pTSP: ProblemContextTSP):
        return self.i >= pTSP.n - 1

    def current(self, pTSP: ProblemContextTSP):
        return MoveSwapClass(self.i, self.j)

assert IteratorSwap in NSIterator.__subclasses__()

class NSSeqSwap(object):
    @staticmethod
    def randomMove(pTSP: ProblemContextTSP, sol: SolutionTSP) -> MoveSwapClass:
        return NSSwap.randomMove(pTSP, sol)

    @staticmethod
    def getIterator(pTSP: ProblemContextTSP, sol: SolutionTSP) -> IteratorSwap:
        return IteratorSwap(-1, -1)

class BasicInitialSearch(object):
    def __init__(self, engine, constructive, evaluator):
        self.engine = engine
        self.constructive = constructive
        self.evaluator = evaluator

    def generateSolution(self):
        sol = self.engine.fconstructive_gensolution(self.constructive)
        eval = self.engine.fevaluator_evaluate(self.evaluator, True, sol)
        return sol, eval

# set random seed for system
random.seed(0)

# loads problem from filesystem
pTSP = ProblemContextTSP()
pTSP.load('instances/00_firsttest.tsp')
print(pTSP)

# Register Basic Components
comp_list = pTSP.engine.setup(pTSP)
print(comp_list)

# get index of new NS
ns_idx = pTSP.engine.add_ns_class(pTSP, NSSwap)
print("ns_idx=", ns_idx)

# get index of new NSSeq
nsseq_idx = pTSP.engine.add_nsseq_class(pTSP, NSSeqSwap)
print("nsseq_idx=", nsseq_idx)

# test each component
gev_idx = comp_list[0] # GeneralEvaluator
ev_idx = comp_list[1] # Evaluator
print("evaluator id:", ev_idx)

c_idx = comp_list[2]
print("c_idx=", c_idx)

fev = pTSP.engine.get_evaluator(ev_idx)
pTSP.engine.print_component(fev)

fc = pTSP.engine.get_constructive(c_idx)
pTSP.engine.print_component(fc)

solxx = pTSP.engine.fconstructive_gensolution(fc)
print("test solution:", solxx)

z1 = pTSP.engine.fevaluator_evaluate(fev, True, solxx)
print("test evaluation:", z1)

initial_search = BasicInitialSearch(pTSP.engine, fc, fev)
print(initial_search)

lns_idx = pTSP.engine.create_component_list("[ OptFrame:NS 0 ]", "OptFrame:NS[]")
print("lns_idx=", lns_idx)

initial_temp = 100000.0
cooling_rate = 0.98
max_iterations = 20000

sa = BasicSimulatedAnnealing(pTSP.engine, gev_idx, IdInitialSearch(0), lns_idx, cooling_rate, max_iterations, initial_temp)
print(sa)

status = sa.search(30.0)
best_solution = status.best_s  # Adjusted here
best_evaluation = status.best_e

print("Instance name: 00_firsttest.tsp;")
print(f"SA variables: initialTemp = {initial_temp}; maxIterations = {max_iterations}; coolingRate = {cooling_rate};")
print("Initial evaluation: ", z1)
print("Initial solution: ", solxx)

print("Best solution found by Simulated Annealing: ")
print(best_solution)
print("Best evaluation: ", best_evaluation)
