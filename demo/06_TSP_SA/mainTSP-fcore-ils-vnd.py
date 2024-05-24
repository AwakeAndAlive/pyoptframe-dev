# OptFrame Python Demo TSP - Traveling Salesman Problem

from typing import List
import random
import time
import sys

from optframe import *
from optframe.protocols import *

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
                self.vx.append(int(id_x_y[1]))
                self.vy.append(int(id_x_y[2]))
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
        f = 0.0
        for i in range(pTSP.n - 1):
            f += pTSP.dist[s.cities[i]][s.cities[i + 1]]
        f += pTSP.dist[s.cities[pTSP.n - 1]][s.cities[0]]
        return f

    @staticmethod
    def generateSolution(problemCtx: 'ProblemContextTSP') -> SolutionTSP:
        sol = SolutionTSP()
        sol.cities = list(range(problemCtx.n))
        random.shuffle(sol.cities)
        sol.n = problemCtx.n
        return sol

from optframe.components import Move

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
    def __init__(self, _i: int, _j: int):
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

class NSSeqSwap(object):
    @staticmethod
    def randomMove(pTSP: ProblemContextTSP, sol: SolutionTSP) -> MoveSwapClass:
        return NSSwap.randomMove(pTSP, sol)

    @staticmethod
    def getIterator(pTSP: ProblemContextTSP, sol: SolutionTSP) -> IteratorSwap:
        return IteratorSwap(-1, -1)

from optframe.heuristics import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python mainTSP-fcore-ils-vnd.py <instance_file>")
        sys.exit(1)

    instance_file = sys.argv[1]

    random.seed(time.time())

    # Load problem instance
    pTSP = ProblemContextTSP()
    pTSP.load(instance_file)
    print(pTSP)

    # Register Basic Components
    comp_list = pTSP.engine.setup(pTSP)
    print(comp_list)

    # Get index of new NS
    ns_idx = pTSP.engine.add_ns_class(pTSP, NSSwap)
    print("ns_idx=", ns_idx)

    # Get index of new NSSeq
    nsseq_idx = pTSP.engine.add_nsseq_class(pTSP, NSSeqSwap)
    print("nsseq_idx=", nsseq_idx)

    gev_idx = comp_list[0]
    ev_idx = comp_list[1]
    print("evaluator id:", ev_idx)

    c_idx = comp_list[2]
    print("c_idx=", c_idx)

    is_idx = IdInitialSearch(c_idx)
    print("is_idx=", is_idx)

    # Create initial solution
    initial_sol = ProblemContextTSP.generateSolution(pTSP)
    initial_eval = ProblemContextTSP.minimize(pTSP, initial_sol)
    print("0000000000Initial solution:", initial_sol)
    print("Initial evaluation:", initial_eval)

    # ILS with VND
    list_vnd_idx = pTSP.engine.create_component_list(
    "[ OptFrame:LocalSearch 0 ]", "OptFrame:LocalSearch[]")
    print("111111111list_vnd_idx=", list_vnd_idx)

    vnd = VariableNeighborhoodDescent(pTSP.engine, gev_idx, list_vnd_idx)
    print("vnd created")

    vnd_idx = vnd.get_id()
    print("vnd_idx=", vnd_idx)

    ilsl_pert = ILSLevelPertLPlus2(pTSP.engine, 0, 0)
    pert_idx = ilsl_pert.get_id()
    print("pert_idx=", pert_idx)

    ilsl = ILSLevels(pTSP.engine, 0, 0, 1, 0, 10, 5)
    print("ilsl created")

    print("will start ILS for 3 seconds")
    lout = ilsl.search(3.0)
    print("Best solution:", lout.best_s)
    print("Best evaluation:", lout.best_e)

    improvement = 100.0 * (initial_eval - lout.best_e) / initial_eval
    print("Improvement: {:.4f}%".format(improvement))

    print("FINISHED")
