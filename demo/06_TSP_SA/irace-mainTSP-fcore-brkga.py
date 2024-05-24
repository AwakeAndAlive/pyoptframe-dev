import time
import random
import argparse
from typing import List, Tuple, Union
from optframe import *
from optframe.protocols import *
from optframe.components import Move, NSIterator
from optframe.heuristics import BRKGA
from optframe.core import LibArrayDouble

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
                id_x_y = lines[i+1].split()
                self.vx.append(float(id_x_y[1]))  # Allow floats
                self.vy.append(float(id_x_y[2]))  # Allow floats
            self.dist = [[0 for col in range(self.n)] for row in range(self.n)]
            for i in range(self.n):
                for j in range(self.n):
                    self.dist[i][j] = self.euclidean(self.vx[i], self.vy[i], self.vx[j], self.vy[j])

    def euclidean(self, x1, y1, x2, y2):
        import math
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __str__(self):
        return f"ProblemContextTSP(n={self.n};vx={self.vx};vy={self.vy};dist={self.dist})"

    @staticmethod
    def minimize(pTSP: 'ProblemContextTSP', s: SolutionTSP) -> float:
        f = 0.0
        for i in range(pTSP.n-1):
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

class MoveSwapClass(Move):
    def __init__(self, _i: int = 0, _j: int = 0):
        self.i = _i
        self.j = _j

    def __str__(self):
        return "MoveSwapClass(i="+str(self.i)+";j="+str(self.j)+")"

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

class IteratorSwap(NSIterator):
    def __init__(self, _i: int, _j: int):
        self.i = _i
        self.j = _j

    def first(self, pTSP: ProblemContextTSP):
        self.i = 0
        self.j = 1

    def next(self, pTSP: ProblemContextTSP):
        if self.j < pTSP.n - 1:
            self.j = self.j + 1
        else:
            self.i = self.i + 1
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

class RKConstructiveTSP(object):
    @staticmethod
    def generateRK(problemCtx: ProblemContextTSP, ptr_array_double: LibArrayDouble) -> int:
        rkeys = []
        for i in range(problemCtx.n):
            key = random.random()
            rkeys.append(key)
        ptr_array_double.contents.size = len(rkeys)
        ptr_array_double.contents.v = engine.callback_adapter_list_to_vecdouble(rkeys)
        return len(rkeys)

class DecoderTSP(object):
    @staticmethod
    def decodeSolution(pTSP: ProblemContextTSP, array_double: LibArrayDouble) -> SolutionTSP:
        sol = SolutionTSP()
        lpairs = []
        for i in range(array_double.size):
            p = [array_double.v[i], i]
            lpairs.append(p)
        sorted_list = sorted(lpairs)
        sol.n = pTSP.n
        sol.cities = []
        for i in range(array_double.size):
            sol.cities.append(sorted_list[i][1])
        return sol

    @staticmethod
    def decodeMinimize(pTSP: ProblemContextTSP, array_double: LibArrayDouble, needsSolution: bool) -> Tuple[Union[SolutionTSP, None], float]:
        sol = DecoderTSP.decodeSolution(pTSP, array_double)
        e = ProblemContextTSP.minimize(pTSP, sol)
        if not needsSolution:
            return (None, e)
        else:
            return (sol, e)

def main():
    parser = argparse.ArgumentParser(description='Run TSP with BRKGA')
    parser.add_argument('instance', type=str, help='Path to the TSP instance file')
    parser.add_argument('population_size', type=int, help='Population size for BRKGA')
    parser.add_argument('num_generations', type=int, help='Number of generations for BRKGA')
    parser.add_argument('elite_proportion', type=float, help='Elite proportion for BRKGA')
    parser.add_argument('mutant_proportion', type=float, help='Mutant proportion for BRKGA')
    parser.add_argument('elite_inheritance_probability', type=float, help='Elite inheritance probability for BRKGA')

    args = parser.parse_args()

    random.seed(0)

    pTSP = ProblemContextTSP()
    pTSP.load(args.instance)

    print("problem=", pTSP)

    import optframe
    print(str(optframe.__version__))
    pTSP.engine.welcome()

    comp_list = pTSP.engine.setup(pTSP)
    print(comp_list)

    ev_idx = comp_list[1]
    print("evaluator id:", ev_idx)

    c_rk_idx = pTSP.engine.add_constructive_rk_class(pTSP, RKConstructiveTSP)
    print("c_rk_idx=", c_rk_idx)

    print("")
    dec_rk_idx = pTSP.engine.add_decoder_rk_class(pTSP, DecoderTSP)
    print("dec_rk_idx=", dec_rk_idx)

    print("")
    print("WILL CREATE DecoderRandomKeys directly with simultaneous evaluation and optional solution!")
    drk_rk_id = pTSP.engine.add_edecoder_op_rk_class(pTSP, DecoderTSP)
    print("drk_rk_id=", drk_rk_id)

    pTSP.engine.list_components("OptFrame:")

    print("")
    print("will start BRKGA for 3 seconds")

    brkga = BRKGA(
        pTSP.engine, drk_rk_id, c_rk_idx,
        args.population_size, args.num_generations,
        args.elite_proportion, args.mutant_proportion, args.elite_inheritance_probability
    )

    pTSP.engine.list_components("OptFrame:")

    # Start timer
    start_time = time.time()

    # Initial solution and evaluation
    initial_sol = ProblemContextTSP.generateSolution(pTSP)
    initial_eval = ProblemContextTSP.minimize(pTSP, initial_sol)

    lout = brkga.search(3000.0)
    best_solution = lout.best_s
    best_evaluation = lout.best_e

    # Calculate improvement
    improvement = 100.0 * (initial_eval - best_evaluation) / initial_eval

    # Print results
    print("Initial evaluation:", initial_eval)
    print("Initial solution:", initial_sol.cities)
    print(f"BRKGA parameters: population size = {args.population_size}, num generations = {args.num_generations}, elite proportion = {args.elite_proportion}, mutant proportion = {args.mutant_proportion}, elite inheritance probability = {args.elite_inheritance_probability}")
    print("Best solution found by BRKGA: Evaluation function value =", best_evaluation)
    print("Best solution:", best_solution.cities)
    print("Improvement:", improvement, "%")

    # Print execution time
    execution_time = time.time() - start_time
    print("Execution Time:", execution_time, "seconds")

if __name__ == "__main__":
    main()
