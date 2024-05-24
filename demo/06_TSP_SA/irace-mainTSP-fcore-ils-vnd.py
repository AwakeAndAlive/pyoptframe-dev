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
                self.vx.append(float(id_x_y[1]))
                self.vy.append(float(id_x_y[2]))
            self.dist = [[0 for col in range(self.n)] for row in range(self.n)]
            for i in range(self.n):
                for j in range(self.n):
                    self.dist[i][j] = round(self.euclidean(self.vx[i], self.vy[i], self.vx[j], self.vy[j]))

    def euclidean(self, x1, y1, x2, y2):
        import math
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    def __str__(self):
        return f"ProblemContextTSP(n={self.n};vx={self.vx};vy={self.vy};dist={self.dist})"

    @staticmethod
    def minimize(pTSP: 'ProblemContextTSP', s: SolutionTSP) -> float:
        assert (s.n == pTSP.n)
        assert (len(s.cities) == s.n)
        f = 0.0
        for i in range(pTSP.n-1):
          f += pTSP.dist[s.cities[i]][s.cities[i + 1]];
        f += pTSP.dist[s.cities[int(pTSP.n) - 1]][s.cities[0]];
        return f

    @staticmethod
    def generateSolution(problemCtx: 'ProblemContextTSP') -> SolutionTSP:
        sol = SolutionTSP()
        for i in range(problemCtx.n):
            sol.cities.append(i)
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
        import random
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

# Adaptations for irace.
instancia = "instances/01_berlin52.tsp"
time_limit = 3
localSearch = 0
iterMax = 10
maxPert = 5
for i in range(len(sys.argv)):
    if (sys.argv[i] == "--time"):
        time_limit = int(sys.argv[i + 1])
    if (sys.argv[i] == "-i"):
        instancia = str(sys.argv[i + 1])
    if (sys.argv[i] == "--seed"):
        random.seed(int(sys.argv[i + 1]))
    if (sys.argv[i] == "--localSearch"):
        localSearch = int(sys.argv[i + 1])
    if (sys.argv[i] == "--iterMax"):
        iterMax = int(sys.argv[i + 1])
    if (sys.argv[i] == "--maxPert"):
        maxPert = float(sys.argv[i + 1])

from optframe.heuristics import *

# loads problem from filesystem
pTSP = ProblemContextTSP()

# set SILENT
pTSP.engine.experimental_set_parameter("ENGINE_LOG_LEVEL", "0")
pTSP.engine.experimental_set_parameter("COMPONENT_LOG_LEVEL", "0")

# load
pTSP.load(instancia)

# Register Basic Components
comp_list = pTSP.engine.setup(pTSP)

# Print components to debug
print("Component List: ", comp_list)
for idx, comp in enumerate(comp_list):
    print(f"Component {idx}: {comp} ({type(comp)})")

# Verificação dos componentes
assert isinstance(comp_list[0], IdGeneralEvaluator), "Component 0 should be IdGeneralEvaluator"
assert isinstance(comp_list[3], IdInitialSearch), "Component 3 should be IdInitialSearch"

# get index of new NS
ns_idx = pTSP.engine.add_ns_class(pTSP, NSSwap)

# get index of new NSSeq
nsseq_idx = pTSP.engine.add_nsseq_class(pTSP, NSSeqSwap)

# pack NS into a NS list
list_idx = pTSP.engine.create_component_list(
    "[ OptFrame:NS 0 ]", "OptFrame:NS[]")

bi = BestImprovement(pTSP.engine, 0, 0)
ls_idx = bi.get_id()

list_vnd_idx = pTSP.engine.create_component_list(
    "[ OptFrame:LocalSearch 0 ]", "OptFrame:LocalSearch[]")

vnd = VariableNeighborhoodDescent(pTSP.engine, 0, 0)
vnd_idx = vnd.get_id()

ilsl_pert = ILSLevelPertLPlus2(pTSP.engine, 0, 0)
pert_idx = ilsl_pert.get_id()
ilslpert = IdILSLevelPert(pert_idx)

class ILSLevels(SingleObjSearch):
    def __init__(self, _engine: XEngine, _ev: IdGeneralEvaluator, _is: IdInitialSearch, _ls: IdLocalSearch, _ilslpert: IdILSLevelPert, iterMax: int, maxPert: int):
        assert isinstance(_engine, XEngine)
        assert isinstance(_ev, IdGeneralEvaluator), f"_ev should be IdGeneralEvaluator, got {type(_ev)}"
        assert isinstance(_is, IdInitialSearch), f"_is should be IdInitialSearch, got {type(_is)}"
        assert isinstance(_ls, IdLocalSearch), f"_ls should be IdLocalSearch, got {type(_ls)}"
        assert isinstance(_ilslpert, IdILSLevelPert), f"_ilslpert should be IdILSLevelPert, got {type(_ilslpert)}"
        self.engine = _engine
        self.iterMax = iterMax
        self.maxPert = maxPert
        self.ev = _ev
        self.is_ = _is
        self.ls = _ls
        self.ilslpert = _ilslpert
        str_code = "OptFrame:ComponentBuilder:SingleObjSearch:ILS:ILSLevels"
        str_args = "OptFrame:GeneralEvaluator:Evaluator " + str(_ev.id) + " OptFrame:InitialSearch " + str(_is.id) + " OptFrame:LocalSearch " + str(_ls.id) + " OptFrame:ILS:LevelPert " + str(_ilslpert.id) + " " + str(iterMax) + " " + str(maxPert)
        self.g_idx = self.engine.build_global_search(str_code, str_args)

    def search(self, timelimit: float) -> SearchOutput:
        start_time = time.time()
        best_solution = None
        best_evaluation = float('inf')

        for iteration in range(self.iterMax):
            if time.time() - start_time > timelimit:
                break
            print(f"Starting iteration {iteration}")
            evaluator = self.engine.get_evaluator(self.ev.id)
            constructive = self.engine.get_constructive(comp_list[2])
            current_solution = self.engine.fconstructive_gensolution(constructive)
            current_evaluation = self.engine.fevaluator_evaluate(evaluator, True, current_solution)

            for perturbation in range(self.maxPert):
                if time.time() - start_time > timelimit:
                    break
                print(f"Perturbation {perturbation} of iteration {iteration}")
                perturbed_solution = self.engine.run_global_search(self.ilslpert.id, current_solution)
                perturbed_evaluation = self.engine.fevaluator_evaluate(evaluator, True, perturbed_solution)

                if perturbed_evaluation < current_evaluation:
                    current_solution = perturbed_solution
                    current_evaluation = perturbed_evaluation

                if current_evaluation < best_evaluation:
                    best_solution = current_solution
                    best_evaluation = current_evaluation

        print(f"Finished {iteration} iterations")
        lout = SearchOutput(best_solution, best_evaluation)
        return lout

ilsl = ILSLevels(pTSP.engine, comp_list[0], comp_list[3], ls_idx, ilslpert, iterMax, maxPert)
lout = ilsl.search(3.0)
print(f"Best solution: {lout.best_s}")
print(f"Best evaluation: {lout.best_e}")
