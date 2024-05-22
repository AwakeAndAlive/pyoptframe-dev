import random
import math
import time
import sys
from optframe import *
from optframe.protocols import *
from optframe.heuristics import BasicSimulatedAnnealing
from optframe.components import Move, IdInitialSearch, IdNS, IdListNS, IdGeneralEvaluator, IdEvaluator, IdConstructive

# Solution class for TSP
class TSPSolution(XSolution):
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost

    def __str__(self):
        return f"TSPSolution(path={self.path}, cost={self.cost})"

# Euclidean distance function
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Total solution cost
def calculate_total_cost(coordinates, path):
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += euclidean_distance(coordinates[path[i]], coordinates[path[i + 1]])
    total_cost += euclidean_distance(coordinates[path[-1]], coordinates[path[0]])  # Closing the loop
    return total_cost

# Initial random solution
def generate_initial_solution(n):
    path = list(range(n))
    random.shuffle(path)
    return path

# Generate with 2-opt swap
class Move2Opt(Move):
    def __init__(self, i, j, coordinates):
        self.i = i
        self.j = j
        self.coordinates = coordinates

    def apply(self, problem, sol):
        new_path = sol.path[:]
        new_path[self.i:self.j+1] = reversed(new_path[self.i:self.j+1])
        new_cost = calculate_total_cost(self.coordinates, new_path)
        return TSPSolution(new_path, new_cost)

    def __str__(self):
        return f"Move2Opt(i={self.i}, j={self.j})"

class NS2Opt(IdNS):
    def __init__(self, id, coordinates):
        super().__init__(id)
        self.coordinates = coordinates

    def randomMove(self, sol):
        i = random.randint(0, len(sol.path) - 2)
        j = random.randint(i + 1, len(sol.path) - 1)
        return Move2Opt(i, j, self.coordinates)

class BasicInitialSearch(IdInitialSearch):
    def __init__(self, id, engine, constructive, evaluator):
        super().__init__(id)
        self.engine = engine
        self.constructive = constructive
        self.evaluator = evaluator

    def generateSolution(self):
        sol = self.engine.fconstructive_gensolution(self.constructive)
        eval = self.engine.fevaluator_evaluate(self.evaluator, True, sol)
        return sol, eval

# Read TSP instance file
def read_instance(instance_file_path):
    try:
        with open(instance_file_path, 'r') as infile:
            lines = infile.readlines()
    except IOError:
        print(f"Error: Could not open TSP file: {instance_file_path}")
        return None

    coordinates = []
    for line in lines[1:]:  # Skip the first line (number of cities)
        index, x, y = map(float, line.split())
        coordinates.append((x, y))

    if not coordinates:
        print(f"Error: No coordinates loaded from TSP file: {instance_file_path}")
        return None

    return coordinates

# Main for irace
def main():
    if len(sys.argv) < 7:
        print("Usage: irace-mainTSP-fcore-sa.py <instance file> <initial_temp> <alpha> <final_temp> <max_iterations> <SAmax>")
        sys.exit(1)

    instance_file = sys.argv[1]
    initial_temp = float(sys.argv[2])
    alpha = float(sys.argv[3])
    final_temp = float(sys.argv[4])
    max_iterations = int(sys.argv[5])
    SAmax = int(sys.argv[6])

    coordinates = read_instance(instance_file)
    if coordinates is None:
        sys.exit(1)

    engine = Engine(APILevel.API1d)
    constructive = IdConstructive(engine.getComponentId())
    
    # Define custom evaluator
    class TSPEvaluator(IdEvaluator):
        def __init__(self, id, coordinates):
            super().__init__(id)
            self.coordinates = coordinates

        def evaluate(self, sol):
            return calculate_total_cost(self.coordinates, sol.path)

    evaluator = TSPEvaluator(engine.getComponentId(), coordinates)
    
    initial_search = BasicInitialSearch(engine.getComponentId(), engine, constructive, evaluator)
    ns = NS2Opt(engine.getComponentId(), coordinates)
    ns_list = IdListNS(engine.getComponentId())
    ns_list.addNS(ns)

    sa = BasicSimulatedAnnealing(engine, evaluator, initial_search, ns_list, alpha, max_iterations, initial_temp, final_temp, SAmax)

    start_time = time.time()

    status = sa.search(30.0)

    end_time = time.time()
    execution_time = end_time - start_time

    best_solution = status.best_s  # Adjusted here
    best_evaluation = status.best_e

    # Calculate improvement
    initial_solution, initial_cost = initial_search.generateSolution()
    final_cost = best_evaluation
    improvement = ((initial_cost - final_cost) / initial_cost) * 100

    print(f"Instance name: {instance_file};")
    print(f"SA variables: initialTemp = {initial_temp}; maxIterations = {max_iterations}; coolingRate = {alpha};")
    print(f"Initial evaluation: {initial_cost}")
    print(f"Initial solution: {initial_solution}")
    print("Best solution found by Simulated Annealing: ")
    print(best_solution)
    print(f"Best evaluation: {final_cost}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()
