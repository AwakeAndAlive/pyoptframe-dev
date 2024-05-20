import random
import math
import time
import os
import sys

# Solution class
class TSPSolution:
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost

# Euclidean distance
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Total cost
def calculate_total_cost(coordinates, path):
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += euclidean_distance(coordinates[path[i]], coordinates[path[i + 1]])
    total_cost += euclidean_distance(coordinates[path[-1]], coordinates[path[0]])
    return total_cost

# Random constructive 
def generate_initial_solution(n):
    path = list(range(n))
    random.shuffle(path)
    return path

# neighborhood
def generate_neighbor(path):
    new_path = path[:]
    i = random.randint(0, len(new_path) - 1)
    j = random.randint(0, len(new_path) - 1)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# Cooling
def exponential_cooling(initial_temp, iteration, cooling_rate):
    return initial_temp * (cooling_rate ** iteration)

# TSP SA
def simulated_annealing_tsp(coordinates, initial_temp, final_temp, max_iterations, cooling_rate):
    random.seed(time.time())

    n = len(coordinates)
    current_solution = generate_initial_solution(n)
    current_cost = calculate_total_cost(coordinates, current_solution)

    best_solution = current_solution
    best_cost = current_cost

    for iter in range(max_iterations):
        temp = exponential_cooling(initial_temp, iter, cooling_rate)
        if temp < final_temp:
            break

        new_solution = generate_neighbor(current_solution)
        new_cost = calculate_total_cost(coordinates, new_solution)

        if new_cost < current_cost or math.exp((current_cost - new_cost) / temp) > random.random():
            current_solution = new_solution
            current_cost = new_cost

        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
            print(f"Best fo: {best_cost} Found on Iter = {iter} and T = {temp};")

    return TSPSolution(best_solution, best_cost)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <instance file>")
        sys.exit(1)

    instance_file_path = sys.argv[1]
    if not os.path.exists(instance_file_path):
        print("Error: Could not find TSP file at the specified path.")
        return
    
    instance_name = os.path.basename(instance_file_path)

    # Reading the instance ex: python3 pytsp-sa.py instances/<name>
    try:
        with open(instance_file_path, 'r') as infile:
            lines = infile.readlines()
    except IOError:
        print("Error: Could not open TSP file.")
        return

    # Coordinates
    coordinates = []
    for line in lines[1:]:
        index, x, y = map(float, line.split())
        coordinates.append((x, y))

    if not coordinates:
        print("Error: No coordinates loaded from TSP file.")
        return

    initial_temp = 99999.0
    final_temp = 0.0001
    max_iterations = 10000
    cooling_rate = 0.98

    start_time = time.time()

    # Saving inicial for comparison
    initial_solution = generate_initial_solution(len(coordinates))
    initial_cost = calculate_total_cost(coordinates, initial_solution)

    best_solution = simulated_annealing_tsp(coordinates, initial_temp, final_temp, max_iterations, cooling_rate)
    end_time = time.time()

    duration = (end_time - start_time) * 1e6  

    final_cost = best_solution.cost
    improvement = ((initial_cost - final_cost) / initial_cost) * 100

    # Results
    print(f"Instance name: {instance_name};")
    print(f"SA variables: initialTemp = {initial_temp}; finalTemp = {final_temp}; maxIterations = {max_iterations}; coolingRate = {cooling_rate};")
    print(f"First solution evaluation: Evaluation function value = {initial_cost};")
    print(f"First solution: vector({len(initial_solution)}) [{', '.join(map(str, initial_solution))}];")
    print(f"Best solution evaluation: Evaluation function value = {best_solution.cost};")
    print(f"Final solution: vector({len(best_solution.path)}) [{', '.join(map(str, best_solution.path))}];")
    print(f"Improvement: {improvement:.2f}%;")
    print(f"Execution time: {duration} microseconds;")

# Executar a função principal
if __name__ == "__main__":
    main()
