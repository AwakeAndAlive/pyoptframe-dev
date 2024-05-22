import random
import math
import time
import sys

# Solution class for TSP
class TSPSolution:
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost

# Euclidean distance
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
def generate_neighbor(path):
    new_path = path[:]
    i = random.randint(0, len(new_path) - 2)
    j = random.randint(i + 1, len(new_path) - 1)
    new_path[i:j + 1] = reversed(new_path[i:j + 1])
    return new_path

# Geometric cooling function
def geometric_cooling(temp, alpha):
    return temp * alpha

# SA function for TSP
def simulated_annealing_tsp(coordinates, initial_temp, final_temp, max_iterations, alpha, SAmax):
    random.seed(time.time())
    
    n = len(coordinates)
    current_solution = generate_initial_solution(n)
    current_cost = calculate_total_cost(coordinates, current_solution)
    
    best_solution = current_solution
    best_cost = current_cost
    
    temp = initial_temp
    
    # Determine the number of cooling steps
    cooling_steps = max_iterations // SAmax
    if cooling_steps == 0:
        cooling_steps = 1
    
    for _ in range(cooling_steps):
        for _ in range(SAmax):
            new_solution = generate_neighbor(current_solution)
            new_cost = calculate_total_cost(coordinates, new_solution)
            
            # Acceptance criteria
            if new_cost < current_cost or math.exp((current_cost - new_cost) / temp) > random.random():
                current_solution = new_solution
                current_cost = new_cost
            
            # Update best solution
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
        
        # Apply geometric cooling
        temp = geometric_cooling(temp, alpha)
        
        # Check if temperature has reached final_temp
        if temp < final_temp:
            break
    
    return TSPSolution(best_solution, best_cost)

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
        print("Usage: irace-pytsp-sa.py <instance file> <initial_temp> <alpha> <final_temp> <max_iterations> <SAmax>")
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

    best_solution = simulated_annealing_tsp(coordinates, initial_temp, final_temp, max_iterations, alpha, SAmax)
    print(best_solution.cost)

if __name__ == "__main__":
    main()
