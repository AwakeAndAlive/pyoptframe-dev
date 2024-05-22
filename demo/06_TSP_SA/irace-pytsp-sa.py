import random
import math
import time
import sys

# Solution class for TSP
class TSPSolution:
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost

# Euclidean distance function
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Calculate the total cost of a solution
def calculate_total_cost(coordinates, path):
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += euclidean_distance(coordinates[path[i]], coordinates[path[i + 1]])
    total_cost += euclidean_distance(coordinates[path[-1]], coordinates[path[0]])  # Closing the loop
    return total_cost

# Generate an initial random solution
def generate_initial_solution(n):
    path = list(range(n))
    random.shuffle(path)
    return path

# Generate a neighboring solution (swap)
def generate_neighbor(path):
    new_path = path[:]
    i = random.randint(0, len(new_path) - 1)
    j = random.randint(0, len(new_path) - 1)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# Geometric cooling function
def geometric_cooling(temp, alpha):
    return temp * alpha

# Main Simulated Annealing function for TSP
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
    for line in lines[1:]:  # Skip the first line if it doesn't contain coordinates
        index, x, y = map(float, line.split())
        coordinates.append((x, y))

    if not coordinates:
        print(f"Error: No coordinates loaded from TSP file: {instance_file_path}")
        return None

    return coordinates

# Main function for execution with irace
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
