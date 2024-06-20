import random
import math
import time
import sys

# Solution class for TSP
class TSPSolution:
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
def generate_neighbor_2opt(path):
    new_path = path[:]
    i = random.randint(0, len(new_path) - 2)
    j = random.randint(i + 1, len(new_path) - 1)
    new_path[i:j + 1] = reversed(new_path[i:j + 1])
    return new_path

# Generate with 3-opt swap
def generate_neighbor_3opt(path):
    new_path = path[:]
    a = random.randint(0, len(new_path) - 3)
    b = random.randint(a + 1, len(new_path) - 2)
    c = random.randint(b + 1, len(new_path) - 1)
    
    segment1 = new_path[:a+1]
    segment2 = new_path[a+1:b+1]
    segment3 = new_path[b+1:c+1]
    segment4 = new_path[c+1:]
    
    options = [
        segment1 + segment2 + segment3 + segment4,
        segment1 + segment3 + segment2 + segment4,
        segment1 + segment2 + segment3[::-1] + segment4,
        segment1 + segment3[::-1] + segment2 + segment4,
        segment1 + segment2[::-1] + segment3 + segment4,
        segment1 + segment2[::-1] + segment3[::-1] + segment4,
        segment1 + segment3 + segment2[::-1] + segment4,
        segment1 + segment3[::-1] + segment2[::-1] + segment4,
    ]
    
    return random.choice(options)

# Geometric cooling function
def geometric_cooling(temp, alpha):
    return temp * alpha

# SA function for TSP with 2-opt and 3-opt neighborhood
def simulated_annealing_tsp(coordinates, initial_temp, final_temp, max_iterations, alpha):
    random.seed(time.time())
    
    n = len(coordinates)
    current_solution = generate_initial_solution(n)
    current_cost = calculate_total_cost(coordinates, current_solution)
    
    best_solution = current_solution
    best_cost = current_cost

    temp = initial_temp
    
    for iteration in range(max_iterations):
        for _ in range(1000):  # Increase the neighborhood exploration significantly
            if random.random() < 0.5:
                new_solution = generate_neighbor_2opt(current_solution)
            else:
                new_solution = generate_neighbor_3opt(current_solution)
            
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

# Main
def main():
    if len(sys.argv) < 6:
        print("Usage: pytsp-sa.py <instance file> <initial_temp> <alpha> <final_temp> <max_iterations>")
        sys.exit(1)

    instance_file = sys.argv[1]
    initial_temp = float(sys.argv[2])
    alpha = float(sys.argv[3])
    final_temp = float(sys.argv[4])
    max_iterations = int(sys.argv[5])

    coordinates = read_instance(instance_file)
    if coordinates is None:
        sys.exit(1)

    start_time = time.time()

    best_solution = simulated_annealing_tsp(coordinates, initial_temp, final_temp, max_iterations, alpha)

    end_time = time.time()
    execution_time = end_time - start_time

    initial_solution = generate_initial_solution(len(coordinates))
    initial_cost = calculate_total_cost(coordinates, initial_solution)
    percent_diff = ((initial_cost - best_solution.cost) / initial_cost) * 100.0

    # Format output as CSV
    result_csv = f"{instance_file};{initial_cost:.2f};{best_solution.cost:.2f};{percent_diff:.2f};{execution_time:.2f};{best_solution.path}"
    print(result_csv)

if __name__ == "__main__":
    main()
