import subprocess
import sys

def main():
    if len(sys.argv) != 7:
        print("Usage: irace-brkga-runner.py <instance> <population_size> <num_generations> <elite_proportion> <mutant_proportion> <elite_inheritance_probability>")
        sys.exit(1)

    instance = sys.argv[1]
    population_size = sys.argv[2]
    num_generations = sys.argv[3]
    elite_proportion = sys.argv[4]
    mutant_proportion = sys.argv[5]
    elite_inheritance_probability = sys.argv[6]

    command = [
        "python3", "irace-mainTSP-fcore-brkga.py",
        instance, population_size, num_generations,
        elite_proportion, mutant_proportion, elite_inheritance_probability
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        # Print only the last line of the output
        print(output[-1])
    except Exception as e:
        print(f"Error running the command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
