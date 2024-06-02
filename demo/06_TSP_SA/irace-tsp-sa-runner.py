import subprocess
import sys

def main():
    if len(sys.argv) != 5:
        print("Usage: irace-tsp-sa-runner.py <instance> <initial_temp> <cooling_rate> <max_iterations>")
        sys.exit(1)

    instance = sys.argv[1]
    initial_temp = sys.argv[2]
    cooling_rate = sys.argv[3]
    max_iterations = sys.argv[4]

    command = [
        "python3", "irace-tsp-sa.py",
        instance, initial_temp, cooling_rate, max_iterations
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        # Print only the integer part of the last line of the output
        final_result = int(float(output[-1]))
        print(final_result)
    except Exception as e:
        print(f"Error running the command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
