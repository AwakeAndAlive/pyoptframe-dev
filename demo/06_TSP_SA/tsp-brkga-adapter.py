import subprocess
import sys
import os

def run_instance(instance):
    command = [
        "python3", "tsp-brkga-log.py",
        instance, "38", "1975", "0.3156", "0.2105", "0.6474"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        return output[-1]
    except Exception as e:
        print(f"Error running the command for instance {instance}: {e}")
        return None

def main():
    instances = [
        "01_berlin52.tsp",
        "02_kroD100.tsp",
        "03_pr226.tsp",
        "04_lin318.tsp",
        "05_TRP-S500-R1.tsp",
        "06_d657.tsp",
        "07_rat784.tsp",
        "08_TRP-S1000-R1.tsp"
    ]
    
    output_file = "tsp_brkga_results.csv"
    
    with open(output_file, "w") as file:
        for instance in instances:
            instance_path = os.path.join("instances", instance)
            for i in range(30):
                print(f"Running instance {instance}, repetition {i+1}/30...")
                result = run_instance(instance_path)
                if result:
                    file.write(result + "\n")
                    file.flush() 
                    print(f"Result: {result}")
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
