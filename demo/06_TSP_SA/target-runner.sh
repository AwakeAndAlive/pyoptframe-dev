#!/bin/bash

# Script path
EXE="python3 irace-tsp-sa-runner.py"

# Parameters from irace
instance=$4
initial_temp=$6
cooling_rate=$8
max_iterations=${10}

# Temporary output
output_file=$(mktemp)

# Logging
echo "Running: $EXE $instance $initial_temp $cooling_rate $max_iterations" >> runner.log

# Execute the algorithm
$EXE $instance $initial_temp $cooling_rate $max_iterations > "$output_file" 2> "$output_file.err"

# Check if not empty
if [ -s "$output_file" ]; then
  # Get the cost
  COST=$(tail -n 1 "$output_file")
  echo $COST
  echo "Cost: $COST" >> runner.log
else
  echo "Inf"
  echo "Output file is empty, returning Inf" >> runner.log
fi

# Remove
rm -f "$output_file" "$output_file.err"
