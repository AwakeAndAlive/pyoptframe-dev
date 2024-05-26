#!/bin/bash

# Script path
EXE="python3 irace-brkga-runner.py"

# Parameters
instance=$4
population_size=$6
num_generations=$8
elite_proportion=${10}
mutant_proportion=${12}
elite_inheritance_probability=${14}

# Temporary output
output_file=$(mktemp)

# Logging
echo "Running: $EXE $instance $population_size $num_generations $elite_proportion $mutant_proportion $elite_inheritance_probability" >> runner.log

# Execute the algorithm
$EXE $instance $population_size $num_generations $elite_proportion $mutant_proportion $elite_inheritance_probability > "$output_file" 2> "$output_file.err"

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
