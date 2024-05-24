#!/bin/bash

# Script path
EXE="python3 irace-mainTSP-fcore-brkga.py"

# Parameters
instance=$4
population_size=$6
num_generations=$8
elite_proportion=${10}
mutant_proportion=${12}
elite_inheritance_probability=${14}

# Temporary output
output_file=$(mktemp)

# Execute the algorithm
$EXE $instance $population_size $num_generations $elite_proportion $mutant_proportion $elite_inheritance_probability > "$output_file" 2> "$output_file.err"

# Check if not empty
if [ -s "$output_file" ]; then
  # Get the cost
  COST=$(tail -n 1 "$output_file")
  echo $COST
else
  echo "Inf"
fi

# Remove
rm -f "$output_file" "$output_file.err"
