#!/bin/bash

# Script path
EXE="python3 irace-pytsp-sa.py"

# Parameters
instance=$4
initial_temp=$6
alpha=$8
final_temp=${10}
max_iterations=${12}
SAmax=10  # Fixed after testing

# Temporary output
output_file=$(mktemp)

# Execute the algorithm
$EXE $instance $initial_temp $alpha $final_temp $max_iterations $SAmax > "$output_file" 2> "$output_file.err"

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
