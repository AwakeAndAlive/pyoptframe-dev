def convert_to_dict(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    num_cities = int(lines[0].strip())
    cities = {}

    for line in lines[1:num_cities + 1]:
        parts = line.strip().split()
        index = int(parts[0]) 
        x = float(parts[1])
        y = float(parts[2])
        cities[index] = (x, y)

    with open(output_file, 'w') as outfile:
        outfile.write("cities = {\n")
        for index, coords in cities.items():
            outfile.write(f"    {index}: {coords},\n")
        outfile.write("}\n")

if __name__ == "__main__":
    input_file = "instances/08_TRP-S1000-R1.tsp" 
    output_file = "output.txt"  
    convert_to_dict(input_file, output_file)
