import subprocess

# Lista de instâncias
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

# Parâmetros fixos
initial_temp = 155016
alpha = 0.9806
final_temp = 0.0001
max_iterations = 523

output_csv = "pytsp-sa-results.csv"

# Limpa o arquivo CSV de saída
with open(output_csv, 'w') as csvfile:
    pass

# Executa 30 vezes cada instância
for instance in instances:
    for i in range(30):
        result = subprocess.run(
            ['python3', 'pytsp-sa.py', f'instances/{instance}', str(initial_temp), str(alpha), str(final_temp), str(max_iterations)],
            stdout=subprocess.PIPE,
            text=True
        )
        output = result.stdout.strip()
        
        # Remove 'instances/' do caminho do arquivo de instância na saída
        output = output.replace(f'instances/{instance}', instance)
        
        with open(output_csv, 'a') as csvfile:
            csvfile.write(output + '\n')

print(f"Resultados salvos em {output_csv}")
