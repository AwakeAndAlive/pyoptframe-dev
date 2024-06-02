# Verificar se o pacote irace está instalado, caso contrário, instalar
if (!requireNamespace("irace", quietly = TRUE)) {
  install.packages("irace")
}

# Carregar o pacote irace
library(irace)

# Definir o caminho do target-runner.sh corretamente
target_runner_path <- file.path(getwd(), "target-runner.sh")

# Verificar se o arquivo target-runner.sh existe
if (!file.exists(target_runner_path)) {
  stop("target-runner.sh not found in the project root directory")
}

# Definir o cenário de configuração
scenario <- list(
  execDir = getwd(),
  targetRunner = target_runner_path,
  trainInstancesDir = file.path(getwd(), "instances"),
  trainInstancesFile = file.path(getwd(), "instances", "instances-list.txt"),
  maxExperiments = 500
)

# Salvar o cenário em um arquivo temporário
scenario_file <- tempfile(fileext = ".txt")
writeLines(c(
  paste0("execDir=\"", scenario$execDir, "\""),
  paste0("targetRunner=\"", scenario$targetRunner, "\""),
  paste0("trainInstancesDir=\"", scenario$trainInstancesDir, "\""),
  paste0("trainInstancesFile=\"", scenario$trainInstancesFile, "\""),
  paste0("maxExperiments=", scenario$maxExperiments)
), scenario_file)

# Ler e executar o cenário
scenario <- readScenario(filename = scenario_file, scenario = defaultScenario())
unlink(scenario_file) # Remover o arquivo temporário

# Executar o irace
irace.main(scenario = scenario)
