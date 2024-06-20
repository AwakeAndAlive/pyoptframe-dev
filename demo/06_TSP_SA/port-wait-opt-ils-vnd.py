import random
import numpy as np

# Parâmetros do ILS-VND
max_iter = 200000  # Aumentado para permitir mais iterações
max_iter_no_improve = 50000  # Aumentado para permitir mais iterações sem melhora
perturbacao_intensidade = 5  # Ajustável para controlar a perturbação
random_seed = 42

# Função para definir a semente aleatória
def definir_semente(seed):
    random.seed(seed)
    np.random.seed(seed)

# Função para gerar instâncias conforme as regras do artigo
def gerar_instancias(nb, T):
    instancias = []
    num_embarcacoes = (nb * T) // 12
    
    for i in range(num_embarcacoes):
        rt = random.uniform(0, T-1)
        te = random.uniform(1, 24)
        prioridade = random.randint(0, 5)
        peso_prioridade = 6 - prioridade
        instancias.append({
            'id': i,
            'release_time': rt,
            'tempo_operacao': te,
            'prioridade': prioridade,
            'peso_prioridade': peso_prioridade
        })
    
    return instancias

# Função para calcular o atraso total
def calcular_atraso_total(solucao, nb):
    atraso_total = 0.0
    tempo_bercos = [0.0] * nb
    
    for embarcacao in solucao:
        berco = embarcacao['berco']
        rt = embarcacao['release_time']
        te = embarcacao['tempo_operacao']
        peso = embarcacao['peso_prioridade']
        
        tempo_inicio = max(rt, tempo_bercos[berco])
        atraso = (tempo_inicio - rt) * peso
        atraso_total += atraso
        tempo_bercos[berco] = tempo_inicio + te
    
    return atraso_total

# Operadores de vizinhança
def troca_2_0(solucao):
    nova_solucao = solucao.copy()
    idx1, idx2 = random.sample(range(len(nova_solucao)), 2)
    nova_solucao[idx1], nova_solucao[idx2] = nova_solucao[idx2], nova_solucao[idx1]
    return nova_solucao

def realocacao(solucao):
    nova_solucao = solucao.copy()
    idx = random.randint(0, len(nova_solucao) - 1)
    nova_solucao[idx]['berco'] = random.randint(0, nb - 1)
    return nova_solucao

def swap_2_1(solucao):
    nova_solucao = solucao.copy()
    idx1, idx2 = random.sample(range(len(nova_solucao)), 2)
    embarcacao1 = nova_solucao.pop(idx1)
    nova_solucao.insert(idx2, embarcacao1)
    return nova_solucao

def shift(solucao):
    nova_solucao = solucao.copy()
    idx1, idx2 = random.sample(range(len(nova_solucao)), 2)
    if idx1 < idx2:
        nova_solucao.insert(idx2, nova_solucao.pop(idx1))
    else:
        nova_solucao.insert(idx1, nova_solucao.pop(idx2))
    return nova_solucao

def perturbacao(solucao, intensidade, nb):
    nova_solucao = solucao.copy()
    for _ in range(intensidade):  # Perturbação com intensidade ajustável
        idx = random.randint(0, len(nova_solucao) - 1)
        nova_solucao[idx]['berco'] = random.randint(0, nb - 1)
    return nova_solucao

# Função ILS-VND
def ils_vnd(instancias, nb, T, max_iter, max_iter_no_improve, perturbacao_intensidade):
    definir_semente(random_seed)
    
    # Inicializa a solução ordenada pelas prioridades
    solucao_inicial = sorted(instancias, key=lambda x: (x['release_time'], x['prioridade']))
    for embarcacao in solucao_inicial:
        embarcacao['berco'] = random.randint(0, nb - 1)
    melhor_solucao = solucao_inicial
    melhor_custo = calcular_atraso_total(melhor_solucao, nb)

    iter_sem_melhora = 0

    while iter_sem_melhora < max_iter_no_improve:
        solucao_corrente = perturbacao(melhor_solucao, perturbacao_intensidade, nb)
        custo_corrente = calcular_atraso_total(solucao_corrente, nb)

        # VND
        while True:
            vizinhancas = [troca_2_0, realocacao, swap_2_1, shift]
            melhorou = False
            for vizinhanca in vizinhancas:
                nova_solucao = vizinhanca(solucao_corrente)
                novo_custo = calcular_atraso_total(nova_solucao, nb)
                if novo_custo < custo_corrente:
                    solucao_corrente = nova_solucao
                    custo_corrente = novo_custo
                    melhorou = True
                    break
            if not melhorou:
                break

        if custo_corrente < melhor_custo:
            melhor_solucao = solucao_corrente
            melhor_custo = custo_corrente
            iter_sem_melhora = 0
        else:
            iter_sem_melhora += 1

        if iter_sem_melhora >= max_iter:
            break

    return melhor_solucao, melhor_custo

# Parâmetros das instâncias fornecidas no artigo
instancias_parametros = [
    (5, 48), (5, 72), (5, 96), (5, 120),
    (6, 48), (6, 72), (6, 96), (6, 120),
    (7, 48), (7, 72), (7, 96), (7, 120),
    (8, 48), (8, 72), (8, 96), (8, 120)
]

# Resultados esperados para as instâncias
resultados_esperados = [
    292.8, 463.6, 1006.2, 1517.2,
    381.4, 710.6, 1205.4, 874.8,
    500.0, 1392.6, 2172.8, 1816.6,
    927.2, 1757.2, 1639.8, 1953.4
]

# Avaliar as instâncias geradas e comparar com os resultados esperados
for parametros, resultado_esperado in zip(instancias_parametros, resultados_esperados):
    nb, T = parametros
    instancias = gerar_instancias(nb, T)
    melhor_solucao, melhor_custo = ils_vnd(instancias, nb, T, max_iter, max_iter_no_improve, perturbacao_intensidade)
    
    print(f"Instância: nb={nb}, T={T}")
    print(f"Resultado esperado: {resultado_esperado}")
    print(f"Resultado obtido: {melhor_custo:.1f}")
    print("------")
