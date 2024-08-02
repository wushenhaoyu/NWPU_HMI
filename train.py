from typing import List

import numpy as np
from main import Game2048


# 假设这是你的游戏函数，它接受参数并返回游戏状态
def play_game(smooth_weight: float, mono_weight: float, empty_weight: float, max_weight: float) -> float:
    # 这里应该是你的游戏逻辑
    # 返回游戏结束时的得分
    game = Game2048(smooth_weight, mono_weight, empty_weight, max_weight)
    game.ai_active = True
    return game.main()


# 适应度函数，计算给定参数下的游戏得分
def fitness(params: List[float]) -> float:
    return play_game(params[0], params[1], params[2], params[3])


# 初始化种群
# 初始化种群
def init_population(pop_size: int, param_range: List[List[float]]) -> List[List[float]]:
    population = []
    for _ in range(pop_size):
        individual = [np.random.uniform(low=low, high=high) for low, high in param_range]
        population.append(individual)
    return population



# 选择操作
def selection(population: List[List[float]], fitnesses: List[float], num_parents: int) -> List[List[float]]:
    parents = []
    for _ in range(num_parents):
        best_idx = np.argmax(fitnesses)
        parents.append(population[best_idx])
        fitnesses[best_idx] = -99999
    return parents


# 交叉操作
def crossover(parents: List[List[float]], offspring_size: int) -> List[List[float]]:
    offspring = []
    while len(offspring) < offspring_size:
        parent1 = np.random.choice(parents)
        parent2 = np.random.choice(parents)

        # 确保 parent1 和 parent2 不相同
        while parent1 == parent2:
            parent2 = np.random.choice(parents)

        idx = np.random.randint(1, len(parent1))
        child = parent1[:idx] + parent2[idx:]
        offspring.append(child)
    return offspring


# 变异操作
def mutation(offspring: List[List[float]], mutation_rate: float, param_range: List[List[float]]) -> List[List[float]]:
    mutated_offspring = []
    for child in offspring:
        for i in range(len(child)):
            if np.random.rand() < mutation_rate:
                child[i] = np.random.uniform(low=param_range[i][0], high=param_range[i][1])
        mutated_offspring.append(child)
    return mutated_offspring


# 主循环
def genetic_algorithm(pop_size: int, num_generations: int, num_parents: int, mutation_rate: float,
                      param_range: List[List[float]]):
    population = init_population(pop_size, param_range)

    for generation in range(num_generations):
        fitnesses = [fitness(individual) for individual in population]
        best_fitness = max(fitnesses)
        best_params = population[np.argmax(fitnesses)]
        print(f"Generation {generation}: Best Fitness = {best_fitness}, Best Parameters = {best_params}")

        parents = selection(population, fitnesses, num_parents)
        offspring = crossover(parents, pop_size - len(parents))
        mutated_offspring = mutation(offspring, mutation_rate, param_range)

        population = parents + mutated_offspring

    return max(population, key=fitness)


if __name__ == '__main__':
    # 参数范围
    param_range = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]

    # 运行遗传算法
    best_params = genetic_algorithm(pop_size=50, num_generations=100, num_parents=10, mutation_rate=0.1,
                                    param_range=param_range)
    print("Best Parameters:", best_params)
