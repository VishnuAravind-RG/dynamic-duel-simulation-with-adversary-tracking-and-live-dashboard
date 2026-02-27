import random
import copy

class GeneticFleetOptimizer:
    def __init__(self, tasks, vehicles, pop_size=60, generations=120,
                 crossover_rate=0.8, mutation_rate=0.15, elite_size=2):
        self.tasks = tasks
        self.vehicles = vehicles
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def fitness(self, chromosome):
        """
        Multi-objective fitness:
        1. Load balance (minimize variance)
        2. Penalize extreme vehicle overuse
        """
        load = {v: 0 for v in self.vehicles}
        for gene in chromosome:
            load[gene] += 1

        values = list(load.values())
        balance_penalty = max(values) - min(values)

        overload_penalty = sum(1 for v in values if v > len(self.tasks) * 0.6)

        return -(balance_penalty + 2 * overload_penalty)

    def tournament_selection(self, population, k=3):
        contenders = random.sample(population, k)
        contenders.sort(key=self.fitness, reverse=True)
        return contenders[0]

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:]

        point = random.randint(1, len(self.tasks) - 2)
        return parent1[:point] + parent2[point:]

    def mutate(self, chromosome):
        mutated = chromosome[:]
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = random.choice(self.vehicles)
        return mutated

    def optimize(self):
        population = [
            [random.choice(self.vehicles) for _ in self.tasks]
            for _ in range(self.pop_size)
        ]

        for _ in range(self.generations):
            population.sort(key=self.fitness, reverse=True)

            # Elitism
            new_population = population[:self.elite_size]

            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                new_population.append(child)

            population = new_population

        best = max(population, key=self.fitness)

        return {
            task: best[i]
            for i, task in enumerate(self.tasks)
        }