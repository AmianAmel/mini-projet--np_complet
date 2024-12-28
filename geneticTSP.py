import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.02

def initpop(cities, taille ):

    return [random.sample(cities, len(cities)) for _ in range(taille)]

def path_length(tour, distance_matrix):
    
    return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1]][tour[0]]

def fitness(individual, distance_matrix):

    return 1 / path_length(individual, distance_matrix)

def selection(population, fitness_scores):

    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind, fit in zip(population, fitness_scores):
        current += fit
        if current > pick:
            return ind
def crossover(parent1, parent2):

    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

def run_genetic_algorithm(cities, distance_matrix, population_size, generations, mutation_rate):
    population = initpop(cities, population_size)
    best_paths_history = []
    best_distances_history = []

    for generation in range(generations):
        fitness_scores = [fitness(ind, distance_matrix) for ind in population]
        best_tour = min(population, key=lambda ind: path_length(ind, distance_matrix))
        best_distance = path_length(best_tour, distance_matrix)

        best_paths_history.append(best_tour)
        best_distances_history.append(best_distance)

        new_population = []
        for _ in range(population_size // 2):
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    return best_paths_history, best_distances_history

def visualize_evolution(best_paths_history, best_distances_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    coords = np.random.rand(len(best_paths_history[0]), 2)

    def update(frame):
        ax1.clear()
        ax2.clear()
        best_tour = best_paths_history[frame]
        tour_coords = coords[best_tour + [best_tour[0]]]

        ax1.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-')
        ax1.set_title(f'Génération {frame}: Meilleur tour')

        ax2.plot(best_distances_history[:frame + 1])
        ax2.set_title('Évolution de la distance')
        ax2.set_xlabel('Génération')
        ax2.set_ylabel('Longueur du chemin')

    anim = animation.FuncAnimation(fig, update, frames=len(best_paths_history),interval=100, repeat=False)
    plt.tight_layout()
    plt.show()

def main():
    distance_matrix = np.array([
        [1, 10, 15, 20, 30],
        [10, 50, 35, 25, 40],
        [15, 5, 0, 30, 45],
        [20, 25, 30, 0, 50],
        [30, 40, 45, 50, 0]
    ])
    
    cities = list(range(len(distance_matrix)))
    
    best_paths_history, best_distances_history = run_genetic_algorithm(
        cities, distance_matrix, POPULATION_SIZE, GENERATIONS, MUTATION_RATE
    )
    
    print("Meilleur tour trouvé:", best_paths_history[-1])
    print("Longueur du meilleur tour:", best_distances_history[-1])
    
    visualize_evolution(best_paths_history, best_distances_history)

if __name__ == "__main__":
    main()
