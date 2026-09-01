"""
Ported from `research/GA_test.ipynb` cell 12 — see `docs/RESTRUCTURING_TODO.md` and the
port plan in `docs/RESEARCH_STRUCTURE.md` / `docs/GA_structure.md`. Extracted mechanically
from the notebook cell source (not retyped) to avoid transcription drift; the notebook
remains the frozen parity reference. Only the import block was touched: names the cell relied
on getting from the shared notebook kernel namespace (cell 0's imports) are now imported
explicitly here, since a standalone module has no such shared kernel state.

`ResultsTracker` (in-memory run tracking + progression plots) - the sibling `DataCache`/
`PredictionCache` classes cells 14/16 originally shared this file with now live in
`caching.py` at the package root (the two LRU caches - one keyed by base feature name, one by
expression identity - are a separate concern from run-history tracking).
"""

import matplotlib.pyplot as plt

class ResultsTracker:
    def __init__(self):
        self.best_fitnesses = []
        self.average_fitnesses = []
        self.worst_fitnesses = []
        self.diversity = []
        self.admitted_rate = []
        self.mutation = []
        self.fitness_distributions = []  # Initialize fitness distributions
        self.best_individuals = []
        self.convergence_count = 0

    def update_fitness(self, evaluated_population):
        # Extract fitness values
        fitness_values = [fitness for _, fitness in evaluated_population]
        
        # Store the fitness distribution for this generation
        self.fitness_distributions.append(fitness_values)

        # Calculate best fitness (closest to zero, which is the highest negative value)
        best_fitness = max(fitness_values)

        # Calculate total fitness manually
        total_fitness = 0
        for fitness in fitness_values:
            total_fitness += fitness

        # Calculate average fitness
        average_fitness = total_fitness / len(fitness_values)

        # Calculate worst fitness (furthest from zero, which is the lowest negative value)
        worst_fitness = min(fitness_values)

        # Update the tracker lists
        self.best_fitnesses.append(best_fitness)
        self.average_fitnesses.append(average_fitness)
        self.worst_fitnesses.append(worst_fitness)

    def get_fitnesses(self):
        """
        Get the tracked fitness values.

        Returns:
            Tuple[List[float], List[float], List[float]]: The best, average, and worst fitness values over generations.
        """
        return self.best_fitnesses, self.average_fitnesses, self.worst_fitnesses

    def plot_fitness_progression(self, save_path = None):
        generations = range(1, len(self.best_fitnesses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_fitnesses, label="Best Fitness", color='blue')
        plt.plot(generations, self.average_fitnesses, label="Average Fitness", color='orange')
        plt.plot(generations, self.worst_fitnesses, label="Worst Fitness", color='green')

        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title('Fitness Progression Over Generations')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
            
        plt.close()
        
    def update_mutation(self, true_mutation_rate):
        self.mutation.append(true_mutation_rate)
        
    def plot_mutation_progression(self, save_path=None):
        generations = range(1, len(self.mutation) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.mutation, label="Mutation Rate", color='pink')

        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Mutation Rate', fontsize=16)
        plt.title('Mutation Rate Over Generations', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
        plt.close()
    
    def update_diversity(self, population):
        unique_individuals = len(set(population))
        self.diversity.append(unique_individuals)

    def plot_diversity_progression(self, save_path=None):
        generations = range(1, len(self.diversity) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.diversity, label="Population Diversity", color='purple')

        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Number of Unique Individuals', fontsize=16)
        plt.title('Diversity Progression Over Generations', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
        plt.close()

    def update_admitted_rate(self, rate):
        """
        rate: fraction of a generation's newly-created individuals whose canonical form
        collided with one already accepted this generation, forcing update_population()'s dedup
        policy to mutate them at least once before acceptance - see GeneticAlgorithm1.__init__'s
        admitted_rate_threshold docstring for why this, not raw diversity, is what run()'s
        early-termination check uses to detect a converged search.
        """
        self.admitted_rate.append(rate)

    def plot_admitted_rate_progression(self, save_path=None):
        generations = range(1, len(self.admitted_rate) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.admitted_rate, label="Admitted-Duplicate Rate", color='crimson')

        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fraction Requiring Forced Mutation', fontsize=16)
        plt.title('Admitted-Duplicate Rate Over Generations', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()
        plt.close()

    def plot_fitness_distribution(self, generation_index, save_path=None):
        if generation_index < len(self.fitness_distributions):
            fitness_values = self.fitness_distributions[generation_index]
            plt.figure(figsize=(10, 6))
            plt.hist(fitness_values, bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('Fitness', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.title(f'Fitness Distribution for Generation {generation_index + 1}', fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')            
                plt.show()
            else:
                plt.show()
            plt.close()
    
    def update_best_individuals(self, best_individual):
        self.best_individuals.append(best_individual)

    def get_best_individuals(self):
        return self.best_individuals