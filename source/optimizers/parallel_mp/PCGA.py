"""
Created on Sat Feb  24 20:18:05 2019

@author: Raneem
"""
from source.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------

import numpy as np
import random
import time

def crossover_populaton(population, scores, population_size, crossover_probability, keep):
	"""
	The crossover of all individuals

	Parameters
	---------- 
	population: list
			The list of individuals
	scores: list
			The list of fitness values for each individual
	population_size: int
			Number of chrmosome in a population  
	crossover_probability: float
			The probability of crossing a pair of individuals
	keep: int
			Number of best individuals to keep without mutating for the next generation

	Returns
	-------
	N/A
	"""
	# initialize a new population

	new_population = np.empty_like(population)
	new_population[0:keep] = population[0:keep]
	# Create pairs of parents. The number of pairs equals the number of individuals divided by 2
	for k in range(keep, population_size, 2):
		# pair of parents selection
		parent1, parent2 = pair_selection(population, scores, population_size)
		crossover_length = min(len(parent1), len(parent2))
		parents_crossover_probability = random.uniform(0.0, 1.0)
		if parents_crossover_probability < crossover_probability:
			offspring1, offspring2 = crossover(crossover_length, parent1, parent2)
		else:
			offspring1 = parent1.copy()
			offspring2 = parent2.copy()

		# Add offsprings to population
		new_population[k] = np.copy(offspring1)
		new_population[k + 1] = np.copy(offspring2)

	return new_population

def mutate_populaton(population, population_size, mutation_probability, keep, lb, ub):
	"""    
	The mutation of all individuals

	Parameters
	---------- 
	population : list
			The list of individuals
	population_size: int
			Number of chrmosome in a population  
	mutation_probability: float
			The probability of mutating an individual
	keep: int
			Number of best individuals to keep without mutating for the next generation
	lb: list
			lower bound limit list
	ub: list
			Upper bound limit list

	Returns
	-------
	N/A
	"""
	for k in range(keep, population_size):
		# Mutation
		offspring_mutation_probability = random.uniform(0.0, 1.0)
		if offspring_mutation_probability < mutation_probability:
			mutation(population[k], len(population[k]), lb, ub)

def elitism(population, scores, best_individual, best_score):
	"""    
	This melitism operator of the population

	Parameters
	----------    
	population : list
			The list of individuals
	scores : list
			The list of fitness values for each individual
	best_individual : list
			An individual of the previous generation having the best fitness value          
	best_score : float
			The best fitness value of the previous generation        

	Returns
	-------
	N/A
	"""

	# get the worst individual
	worst_fitness_id = select_worst_individual(scores)

	# replace worst cromosome with best one from previous generation if its fitness is less than the other
	if scores[worst_fitness_id] > best_score:
		population[worst_fitness_id] = np.copy(best_individual)
		scores[worst_fitness_id] = np.copy(best_score)

def select_worst_individual(scores):
	"""    
	It is used to get the worst individual in a population based n the fitness value

	Parameters
	---------- 
	scores : list
			The list of fitness values for each individual

	Returns
	-------
	int
			max_fitness_id: The individual id of the worst fitness value
	"""

	max_fitness_id = np.where(scores == np.max(scores))
	max_fitness_id = max_fitness_id[0][0]
	return max_fitness_id

def pair_selection(population, scores, population_size):
	"""    
	This is used to select one pair of parents using roulette Wheel Selection mechanism

	Parameters
	---------- 
	population : list
			The list of individuals
	scores : list
			The list of fitness values for each individual
	population_size: int
			Number of chrmosome in a population

	Returns
	-------
	list
			parent1: The first parent individual of the pair
	list
			parent2: The second parent individual of the pair
	"""
	parent1_id = roulette_wheel_selection_id(scores, population_size)
	parent1 = population[parent1_id].copy()

	parent2_id = roulette_wheel_selection_id(scores, population_size)
	parent2 = population[parent2_id].copy()

	return parent1, parent2

def roulette_wheel_selection_id(scores, population_size):
	"""    
	A roulette Wheel Selection mechanism for selecting an individual

	Parameters
	---------- 
	scores : list
			The list of fitness values for each individual
	population_size: int
			Number of chrmosome in a population

	Returns
	-------
	id
			individual_id: The id of the individual selected
	"""

	# reverse score because minimum value should have more chance of selection
	reverse = max(scores) + min(scores)
	if reverse == float("inf"):
		return random.randint(0, population_size - 1)
	reverse_scores = reverse - scores.copy()
	sum_scores = sum(reverse_scores)
	
	if sum_scores == float("inf"):
		return random.randint(0, population_size - 1)
	pick = random.uniform(0, sum_scores)
	current = 0
	for individual_id in range(population_size):
		current += reverse_scores[individual_id]
		if current > pick:
			return individual_id

def crossover(individual_length, parent1, parent2):
	"""    
	The crossover operator of a two individuals

	Parameters
	---------- 
	individual_length: int
			The maximum index of the crossover
	parent1 : list
			The first parent individual of the pair
	parent2 : list
			The second parent individual of the pair

	Returns
	-------
	list
			offspring1: The first updated parent individual of the pair
	list
			offspring2: The second updated parent individual of the pair
	"""
	# The point at which crossover takes place between two parents.
	crossover_point = random.randint(0, individual_length - 1)
	# The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
	offspring1 = np.concatenate([parent1[0:crossover_point], parent2[crossover_point:]])
	# The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
	offspring2 = np.concatenate([parent2[0:crossover_point], parent1[crossover_point:]])

	return offspring1, offspring2

def mutation(offspring, individual_length, lb, ub):
	"""    
	The mutation operator of a single individual

	Parameters
	---------- 
	offspring : list
			A generated individual after the crossover
	individual_length: int
			The maximum index of the crossover
	lb: list
			lower bound limit list
	ub: list
			Upper bound limit list

	Returns
	-------
	N/A
	"""

	mutation_index = random.randint(0, individual_length - 1)
	mutation_value = random.uniform(lb, ub)
	offspring[mutation_index] = mutation_value

def clear_dups(population, lb, ub):
	"""    
	It removes individuals duplicates and replace them with random ones

	Parameters
	----------    
	objective_function : function
			The objective function selected
	lb: list
			lower bound limit list
	ub: list
			Upper bound limit list

	Returns
	-------
	list
			new_population: the updated list of individuals
	"""
	new_population = np.unique(population, axis=0)
	old_len = len(population)
	new_len = len(new_population)
	if new_len < old_len:
		n_duplicates = old_len - new_len
		new_population = np.append(new_population, np.random.uniform(0, 1, (n_duplicates, len(
			population[0]))) * (np.array(ub) - np.array(lb)) + np.array(lb), axis=0)

	return new_population

def calculate_cost(objective_function, population, dimension, population_size, lb, ub, num_clusters, points, metric, cores):
	"""    
	It calculates the fitness value of each individual in the population

	Parameters
	----------    
	objective_function : function
			The objective function selected    
	population : list
			The list of individuals
	population_size: int
			Number of chrmosomes in a population
	lb: list
			lower bound limit list
	ub: list
			Upper bound limit list

	Returns
	-------
	list
			scores: fitness values of all individuals in the population
	"""
	num_features = int(dimension / num_clusters)

	# ------- Parallel -------
	scores = pymp.shared.array(population_size, dtype="float")
	scores.fill(float("inf"))
	best_score = pymp.shared.array(1, dtype="float")
	best_score.fill(float("inf"))
	best_labels_pred = pymp.shared.array(len(points), dtype="float")
	best_labels_pred.fill(float("inf"))

	best_individual = pymp.shared.array(dimension, dtype="float")
	# ------------------------

	# Loop through individuals in population
	# ------- Parallel -------
	with pymp.Parallel(cores) as p:
		for i in p.range(population_size):
			# Return back the search agents that go beyond the boundaries of the search space
			population[i] = np.clip(population[i], lb, ub)

			# Calculate objective function for each search agent
			# scores[i] = objective_function(population[i,:])

			startpts = np.reshape(population[i, :], (num_clusters, num_features))

			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness, labels_pred = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness, labels_pred = objective_function(startpts, points, num_clusters)

			scores[i] = fitness
			with p.lock:
				if fitness < best_score[0]:
					best_labels_pred[:] = labels_pred
					best_score[0] = fitness
					best_individual[:] = population[i]
	# ------------------------

	return scores, best_labels_pred, best_individual

def sort_population(population, scores):
	"""    
	This is used to sort the population according to the fitness values of the individuals

	Parameters
	---------- 
	population : list
			The list of individuals
	scores : list
			The list of fitness values for each individual

	Returns
	-------
	list
			population: The new sorted list of individuals
	list
			scores: The new sorted list of fitness values of the individuals
	"""
	sorted_indices = scores.argsort()
	population = population[sorted_indices]
	scores = scores[sorted_indices]

	return population, scores

def PGA(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	"""    
	This is the main method which implements GA

	Parameters
	----------    
	objective_function : function
			The objective function selected
	lb: list
			lower bound limit list
	ub: list
			Upper bound limit list
	dimension: int
			The dimension of the indivisual
	population_size: int
			Number of chrmosomes in a population
	iterations: int
			Number of iterations / generations of GA

	Returns
	-------
	obj
			sol: The solution obtained from running the algorithm
	"""

	cp = 1  # crossover Probability
	mp = 0.01  # Mutation Probability
	keep = 2  # elitism parameter: how many of the best individuals to keep from one generation to the next

	sol = Solution()

	best_individual = np.zeros(dimension)
	best_score = float("inf")
	best_labels_pred = np.full(len(points), np.inf)

	# ------- Parallel -------
	ga = pymp.shared.array((population_size, dimension), dtype="float")
	ga[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	# ------------------------
	scores = np.full(population_size, float("inf"))

	for k in range(dimension):
		ga[:, k] = np.random.uniform(0, 1, population_size) * (ub - lb) + lb

	convergence = np.zeros(iterations)

	print("GA_mp is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	for k in range(iterations):
		# crossover
		ga = crossover_populaton(ga, scores, population_size, cp, keep)

		# mutation
		mutate_populaton(ga, population_size, mp, keep, lb, ub)

		ga = clear_dups(ga, lb, ub)

		scores, best_labels_pred, best_individual = calculate_cost(
			objective_function, ga, dimension, population_size, lb, ub, num_clusters, points, metric, cores)

		best_score = min(scores)

		# Sort from best to worst
		ga, scores = sort_population(ga, scores)

		convergence[k] = best_score
		print(["At iteration " + str(k) + " the best fitness is " + str(best_score)])

	best_labels_pred = np.asarray(best_labels_pred)
	timer_end = time.time()
	sol.best_individual = best_individual
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "GA_mp"
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(best_labels_pred, dtype=np.int64)
	sol.objf_name = objective_function.__name__
	sol.fitness = best_score

	sol.save()
	# return sol