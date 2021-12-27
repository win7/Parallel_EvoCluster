# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:42:18 2016

@author: hossam
"""
from source.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------

import numpy as np
import math
import time
import random

def PMFO(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)
	# iterations = 1000
	# lb = -100
	# ub = 100
	# dimension = 30
	# population_size = 50  # Number of search agents

	# ------- Parallel -------
	# Initialize the positions of moths
	moth_pos = pymp.shared.array((population_size, dimension), dtype="float")
	moth_pos[:] = population # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	moth_fitness = pymp.shared.array(population_size, dtype="float")
	moth_fitness.fill(float("inf"))
	moth_labels = pymp.shared.array((population_size, len(points)), dtype="float")
	# moth_fitness=np.fell(float("inf"))
	# ------------------------

	convergence_curve = np.zeros(iterations)

	sorted_population = np.copy(moth_pos)
	sorted_labels = np.copy(moth_labels)
	fitness_sorted = np.zeros(population_size)

	best_flames = np.copy(moth_pos)
	best_labels = np.copy(moth_labels)
	best_flame_fitness = np.zeros(population_size)

	double_population = np.zeros((2 * population_size, dimension))
	double_labels = np.zeros((2 * population_size, len(points)))
	double_fitness = np.zeros(2 * population_size)

	double_sorted_population = np.zeros((2 * population_size, dimension))
	double_sorted_labels = np.zeros((2 * population_size, len(points)))
	double_fitness_sorted = np.zeros(2 * population_size)

	previous_population = np.zeros((population_size, dimension))
	previous_labels = np.zeros((population_size, len(points)))
	previous_fitness = np.zeros(population_size)

	sol = Solution()

	print("MP_MFO is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	iteration = 1

	# Main loop
	while (iteration < iterations + 1): # Obs: iteration < iterations
		# Number of flames Eq. (3.14) in the paper
		flame_no = round(population_size - iteration * ((population_size - 1) / iterations))

		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for k in p.range(population_size):
				# Check if moths go out of the search spaceand bring it back
				moth_pos[k, :] = np.clip(moth_pos[k, :], lb, ub)

				# evaluate moths
				startpts = np.reshape(moth_pos[k, :], (num_clusters, num_features))
				if objective_function.__name__ in ["SSE", "SC", "DI"]:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
				else:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
				moth_fitness[k] = fitness_value
				moth_labels[k, :] = labels_pred_values
		# ------------------------

		if iteration == 1:
			# Sort the first population of moths
			fitness_sorted = np.sort(moth_fitness)
			I = np.argsort(moth_fitness)

			sorted_population = moth_pos[I, :]
			sorted_labels = moth_labels[I, :]

			# Update the flames
			best_flames = sorted_population
			best_flame_fitness = fitness_sorted
			best_labels = sorted_labels
		else:
			# Sort the moths
			double_population = np.concatenate((previous_population, best_flames), axis=0)
			double_labels = np.concatenate((previous_labels, best_labels), axis=0)
			double_fitness = np.concatenate((previous_fitness, best_flame_fitness), axis=0)

			double_fitness_sorted = np.sort(double_fitness)
			I2 = np.argsort(double_fitness)

			for new_index in range(2 * population_size):
				double_sorted_population[new_index, :] = np.array(double_population[I2[new_index], :])
				double_sorted_labels[new_index, :] = np.array(double_labels[I2[new_index], :])

			fitness_sorted = double_fitness_sorted[0:population_size]
			sorted_population = double_sorted_population[0:population_size, :]
			sorted_labels = double_sorted_labels[0:population_size, :]

			# Update the flames
			best_flames = sorted_population
			best_labels = sorted_labels
			best_flame_fitness = fitness_sorted

		# Update the position best flame obtained so far
		best_flame_score = fitness_sorted[0]
		best_flame_pos = sorted_population[0, :]
		best_labels_pred = sorted_labels[0, :]

		previous_population = moth_pos
		previous_labels = moth_labels
		previous_fitness = moth_fitness

		# a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
		a = -1 + iteration * (-1 / iterations)

		# Loop counter
		for i in range(population_size):
			for j in range(dimension):
				if (i <= flame_no): # Update the position of the moth with respect to its corresponsing flame
					# D in Eq. (3.13)
					distance_to_flame = abs(sorted_population[i, j] - moth_pos[i, j])
					b = 1
					t = (a - 1) * random.random() + 1

					# Eq. (3.12)
					moth_pos[i, j] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + sorted_population[i, j]

				if i > flame_no: # Update the position of the moth with respct to one flame
					# Eq. (3.13)
					distance_to_flame = abs(sorted_population[i, j] - moth_pos[i, j])
					b = 1
					t = (a - 1) * random.random() + 1
					
					# Eq. (3.12)
					moth_pos[i, j] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + sorted_population[flame_no, j]

		convergence_curve[iteration - 1] = best_flame_score # Obs: convergence_curve[iteration]
		iteration += 1
		# Display best fitness along the iteration
		print(["At iteration " + str(iteration - 2) + " the best fitness is " + str(best_flame_score)])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "MP_MFO"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.best_individual = best_flame_pos
	sol.labels_pred = np.array(best_labels_pred, dtype=np.int64)
	sol.fitness = best_flame_score

	sol.save()
	# return sol