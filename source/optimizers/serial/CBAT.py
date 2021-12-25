# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016

@author: hossam
"""
from source.solution import Solution

import numpy as np
import random
import time

def BAT(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population):
	num_features = int(dimension / num_clusters)

	# lb = -50
	# ub = 50

	A = 0.5 # Loudness  (constant or decreasing)
	r = 0.5 # Pulse rate (constant or decreasing)

	Qmin = 0 # Frequency minimum
	Qmax = 2 # Frequency maximum

	# Initializing arrays
	Q = np.zeros(population_size)  # Frequency
	v = np.zeros((population_size, dimension))  # Velocities
	convergence_curve = []

	# Initialize the population/solutions
	pop = np.copy(population) # np.random.rand(population_size, dimension) * (ub - lb) + lb
	labels_pred = np.zeros((population_size, len(points)))
	fitness = np.zeros(population_size)

	S = np.zeros((population_size, dimension))
	S = np.copy(pop)

	# Initialize solution for the final results
	sol = Solution()
	print("BAT is optimizing \"" + objective_function.__name__ + "\"")

	# Initialize timer for the experiment
	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	# Evaluate initial random solutions
	for k in range(population_size):
		startpts = np.reshape(pop[k, :], (num_clusters, num_features))
		if objective_function.__name__ in ["SSE", "SC", "DI"]:
			fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
		else:
			fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
		fitness[k] = fitness_value
		labels_pred[k, :] = labels_pred_values

	# Find the initial best solution
	fmin = min(fitness)
	I = np.argmin(fitness)
	best = pop[I, :]	
	best_labels_pred = labels_pred[I, :]

	# Main loop
	for i in range(iterations):
		# Loop over all bats(solutions)
		for j in range(population_size):
			Q[j] = Qmin + (Qmin - Qmax) * random.random()
			v[j, :] = v[j, :] + (pop[j, :] - best) * Q[j]
			S[j, :] = pop[j, :] + v[j, :]

			# Check boundaries
			pop = np.clip(pop, lb, ub)

			# Pulse rate
			if random.random() > r:
				S[j, :] = best + 0.001 * np.random.randn(dimension)

			# Evaluate new solutions
			startpts = np.reshape(S[j, :], (num_clusters, num_features))

			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)

			fnew = fitness_value
			labels_pred_new = labels_pred_values

			# Update if the solution improves
			if ((fnew != np.inf) and (fnew <= fitness[j]) and (random.random() < A)):
				pop[j, :] = np.copy(S[j, :])
				fitness[j] = fnew
				labels_pred[j, :] = labels_pred_new

			# Update the current best solution
			if fnew != np.inf and fnew <= fmin:
				best = np.copy(S[j, :])
				fmin = fnew
				best_labels_pred = labels_pred_new

		# update convergence curve
		convergence_curve.append(fmin)
		print(["At iteration " + str(i) + " the best fitness is " + str(fmin)])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "BAT"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(best_labels_pred, dtype=np.int64)
	sol.best_individual = best
	sol.fitness = fmin

	sol.save()
	# return sol