# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016

@author: hossam
"""
from utils.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------
import numpy as np
import time

def PBAT(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)

	# lb = -50
	# ub = 50

	A = 0.5 # Loudness  (constant or decreasing)
	r = 0.5 # Pulse rate (constant or decreasing)

	Qmin = 0 # Frequency minimum
	Qmax = 2 # Frequency maximum

	# Initializing arrays
	# Q = np.zeros(population_size)  # Frequency
	Q = pymp.shared.array(population_size, dtype="float")
	# v = np.zeros((population_size, dimension))  # Velocities
	v = pymp.shared.array((population_size, dimension), dtype="float")
	convergence_curve = []

	# Initialize the population/solutions
	pop = pymp.shared.array((population_size, dimension), dtype="float")
	pop[:] = np.copy(population) # np.random.rand(population_size, dimension) * (ub - lb) + lb
	# labels_pred = np.zeros((population_size, len(points)))
	labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")
	# fitness = np.zeros(population_size)
	fitness = pymp.shared.array(population_size, dtype="float")

	# S = np.zeros((population_size, dimension))
	S = pymp.shared.array((population_size, dimension), dtype="float")
	S[:] = np.copy(pop)

	# Initialize solution for the final results
	sol = Solution()
	print("P_MP_BAT is optimizing \"" + objective_function.__name__ + "\"")

	# Initialize timer for the experiment
	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	# Evaluate initial random solutions
	# ------- Parallel -------
	with pymp.Parallel(cores) as p:
		for k in p.range(population_size):
			startpts = np.reshape(pop[k, :], (num_clusters, num_features))
			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
			fitness[k] = fitness_value
			labels_pred[k, :] = labels_pred_values
	# ------------------------

	# Find the initial best solution
	fmin = pymp.shared.array(1, dtype="float")
	fmin[0] = np.min(fitness)
	I = np.argmin(fitness)
	best = pymp.shared.array(dimension, dtype="float")
	best[:] = pop[I, :]
	best_labels_pred = pymp.shared.array(len(points), dtype="float")
	best_labels_pred[:] = labels_pred[I, :]

	# Main loop
	for i in range(iterations):
		# Loop over all bats(solutions)
		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for j in p.range(population_size):
				Q[j] = Qmin + (Qmin - Qmax) * np.random.random()
				v[j, :] = v[j, :] + (pop[j, :] - best) * Q[j]
				S[j, :] = pop[j, :] + v[j, :]

				# Check boundaries
				pop[:] = np.clip(pop, lb, ub)

				# Pulse rate
				if np.random.random() > r:
					S[j, :] = best + 0.001 * np.random.randn(dimension)

				# Evaluate new solutions
				startpts = np.reshape(S[j, :], (num_clusters, num_features))

				if objective_function.__name__ in ["SSE", "SC", "DI"]:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
				else:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)

				fnew = fitness_value
				labels_pred_new = labels_pred_values

				with p.lock:
					# Update if the solution improves
					if ((fnew != np.inf) and (fnew <= fitness[j]) and (np.random.random() < A)):
						pop[j, :] = np.copy(S[j, :])
						fitness[j] = fnew
						labels_pred[j, :] = labels_pred_new

					# Update the current best solution
					if fnew != np.inf and fnew <= fmin[0]:
						best[:] = np.copy(S[j, :])
						fmin[0] = fnew
						best_labels_pred[:] = labels_pred_new
		# ------------------------

		# update convergence curve
		convergence_curve.append(fmin[0])
		print(["At iteration " + str(i) + " the best fitness is " + str(fmin[0])])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "P_MP_BAT"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(best_labels_pred, dtype=np.int64)
	sol.best_individual = best
	sol.fitness = fmin[0]

	sol.save()
	# return sol