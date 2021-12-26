# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016

@author: hossam
"""
# ------- Parallel -------
from mpi4py import MPI
from utils.models import run_migration
# ------------------------

from utils.solution import Solution

import numpy as np
import time

def PBAT(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population):
	# ------- Parallel -------
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	population_size = int(population_size / 24)
	# ------------------------

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
	pop = population[rank * population_size:population_size * (rank + 1)] # np.random.rand(population_size, dimension) * (ub - lb) + lb
	labels_pred = np.zeros((population_size, len(points)))
	fitness = np.zeros(population_size)

	S = np.zeros((population_size, dimension))
	S = np.copy(pop)

	# Initialize solution for the final best_sols
	sol = Solution()
	print("P_MPI_BAT is optimizing \"" + objective_function.__name__ + "\"")

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
	fmin = np.min(fitness)
	I = np.argmin(fitness)
	best = pop[I, :]
	best_labels_pred = labels_pred[I, :]

	# Main loop
	for i in range(iterations):
		# Loop over all bats(solutions)
		for j in range(population_size):
			Q[j] = Qmin + (Qmin - Qmax) * np.random.random()
			v[j, :] = v[j, :] + (pop[j, :] - best) * Q[j]
			S[j, :] = pop[j, :] + v[j, :]

			# Check boundaries
			pop = np.clip(pop, lb, ub)

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

			# Update if the solution improves
			if ((fnew != np.inf) and (fnew <= fitness[j]) and (np.random.random() < A)):
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
		print(["Core: " + str(rank) + " at iteration " + str(i) + " the best fitness is " + str(fmin)])

		# ------- Parallel -------
		# Migrations
		if i % policy["interval_emi_imm"] == 0:
			migration_index = np.zeros(policy["number_emi_imm"] * size, dtype=int)
			run_migration(comm, pop, dimension, migration_index, policy, rank, size)
		# ------------------------

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "P_MPI_BAT"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(best_labels_pred, dtype=np.int64)
	sol.best_individual = best
	sol.fitness = fmin
	sol.policy = policy

	# ------- Parallel -------
	# Select best solution
	comm.Barrier()
	best_sol = None
	if rank == 0:
		# sol.ShowInformation()
		best_fitness = sol.fitness
		best_sol = sol
		for k in range(1, size):
			sol = comm.recv(source=k)
			if best_fitness < sol.fitness:
				best_fitness = sol.fitness
				best_sol = sol
		best_sol.save()

		""" # Save fitness into file
		file = open(path_file_metric, "a")
		file.write("{},{}\n".format(best_sol.fitness, best_sol.execution_time))
		file.close()

		# Save convergence into file
		file = open(path_file_convergence, "a")
		for item in best_sol.convergence:
			file.write("{}\n".format(item))
		file.close() """
	else:
		comm.send(sol, dest=0)
	# ------------------------