# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""
# ------- Parallel -------
from mpi4py import MPI
from source.models import run_migration
# ------------------------

from source.solution import Solution

import numpy as np
import time
import random

def PGWO(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population):
	# ------- Parallel -------
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	population_size = int(population_size / 24)
	# ------------------------

	num_features = int(dimension / num_clusters)
	# iterations = 1000
	# lb = -100
	# ub = 100
	# dimension = 30
	# population_size = 5 (SearchAVgents_no)

	# Initialize alpha, beta, and delta_pos
	alpha_score = float("inf")
	alpha_pos = np.zeros(dimension)
	alpha_labels = np.zeros(len(points))

	beta_score = float("inf")
	beta_pos = np.zeros(dimension)
	beta_labels = np.zeros(len(points))

	delta_score = float("inf")
	delta_pos = np.zeros(dimension)
	delta_labels = np.zeros(dimension)

	# Initialize the positions of search agents
	# positions = np.zeros((population_size, dimension))
	# ------- Parallel -------
	positions = population[rank * population_size:population_size * (rank + 1)] # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	# ------------------------
	labels_pred = np.zeros((population_size, len(points)))

	convergence = np.zeros(iterations)
	sol = Solution()

	# Loop counter
	print("GWO_mpi is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	# Main loop
	for k in range(iterations):
		for i in range(population_size):
			# Return back the search agents that go beyond the boundaries of the search space
			positions[i, :] = np.clip(positions[i, :], lb, ub)

			# Calculate objective function for each search agent
			startpts = np.reshape(positions[i, :], (num_clusters, num_features))
			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
			fitness = fitness_value
			labels_pred[i, :] = labels_pred_values

			# Update Alpha, Beta, and Delta
			if fitness < alpha_score:
				delta_score = beta_score # Update delte
				delta_pos = beta_pos.copy()
				# delta_labels = beta_labels.copy()
				beta_score = alpha_score # Update beta
				beta_pos = alpha_pos.copy()
				beta_labels = alpha_labels.copy()
				alpha_score = fitness # Update alpha
				alpha_pos = positions[i, :].copy()
				alpha_labels = labels_pred[i, :].copy()

			if (fitness > alpha_score and fitness < beta_score):
				delta_score = beta_score # Update delte
				delta_pos = beta_pos.copy()
				# delta_labels = beta_labels.copy()
				beta_score = fitness # Update beta
				beta_pos = positions[i, :].copy()
				beta_labels = labels_pred[i, :].copy()

			if (fitness > alpha_score and fitness > beta_score and fitness < delta_score):
				delta_score = fitness # Update delta
				delta_pos = positions[i, :].copy()
				# delta_labels = labels_pred[i, :].copy()

		a = 2 - k * (2 / iterations) # a decreases linearly fron 2 to 0

		# Update the Position of search agents including omegas
		for i in range(population_size):
			for j in range(dimension):
				r1 = random.random() # r1 is a random number in [0,1]
				r2 = random.random() # r2 is a random number in [0,1]

				A1 = 2 * a * r1 - a # Equation (3.3)
				C1 = 2 * r2 # Equation (3.4)

				# Equation (3.5)-part 1
				D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
				X1 = alpha_pos[j] - A1 * D_alpha # Equation (3.6)-part 1

				r1 = random.random()
				r2 = random.random()

				A2 = 2 * a * r1 - a # Equation (3.3)
				C2 = 2 * r2 # Equation (3.4)

				# Equation (3.5)-part 2
				D_beta = abs(C2 * beta_pos[j] - positions[i, j])
				X2 = beta_pos[j]-A2*D_beta # Equation (3.6)-part 2

				r1 = random.random()
				r2 = random.random()

				A3 = 2 * a * r1 - a # Equation (3.3)
				C3 = 2 * r2 # Equation (3.4)

				# Equation (3.5)-part 3
				D_delta = abs(C3 * delta_pos[j] - positions[i, j])
				X3 = delta_pos[j] - A3 * D_delta # Equation (3.5)-part 3

				positions[i, j] = (X1 + X2 + X3) / 3 # Equation (3.7)

		convergence[k] = alpha_score
		print(["Core: " + str(rank) + " at iteration " + str(k) + " the best fitness is " + str(alpha_score)])

		# ------- Parallel -------
		# Migrations
		if k % policy["interval_emi_imm"] == 0:
			migration_index = np.zeros(policy["number_emi_imm"] * size, dtype=int)
			run_migration(comm, positions, dimension, migration_index, policy, rank, size)
		# ------------------------

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "GWO_mpi"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(alpha_labels, dtype=np.int64)
	sol.best_individual = alpha_pos
	sol.fitness = alpha_score
	sol.policy = policy

	# ------- Parallel -------
	# Select best solution
	comm.Barrier()
	best_sol = None
	if rank == 0:
		best_fitness = sol.fitness
		best_sol = sol
		for k in range(1, size):
			sol = comm.recv(source=k)
			if sol.fitness < best_fitness:
				best_fitness = sol.fitness
				best_sol = sol
		best_sol.save()
	else:
		comm.send(sol, dest=0)
	# ------------------------