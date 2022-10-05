# ------- Parallel -------
from mpi4py import MPI
from source.models import run_migration
# ------------------------

from source.solution import Solution

import numpy as np
import math
import time
import random

def PSSA(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population):
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
	# population_size = 50 # Number of search agents

	convergence_curve = np.zeros(iterations)

	# Initialize the positions of salps
	# ------- Parallel -------
	salp_positions = population[rank * population_size:population_size * (rank + 1)] # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	# ------------------------
	salp_fitness = np.full(population_size, float("inf"))
	salp_labels_pred = np.full((population_size, len(points)), np.inf)

	food_position = np.zeros(dimension)
	food_fitness = float("inf")
	food_labels_pred = np.full(len(points), np.inf)
	# Moth_fitness=np.fell(float("inf"))

	sol = Solution()

	print("MPI_SSA is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	for k in range(population_size):
		# Evaluate moths
		startpts = np.reshape(salp_positions[k, :], (num_clusters, num_features))

		if objective_function.__name__ in ["SSE", "SC", "DI"]:
			fitness, labels_pred = objective_function(startpts, points, num_clusters, metric)
		else:
			fitness, labels_pred = objective_function(startpts, points, num_clusters)
		salp_fitness[k] = fitness
		salp_labels_pred[k, :] = labels_pred

	sorted_salps_fitness = np.sort(salp_fitness)
	I = np.argsort(salp_fitness)

	sorted_salps = np.copy(salp_positions[I, :])
	sorted_labels_pred = np.copy(salp_labels_pred[I, :])

	food_position = np.copy(sorted_salps[0, :])
	food_fitness = sorted_salps_fitness[0]
	food_labels_pred = sorted_labels_pred[0]
	
	convergence_curve[0] = food_fitness
	print(["Core: " + str(rank) + " at iteration 0 the best fitness is " + str(food_fitness)])
	
	iteration = 1

	# Main loop
	while (iteration < iterations):
		# Number of flames Eq. (3.14) in the paper
		# Flame_no=round(population_size-iteration*((population_size-1)/iterations));

		c1 = 2 * math.exp(-(4 * iteration / iterations) ** 2) # Eq. (3.2) in the paper

		for i in range(population_size):
			salp_positions = np.transpose(salp_positions)

			if i < population_size / 2:
				for j in range(dimension):
					c2 = random.random()
					c3 = random.random()
					# Eq. (3.1) in the paper
					if c3 < 0.5:
						salp_positions[j, i] = food_position[j] + 0.1 * c1 * ((ub - lb) * c2 + lb)
					else:
						salp_positions[j, i] = food_position[j] - 0.1 * c1 * ((ub - lb) * c2 + lb)
			elif i >= population_size / 2 and i < population_size:
				point1 = np.copy(salp_positions[:, i - 1])
				point2 = np.copy(salp_positions[:, i])

				# Eq. (3.4) in the paper
				salp_positions[:, i] = (point2 + point1) / 2
			salp_positions = np.transpose(salp_positions)

		for k in range(population_size):
			# Check if salps go out of the search spaceand bring it back
			salp_positions[k, :] = np.clip(salp_positions[k, :], lb, ub)

			startpts = np.reshape(salp_positions[k, :], (num_clusters, num_features))

			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness, labels_pred = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness, labels_pred = objective_function(startpts, points, num_clusters)

			salp_fitness[k] = fitness
			salp_labels_pred[k, :] = np.copy(labels_pred)

			if salp_fitness[k] < food_fitness:
				food_position = np.copy(salp_positions[k, :])
				food_fitness = salp_fitness[k]
				food_labels_pred = np.copy(salp_labels_pred[k, :])

		convergence_curve[iteration] = food_fitness
		iteration += 1
		# Display best fitness along the iteration
		print(["Core: " + str(rank) + " at iteration " + str(iteration - 1) + " the best fitness is " + str(food_fitness)])

		# ------- Parallel -------
		# Migrations
		if iteration % policy["interval_emi_imm"] == 0:
			migration_index = np.zeros(policy["number_emi_imm"] * size, dtype=int)
			run_migration(comm, salp_positions, dimension, migration_index, policy, rank, size)
		# ------------------------

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "MPI_SSA"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(food_labels_pred, dtype=np.int64)
	sol.best_individual = food_position
	sol.fitness = food_fitness
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
			if best_fitness < sol.fitness:
				best_fitness = sol.fitness
				best_sol = sol
		best_sol.save()
	else:
		comm.send(sol, dest=0)
	# ------------------------