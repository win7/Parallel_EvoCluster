# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam
"""
# ------- Parallel -------
from mpi4py import MPI
from source.models import run_migration
# ------------------------

from source.solution import Solution

import numpy as np
import math
import time
import random

def PWOA(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population):
	# ------- Parallel -------
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	population_size = int(population_size / 24)
	# ------------------------

	num_features = int(dimension / num_clusters)
	# dimension = 30
	# population_size = SearchAgents_no = 50
	# lb = -100
	# ub = 100
	# iterations = 500

	# Initialize position vector and score for the leader
	leader_pos = np.zeros(dimension)
	# Change this to -inf for maximization problems
	leader_score = float("inf")

	# Initialize the positions of search agents
	# positions = np.zeros((population_size, dimension))
	# ------- Parallel -------
	positions = population[rank * population_size:population_size * (rank + 1)] # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	# ------------------------
	labels_pred = np.zeros((population_size, len(points)))

	# Initialize convergence
	convergence = np.zeros(iterations)

	sol = Solution()

	print("WOA_mpi is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	iteration = 0  # Loop counter

	# Main loop
	while iteration < iterations:
		for i in range(population_size):
			# Return back the search agents that go beyond the boundaries of the search space

			# positions[i,:]=checkBounds(positions[i,:],lb,ub)
			positions[i, :] = np.clip(positions[i, :], lb, ub)

			# Calculate objective function for each search agent
			startpts = np.reshape(positions[i, :], (num_clusters, num_features))

			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)

			fitness = fitness_value
			labels_pred[i, :] = labels_pred_values

			# Update the leader
			if fitness < leader_score: # Change this to > for maximization problem
				leader_score = fitness # Update alpha
				# copy current whale position into the leader position
				leader_pos = positions[i, :].copy()
				# copy current whale position into the leader position
				leader_labels = labels_pred[i, :].copy()

		a = 2 - iteration * (2 / iterations) # a decreases linearly fron 2 to 0 in Eq. (2.3)

		# a2 linearly decreases from -1 to -2 to calculate iteration in Eq. (3.12)
		a2 = -1 + iteration * (-1 / iterations)

		# Update the Position of search agents
		for i in range(population_size):
			r1 = random.random() # r1 is a random number in [0,1]
			r2 = random.random() # r2 is a random number in [0,1]

			A = 2 * a * r1 - a # Eq. (2.3) in the paper
			C = 2 * r2 # Eq. (2.4) in the paper

			b = 1 # parameters in Eq. (2.5)
			l = (a2 - 1) * random.random() + 1 # parameters in Eq. (2.5)

			p = random.random() # p in Eq. (2.6)

			for j in range(dimension):
				if p < 0.5:
					if abs(A) >= 1:
						rand_leader_index = math.floor(population_size * random.random())
						X_rand = positions[rand_leader_index, :]
						D_X_rand = abs(C * X_rand[j] - positions[i, j])
						positions[i, j] = X_rand[j] - A * D_X_rand
					elif abs(A) < 1:
						D_leader = abs(C * leader_pos[j] - positions[i, j])
						positions[i, j] = leader_pos[j] - A * D_leader
				elif p >= 0.5:
					distance2_leader = abs(leader_pos[j] - positions[i, j])
					# Eq. (2.5)
					positions[i, j] = distance2_leader * math.exp(b * l) * math.cos(l * 2 * math.pi) + leader_pos[j]

		convergence[iteration] = leader_score
		iteration += 1
		print(["Core: " + str(rank) + " at iteration " + str(iteration - 1) + " the best fitness is " + str(leader_score)])

		# ------- Parallel -------
		# Migrations
		if iteration % policy["interval_emi_imm"] == 0:
			migration_index = np.zeros(policy["number_emi_imm"] * size, dtype=int)
			run_migration(comm, positions, dimension, migration_index, policy, rank, size)
		# ------------------------

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "WOA_mpi"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.best = leader_score
	sol.best_individual = leader_pos
	sol.labels_pred = np.array(leader_labels, dtype=np.int64)
	sol.fitness = leader_score
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