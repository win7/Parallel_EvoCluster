# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:13:28 2016

@author: Hossam Faris
"""
# ------- Parallel -------
from mpi4py import MPI
from source.models import run_migration
# ------------------------

from source.solution import Solution

import math
import numpy as np
import random
import time

def get_cuckoos(nest, best, lb, ub, population_size, dimension):
	# perform Levy flights
	temp_nest = np.zeros((population_size, dimension))
	temp_nest = np.array(nest)
	beta = 3 / 2
	sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
			 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

	s = np.zeros(dimension)
	for k in range(population_size):
		s = nest[k, :]
		u = np.random.randn(len(s)) * sigma
		v = np.random.randn(len(s))
		step = u / abs(v) ** (1 / beta)

		step_size = 0.01 * (step * (s - best))
		s += step_size * np.random.randn(len(s))
		temp_nest[k, :] = np.clip(s, lb, ub)
	return temp_nest

def get_best_nest(nest, labels_pred, new_nest, fitness, population_size, dimension, objective_function, num_clusters, points, metric):
	num_features = int(dimension / num_clusters)
	
	# Evaluating all new solutions
	temp_nest = np.zeros((population_size, dimension))
	temp_nest = np.copy(nest)
	temp_labels = np.copy(labels_pred)

	for k in range(population_size):
		# for k=1:size(nest,1),
		startpts = np.reshape(new_nest[k, :], (num_clusters, num_features))
		if objective_function.__name__ in ["SSE", "SC", "DI"]:
			fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
		else:
			fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
		fnew = fitness_value
		new_labels = labels_pred_values

		if fnew <= fitness[k]:
			fitness[k] = fnew
			temp_nest[k, :] = new_nest[k, :]
			temp_labels[k, :] = new_labels

	# Find the current best
	fmin = min(fitness)
	I = np.argmin(fitness)
	best_local = temp_nest[I, :]
	best_labels = temp_labels[I, :]

	return fmin, best_local, best_labels, temp_nest, fitness, temp_labels

# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, population_size, dimension):
	# Discovered or not
	temp_nest = np.zeros((population_size, dimension))
	K = np.random.uniform(0, 1, (population_size, dimension)) > pa
	step_size = random.random() * (nest[np.random.permutation(population_size), :] - nest[np.random.permutation(population_size), :])
	temp_nest = nest + step_size * K

	return temp_nest

def PCS(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population):
	# ------- Parallel -------
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	population_size = int(population_size / 24)
	# ------------------------

	# lb = -1
	# ub = 1
	# population_size = 50
	# iterations = 1000
	# dimension = 30

	# Discovery rate of alien eggs/solutions
	pa = 0.25

	# Lb = [lb] * nd
	# Ub = [ub] * nd
	convergence = []

	# RInitialize nests randomely
	# ------- Parallel -------
	nest = population[rank * population_size:population_size * (rank + 1)] # np.random.rand(population_size, dimension) * (ub - lb) + lb
	# ------------------------
	labels_pred = np.zeros((population_size, len(points)))

	new_nest = np.zeros((population_size, dimension))
	new_nest = np.copy(nest)

	best_nest = [0] * dimension
	best_labels_pred = [0] * len(points)

	fitness = np.zeros(population_size)
	fitness.fill(float("inf"))

	sol = Solution()

	print("CS_mpi is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	fmin, best_nest, best_labels, nest, fitness, labels_pred = get_best_nest(
		nest, labels_pred, new_nest, fitness, population_size, dimension, objective_function, num_clusters, points, metric)
	
	# Main loop counter
	for k in range(iterations):
		# Generate new solutions (but keep the current best)

		new_nest = get_cuckoos(nest, best_nest, lb, ub, population_size, dimension)

		# Evaluate new solutions and find best
		fnew, best, best_labels, nest, fitness, labels_pred = get_best_nest(
			nest, labels_pred, new_nest, fitness, population_size, dimension, objective_function, num_clusters, points, metric)

		new_nest = empty_nests(new_nest, pa, population_size, dimension)

		# Evaluate new solutions and find best
		fnew, best, best_labels, nest, fitness, labels_pred = get_best_nest(
			nest, labels_pred, new_nest, fitness, population_size, dimension, objective_function, num_clusters, points, metric)

		if fnew < fmin:
			fmin = fnew
			best_nest = best
			best_labels_pred = best_labels

		convergence.append(fmin)
		print(["Core: " + str(rank) + " at iteration " + str(k) + " the best fitness is " + str(fmin)])

		# ------- Parallel -------
		# Migrations
		if k % policy["interval_emi_imm"] == 0:
			migration_index = np.zeros(policy["number_emi_imm"] * size, dtype=int)
			run_migration(comm, nest, dimension, migration_index, policy, rank, size)
		# ------------------------

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "CS_mpi"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.best_individual = best_nest
	sol.labels_pred = np.array(best_labels_pred, dtype=np.int64)
	sol.fitness = fmin
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