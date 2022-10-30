# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:06:34 2016

@author: hossam
"""
from sklearn.preprocessing import normalize
from source.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------

import numpy as np
import math
import time
import random

def normr(matrix):
	""" 
	normalize the columns of the matrix
	B = normr(A) normalizes the row
	the dtype of A is float """

	matrix = matrix.reshape(1, -1)
	# Enforce dtype float
	if matrix.dtype != "float":
		matrix = np.asarray(matrix, dtype=float)

	# if statement to enforce dtype float
	# B = normalize(matrix,norm="l2",axis=1)
	B = (matrix - min(matrix)) / (max(matrix) - min(matrix))
	B = np.reshape(B, -1)
	return B

def randk(t):
	if (t % 2) == 0:
		s = 0.25
	else:
		s = 0.75
	return s

def roulette_wheel_selection(weights):
	accumulation = np.cumsum(weights)
	p = random.random() * accumulation[-1]
	chosen_index = -1
	for index in range(len(accumulation)):
		if (accumulation[index] > p):
			chosen_index = index
			break
	choice = chosen_index
	return choice

def PMVO(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)

	# dimension = 30
	# lb = -100
	# ub = 100
	# iterations = 1000
	# population_size = 50

	WEP_max = 1
	WEP_min = 0.2

	# ------- Parallel -------
	universes = pymp.shared.array((population_size, dimension), dtype="float")
	universes[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb

	sorted_universes = np.copy(universes)

	# labels_pred = np.zeros((population_size, len(points)))
	labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")
	sorted_labels = np.copy(labels_pred)

	convergence = np.zeros(iterations)

	best_universe = pymp.shared.array(dimension, dtype="float")
	best_universe_inflation_rate = pymp.shared.array(1, dtype="float")
	best_universe_inflation_rate.fill(float("inf"))
	labels_pred_best = pymp.shared.array(len(points), dtype="float")
	# ------------------------

	sol = Solution()

	iteration = 1

	print("MVO_mp is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	while (iteration < iterations + 1):
		# Eq. (3.3) in the paper
		WEP = WEP_min + iteration * ((WEP_max - WEP_min) / iterations)

		TDR = 1 - (math.pow(iteration, 1 / 6) / math.pow(iterations, 1 / 6))

		# ------- Parallel -------
		inflation_rates = pymp.shared.array(len(universes), dtype="float")

		with pymp.Parallel(cores) as p:
			for k in p.range(population_size):
				universes[k, :] = np.clip(universes[k, :], lb, ub)

				startpts = np.reshape(universes[k, :], (num_clusters, num_features))
				if objective_function.__name__ in ["SSE", "SC", "DI"]:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
				else:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
				
				inflation_rates[k] = fitness_value
				labels_pred[k, :] = labels_pred_values

				with p.lock:
					if inflation_rates[k] < best_universe_inflation_rate[0]:
						best_universe_inflation_rate[0] = inflation_rates[k]
						best_universe[:] = np.array(universes[k, :])
						labels_pred_best[:] = np.array(labels_pred[k, :])
		# ------------------------

		sorted_inflation_rates = np.sort(inflation_rates)
		sorted_indexes = np.argsort(inflation_rates)

		for new_index in range(population_size):
			sorted_universes[new_index, :] = np.array(universes[sorted_indexes[new_index], :])
			sorted_labels[new_index, :] = np.array(labels_pred[sorted_indexes[new_index], :])

		normalized_sorted_Inflation_rates = np.copy(normr(sorted_inflation_rates))

		universes[0, :] = np.array(sorted_universes[0, :])
		labels_pred[0, :] = sorted_labels[0, :]

		for i in range(1, population_size):
			back_hole_index = i
			for j in range(dimension):
				r1 = random.random()
				if r1 < normalized_sorted_Inflation_rates[i]:
					white_hole_index = roulette_wheel_selection(-sorted_inflation_rates)

					if white_hole_index == -1:
						white_hole_index = 0
					white_hole_index = 0
					universes[back_hole_index, j] = sorted_universes[white_hole_index, j]
					labels_pred[back_hole_index, j] = sorted_labels[white_hole_index, j]

				r2 = random.random()

				if r2 < WEP:
					r3 = random.random()
					if r3 < 0.5:
						# random.uniform(0, 1) + lb);
						universes[i, j] = best_universe[j] + TDR * ((ub - lb) * random.random() + lb)
					if r3 > 0.5:
						# random.uniform(0, 1) + lb);
						universes[i, j] = best_universe[j] - TDR * ((ub - lb) * random.random() + lb)

		convergence[iteration - 1] = best_universe_inflation_rate[0]
		iteration += 1
		print(["At iteration " + str(iteration - 2) + " the best fitness is " + str(best_universe_inflation_rate[0])])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "MVO_mp"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.best_individual = best_universe
	sol.labels_pred = np.array(labels_pred_best, dtype=np.int64)
	sol.fitness = best_universe_inflation_rate[0]

	sol.save()
	# return sol