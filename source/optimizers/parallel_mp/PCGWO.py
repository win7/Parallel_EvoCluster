# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""
from source.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------

import numpy as np
import time
import random

def PGWO(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)
	# iterations = 1000
	# lb = -100
	# ub = 100
	# dimension = 30
	# population_size = 5 (SearchAVgents_no)

	# Initialize alpha, beta, and delta_pos
	# ------- Parallel -------
	alpha_score = pymp.shared.array(1, dtype="float")
	alpha_score.fill(float("inf"))
	alpha_pos = pymp.shared.array(dimension, dtype="float")
	alpha_labels = pymp.shared.array(len(points), dtype="float") # OBS: before dimension

	beta_score = pymp.shared.array(1, dtype="float")
	beta_score.fill(float("inf"))
	beta_pos = pymp.shared.array(dimension, dtype="float")
	beta_labels = pymp.shared.array(len(points), dtype="float") # OBS: before dimension

	delta_score = pymp.shared.array(1, dtype="float")
	delta_score.fill(float("inf"))
	delta_pos = pymp.shared.array(dimension, dtype="float")
	delta_labels = pymp.shared.array(dimension, dtype="float")

	# Initialize the positions of search agents
	# positions = np.zeros((population_size, dimension))
	positions = pymp.shared.array((population_size, dimension), dtype="float")
	positions[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")
	# ------------------------

	convergence_curve = np.zeros(iterations)
	sol = Solution()

	# Loop counter
	print("GWO_mp is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	# Main loop
	for k in range(iterations):
		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for i in p.range(population_size):
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

				with p.lock:
					# Update Alpha, Beta, and Delta
					if fitness < alpha_score[0]:
						delta_score[0] = beta_score[0] # Update delte
						delta_pos[:] = beta_pos.copy()
						# delta_labels[:] = beta_labels.copy()
						beta_score[0] = alpha_score[0] # Update beta
						beta_pos[:] = alpha_pos.copy()
						beta_labels[:] = alpha_labels.copy()
						alpha_score[0] = fitness # Update alpha
						alpha_pos[:] = positions[i, :].copy()
						alpha_labels[:] = labels_pred[i, :].copy()

					if (fitness > alpha_score[0] and fitness < beta_score[0]):
						delta_score[0] = beta_score[0] # Update delte
						delta_pos[:] = beta_pos.copy()
						# delta_labels[:] = beta_labels.copy()
						beta_score[0] = fitness # Update beta
						beta_pos[:] = positions[i, :].copy()
						beta_labels[:] = labels_pred[i, :].copy()

					if (fitness > alpha_score[0] and fitness > beta_score[0] and fitness < delta_score[0]):
						delta_score[0] = fitness # Update delta
						delta_pos[:] = positions[i, :].copy()
						# delta_labels[:] = labels_pred[i, :].copy() # OBS: this variable no used
		# ------------------------

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

		convergence_curve[k] = alpha_score[0]
		print(["At iteration " + str(k) + " the best fitness is " + str(alpha_score[0])])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "GWO_mp"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(alpha_labels, dtype=np.int64)
	sol.best_individual = alpha_pos
	sol.fitness = alpha_score[0]

	sol.save()
	# return sol