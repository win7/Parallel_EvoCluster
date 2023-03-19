# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

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

def PWOA(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)
	# dimension = 30
	# population_size = SearchAgents_no = 50
	# lb = -100
	# ub = 100
	# iterations = 500

	# ------- Parallel -------
	# Initialize position vector and score for the leader
	leader_pos = pymp.shared.array(dimension, dtype="float")
	# Change this to -inf for maximization problems
	leader_score = pymp.shared.array(1, dtype="float")
	leader_score.fill(float("inf"))

	# Initialize the positions of search agents
	# positions = np.zeros((population_size, dimension))
	positions = pymp.shared.array((population_size, dimension), dtype="float")
	positions[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")

	leader_labels = pymp.shared.array(len(points), dtype="float")
	# ------------------------
	
	# Initialize convergence
	convergence = np.zeros(iterations)

	sol = Solution()

	print("WOA_mp is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	iteration = 0  # Loop counter

	# Main loop
	while iteration < iterations:
		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for i in p.range(population_size):
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

				with p.lock:
					# Update the leader
					if fitness < leader_score[0]: # Change this to > for maximization problem
						leader_score[0] = fitness # Update alpha
						# copy current whale position into the leader position
						leader_pos[:] = positions[i, :].copy()
						# copy current whale position into the leader position
						leader_labels[:] = labels_pred[i, :].copy()
		# ------------------------

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

		convergence[iteration] = leader_score[0]
		iteration += 1
		print(["At iteration " + str(iteration - 1) + " the best fitness is " + str(leader_score[0])])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "WOA_mp"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.best = leader_score[0]
	sol.best_individual = leader_pos
	sol.labels_pred = np.array(leader_labels, dtype=np.int64)
	sol.fitness = leader_score[0]

	sol.save()
	# return sol