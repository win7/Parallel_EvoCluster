# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:04:15 2019

@author: Raneem
"""
from utils.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------
import numpy as np
import time
from datetime import datetime

def PPSO(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)
	# PSO parameters
	# dimension = 30
	# iterations = 200
	# Vmax = 6
	# population_size = 50     #population size
	# lb = -10
	# ub = 10

	Vmax = 6  # raneem
	w_max = 0.9
	w_min = 0.2
	c1 = 2
	c2 = 2

	sol = Solution()

	# Initializations

	vel = np.zeros((population_size, dimension))

	# p_best_score = np.zeros(population_size)
	p_best_score = pymp.shared.array(population_size, dtype="float")
	p_best_score.fill(float("inf"))
	# p_best = np.zeros((population_size, dimension))
	p_best = pymp.shared.array((population_size, dimension), dtype="float")
	# p_best_labels_pred = np.full((population_size, len(points)), np.inf)
	p_best_labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")
	p_best_labels_pred.fill(float("inf"))

	# g_best = np.zeros(dimension)
	g_best = pymp.shared.array(dimension, dtype="float")
	# g_best_score = float("inf")
	g_best_score = pymp.shared.array(1, dtype="float")
	g_best_score.fill(float("inf"))
	# g_best_labels_pred = np.full(len(points), np.inf)
	g_best_labels_pred = pymp.shared.array(len(points), dtype="int")
	g_best_labels_pred.fill(float("inf"))

	pos = pymp.shared.array((population_size, dimension), dtype="float")
	pos[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb

	convergence_curve = np.zeros(iterations)

	print("P_MP_PSO is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	for k in range(iterations):
		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for i in p.range(population_size):
				# pos[i,:]=checkBounds(pos[i,:],lb,ub)
				pos[i, :] = np.clip(pos[i, :], lb, ub)
				# Calculate objective function for each particle

				startpts = np.reshape(pos[i, :], (num_clusters, num_features))
				if objective_function.__name__ in ["SSE", "SC", "DI"]:
					fitness, labels_pred = objective_function(startpts, points, num_clusters, metric)
				else:
					fitness, labels_pred = objective_function(startpts, points, num_clusters)

				with p.lock:
					if(p_best_score[i] > fitness):
						p_best_score[i] = fitness
						p_best[i, :] = pos[i, :].copy()
						p_best_labels_pred[i, :] = np.copy(labels_pred)

					if(g_best_score[0] > fitness):
						g_best_score[0] = fitness
						g_best[:] = pos[i, :].copy()
						g_best_labels_pred[:] = np.copy(labels_pred)
		# ------------------------
		# Update the W of PSO
		w = w_max - k * ((w_max - w_min) / iterations)  # check this

		for i in range(population_size):
			for j in range(dimension):
				r1 = np.random.random()
				r2 = np.random.random()
				vel[i, j] = w * vel[i, j] + c1 * r1 * (p_best[i, j] - pos[i, j]) + c2 * r2 * (g_best[j] - pos[i, j])

				if(vel[i, j] > Vmax):
					vel[i, j] = Vmax

				if(vel[i, j] < -Vmax):
					vel[i, j] = -Vmax

				pos[i, j] = pos[i, j] + vel[i, j]

		convergence_curve[k] = g_best_score[0]
		print(["At iteration " + str(k) + " the best fitness is " + str(g_best_score[0])])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "P_MP_PSO"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(g_best_labels_pred, dtype=np.int64)
	sol.best_individual = g_best
	sol.fitness = g_best_score[0]

	sol.save()
	# return sol