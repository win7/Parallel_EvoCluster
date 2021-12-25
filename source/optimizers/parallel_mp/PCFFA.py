# -*- coding: utf-8 -*-
"""
Created on Sun May 29 00:49:35 2016

@author: hossam
"""

# % ======================================================== %
# % Files of the Matlab programs included in the book:       %
# % Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
# % Second Edition, Luniver Press, (2010).   www.luniver.com %
# % ======================================================== %
#
# % -------------------------------------------------------- %
# % Firefly Algorithm for constrained optimization using     %
# % for the design of a spring (benchmark)                   %
# % by Xin-She Yang (Cambridge University) Copyright @2009   %
# % -------------------------------------------------------- %
from utils.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------
import numpy as np
import time

def alpha_new(alpha, num_generations):
	# % alpha_n=alpha_0(1-delta)^num_generations=10^(-4);
	# % alpha_0=0.9
	delta = 1 - (10 ** (-4) / 0.9) ** (1 / num_generations)
	alpha = (1 - delta) * alpha
	return alpha

def PFFA(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)

	# General parameters
	# population_size = 50 # number of fireflies
	# dimension = 30 #dimension
	# lb = -50
	# ub = 50
	# iterations = 500

	# FFA parameters
	alpha = 0.5 # Randomness 0--1 (highly random)
	beta_min = 0.20 # minimum value of beta
	gamma = 1 # Absorption coefficient

	# zn = np.ones(population_size)
	# zn.fill(float("inf"))
	zn = pymp.shared.array(population_size, dtype="float")
	zn.fill(float("inf"))

	# ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
	# ns = np.zeros((population_size, dimension))
	ns = pymp.shared.array((population_size, dimension), dtype="float")
	ns[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb

	# lightn = np.ones(population_size)
	# lightn.fill(float("inf"))
	lightn = pymp.shared.array(population_size, dtype="float")
	lightn.fill(float("inf"))

	# labels_pred = np.zeros((population_size, len(points)))
	labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")

	# [ns,lightn]=init_ffa(population_size,d,Lb,Ub,u0)

	convergence = []
	sol = Solution()

	print("P_MP_FFA is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	# Main loop
	for k in range(iterations): # start iterations
		# This line of reducing alpha is optional
		alpha = alpha_new(alpha, iterations)

		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			# Evaluate new solutions (for all population_size fireflies)
			for i in p.range(population_size):
				startpts = np.reshape(ns[i, :], (num_clusters, num_features))
				if objective_function.__name__ in ["SSE", "SC", "DI"]:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters, metric)
				else:
					fitness_value, labels_pred_values = objective_function(startpts, points, num_clusters)
				zn[i] = fitness_value
				lightn[i] = zn[i]
				labels_pred[i, :] = labels_pred_values
		# ------------------------

		# Ranking fireflies by their light intensity/objectives
		lightn = np.sort(zn)
		index = np.argsort(zn)
		ns[:] = ns[index, :]

		# Find the current best
		nso = ns
		lighto = lightn
		nbest = ns[0, :]
		light_best = lightn[0]
		labels_pred_best = labels_pred[0]

		# % For output only
		fbest = light_best

		# % Move all fireflies to the better locations
		# [ns]=ffa_move(population_size,d,ns,lightn,nso,lighto,nbest,...
		# light_best,alpha,beta_min,gamma,Lb,Ub);

		scale = np.ones(dimension) * abs(ub - lb)
		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for i in p.range(population_size):
				# The attractiveness parameter beta=exp(-gamma*r)
				for j in range(population_size):
					# r = np.sqrt(np.sum(np.power((ns[i, :] - ns[j, :]), 2)))
					r = np.sqrt(np.sum((ns[i, :] - ns[j, :]) ** 2))
					
					# r = 1
					# Update moves
					if lightn[i] > lighto[j]: # Brighter and more attractive
						beta0 = 1
						beta = (beta0 - beta_min) * np.exp(-gamma * r ** 2) + beta_min
						tmpf = alpha * (np.random.rand(dimension) - 0.5) * scale
						ns[i, :] = ns[i, :] * (1 - beta) + nso[j, :] * beta + tmpf
		# ------------------------

		# ns = np.clip(ns, lb, ub)
		convergence.append(fbest)
		print(["At iteration " + str(k) + " the best fitness is " + str(fbest)])
	# End main loop
	
	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "P_MP_FFA"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(labels_pred_best, dtype=np.int64)
	sol.best_individual = nbest
	sol.fitness = fbest

	sol.save()
	# return sol