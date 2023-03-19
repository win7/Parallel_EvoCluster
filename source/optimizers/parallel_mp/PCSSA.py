from source.solution import Solution

# ------- Parallel -------
import pymp
# ------------------------

import numpy as np
import math
import time
import random

def PSSA(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population, cores):
	num_features = int(dimension / num_clusters)
	# iterations = 1000
	# lb = -100
	# ub = 100
	# dimension = 30
	# population_size = 50 # Number of search agents

	convergence = np.zeros(iterations)

	# Initialize the positions of salps
	# ------- Parallel -------
	salp_positions = pymp.shared.array((population_size, dimension), dtype="float")
	salp_positions[:] = np.copy(population) # np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
	salp_fitness = pymp.shared.array(population_size, dtype="float")
	salp_fitness.fill(float("inf"))
	salp_labels_pred = pymp.shared.array((population_size, len(points)), dtype="float")
	salp_labels_pred.fill(float("inf"))

	food_position = pymp.shared.array(dimension, dtype="float")
	food_fitness = pymp.shared.array(1, dtype="float")
	food_fitness.fill(float("inf"))
	food_labels_pred = pymp.shared.array(len(points), dtype="float")
	food_labels_pred.fill(float("inf"))
	# ------------------------
	# Moth_fitness=np.fell(float("inf"))

	sol = Solution()

	print("SSA_mp is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	# ------- Parallel -------
	with pymp.Parallel(cores) as p:
		for k in p.range(population_size):
			# Evaluate moths
			startpts = np.reshape(salp_positions[k, :], (num_clusters, num_features))

			if objective_function.__name__ in ["SSE", "SC", "DI"]:
				fitness, labels_pred = objective_function(startpts, points, num_clusters, metric)
			else:
				fitness, labels_pred = objective_function(startpts, points, num_clusters)
			salp_fitness[k] = fitness
			salp_labels_pred[k, :] = labels_pred
	# ------------------------

	sorted_salps_fitness = np.sort(salp_fitness)
	I = np.argsort(salp_fitness)

	sorted_salps = np.copy(salp_positions[I, :])
	sorted_labels_pred = np.copy(salp_labels_pred[I, :])

	food_position[:] = np.copy(sorted_salps[0, :])
	food_fitness[0] = sorted_salps_fitness[0]
	food_labels_pred[:] = sorted_labels_pred[0]
	
	convergence[0] = food_fitness[0]
	print(["At iteration 0 the best fitness is " + str(food_fitness)])
	
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

		# ------- Parallel -------
		with pymp.Parallel(cores) as p:
			for k in p.range(population_size):
				# Check if salps go out of the search spaceand bring it back
				salp_positions[k, :] = np.clip(salp_positions[k, :], lb, ub)

				startpts = np.reshape(salp_positions[k, :], (num_clusters, num_features))

				if objective_function.__name__ in ["SSE", "SC", "DI"]:
					fitness, labels_pred = objective_function(startpts, points, num_clusters, metric)
				else:
					fitness, labels_pred = objective_function(startpts, points, num_clusters)

				salp_fitness[k] = fitness
				salp_labels_pred[k, :] = np.copy(labels_pred)
				
				with p.lock:
					if salp_fitness[k] < food_fitness[0]:
						food_position[:] = np.copy(salp_positions[k, :])
						food_fitness[0] = salp_fitness[k]
						food_labels_pred[:] = np.copy(salp_labels_pred[k, :])
		# ------------------------

		convergence[iteration] = food_fitness[0]
		iteration += 1
		# Display best fitness along the iteration
		print(["At iteration " + str(iteration - 1) + " the best fitness is " + str(food_fitness[0])])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence
	sol.optimizer = "SSA_mp"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(food_labels_pred, dtype=np.int64)
	sol.best_individual = food_position
	sol.fitness = food_fitness[0]

	sol.save()
	# return sol