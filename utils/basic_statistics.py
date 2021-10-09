from utils import algorithms, dataset_name, clusters, features, seeds

import math
import numpy as np
import os

index_algorithms = [0, 1] # Change
index_datasets = [0, 3, 6, 7, 8] # Change

for i in index_algorithms:
	print("Algorithm: {}".format(algorithms[i]))
	for j in index_datasets:
		# Read data
		filename = "output/{}_{}_metric.out".format(algorithms[i], dataset_name[j])
		raw_data = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), filename), "rt")
		data = np.loadtxt(raw_data, delimiter=",")
		# print(data)
		metric = data[:,:1].tolist()
		run_time = data[:,1:].tolist()

		print("Dataset: {}".format(dataset_name[j]))
		print("Metrics")
		print("---------------------")
		print(np.amin(metric))
		print(np.amax(metric))
		print(np.mean(metric))
		print(np.median(metric))
		print("---------------------")

		print("Runtime")
		print("---------------------")
		print(np.amin(run_time))
		print(np.amax(run_time))
		print(np.mean(run_time))
		print(np.median(run_time))
		print("---------------------")