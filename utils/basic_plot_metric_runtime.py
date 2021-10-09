from utils import algorithms, dataset_name, clusters, features, seeds

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

index_algorithms = [0, 1, 2, 3, 4, 5, 9] # Change [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 9: SSA
index_datasets = [0, 3, 6, 7, 8] # [0, 3, 6, 7, 8] # Change
values = "metric" # Change metric, runtime

title = "Runtime "
ylabel = "Runtime (ms)"

labels = [] # Dataset label
for index in index_datasets:
	labels.append(dataset_name[index])

# global_data = res = [[0 for i in range(len(index_datasets))] for j in range(len(index_algorithms))]
global_data = np.zeros((len(index_algorithms), len(index_datasets)))

for count_i, i in enumerate(index_datasets):
	print("Dataset: {}".format(dataset_name[i]))

	aux = []
	for count_j, j in enumerate(index_algorithms):
		print(j)
		print("Algorithm: {}".format(algorithms[j]))

		# Read data
		filename = "output/{}_{}_metric.out".format(algorithms[j], dataset_name[i])
		raw_data = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), filename), "rt")
		data = np.loadtxt(raw_data, delimiter=",")
		
		if values == "metric":
			metric = data[:,:1].tolist()
			global_data[count_j][count_i] = np.mean(metric)
			title = "Metric "
			ylabel = "Metric"
		else:
			run_time = data[:,1:].tolist()
			global_data[count_j][count_i] = np.mean(run_time)

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()

for k, item in enumerate(global_data):
	rects1 = ax.bar(x + k * width, item, width, label=algorithms[index_algorithms[k]])
	ax.bar_label(rects1, padding=3, rotation=60)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(ylabel)
ax.set_title("{} by dataset and algorithms".format(title))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.grid()
plt.show()
