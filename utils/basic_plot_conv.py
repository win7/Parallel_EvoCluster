from utils import algorithms, dataset_name, clusters, features, seeds

import math
import numpy as np
import matplotlib.pyplot as plt
import os

number_runs = 2 # Change
max_iteration = 100 # Change
index_algorithms = [0, 1, 2, 3, 4, 9] # Change
index_datasets = [0, 3, 6, 7, 8] # Change

columns = 2
rows = int(math.ceil(len(index_datasets) / columns))
plt.figure(0)

for count, i in enumerate(index_datasets):
	print("Dataset: {}".format(dataset_name[i]))

	plt.subplot2grid((rows, columns), (int((count - (count % columns)) / columns), count % columns))
	for j in index_algorithms:
		print("Algorithm: {}".format(algorithms[j]))

		# Read data
		filename = "output/{}_{}_conv.out".format(algorithms[j], dataset_name[i])
		raw_data = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), filename), "rt")
		data = np.loadtxt(raw_data, delimiter=",")
		points = data[:].tolist()
		# print(points)
		
		convergence = [0] * max_iteration
		for k in range(number_runs):
			aux = points[k * max_iteration:k * max_iteration + max_iteration]
			# print(aux)
			for c in range(max_iteration):
				convergence[c] += aux[c]

		for k in range(len(convergence)):
			convergence[k] /= number_runs
		
		# Plot mectric vs iteration
		y = convergence
		x = list(range(len(y)))
		
		plt.plot(x, y, linestyle="-", label=algorithms[j], linewidth=2, marker="")

	# Naming the x axis 
	plt.xlabel("Iterations", fontsize=12)
	plt.ylabel("Metrics", fontsize=12) 
	plt.title("Dataset: {}".format(dataset_name[i]), fontsize=14)
	plt.grid()

	# Show a legend on the plot 
	plt.legend(loc="upper right", ncol=3, borderaxespad=0.2, prop={"size":8})
	# plt.legend(loc="upper right", ncol=3, borderaxespad=0.2, prop={"size":12})

# plt.subplots_adjust(top=0.96, bottom=0.045, left=0.06, right=0.985, hspace=0.280, wspace=0.150)
plt.subplots_adjust(top=0.96, bottom=0.066, left=0.06, right=0.985, hspace=0.280, wspace=0.150)

# Function to show the plot 
plt.show()
plt.ion()
	
# Run:
# python basic_plot_conv.py