from operator import itemgetter
from source.optimizer import run

import numpy as np

if __name__ == "__main__":
	# Number cores for MPI
	cores = 24

	# Select optimizers
	# "SSO", "SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS"
	# "MPI_SSA", "MPI_PSO", "MPI_GA", "MPI_BAT", "MPI_FFA", "MPI_GWO", "MPI_WOA", "MPI_MVO", "MPI_MFO", "MPI_CS"
	# "MP_SSA", "MP_PSO", "MP_GA", "MP_BAT", "MP_FFA", "MP_GWO", "MP_WOA", "MMVO", "MMFO", "MP_CS"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				"MPI_SSA", "MPI_PSO", "MPI_GA", "MPI_BAT", "MPI_FFA", "MPI_GWO", "MPI_WOA", "MPI_MVO", "MPI_MFO", "MPI_CS",
				"MP_SSA", "MP_PSO", "MP_GA", "MP_BAT", "MP_FFA", "MP_GWO", "MP_WOA", "MP_MVO", "MP_MFO", "MP_CS"]
	optimizer = ["MPI_SSA", "MPI_PSO", "MPI_GA", "MPI_BAT", "MPI_FFA", "MPI_GWO", "MPI_WOA", "MPI_MVO", "MPI_MFO", "MPI_CS"]

	# Select objective function
	# "SSE", "TWCV", "SC", "DB", "DI"
	objective_function = ["SSE", "TWCV", "SC", "DB", "DI"]
	objective_function=["SSE"] 

	# Select datasets
	# "aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "blood", "circles", "diagnosis_II", "ecoli", "flame","glass", "heart", "ionosphere", "iris", 
	# "iris2D", "jain", "liver", "moons", "mouse", "pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"
	dataset_list = ["aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "blood", "circles", "diagnosis_II", "ecoli", 
                    "flame", "glass", "heart", "ionosphere", "iris", "iris2D", "jain", "liver", "moons", "mouse", 
                    "pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"]

	# Select cluster numbers for dataset
	clusters = [7, 3, 2, 3, 2, 3, 2, 2, 2, 5, 2, 6, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 4, 2, 3, 3, 2, 3, 2, 3]

	# Select index for dataset and clusters numbers
	index = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

	# Select number of repetitions for each experiment.
	# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
	num_runs = 10

	# Select general parameters for all optimizers (population size, number of iterations, number of cores for MP)
	params = {"population_size": cores * 30, "iterations": 100, "cores": cores}

	# Choose whether to Export the results in different formats
	export_flags = {
		"export_avg": True,
		"export_details": False,
		"export_details_labels": False,
		"export_convergence": True,
		"export_boxplot": False,
		"export_runtime": False
	}

	# Policy for migrations
	topology = ["RING", "TREE", "NETA", "NETB", "TORUS", "GRAPH", "SAME", "GOODBAD", "RAND"]
	emigration = ["CLONE", "REMOVE"]
	choice_emi = ["BEST", "WORST", "RANDOM"]
	choice_imm = ["BEST", "WORST", "RANDOM"]
	number_emi_imm = [1, 2, 3, 4, 5]
	interval_emi_imm = [1, 2, 4, 6, 8, 10]

	for item1 in topology:
		for item2 in emigration:
			for item3 in choice_emi:
				for item3 in choice_imm:
					policy = {
						"topology": item1,
						"emigration": item2,
						"choice_emi": item3,
						"choice_imm": item4,
						"number_emi_imm": 5,
						"interval_emi_imm": 4
					}

					# run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, policy)
					run(optimizer, objective_function, itemgetter(*index)(dataset_list), num_runs, params, export_flags, policy, auto_cluster=False, num_clusters=itemgetter(*index)(clusters), labels_exist=True, metric="euclidean")

# Run:
# python experiments.py