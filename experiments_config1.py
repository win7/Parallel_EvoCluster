from operator import itemgetter
from source.optimizer import run

import numpy as np

if __name__ == "__main__":
	# Number cores for MPI
	cores = 24

	# Select optimizers
	# "SSO", "SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS"
	# "SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_pmi", "MFO_mpi", "CS_mpi",
	# "SSA_mp", "PSO_mp", "GA_mp", "BAT_mp", "FFA_mp", "GWO_mp", "WOA_mp", "MVO_mp", "MFO_mp", "CS_mp"
	optimizer = ["SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_mpi", "MFO_mpi", "CS_mpi"]

	# Select objective function
	# "SSE", "TWCV", "SC", "DB", "DI"
	objective_function = ["SSE", "TWCV", "SC", "DB", "DI"]
	objective_function = ["SSE"] 

	# Select datasets
	# "aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "blood", "circles", "diagnosis_II", "ecoli", "flame","glass", "heart", "ionosphere", "iris", 
	# "iris2D", "jain", "liver", "moons", "mouse", "pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"
	dataset_list = np.array(["aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "blood", "circles", "diagnosis_II", "ecoli", 
					"flame", "glass", "heart", "iris", "iris2D", "ionosphere", "jain", "liver", "moons", "mouse", 
					"pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"])

	# Select cluster numbers for dataset
	clusters = np.array([7, 3, 2, 3, 2, 3, 2, 2, 2, 5, 2, 6, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 4, 2, 3, 3, 2, 3, 2, 3])

	# Select index for dataset and clusters numbers
	index = [3, 6, 20, 22, 25, 29]

	# Select number of repetitions for each experiment.
	# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
	num_runs = 5

	# Select general parameters for all optimizers (population size, number of iterations, number of cores for MP)
	params = {"population_size": cores * 30, "iterations": 100, "cores": cores}

	# Choose whether to Export the results in different formats
	export_flags = {
		"export_avg": False,
		"export_details": False,
		"export_details_labels": False,
		"export_best_params": True,
		"export_convergence": False,
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
				for item4 in choice_imm:
					policy = {
						"topology": item1,
						"emigration": item2,
						"choice_emi": item3,
						"choice_imm": item4,
						"number_emi_imm": 4,
						"interval_emi_imm": 4
					}
					print(policy)
					# run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, policy)
					run(optimizer, objective_function, list(dataset_list[index]), num_runs, params, export_flags, [policy], 
						auto_cluster=False, num_clusters=list(clusters[index]), labels_exist=True, metric="euclidean")

# Run:
# python experiments_config1.py