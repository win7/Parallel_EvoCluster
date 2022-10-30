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

	params_policy = [
		# MPI_SSA
		[
			[0, 1, 0, 4, 1], # RING: 0
			[0, 0, 1, 4, 0], # TREE: 1
			[0, 1, 0, 4, 0], # NETA: 2
			[0, 1, 0, 3, 2], # NETB: 3
			[0, 1, 1, 4, 0], # TORUS: 4
			[0, 0, 2, 1, 2], # GRAPH: 5
			[0, 2, 0, 2, 2], # SAME: 6
			[1, 2, 1, 3, 4], # GOODBAD: 7
			[1, 1, 1, 0, 0]  # RAND: 8
		],
		# MPI_PSO
		[
			[1, 2, 0, 4, 1], # RING: 0
			[0, 2, 1, 4, 0], # TREE: 1
			[0, 1, 0, 4, 0], # NETA: 2
			[0, 1, 1, 4, 0], # NETB: 3
			[0, 1, 0, 2, 0], # TORUS: 4
			[0, 2, 0, 3, 0], # GRAPH: 5
			[1, 2, 0, 2, 1], # SAME: 6
			[1, 2, 0, 1, 0], # GOODBAD: 7
			[0, 2, 0, 0, 5]  # RAND: 8
		],
		# MPI_GA
		[
			[0, 0, 0, 4, 0], # RING: 0
			[0, 0, 1, 3, 0], # TREE: 1
			[0, 0, 0, 3, 0], # NETA: 2
			[0, 1, 0, 2, 0], # NETB: 3
			[0, 1, 1, 3, 0], # TORUS: 4
			[0, 1, 2, 0, 0], # GRAPH: 5
			[0, 0, 2, 2, 4], # SAME: 6
			[1, 2, 1, 3, 2], # GOODBAD: 7
			[0, 0, 0, 0, 1]  # RAND: 8
		],
		# MPI_BAT
		[
			[0, 0, 1, 1, 0], # RING: 0
			[1, 1, 2, 4, 0], # TREE: 1
			[1, 2, 2, 3, 2], # NETA: 2
			[1, 1, 0, 2, 4], # NETB: 3
			[1, 2, 2, 3, 3], # TORUS: 4
			[0, 1, 1, 4, 0], # GRAPH: 5
			[0, 0, 2, 1, 4], # SAME: 6
			[1, 2, 2, 3, 1], # GOODBAD: 7
			[0, 0, 2, 3, 3]  # RAND: 8
		],
		# MPI_FFA
		[
			[0, 1, 2, 4, 1], # RING: 0
			[0, 1, 2, 1, 0], # TREE: 1
			[0, 1, 2, 4, 1], # NETA: 2
			[1, 2, 1, 4, 1], # NETB: 3
			[0, 2, 2, 4, 1], # TORUS: 4
			[0, 2, 2, 4, 0], # GRAPH: 5
			[0, 0, 2, 2, 1], # SAME: 6
			[1, 2, 1, 4, 0], # GOODBAD: 7
			[0, 0, 2, 0, 1]  # RAND: 8
		],
		# MPI_GWO
		[
			[0, 0, 1, 4, 0], # RING: 0
			[0, 1, 1, 4, 0], # TREE: 1
			[0, 1, 0, 4, 0], # NETA: 2
			[0, 1, 0, 2, 0], # NETB: 3
			[0, 0, 0, 4, 0], # TORUS: 4
			[0, 2, 2, 2, 1], # GRAPH: 5
			[0, 0, 2, 1, 4], # SAME: 6
			[0, 2, 0, 4, 0], # GOODBAD: 7
			[1, 1, 0, 2, 2]  # RAND: 8
		],
		# MPI_WOA
		[
			[0, 0, 1, 3, 0], # RING: 0
			[1, 0, 1, 3, 4], # TREE: 1
			[0, 2, 2, 3, 0], # NETA: 2
			[1, 0, 2, 3, 0], # NETB: 3
			[1, 2, 2, 1, 0], # TORUS: 4
			[0, 1, 2, 3, 2], # GRAPH: 5
			[1, 1, 0, 1, 3], # SAME: 6
			[1, 0, 1, 3, 0], # GOODBAD: 7
			[1, 0, 0, 0, 2]  # RAND: 8
		],
		# MPI_MVO
		[
			[0, 1, 0, 3, 0], # RING: 0
			[1, 0, 1, 1, 2], # TREE: 1
			[0, 0, 2, 4, 2], # NETA: 2
			[1, 0, 1, 4, 0], # NETB: 3
			[0, 0, 2, 2, 0], # TORUS: 4
			[0, 2, 2, 0, 0], # GRAPH: 5
			[1, 1, 0, 2, 1], # SAME: 6
			[0, 2, 2, 4, 4], # GOODBAD: 7
			[1, 1, 1, 1, 4]  # RAND: 8
		],
		# MPI_MFO
		[
			[0, 0, 0, 3, 0], # RING: 0
			[0, 1, 1, 3, 0], # TREE: 1
			[0, 0, 0, 2, 0], # NETA: 2
			[0, 0, 0, 3, 0], # NETB: 3
			[0, 0, 1, 3, 2], # TORUS: 4
			[0, 0, 2, 1, 5], # GRAPH: 5
			[1, 1, 2, 4, 4], # SAME: 6
			[1, 2, 1, 4, 5], # GOODBAD: 7
			[0, 1, 0, 4, 4]  # RAND: 8
		],
		# MPI_CS
		[
			[1, 2, 0, 3, 1], # RING: 0
			[0, 2, 0, 1, 1], # TREE: 1
			[1, 1, 1, 4, 5], # NETA: 2
			[0, 2, 0, 3, 5], # NETB: 3
			[1, 2, 0, 4, 4], # TORUS: 4
			[0, 2, 0, 1, 2], # GRAPH: 5
			[1, 2, 0, 1, 5], # SAME: 6
			[0, 0, 0, 3, 3], # GOODBAD: 7
			[1, 1, 1, 1, 0]  # RAND: 8
		]
	]

	for i, item1 in enumerate(optimizer):
		for j, item2 in enumerate(topology):
			policy = {
                "topology": item2,
                "emigration": emigration[params_policy[i][j][0]],
                "choice_emi": choice_emi[params_policy[i][j][1]],
                "choice_imm": choice_imm[params_policy[i][j][2]],
                "number_emi_imm": number_emi_imm[params_policy[i][j][3]],
                "interval_emi_imm": interval_emi_imm[params_policy[i][j][4]]
            }
			print(policy)
			# run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, policy)
			run([item1], objective_function, list(dataset_list[index]), num_runs, params, export_flags, policy, 
                auto_cluster=False, num_clusters=list(clusters[index]), labels_exist=True, metric="euclidean")
					
# Run:
# python experiments_config(1_2).py