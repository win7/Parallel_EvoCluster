from source.optimizer import run
import numpy as np

if __name__ == "__main__":
	# Number cores for MPI
	cores = 24

	# Select optimizers
	# "SSO", "SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS"
	# "SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_pmi", "MFO_mpi", "CS_mpi",
	# "SSA_mp", "PSO_mp", "GA_mp", "BAT_mp", "FFA_mp", "GWO_mp", "WOA_mp", "MVO_mp", "MFO_mp", "CS_mp"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				 "SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_mpi", "MFO_mpi", "CS_mpi",
				 "SSA_mp", "PSO_mp", "GA_mp", "BAT_mp", "FFA_mp", "GWO_mp", "WOA_mp", "MVO_mp", "MFO_mp", "CS_mp"]
	# optimizer = ["BAT", "BAT_mpi", "BAT_mp"]

	# Select objective function
	# "SSE", "TWCV", "SC", "DB", "DI"
	objective_function = ["SSE", "TWCV", "SC", "DB", "DI"]
	objective_function=["SSE"] 

	# Select data sets
	# "aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "blood", "circles", "diagnosis_II", "ecoli", "flame","glass", "heart", "ionosphere", "iris", 
	# "iris2D", "jain", "liver", "moons", "mouse", "pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"
	dataset_list = np.array(["aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "blood", "circles", "diagnosis_II", "ecoli", 
					"flame", "glass", "heart", "iris", "iris2D", "ionosphere", "jain", "liver", "moons", "mouse", 
					"pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"])

	# Select cluster numbers for dataset
	clusters = np.array([7, 3, 2, 3, 2, 3, 2, 2, 2, 5, 2, 6, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 4, 2, 3, 3, 2, 3, 2, 3])

	# Select index for dataset and clusters numbers
	index = [13] # [1, 4, 5, 9, 15, 17, 19, 28]

	# Select number of repetitions for each experiment.
	# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
	num_runs = 2 # 5, 10

	# Select general parameters for all optimizers (population size, number of iterations, number of cores for MP)
	params = {"population_size": cores * 30, "iterations": 100, "cores": cores}

	# Choose whether to Export the results in different formats
	export_flags = {
		"export_avg": True,
		"export_details": True,
		"export_details_labels": True,
		"export_best_params": False,
		"export_convergence": True,
		"export_boxplot": False,
		"export_runtime": True
	}

	# Policy for migrations
	topology = ["RING", "TREE", "NETA", "NETB", "TORUS", "GRAPH", "SAME", "GOODBAD", "RAND"]
	emigration = ["CLONE", "REMOVE"]
	choice_emi = ["BEST", "WORST", "RANDOM"]
	choice_imm = ["BEST", "WORST", "RANDOM"]
	number_emi_imm = [1, 2, 3, 4, 5]
	interval_emi_imm = [1, 2, 4, 6, 8, 10]

	best_params_policy = [
		[2, 0, 1, 0, 4, 0], # MPI_SSA: NETA
		[3, 0, 1, 1, 4, 0], # MPI_PSO: NETB
		[2, 0, 0, 0, 3, 0], # MPI_GA: NETA
		[3, 1, 1, 0, 2, 4], # MPI_BAT: NETB
		[2, 0, 1, 2, 4, 1], # MPI_FFA: NETA
		[5, 0, 2, 2, 2, 1], # MPI_GWO: GRAPH
		[3, 1, 0, 2, 3, 0], # MPI_WOA: NETB
		[5, 0, 2, 2, 0, 0], # MPI_MVO: GRAPH
		[5, 0, 0, 2, 1, 5], # MPI_MFO: GRAPH
		[1, 0, 2, 0, 1, 1]  # MPI_CS: TREE
	]   # Change under best params for config1 and config2
	# Select index for params_policy (topology)
	# 0, 1, 2, 3, 4, 5, 6, 7, 8

	mpi_optimizer = ["SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_mpi", "MFO_mpi", "CS_mpi"]
	list_policy = []
	for i, item in enumerate(optimizer):
		if item in mpi_optimizer:
			index_policy = mpi_optimizer.index(item)
			policy = {
				"topology": topology[best_params_policy[index_policy][0]],
				"emigration": emigration[best_params_policy[index_policy][1]],
				"choice_emi": choice_emi[best_params_policy[index_policy][2]],
				"choice_imm": choice_imm[best_params_policy[index_policy][3]],
				"number_emi_imm": number_emi_imm[best_params_policy[index_policy][4]],
				"interval_emi_imm": interval_emi_imm[best_params_policy[index_policy][5]]
			}
		else:
			policy = {
				"topology": "-",
				"emigration": "-",
				"choice_emi": "-",
				"choice_imm": "-",
				"number_emi_imm": "-",
				"interval_emi_imm": "-"
			}
		list_policy.append(policy)
	# print(list_policy)

	# run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, policy)
	run(optimizer, objective_function, list(dataset_list[index]), num_runs, params, export_flags, list_policy,
		auto_cluster=False, num_clusters=list(clusters[index]), labels_exist=True, metric="euclidean")

# Run:
# python example.py
# python -m profile example.py
# py-spy top -- python example.py 
# py-spy record -o profile.svg -- python example.py