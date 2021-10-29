from optimizer import run
import numpy as np

if __name__ == "__main__":
	# Number cores for MPI
	cores = 24

	# Select optimizers
	# "SSO", "SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS"
	# "P_MPI_SSA", "P_MPI_PSO", "P_MPI_GA", "P_MPI_BAT", "P_MPI_FFA", "P_MPI_GWO", "P_MPI_WOA", "P_MPI_MVO", "P_MPI_MFO", "P_MPI_CS"
	# "P_MP_SSA", "P_MP_PSO", "P_MP_GA", "P_MP_BAT", "P_MP_FFA", "P_MP_GWO", "P_MP_WOA", "P_MP_MVO", "P_MP_MFO", "P_MP_CS"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				"P_MPI_SSA", "P_MPI_PSO", "P_MPI_GA", "P_MPI_BAT", "P_MPI_FFA", "P_MPI_GWO", "P_MPI_WOA", "P_MPI_MVO", "P_MPI_MFO", "P_MPI_CS",
				"P_MP_SSA", "P_MP_PSO", "P_MP_GA", "P_MP_BAT", "P_MP_FFA", "P_MP_GWO", "P_MP_WOA", "P_MP_MVO", "P_MP_MFO", "P_MP_CS"]

	# Select objective function
	# "SSE", "TWCV", "SC", "DB", "DI"
	objective_function = ["SSE", "TWCV", "SC", "DB", "DI"]

	# Select data sets
	# "aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "Blood", "circles", "diagnosis_II", "ecoli", "flame","glass", "heart", "ionosphere", "iris", 
	# "iris2D", "jain", "liver", "moons", "mouse", "pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"
	dataset_list = ["aggregation", "aniso", "appendicitis", "balance", "banknote", "blobs", "Blood", "circles", "diagnosis_II", "ecoli", "flame", "glass", "heart", "ionosphere",
					"iris", "iris2D", "jain", "liver", "moons", "mouse", "pathbased", "seeds", "smiley", "sonar", "varied", "vary-density", "vertebral2", "vertebral3", "wdbc", "wine"]

	# Select number of repetitions for each experiment.
	# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
	num_runs = 1

	# Select general parameters for all optimizers (population size, number of iterations, number of cores for MP)
	params = {"population_size": cores * 10, "iterations": 2, "cores": 3}

	# Choose whether to Export the results in different formats
	export_flags = {
		"export_avg": True,
		"export_details": True,
		"export_details_labels": True,
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

	params_policy = [
		[0, 0, 0, 1, 4, 1], # RING: 0
		[1, 0, 0, 2, 4, 0], # TREE: 1
		[2, 0, 0, 2, 1, 0], # NETA: 2
		[3, 0, 0, 2, 2, 0], # NETB: 3
		[4, 0, 0, 2, 0, 0], # TORUS: 4
		[5, 0, 0, 2, 0, 1], # GRAPH: 5
		[6, 0, 0, 2, 1, 0], # SAME: 6
		[7, 0, 0, 2, 3, 1], # GOODBAD: 7
		[8, 0, 0, 1, 1, 0]  # RAND: 8
	]   # Change under best params for congig1 and config2
	# Select index for params_policy
	# 0, 1, 2, 3, 4, 5, 6, 7, 8
	index_policy = 2

	policy = {
		"topology": topology[params_policy[index_policy][0]],
		"emigration": emigration[params_policy[index_policy][1]],
		"choice_emi": choice_emi[params_policy[index_policy][2]],
		"choice_imm": choice_imm[params_policy[index_policy][3]],
		"number_emi_imm": number_emi_imm[params_policy[index_policy][4]],
		"interval_emi_imm": interval_emi_imm[params_policy[index_policy][5]]
	}

	run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, policy)

# Run:
# python example.py
# python -m profile example.py
# py-spy top -- python example.py 
