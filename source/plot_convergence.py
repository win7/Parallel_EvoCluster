import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run(results_directory, optimizer, objective_function, dataset_list, iterations, show=False):
	plt.ioff()
	file_results_data = pd.read_csv(results_directory + "/experiment.csv")
	
	# Set values for colors
	# colormap = plt.cm.gist_ncar
	# plt.gca().set_prop_cycle(plt.cycler("color", plt.cm.jet(np.linspace(0, 1, len(optimizer)))))

	for i in range(len(dataset_list)):
		# dataset_filename = dataset_list[i] + ".csv" 
		for j in range (len(objective_function)):
			objective_name = objective_function[j]

			start_iteration = 0
			""" if "SSA" in optimizer: # Obs
				start_iteration = 1 """
			all_generations = [x + 1 for x in range(start_iteration, iterations)]
			for k in range(len(optimizer)):
				optimizer_name = optimizer[k]
				
				if "MPI_" == optimizer_name[:4]:
					line_style = "dashed"
				elif "MP_" == optimizer_name[:3]:
					line_style = "dotted"
				else:
					line_style = "solid"

				file_results_data = file_results_data.drop(["SSE", "Purity", "Entropy", "HS", "CS", "VM", "AMI", "ARI", 
					"Fmeasure", "TWCV", "SC", "Accuracy", "DI", "DB", "STDev"], errors="ignore", axis=1)
				row = file_results_data[(file_results_data["Dataset"] == dataset_list[i]) & 
					(file_results_data["Optimizer"] == optimizer_name) & (file_results_data["ObjfName"] == objective_name)]
				row = row.iloc[:, 6 + start_iteration:]
				plt.plot(all_generations, row.values.tolist()[0], label=optimizer_name, linestyle=line_style, linewidth=1.5, marker="")

			fig_name = results_directory + "/convergence-" + dataset_list[i] + "-" + objective_name + ".png"
			plt.suptitle("Convergence Graph - Dataset: {}".format(dataset_list[i]))
			plt.xlabel("Iterations")
			plt.ylabel("Fitness")
			# plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
			plt.legend(loc="upper right", ncol=int(np.ceil(len(optimizer) / 10)), borderaxespad=0.2, prop={"size":9})
			plt.grid()
			plt.savefig(fig_name, bbox_inches="tight")
			if show:
				plt.show()
			plt.clf()

if __name__ == "__main__":
	results_directory = "2021-05-20_23:35:35"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				"MPI_SSA", "MPI_PSO", "MPI_GA", "MPI_BAT", "MPI_FFA", "MPI_GWO", "MPI_WOA", "MPI_MVO", "MPI_MFO", "MPI_CS",
				"MP_SSA", "MP_PSO", "MP_GA", "MP_BAT", "MP_FFA", "MP_GWO", "MP_WOA", "MP_MVO", "MP_MFO", "MP_CS"]

	objective_function = ["SSE"]
	dataset_list = ["iris", "wine"]
	iterations = 100
	
	run(results_directory, optimizer, objective_function, dataset_list, iterations, show=True)

# Run:
# $ python utils/plot_convergence.py