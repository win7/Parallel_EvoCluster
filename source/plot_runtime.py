import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run(results_directory, optimizer, objective_function, dataset_list, iterations, show=False):
	plt.ioff()
	file_results_data = pd.read_csv(results_directory + "/experiment_avg.csv")
	
	for i in range(len(dataset_list)):
		for j in range (len(objective_function)):
			objective_name = objective_function[j]

			fig_name = results_directory + "/runtime-" + dataset_list[i] + "-" + objective_name + ".png"

			detailed_data = file_results_data[(file_results_data["Dataset"] == dataset_list[i]) &  (file_results_data["ObjfName"] == objective_name)]

			bars = pd.DataFrame(detailed_data, columns=["Optimizer", "ExecutionTime"])
			bars = bars.sort_values(by=["ExecutionTime"], ascending=False)
			ax = bars.plot(x="Optimizer", y="ExecutionTime", kind="barh", xlabel="Optimizers", ylabel="Runtime (s)", title="", legend=True)
			# annotate
			ax.bar_label(ax.containers[0], label_type="edge", rotation=0)
			ax.set_xlabel("Runtime (s)")
			plt.suptitle("Runtime: {} dataset".format(dataset_list[i]))
			plt.xticks(rotation=0)
			plt.legend(loc="upper right")
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
# $ python utils/plot_runtime.py