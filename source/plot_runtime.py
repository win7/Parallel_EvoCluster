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
			ax = bars.plot(x="Optimizer", y="ExecutionTime", kind="barh", xlabel="Runtime (s)", ylabel="Optimizers", title="", legend=True)
			# annotate
			ax.bar_label(ax.containers[0], label_type="edge", rotation=0)
			
			plt.suptitle("Runtime: {} dataset".format(dataset_list[i]))
			plt.xticks(rotation=0)
			plt.legend(loc="upper right")
			plt.grid()
			plt.savefig(fig_name, bbox_inches="tight")
			if show:
				plt.show()
			plt.clf()

if __name__ == "__main__":
	results_directory = "results/2022-10-30_13:09:03"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				 "SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_mpi", "MFO_mpi", "CS_mpi",
				 "SSA_mp", "PSO_mp", "GA_mp", "BAT_mp", "FFA_mp", "GWO_mp", "WOA_mp", "MVO_mp", "MFO_mp", "CS_mp"]

	objective_function = ["SSE"]
	dataset_list = ["iris"]
	iterations = 30
	
	run(results_directory, optimizer, objective_function, dataset_list, iterations, show=True)

# Run:
# $ python source/plot_runtime.py