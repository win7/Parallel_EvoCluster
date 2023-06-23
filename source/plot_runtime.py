import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import mplp
# help(mplp)

def run(results_directory, optimizer, objective_function, dataset_list, iterations, show=False):
	mfig = mplp.Mfig(format="double", formatting="landscape")
	fig, ax = mfig.subplots(figsize=(8, 5))
	colors = mfig.get_color_cycle()
	# plt.ioff()
	
	file_results_data = pd.read_csv(results_directory + "/experiment_avg.csv")

	for i in range(len(dataset_list)):
		for j in range (len(objective_function)):
			objective_name = objective_function[j]

			fig_name = results_directory + "/runtime-" + dataset_list[i] + "-" + objective_name

			detailed_data = file_results_data[(file_results_data["Dataset"] == dataset_list[i]) &  (file_results_data["ObjfName"] == objective_name)]

			bars = pd.DataFrame(detailed_data, columns=["Optimizer", "ExecutionTime"])
			bars = bars.sort_values(by=["ExecutionTime"], ascending=False)

			optimizers = bars.iloc[:, 0].values
			x = []
			for optimizer_name in optimizers:
				if not "_" in optimizer_name:
					optimizer_name = r"%s" % (optimizer_name)
				else:
					optimizer_name = optimizer_name.split("_")
					optimizer_name = r"%s\textsubscript{%s}" % (optimizer_name[0], optimizer_name[1])
				x.append(optimizer_name)
							
			y = bars.iloc[:, 1].values
			ax.barh(x, y,  height=0.8)
			
			# annotate
			ax.bar_label(ax.containers[0], label_type="edge", rotation=0)

			# plt.suptitle("Runtime: {} dataset".format(dataset_list[i]))
			# plt.xticks(rotation=0)
			ax.set_xlabel("Runtime (sec.)")
			ax.set_ylabel("Optimizer")
			# ax.legend(loc="upper right")
			# plt.grid()
			plt.savefig(fig_name + ".png", bbox_inches="tight")
			if show:
				plt.show()
			# plt.clf()
			mfig.savefig(fig_name)

if __name__ == "__main__":
	results_directory =  "results_v1_update/2022-10-18_09_44_21"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				 "SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_mpi", "MFO_mpi", "CS_mpi",
				 "SSA_mp", "PSO_mp", "GA_mp", "BAT_mp", "FFA_mp", "GWO_mp", "WOA_mp", "MVO_mp", "MFO_mp", "CS_mp"]

	objective_function = ["SSE"]
	dataset_list = ["aniso"]
	iterations = 100
	
	run(results_directory, optimizer, objective_function, dataset_list, iterations, show=False)

# Run:
# $ python source/plot_runtime.py