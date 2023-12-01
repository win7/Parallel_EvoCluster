import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys

sys.path.insert(0, os.getcwd() + "/source")
import mplp
# help(mplp)

def run(results_directory, optimizer, objective_function, dataset_list, iterations, show=False):
	mfig = mplp.Mfig(format="double", formatting="landscape")
	fig, ax = mfig.subplots(figsize=(8, 5))
	colors = mfig.get_color_cycle()
	# plt.ioff()

	file_results_data = pd.read_csv(results_directory + "/experiment_avg.csv")
	
	# Set values for colors
	# colormap = plt.cm.gist_ncar
	# plt.gca().set_prop_cycle(plt.cycler("color", plt.cm.jet(np.linspace(0, 1, len(optimizer)))))
	
	""" plt.rcParams.update({
		"text.usetex": True,
		"font.family": "Helvetica"
	}) """
	
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

				if "_mpi" == optimizer[k][-4:]:
					line_style = "dashed"
				elif "_mp" == optimizer[k][-3:]:
					line_style = "dotted"
				else:
					line_style = "solid"

				file_results_data = file_results_data.drop(["SSE", "Purity", "Entropy", "HS", "CS", "VM", "AMI", "ARI", 
					"Fmeasure", "TWCV", "SC", "Accuracy", "DI", "DB", "STDev"], errors="ignore", axis=1)
				row = file_results_data[(file_results_data["Dataset"] == dataset_list[i]) & 
					(file_results_data["Optimizer"] == optimizer_name) & (file_results_data["ObjfName"] == objective_name)]
				row = row.iloc[:, 6 + start_iteration:]
				
				if line_style == "solid":
					optimizer_name = r"%s" % (optimizer_name)
				else:
					optimizer_name = optimizer_name.split("_")
					optimizer_name = r"%s\textsubscript{%s}" % (optimizer_name[0], optimizer_name[1])
				
				ax.plot(all_generations, row.values.tolist()[0], label=optimizer_name, linestyle=line_style, linewidth=1.5, marker="")

			fig_name = results_directory + "/convergence-" + dataset_list[i] + "-" + objective_name # + ".png"
			# plt.suptitle("Convergence: {} dataset".format(dataset_list[i]))
			ax.set_xlabel("Iterations")
			ax.set_ylabel("Fitness")
			# plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
			ax.legend(loc="upper right", ncol=int(np.ceil(len(optimizer) / 10)), borderaxespad=0.2, prop={"size":9})
			# plt.grid()
			# plt.savefig(fig_name, bbox_inches="tight")
			if show:
				ax.show()
				# plt.clf()
			mfig.savefig(fig_name)

if __name__ == "__main__":
	results_directory =  "results_v1_update/2022-10-30_09_51_52" # "results/2022-10-30_09:51:52"
	optimizer = ["SSA", "PSO", "GA", "BAT", "FFA", "GWO", "WOA", "MVO", "MFO", "CS", 
				"SSA_mpi", "PSO_mpi", "GA_mpi", "BAT_mpi", "FFA_mpi", "GWO_mpi", "WOA_mpi", "MVO_mpi", "MFO_mpi", "CS_mpi",
				"SSA_mp", "PSO_mp", "GA_mp", "BAT_mp", "FFA_mp", "GWO_mp", "WOA_mp", "MVO_mp", "MFO_mp", "CS_mp"]

	objective_function = ["SSE"]
	dataset_list = ["ecoli"]
	iterations = 100
	
	run(results_directory, optimizer, objective_function, dataset_list, iterations, show=False)

# Run:
# $ python source/plot_convergence.py