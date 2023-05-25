# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:48 2019

@author: Raneem
"""
from source.params import Params
from source.solution import Solution

from sklearn import preprocessing
from pathlib import Path

import csv
import numpy as np
import os
import source.cluster_detection as clus_det
import source.measures as measures
# import source.objectives as objectives
import source.plot_convergence as convergence_plot
import source.plot_boxplot as box_plot
import source.plot_runtime as runtime_plot
import sys
import subprocess
import time
import warnings

warnings.simplefilter(action="ignore")

def run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, list_policy, auto_cluster=True, num_clusters="supervised", labels_exist=True, metric="euclidean"):
	"""
	It serves as the main interface of the framework for running the experiments.

	Parameters
	----------    
	optimizer: list
		The list of optimizers names
	objective_function: list
		The list of objective functions
	dataset_list: list
		The list of the names of the data sets files
	num_runs: int
		The number of independent runs 
	params: set
		The set of parameters which are: 
		1. Size of population (population_size)
		2. The number of iterations (iterations)
	export_flags: set
		The set of Boolean flags which are: 
		1. Export (Exporting the results in a file)
		2. export_details (Exporting the detailed results in files)
		3. export_details_labels (Exporting the labels detailed results in files)
		4. export_convergence (Exporting the covergence plots)
		5. export_boxplot (Exporting the box plots)
	list_policy: set
		[{
			"topology": "RING",		# Change: RING, TREE, NETA. NETB, TORUS, GRAPH, SAME, GOODBAD, RAND
			"emigration": "CLONE", 	# Change: CLONE, REMOVE
			"choice_emi": "BEST", 	# Change: BEST, WORST, RANDOM
			"choice_imm": "WORST", 	# Change: BEST, WORST, RANDOM
			"number_emi_imm": 5, 	# Change: 1, 2, 3, 4, 5
			"interval_emi_imm": 2 	# Change: 1, 2, 4, 6, 8, 10
		}, {}, {}]
	auto_cluster: boolean, default = True
		Choose whether the number of clusters is detected automatically. 
		If True, select one of the following: "supervised", "CH", "silhouette", "elbow", "gap", "min", "max", "median" for num_clusters. 
		If False, specify a list of integers for num_clusters. 
	num_clusters: string, or list, default = "supervised"
		A list of the number of clusters for the datasets in dataset_list
		Other values can be considered instead of specifying the real value, which are as follows:
		- supervised: The number of clusters is derived from the true labels of the datasets
		- elbow: The number of clusters is automatically detected by elbow method
		- gap: The number of clusters is automatically detected by gap analysis methos
		- silhouette: The number of clusters is automatically detected by silhouette coefficient method
		- CH: The number of clusters is automatically detected by Calinski-Harabasz index
		- DB: The number of clusters is automatically detected by Davies Bouldin index
		- BIC: The number of clusters is automatically detected by Bayesian Information Criterion score
		- min: The number of clusters is automatically detected by the minimum value of the number of clusters detected by all detection techniques
		- max: The number of clusters is automatically detected by the maximum value of the number of clusters detected by all detection techniques
		- median: The number of clusters is automatically detected by the median value of the number of clusters detected by all detection techniques
		- majority: The number of clusters is automatically detected by the majority vote of the number of clusters detected by all detection techniques
	labels_exist: boolean, default = True
		Specify if labels exist as the last column of the csv file of the datasets in dataset_list
		if the value is False, the following hold:
		- supervised value for num_clusters is not allowed
		- experiments, and experiments_details files contain only the evaluation measures for 
		  "SSE","TWCV","SC","DB","DI","STDev"
		- export_boxplot is set for "SSE","TWCV","SC","DB","DI","STDev"   
	metric: string, default = "euclidean"
		The metric to use when calculating the distance between points if applicable for the objective function selected. 
		It must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter
	Returns
	-----------
	population_size/A
	"""

	if not labels_exist and num_clusters == "supervised":
		print("Syupervised value for num_clusters is not allowed when labels_exist value is false")
		sys.exit()

	if isinstance(num_clusters, list):
		if len(num_clusters) != len(dataset_list):
			print(
				"Length of num_clusters list should equal the length of dataset_list list")
			sys.exit()
		if min(num_clusters) < 2:
			print("num_clusters value should be larger than 2")
			sys.exit()
		if auto_cluster:
			print("num_clusters should be string if auto_cluster is true")
			sys.exit()
	else:
		if auto_cluster == False:
			print("num_clusters should be a list of integers if auto_cluster is false")
			sys.exit()

	# Select general parameters for all optimizers (population size, number of iterations) ....
	population_size = params["population_size"]
	iterations = params["iterations"]
	cores = params["cores"]

	# Export results ?
	export_avg = export_flags["export_avg"]
	export_details = export_flags["export_details"]
	export_details_labels = export_flags["export_details_labels"]
	export_best_params = export_flags["export_best_params"]
	export_convergence = export_flags["export_convergence"]
	export_boxplot = export_flags["export_boxplot"]
	export_runtime = export_flags["export_runtime"]

	# Check if it works at least once
	flag_avg = False
	flag_details = False
	flag_details_Labels = False
	flag_best_params = False

	# CSV Header for for the cinvergence
	CnvgHeader = []

	if labels_exist:
		datasets_directory = "../datasets/supervised/"  # the directory where the dataset is stored
	else:
		# the directory where the dataset is stored
		datasets_directory = "../datasets/unsupervised/"

	results_directory = "results/{}/".format(time.strftime("%Y-%m-%d_%H:%M:%S"))
	Path(results_directory).mkdir(parents=True, exist_ok=True)

	dataset_len = len(dataset_list)

	k = [-1] * dataset_len
	f = [-1] * dataset_len
	points = [0] * dataset_len
	labels_true = [0] * dataset_len

	for l in range(0, iterations):
		CnvgHeader.append("Iterations" + str(l + 1))

	# read all datasets
	for h in range(dataset_len):
		dataset_filename = dataset_list[h] + ".csv"
		# Read the dataset file and generate the points list and true values
		raw_data = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), datasets_directory + dataset_filename), "rt")
		data = np.loadtxt(raw_data, delimiter=",")

		num_points, num_values = data.shape  # Number of points and Number of values for each point

		if labels_exist:
			f[h] = num_values - 1  # Dimension value
			points[h] = data[:, :-1].tolist()  # list of points
			# List of actual cluster of each points (last field)
			labels_true[h] = data[:, -1].tolist()
		else:
			f[h] = num_values  # Dimension value
			points[h] = data.copy().tolist()  # list of points
			# List of actual cluster of each points (last field)
			labels_true[h] = None

		points[h] = preprocessing.normalize(points[h], norm="max", axis=0)

		if num_clusters == "supervised":
			k[h] = len(np.unique(data[:, -1]))  # k: Number of clusters
		elif num_clusters == "elbow":
			k[h] = clus_det.ELBOW(points[h])  # k: Number of clusters
		elif num_clusters == "gap":
			k[h] = clus_det.GAP_STATISTICS(points[h])  # k: Number of clusters
		elif num_clusters == "silhouette":
			k[h] = clus_det.SC(points[h])  # k: Number of clusters
		elif num_clusters == "DB":
			k[h] = clus_det.DB(points[h])  # k: Number of clusters
		elif num_clusters == "CH":
			k[h] = clus_det.CH(points[h])  # k: Number of clusters
		elif num_clusters == "DB":
			k[h] = clus_det.DB(points[h])  # k: Number of clusters
		elif num_clusters == "BIC":
			k[h] = clus_det.BIC(points[h])  # k: Number of clusters
		elif num_clusters == "min":
			k[h] = clus_det.min_clusters(points[h])  # k: Number of clusters
		elif num_clusters == "max":
			k[h] = clus_det.max_clusters(points[h])  # k: Number of clusters
		elif num_clusters == "median":
			k[h] = clus_det.median_clusters(points[h])  # k: Number of clusters
		elif num_clusters == "majority":
			k[h] = clus_det.majority_clusters(points[h])  # k: Number of clusters
		else:
			k[h] = num_clusters[h]  # k: Number of clusters

	for i in range(0, len(optimizer)):
		for j in range(0, len(objective_function)):
			for h in range(len(dataset_list)):
				HS = [0]*num_runs
				CS = [0]*num_runs
				VM = [0]*num_runs
				AMI = [0]*num_runs
				ARI = [0]*num_runs
				Fmeasure = [0]*num_runs
				SC = [0]*num_runs
				accuracy = [0]*num_runs
				DI = [0]*num_runs
				DB = [0]*num_runs
				stdev = [0]*num_runs
				exSSE = [0]*num_runs
				exTWCV = [0]*num_runs
				purity = [0]*num_runs
				entropy = [0]*num_runs
				convergence = [0]*num_runs
				runtime = [0]*num_runs
				# Agg = [0]*num_runs

				for z in range(num_runs):
					print("Dataset: " + dataset_list[h])
					print("Num. Clusters: " + str(k[h]))
					print("Run num.: " + str(z + 1))
					print("Population Size: " + str(population_size))
					print("Iterations: " + str(iterations))

					objective_name = objective_function[j]

					# sol = selector(optimizer[i], objective_name, k[h], f[h], population_size, iterations, points[h], metric, dataset_list[h], list_policy[i], population)

					# ---------------------
					debug = False	# True for testing (serial)
					if debug:
						from selector import selector # Important only here

						np.random.seed(123123123)
						cores
						lb = 0
						ub = 1
						dimension = k[h] * f[h] # Number of dimensions
						population = np.random.uniform(0, 1, (population_size, dimension)) * (ub - lb) + lb
						sol = selector(optimizer[i], objective_name, k[h], f[h], population_size, iterations, points[h], metric, dataset_list[h], list_policy[i], population, cores)
					else:
						params = Params()
						params.algorithm = optimizer[i]
						params.objective_name = objective_name
						params.num_clusters = k[h]
						params.num_features = f[h]
						params.population_size = population_size
						params.iterations = iterations
						params.iteration = z
						params.points = points[h]
						params.metric = metric
						params.dataset_name = dataset_list[h]
						params.policy = list_policy[i] # for get best params use: list_policy[0] else use list_policy[i]
						params.cores = cores
						params.save()

						# print(optimizer[i])
						if "_mpi" == optimizer[i][-4:]:
							print("Parallel MPI version")
							# print(os.system("pwd"))
							# result = os.system("mpirun -np 2 python -m mpi4py pyfile hello_mpi.py")
							# result = os.system("mpirun -np 2 python -m mpi4py pyfile hello_mpi.py")
							# result = subprocess.Popen("mpiexec -n 24 --oversubscribe python selector.py")
							# print(result)
							# result = subprocess.run(["mpirun -np 24", "python", "selector.py"])
							# prog = subprocess.Popen('mpirun -n 2 python hello_mpi.py', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
							# prog = subprocess.Popen('mpirun -np 24 --oversubscribe python selector.py', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
							# result = os.system("mpiexec -n 24 python selector.py")
							os.system("mpirun -np 24 --oversubscribe python3 selector.py")
						elif "_mp" == optimizer[i][-3:]:
							print("Parallel MP version")
							os.system("python3 selector.py")
						else:
							print("Sequential version")
							# print(os.system("ls"))
							# os.system("cd source")
							# os.system("python selector.py")
							# subprocess.call('cd source', shell=True)
							# result = subprocess.run(["python", "selector.py"])
							# subprocess.run('python source/selector.py', shell=True)
							# print(output)
							# subprocess.check_output("python source/selector.py", shell=True)
							# result = subprocess.run(["python", "sources/selector.py"])
							os.system("python3 selector.py")
					# ---------------------
					sol = Solution().get("{}_{}_{}".format(optimizer[i], objective_name, dataset_list[h]))
					
					if labels_exist:
						HS[z] = measures.HS(labels_true[h], sol.labels_pred)
						CS[z] = measures.CS(labels_true[h], sol.labels_pred)
						VM[z] = measures.VM(labels_true[h], sol.labels_pred)
						AMI[z] = measures.AMI(labels_true[h], sol.labels_pred)
						ARI[z] = measures.ARI(labels_true[h], sol.labels_pred)
						Fmeasure[z] = measures.Fmeasure(labels_true[h], sol.labels_pred)
						accuracy[z] = measures.accuracy(labels_true[h], sol.labels_pred)
						purity[z] = measures.purity(labels_true[h], sol.labels_pred)
						entropy[z] = measures.entropy(labels_true[h], sol.labels_pred)
						# Agg[z] = float("%0.2f"%(float("%0.2f"%(HS[z] + CS[z] + VM[z] + AMI[z] + ARI[z])) / 5))

					SC[z] = measures.SC(points[h], sol.labels_pred)
					DI[z] = measures.DI(points[h], sol.labels_pred)
					DB[z] = measures.DB(points[h], sol.labels_pred)
					stdev[z] = measures.stdev(sol.best_individual, sol.labels_pred, k[h], points[h])
					exSSE[z] = measures.SSE(sol.best_individual, sol.labels_pred, k[h], points[h])
					exTWCV[z] = measures.TWCV(sol.best_individual, sol.labels_pred, k[h], points[h])

					runtime[z] = sol.runtime
					convergence[z] = sol.convergence
					optimizerName = sol.optimizer
					objf_name = sol.objf_name

					if(export_details_labels):
						export_to_file_details_labels = results_directory + "experiment_details_labels.csv"
						with open(export_to_file_details_labels, "a", newline="\n") as out_details_labels:
							writer_details = csv.writer(
								out_details_labels, delimiter=",")
							# just one time to write the header of the CSV file
							if (flag_details_Labels == False):
								header_details = np.concatenate(
									[["Dataset", "Optimizer", "Topology", "ObjfName", "k"]])
								writer_details.writerow(header_details)
								flag_details_Labels = True
							a = np.concatenate(
								[[dataset_list[h], optimizerName, sol.policy["topology"], objf_name, k[h]], sol.labels_pred])
							writer_details.writerow(a)
						out_details_labels.close()

					if(export_details):
						export_to_file_details = results_directory + "experiment_details.csv"
						with open(export_to_file_details, "a", newline="\n") as out_details:
							writer_details = csv.writer(out_details, delimiter=",")
							# just one time to write the header of the CSV file
							if (flag_details == False):
								if labels_exist:
									header_details = np.concatenate([["Dataset", "Optimizer", "Topology", "ObjfName", "k", "ExecutionTime", "SSE", "Purity",
																		 "Entropy", "HS", "CS", "VM", "AMI", "ARI", "Fmeasure", "TWCV", "SC", "Accuracy", "DI", "DB", "STDev"], CnvgHeader])
								else:
									header_details = np.concatenate(
										[["Dataset", "Optimizer", "Topology", "ObjfName", "k", "ExecutionTime", "SSE", "TWCV", "SC", "DI", "DB", "STDev"], CnvgHeader])
								writer_details.writerow(header_details)
								flag_details = True
							if labels_exist:
								a = np.concatenate([[dataset_list[h], optimizerName, sol.policy["topology"], objf_name, k[h], float("%0.2f" % (runtime[z])), float("%0.2f" % (exSSE[z])), float("%0.2f" % (purity[z])), float("%0.2f" % (entropy[z])), float("%0.2f" % (HS[z])), float("%0.2f" % (CS[z])),  float("%0.2f" % (VM[z])),  float(
									"%0.2f" % (AMI[z])),  float("%0.2f" % (ARI[z])), float("%0.2f" % (Fmeasure[z])),  float("%0.2f" % (exTWCV[z])),  float("%0.2f" % (SC[z])),  float("%0.2f" % (accuracy[z])),  float("%0.2f" % (DI[z])), float("%0.2f" % (DB[z])), float("%0.2f" % (stdev[z]))], np.around(convergence[z], decimals=2)])
							else:
								a = np.concatenate([[dataset_list[h], optimizerName, sol.policy["topology"], objf_name, k[h], float("%0.2f" % (runtime[z])), float("%0.2f" % (exSSE[z])), float("%0.2f" % (
									exTWCV[z])),  float("%0.2f" % (SC[z])),  float("%0.2f" % (DI[z])), float("%0.2f" % (DB[z])), float("%0.2f" % (stdev[z]))], np.around(convergence[z], decimals=2)])

							writer_details.writerow(a)
						out_details.close()
				
				if(export_avg):
					export_to_file = results_directory + "experiment_avg.csv"

					with open(export_to_file, "a", newline="\n") as out:
						writer = csv.writer(out, delimiter=",")
						if (flag_avg == False):  # just one time to write the header of the CSV file
							if labels_exist:
								header = np.concatenate([["Dataset", "Optimizer", "Topology", "ObjfName", "k", "ExecutionTime", "SSE", "Purity", "Entropy",
															 "HS", "CS", "VM", "AMI", "ARI", "Fmeasure", "TWCV", "SC", "Accuracy", "DI", "DB", "STDev"], CnvgHeader])
							else:
								header = np.concatenate(
									[["Dataset", "Optimizer", "Topology", "ObjfName", "k", "ExecutionTime", "SSE", "TWCV", "SC", "DI", "DB", "STDev"], CnvgHeader])
							writer.writerow(header)
							flag_avg = True  # at least one experiment

						avgSSE = str(float("%0.2f" % (sum(exSSE) / num_runs))) # Important for find best parameters for Parallel_MPI
						avgTWCV = str(float("%0.2f" % (sum(exTWCV) / num_runs)))
						avgPurity = str(float("%0.2f" % (sum(purity) / num_runs)))
						avgEntropy = str(float("%0.2f" % (sum(entropy) / num_runs)))
						avgHomo = str(float("%0.2f" % (sum(HS) / num_runs)))
						avgComp = str(float("%0.2f" % (sum(CS) / num_runs)))
						avgVmeas = str(float("%0.2f" % (sum(VM) / num_runs)))
						avgAMI = str(float("%0.2f" % (sum(AMI) / num_runs)))
						avgARI = str(float("%0.2f" % (sum(ARI) / num_runs)))
						avgFmeasure = str(float("%0.2f" % (sum(Fmeasure) / num_runs)))
						avgSC = str(float("%0.2f" % (sum(SC) / num_runs)))
						avgAccuracy = str(float("%0.2f" % (sum(accuracy) / num_runs)))
						avgDI = str(float("%0.2f" % (sum(DI) / num_runs)))
						avgDB = str(float("%0.2f" % (sum(DB) / num_runs)))
						avgStdev = str(float("%0.2f" % (sum(stdev) / num_runs)))
						# avgAgg = str(float("%0.2f"%(np.sum(Agg) / num_runs)))

						avgExecutionTime = float("%0.2f" % (sum(runtime) / num_runs))
						avgConvergence = np.around(np.mean(convergence, axis=0, dtype=np.float64), decimals=2).tolist()
						if labels_exist:
							a = np.concatenate([[dataset_list[h], optimizerName, sol.policy["topology"], objf_name, k[h], avgExecutionTime, avgSSE, avgPurity, avgEntropy, avgHomo,
													avgComp, avgVmeas, avgAMI, avgARI, avgFmeasure, avgTWCV, avgSC, avgAccuracy, avgDI, avgDB, avgStdev], avgConvergence])
						else:
							a = np.concatenate([[dataset_list[h], optimizerName, sol.policy["topology"], objf_name, k[h], avgExecutionTime,
													avgSSE, avgTWCV, avgSC, avgDI, avgDB, avgStdev], avgConvergence])
						writer.writerow(a)
					out.close()

				if(export_best_params):
					export_to_file = results_directory + "experiment_best_params.csv"

					with open(export_to_file, "a", newline="\n") as out:
						writer = csv.writer(out, delimiter=",")
						if (flag_best_params == False):  # just one time to write the header of the CSV file
							header = np.concatenate(
									[["Dataset", "Optimizer", "Topology", "Emigration", "ChoiceEmi", "ChoiceImm", "NumberEmiImm", "IntervalEmiImm",
									 "ObjfName", "k", "ExecutionTime", "SSE"]])
							writer.writerow(header)
							flag_best_params = True  # at least one experiment

						avgSSE = str(float("%0.2f" % (sum(exSSE) / num_runs))) # Important for find best parameters for Parallel_MPI
						avgExecutionTime = float("%0.2f" % (sum(runtime) / num_runs))

						a = np.concatenate([[dataset_list[h], optimizerName, sol.policy["topology"], sol.policy["emigration"], sol.policy["choice_emi"],
											sol.policy["choice_imm"], sol.policy["number_emi_imm"], sol.policy["interval_emi_imm"], objf_name, k[h],
											avgExecutionTime, avgSSE]])
						writer.writerow(a)
					out.close()

	if export_convergence:
		convergence_plot.run(results_directory, optimizer, objective_function, dataset_list, iterations)

	if export_boxplot:
		if labels_exist:
			ev_measures = ["SSE", "Purity", "Entropy", "HS", "CS", "VM", "AMI",
						   "ARI", "Fmeasure", "TWCV", "SC", "Accuracy", "DI", "DB", "STDev"]
		else:
			ev_measures = ["SSE", "TWCV", "SC", "DI", "DB", "STDev"]
		box_plot.run(results_directory, optimizer, objective_function, dataset_list, ev_measures, iterations)

	if export_runtime:
		runtime_plot.run(results_directory, optimizer, objective_function, dataset_list, iterations)

	print("Execution completed")