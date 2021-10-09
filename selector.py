from utils.params import Params
import numpy as np
import utils.objectives as objectives

import serial_optimizers.CSSA as cssa
import serial_optimizers.CPSO as cpso
import serial_optimizers.CGA as cga
import serial_optimizers.CBAT as cbat
import serial_optimizers.CFFA as cffa
import serial_optimizers.CGWO as cgwo
import serial_optimizers.CWOA as cwoa
import serial_optimizers.CMVO as cmvo
import serial_optimizers.CMFO as cmfo
import serial_optimizers.CCS as ccs
# import serial_optimizers.CSSO as csso

import parallel_mpi_optimizers.PCSSA as p_mpi_cssa
import parallel_mpi_optimizers.PCPSO as p_mpi_cpso
import parallel_mpi_optimizers.PCGA as p_mpi_cga
import parallel_mpi_optimizers.PCBAT as p_mpi_cbat
import parallel_mpi_optimizers.PCFFA as p_mpi_cffa
import parallel_mpi_optimizers.PCGWO as p_mpi_cgwo
import parallel_mpi_optimizers.PCWOA as p_mpi_cwoa
import parallel_mpi_optimizers.PCMVO as p_mpi_cmvo
import parallel_mpi_optimizers.PCMFO as p_mpi_cmfo
import parallel_mpi_optimizers.PCCS as p_mpi_ccs
# import parallel_mpi_optimizers.PCSSO as p_mpi_csso

import parallel_mp_optimizers.PCSSA as p_mp_cssa
import parallel_mp_optimizers.PCPSO as p_mp_cpso
import parallel_mp_optimizers.PCGA as p_mp_cga
import parallel_mp_optimizers.PCBAT as p_mp_cbat
import parallel_mp_optimizers.PCFFA as p_mp_cffa
import parallel_mp_optimizers.PCGWO as p_mp_cgwo
import parallel_mp_optimizers.PCWOA as p_mp_cwoa
import parallel_mp_optimizers.PCMVO as p_mp_cmvo
import parallel_mp_optimizers.PCMFO as p_mp_cmfo
import parallel_mp_optimizers.PCCS as p_mp_ccs
# import parallel_pi_optimizers.PCSSO as p_mp_csso

def selector(algorithm, objective_name, num_clusters, num_features, population_size, iterations, points, metric, dataset_name, policy, population, cores):
	"""
	This is used to call the algorithm which is selected

	Parameters
	----------
	algorithm: int
		The index of the selected algorithm
	objective_name: str
		The name of the selected function
	num_clusters: int
		Number of clusters
	num_features: int
		Number of features
	population_size: int
		Size of population (the number of individuals at each iteration)
	iterations: int
		The number of iterations / Number of generations
	points: np.ndaarray
		The attribute values of all the points / dataset

	Returns
	-----------
	obj
		sol: Solution() object returned by the selected algorithm
	"""
	lb = 0
	ub = 1
	dimension = num_clusters * num_features # Number of dimensions

	if (algorithm == "SSA"):
		sol = cssa.SSA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "PSO"):
		sol = cpso.PSO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "GA"):
		sol = cga.GA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					 iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "BAT"):
		sol = cbat.BAT(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "FFA"):
		sol = cffa.FFA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "GWO"):
		sol = cgwo.GWO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "WOA"):
		sol = cwoa.WOA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "MVO"):
		sol = cmvo.MVO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "MFO"):
		sol = cmfo.MFO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					   iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "CS"):
		sol = ccs.CS(getattr(objectives, objective_name), lb, ub, dimension, population_size,
					 iterations, num_clusters, points, metric, dataset_name, population)
	# elif (algorithm == "SSO"):
	# 	sol = csso.CSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population)

	elif (algorithm == "P_MP_SSA"):
		sol = p_mp_cssa.PSSA(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_PSO"):
		sol = p_mp_cpso.PPSO(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_GA"):
		sol = p_mp_cga.PGA(getattr(objectives, objective_name), lb, ub, dimension,
						   population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_BAT"):
		sol = p_mp_cbat.PBAT(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_FFA"):
		sol = p_mp_cffa.PFFA(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_GWO"):
		sol = p_mp_cgwo.PGWO(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_WOA"):
		sol = p_mp_cwoa.PWOA(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_MVO"):
		sol = p_mp_cmvo.PMVO(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_MFO"):
		sol = p_mp_cmfo.PMFO(getattr(objectives, objective_name), lb, ub, dimension,
							 population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "P_MP_CS"):
		sol = p_mp_ccs.PCS(getattr(objectives, objective_name), lb, ub, dimension,
						   population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	# elif (algorithm == "P_MP_SSO"):
	# 	sol = p_mp_csso.PCSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population)

	elif (algorithm == "P_MPI_SSA"):
		sol = p_mpi_cssa.PSSA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_PSO"):
		sol = p_mpi_cpso.PPSO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_GA"):
		sol = p_mpi_cga.PGA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_BAT"):
		sol = p_mpi_cbat.PBAT(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_FFA"):
		sol = p_mpi_cffa.PFFA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_GWO"):
		sol = p_mpi_cgwo.PGWO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_WOA"):
		sol = p_mpi_cwoa.PWOA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_MVO"):
		sol = p_mpi_cmvo.PMVO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_MFO"):
		sol = p_mpi_cmfo.PMFO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "P_MPI_CS"):
		sol = p_mpi_ccs.PCS(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	# elif (algorithm == "P_MPI_SSO"):
	#	sol = p_mpi_csso.PCSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population)

if __name__ == "__main__":
	seeds = [169735477, 160028434, 160897947, 157407246, 153881302,
				172694171, 171070236, 154302761, 165786948, 159504387]
	p = Params().get()
	np.random.seed(seeds[p.iteration])
	# Generate population
	lb = 0
	ub = 1
	dimension = p.num_clusters * p.num_features # Number of dimensions
	population = np.random.uniform(0, 1, (p.population_size, dimension)) * (ub - lb) + lb
	# print(population[:5])
	
	selector(p.algorithm, p.objective_name, p.num_clusters, p.num_features, p.population_size,
		 p.iterations, p.points, p.metric, p.dataset_name, p.policy, population, p.cores)

# selector(p["algorithm"], p["objective_name"], p["num_clusters"], p["num_features"], p["population_size"], p["iterations"], p["points"], p["metric"], p["dataset_name"], p["policy"])
