from source.params import Params

from source.optimizers.serial import CSSA as cssa
from source.optimizers.serial import CPSO as cpso
from source.optimizers.serial import CGA as cga
from source.optimizers.serial import CBAT as cbat
from source.optimizers.serial import CFFA as cffa
from source.optimizers.serial import CGWO as cgwo
from source.optimizers.serial import CWOA as cwoa
from source.optimizers.serial import CMVO as cmvo
from source.optimizers.serial import CMFO as cmfo
from source.optimizers.serial import CCS as ccs
# from source.optimizers.serial import CSSO as csso

import source.optimizers.parallel_mp.PCSSA as mp_cssa
import source.optimizers.parallel_mp.PCPSO as mp_cpso
import source.optimizers.parallel_mp.PCGA as mp_cga
import source.optimizers.parallel_mp.PCBAT as mp_cbat
import source.optimizers.parallel_mp.PCFFA as mp_cffa
import source.optimizers.parallel_mp.PCGWO as mp_cgwo
import source.optimizers.parallel_mp.PCWOA as mp_cwoa
import source.optimizers.parallel_mp.PCMVO as mp_cmvo
import source.optimizers.parallel_mp.PCMFO as mp_cmfo
import source.optimizers.parallel_mp.PCCS as mp_ccs
# import source.optimizers.parallel_mp.PCSSO as mp_csso

import source.optimizers.parallel_mpi.PCSSA as mpi_cssa
import source.optimizers.parallel_mpi.PCPSO as mpi_cpso
import source.optimizers.parallel_mpi.PCGA as mpi_cga
import source.optimizers.parallel_mpi.PCBAT as mpi_cbat
import source.optimizers.parallel_mpi.PCFFA as mpi_cffa
import source.optimizers.parallel_mpi.PCGWO as mpi_cgwo
import source.optimizers.parallel_mpi.PCWOA as mpi_cwoa
import source.optimizers.parallel_mpi.PCMVO as mpi_cmvo
import source.optimizers.parallel_mpi.PCMFO as mpi_cmfo
import source.optimizers.parallel_mpi.PCCS as mpi_ccs
# import source.optimizers.parallel_mpi.PCSSO as mpi_csso

import numpy as np
import source.objectives as objectives

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
	# 	sol = csso.CSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population) """

	elif (algorithm == "MP_SSA"):
		sol = mp_cssa.PSSA(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_PSO"):
		sol = mp_cpso.PPSO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_GA"):
		sol = mp_cga.PGA(getattr(objectives, objective_name), lb, ub, dimension,
						population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_BAT"):
		sol = mp_cbat.PBAT(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_FFA"):
		sol = mp_cffa.PFFA(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_GWO"):
		sol = mp_cgwo.PGWO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_WOA"):
		sol = mp_cwoa.PWOA(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_MVO"):
		sol = mp_cmvo.PMVO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MP_MFO"):
		sol = mp_cmfo.PMFO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)								
	elif (algorithm == "MP_CS"):
		sol = mp_ccs.PCS(getattr(objectives, objective_name), lb, ub, dimension,
						population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	# elif (algorithm == "MP_SSO"):
	# 	sol = mp_csso.PCSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "MPI_SSA"):
		sol = mpi_cssa.PSSA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_PSO"):
		sol = mpi_cpso.PPSO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_GA"):
		sol = mpi_cga.PGA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_BAT"):
		sol = mpi_cbat.PBAT(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_FFA"):
		sol = mpi_cffa.PFFA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_GWO"):
		sol = mpi_cgwo.PGWO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_WOA"):
		sol = mpi_cwoa.PWOA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_MVO"):
		sol = mpi_cmvo.PMVO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_MFO"):
		sol = mpi_cmfo.PMFO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							  iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MPI_CS"):
		sol = mpi_ccs.PCS(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	# elif (algorithm == "mpi_SSO"):
	#	sol = mpi_csso.PCSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population)

if __name__ == "__main__":
	seeds = [169735477, 160028434, 160897947, 157407246, 153881302,
				172694171, 171070236, 154302761, 165786948, 159504387]
	p = Params().get()
	# np.random.seed(seeds[p.iteration])
	np.random.seed(123123123)
	# Generate population
	lb = 0
	ub = 1
	dimension = p.num_clusters * p.num_features # Number of dimensions
	population = np.random.uniform(0, 1, (p.population_size, dimension)) * (ub - lb) + lb
	# print(population[:5])

	selector(p.algorithm, p.objective_name, p.num_clusters, p.num_features, p.population_size,
				p.iterations, p.points, p.metric, p.dataset_name, p.policy, population, p.cores)

	# selector(p["algorithm"], p["objective_name"], p["num_clusters"], p["num_features"], p["population_size"], p["iterations"], p["points"], p["metric"], p["dataset_name"], p["policy"])
