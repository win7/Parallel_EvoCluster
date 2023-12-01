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

import source.optimizers.parallel_mp.PCSSA as cssa_mp
import source.optimizers.parallel_mp.PCPSO as cpso_mp
import source.optimizers.parallel_mp.PCGA as cga_mp
import source.optimizers.parallel_mp.PCBAT as cbat_mp
import source.optimizers.parallel_mp.PCFFA as cffa_mp
import source.optimizers.parallel_mp.PCGWO as cgwo_mp
import source.optimizers.parallel_mp.PCWOA as cwoa_mp
import source.optimizers.parallel_mp.PCMVO as cmvo_mp
import source.optimizers.parallel_mp.PCMFO as cmfo_mp
import source.optimizers.parallel_mp.PCCS as ccs_mp
# import source.optimizers.parallel_mp.PCSSO as csso_mp

import source.optimizers.parallel_mpi.PCSSA as cssa_mpi
import source.optimizers.parallel_mpi.PCPSO as cpso_mpi
import source.optimizers.parallel_mpi.PCGA as cga_mpi
import source.optimizers.parallel_mpi.PCBAT as cbat_mpi
import source.optimizers.parallel_mpi.PCFFA as cffa_mpi
import source.optimizers.parallel_mpi.PCGWO as cgwo_mpi
import source.optimizers.parallel_mpi.PCWOA as cwoa_mpi
import source.optimizers.parallel_mpi.PCMVO as cmvo_mpi
import source.optimizers.parallel_mpi.PCMFO as cmfo_mpi
import source.optimizers.parallel_mpi.PCCS as ccs_mpi
# import source.optimizers.parallel_mpi.PCSSO as csso_mpi

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

	elif (algorithm == "SSA_mp"):
		sol = cssa_mp.PSSA(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "PSO_mp"):
		sol = cpso_mp.PPSO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "GA_mp"):
		sol = cga_mp.PGA(getattr(objectives, objective_name), lb, ub, dimension,
						population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "BAT_mp"):
		sol = cbat_mp.PBAT(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "FFA_mp"):
		sol = cffa_mp.PFFA(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "GWO_mp"):
		sol = cgwo_mp.PGWO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "WOA_mp"):
		sol = cwoa_mp.PWOA(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MVO_mp"):
		sol = cmvo_mp.PMVO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	elif (algorithm == "MFO_mp"):
		sol = cmfo_mp.PMFO(getattr(objectives, objective_name), lb, ub, dimension,
							population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)								
	elif (algorithm == "CS_mp"):
		sol = ccs_mp.PCS(getattr(objectives, objective_name), lb, ub, dimension,
						population_size, iterations, num_clusters, points, metric, dataset_name, population, cores)
	# elif (algorithm == "SSO_mp"):
	# 	sol = csso_mp.PCSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, population)
	elif (algorithm == "SSA_mpi"):
		sol = cssa_mpi.PSSA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "PSO_mpi"):
		sol = cpso_mpi.PPSO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "GA_mpi"):
		sol = cga_mpi.PGA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "BAT_mpi"):
		sol = cbat_mpi.PBAT(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "FFA_mpi"):
		sol = cffa_mpi.PFFA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "GWO_mpi"):
		sol = cgwo_mpi.PGWO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "WOA_mpi"):
		sol = cwoa_mpi.PWOA(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MVO_mpi"):
		sol = cmvo_mpi.PMVO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "MFO_mpi"):
		sol = cmfo_mpi.PMFO(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	elif (algorithm == "CS_mpi"):
		sol = ccs_mpi.PCS(getattr(objectives, objective_name), lb, ub, dimension, population_size,
							iterations, num_clusters, points, metric, dataset_name, policy, population)
	# elif (algorithm == "SSO_mpi"):
	#	sol = csso_mpi.PCSSO(getattr(objectives, objective_name), lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name, policy, population)

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

	selector(p.algorithm, p.objective_name, p.num_clusters, p.num_features, p.population_size,
				p.iterations, p.points, p.metric, p.dataset_name, p.policy, population, p.cores)

	# selector(p["algorithm"], p["objective_name"], p["num_clusters"], p["num_features"], p["population_size"], p["iterations"], p["points"], p["metric"], p["dataset_name"], p["policy"])
