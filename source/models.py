from mpi4py import MPI
import numpy as np

def package(population, dimension, migration_index, policy, order_portion):
	data = np.zeros((policy["number_emi_imm"], dimension), dtype=float)
	length = len(population)

	if policy["choice_emi"] == "BEST":
		if policy["emigration"] == "REMOVE":
			start_portion = order_portion * policy["number_emi_imm"]
			end_portion = start_portion + policy["number_emi_imm"]
			data = population[start_portion:end_portion]
			migration_index[start_portion:end_portion] = np.arange(start_portion, end_portion, dtype=int)
		else: # CLONE
			start_portion = 0
			end_portion = policy["number_emi_imm"]
			data = population[start_portion:end_portion]
	elif policy["choice_emi"] == "WORST":
		if policy["emigration"] == "REMOVE":
			start_portion = length - (policy["number_emi_imm"] * (order_portion + 1))
			end_portion = start_portion + policy["number_emi_imm"]
			data = population[start_portion:end_portion]
			migration_index[start_portion:end_portion] = np.arange(start_portion, end_portion, dtype=int)
		else: # CLONE
			start_portion = 0
			end_portion = policy["number_emi_imm"]
			data = population[start_portion:end_portion]
	else: # RANDOM
		if policy["emigration"] == "REMOVE":
			start_portion = order_portion * policy["number_emi_imm"]
			end_portion = start_portion + policy["number_emi_imm"]
			for i, j in enumerate(range(start_portion, end_portion)):
				index = np.random.randint(length)
				# data = np.append(data, population[index])
				data[i] = population[index]
				migration_index[j] = index
		else: # CLONE
			for k in range(policy["number_emi_imm"]):
				index = np.random.randint(length)
				# data = np.append(data, population[index])
				data[k] = population[index]
	return data

def unpack(population, migration_index, policy, order_portion, data):
	length = len(population)
	
	if policy["choice_imm"] == "BEST":
		if policy["emigration"] == "REMOVE":
			start_portion = order_portion * policy["number_emi_imm"]
			end_portion = start_portion + policy["number_emi_imm"]
			population[migration_index[start_portion:end_portion]] = data
		else: # REPLACE
			start_portion = length - (policy["number_emi_imm"] * (order_portion + 1))
			end_portion = start_portion + policy["number_emi_imm"]
			population[start_portion:end_portion] = data
	elif policy["choice_imm"] == "WORST":
		if policy["emigration"] == "REMOVE":
			start_portion = order_portion * policy["number_emi_imm"]
			end_portion = start_portion + policy["number_emi_imm"]
			population[migration_index[start_portion:end_portion]] = data
		else: # REPLACE
			start_portion = length - (policy["number_emi_imm"] * (order_portion + 1))
			end_portion = start_portion + policy["number_emi_imm"]
			population[start_portion:end_portion] = data
	else: # RANDOM
		if policy["emigration"] == "REMOVE":
			start_portion = order_portion * policy["number_emi_imm"]
			end_portion = start_portion + policy["number_emi_imm"]
			for k, index in enumerate(migration_index[start_portion: end_portion]):
				population[index] = data[k]
		else: # REPLACE
			for k in range(policy["number_emi_imm"]):
				index = np.random.randint(length)
				population[index] = data[k]

def ring(comm, population, dimension, migration_index, policy, rank, size):
	data_r = np.zeros((policy["number_emi_imm"], dimension), dtype=float)

	data_s = package(population, dimension, migration_index, policy, 0)
	comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)

	comm.Recv([data_r, MPI.FLOAT], source=(rank + size - 1) % size)
	unpack(population, migration_index, policy, 0, data_r)
	
	""" data = package(population, dimension, migration_index, policy, 0)
	comm.send(data, dest=(rank + 1) % size)

	-data = comm.recv(source=(rank + size - 1) % size)
	unpack(population, migration_index, policy, 0, data) """

def tree(comm, population, dimension, migration_index, policy, rank, size):
	data_r = np.zeros((policy["number_emi_imm"], dimension), dtype=float)
	if rank == 0:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(2 * rank) + 1)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(2 * rank) + 2)

		comm.Recv([data_r, MPI.FLOAT], source=(2 * rank) + 1)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(2 * rank) + 2)
		unpack(population, migration_index, policy, 1, data_r)
	elif rank >= 1 and rank <= 6:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1) / 2)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(2 * rank) + 1)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(2 * rank) + 2)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1) / 2)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(2 * rank) + 1)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(2 * rank) + 2)
		unpack(population, migration_index, policy, 2, data_r)
	elif rank >= 7 and rank <= 15:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1) / 2)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=rank + 8)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1) / 2)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=rank + 8)
		unpack(population, migration_index, policy, 1, data_r)
	else:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=rank - 8)

		comm.Recv([data_r, MPI.FLOAT], source=rank - 8)
		unpack(population, migration_index, policy, 0, data_r)

def net_a(comm, population, dimension, migration_index, policy, rank, size):
	data_r = np.zeros((policy["number_emi_imm"], dimension), dtype=float)
	if rank == 1 or rank == 2:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % size)
		unpack(population, migration_index, policy, 2, data_r)
	elif rank == 21 or rank == 22:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank -1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank -1 + 24) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 2, data_r)
	elif rank % 4 == 0:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % size)
		unpack(population, migration_index, policy, 2, data_r)
	elif (rank + 1) % 4 == 0:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank  + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank  + 4) % size)
		unpack(population, migration_index, policy, 2, data_r)
	else:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)
		data_s = package(population, dimension, migration_index, policy, 3)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 2, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % size)
		unpack(population, migration_index, policy, 3, data_r)

def net_b(comm, population, dimension, migration_index, policy, rank, size):
	data_r = np.zeros((policy["number_emi_imm"], dimension), dtype=float)
	if rank == 1 or rank == 2:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % size)
		unpack(population, migration_index, policy, 2, data_r)
	elif rank == 21 or rank == 22:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank -1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank -1 + 24) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 2, data_r)
	elif rank % 6 == 0 or rank % 4 == 0:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % size)
		unpack(population, migration_index, policy, 2, data_r)
	elif rank == 9 or rank == 10 or rank == 13 or rank == 14:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % size)
		data_s = package(population, dimension, migration_index, policy, 3)
		comm.Send([data_s, MPI.FLOAT], dest=(rank  + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % size)
		unpack(population, migration_index, policy, 2, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank  + 4) % size)
		unpack(population, migration_index, policy, 3, data_r)
	else:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % size)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % size)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % size)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % size)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % size)
		unpack(population, migration_index, policy, 2, data_r)

def torus(comm, population, dimension, migration_index, policy, rank, size):
	data_r = np.zeros((policy["number_emi_imm"], dimension), dtype=float)
	if rank % 4 == 0:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % 24)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % 24)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 3) % 24)
		data_s = package(population, dimension, migration_index, policy, 3)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % 24)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % 24)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % 24)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 3) % 24)
		unpack(population, migration_index, policy, 2, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % 24)
		unpack(population, migration_index, policy, 3, data_r)
	elif rank % 4 == 3:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % 24)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 3 + 24) % 24)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % 24)
		data_s = package(population, dimension, migration_index, policy, 3)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % 24)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % 24)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 3 + 24) % 24)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % 24)
		unpack(population, migration_index, policy, 2, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % 24)
		unpack(population, migration_index, policy, 3, data_r)
	else:
		data_s = package(population, dimension, migration_index, policy, 0)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 4 + 24) % 24)
		data_s = package(population, dimension, migration_index, policy, 1)
		comm.Send([data_s, MPI.FLOAT], dest=(rank - 1 + 24) % 24)
		data_s = package(population, dimension, migration_index, policy, 2)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 1) % 24)
		data_s = package(population, dimension, migration_index, policy, 3)
		comm.Send([data_s, MPI.FLOAT], dest=(rank + 4) % 24)

		comm.Recv([data_r, MPI.FLOAT], source=(rank - 4 + 24) % 24)
		unpack(population, migration_index, policy, 0, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank - 1 + 24) % 24)
		unpack(population, migration_index, policy, 1, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 1) % 24)
		unpack(population, migration_index, policy, 2, data_r)
		comm.Recv([data_r, MPI.FLOAT], source=(rank + 4) % 24)
		unpack(population, migration_index, policy, 3, data_r)

def graph(comm, population, dimension, migration_index, policy, rank, size):
	data_r = np.zeros((policy["number_emi_imm"], dimension), dtype=float)
	# Ayuda a ver que porcion del arreglo de fitness (ordenado) se empaquetara para enviar, segun en number_emi_imm
	order_portion = 0

	for k in range(size):
		if k != rank:
			data_s = package(population, dimension, migration_index, policy, order_portion)
			comm.Send([data_s, MPI.FLOAT], dest=k)
			order_portion += 1

	order_portion = 0
	for k in range(size):
		if k != rank:
			comm.Recv([data_r, MPI.FLOAT], source=k)
			unpack(population, migration_index, policy, 0, data_r)
			order_portion += 1

def run_migration(comm, population, dimension, migration_index, policy, rank, size):
	if policy["topology"] == "RING":
		ring(comm, population,dimension, migration_index, policy, rank, size)
	elif policy["topology"] == "TREE":
		tree(comm, population,dimension, migration_index, policy, rank, size)
	elif policy["topology"] == "NETA":
		net_a(comm, population,dimension, migration_index, policy, rank, size)
	elif policy["topology"] == "NETB":
		net_b(comm, population,dimension, migration_index, policy, rank, size)
	elif policy["topology"] == "TORUS":
		torus(comm, population,dimension, migration_index, policy, rank, size)
	elif policy["topology"] == "GRAPH":
		graph(comm, population,dimension, migration_index, policy, rank, size)

"""
policy = {
	"topology": "RING",
	"emigration": "CLONE",
	"choice_emi": "BEST",
	"choice_imm": "WORST",
	"number_emi_imm": 5,
	"interval_emi_imm": 2
}
"""

""" Política de inmigración: Indica que los individuos inmigrantes pueden
reemplazaralos individuos de la isla local.

- Si se elige la política de emigración remover,
entonceslos individuos inmigrantes sustituyen los espacios
libres existentes en la isla local.Por el contrario.
- Si se elige la política de migración clonar,
entonces los individuos inmigrantes reemplazan a los 
individuos de la isla local según el tipo de individuo inmigrante """