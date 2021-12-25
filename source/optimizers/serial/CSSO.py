import math
import numpy as np
import random
import time
import sys
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import distance
from utils.solution import Solution

class ClusterCenter(object):
	"""docstring for ClusterCenter"""

	point = [] # N-dimensional point
	def __init__(self, length):
		super(ClusterCenter, self).__init__()
		self.point = np.zeros(length) # N = length
	
	def get_point(self):
		return self.point

	def calculate_distance_cc(self, p_point):
		""" start = time.time()
		point_dimension = len(self.point)
		total_sum = 0
		for k in range(point_dimension):
			total_sum += (self.point[k] - p_point[k]) * (self.point[k] - p_point[k])
		
		end = time.time()
		print(f"Runtime of the program is {end - start}")
		return np.sqrt(total_sum) """

		start = time.time()
		total_sum = distance.euclidean(self.point, p_point)
		end = time.time()
		print(f"Runtime of the program is {end - start}")
		return total_sum

	def calculate_distance_(self, dataset, i, j):
		point_dimension = len(dataset[0]) # point length
		total_sum = 0
		for k in range(point_dimension):
			total_sum += (dataset[i][k] - dataset[j][k]) * (dataset[i][k] - dataset[j][k])
		return np.sqrt(total_sum)

class Spider(object):
	"""docstring for Spider"""
	type = 0 # female = 0, male = 1 
	centers = []
	dataset_clusters = []
	fitness = 0
	weight = 0
	# Random generator # random generator of numbers

	def __init__(self, type, length, dataset):
		super(Spider, self).__init__()
		self.type = type
		self.centers = np.empty(length, dtype=ClusterCenter)
		self.dataset_clusters = np.zeros(len(dataset)).astype(int)
		self.fitness = sys.maxsize
		self.weight = 0

	def get_type(self):
		return self.type

	def get_cluster_centers(self):
		return self.centers

	def get_dataset_clusters(self):
		return self.dataset_clusters

	def get_fitness(self):
		return self.fitness

	def get_weight(self):
		return self.weight

	def set_fitness(self, fitness):
		self.fitness = fitness

	def set_weight(self, weight):
		self.weight = weight

	def evaluate_fitness(self, dataset):
		# build clusters according to its centers
		# dataset_clusters = buildClusters(dataset);
		self.build_clusters(dataset)
		# compute new cluster centers
		self.compute_new_cluster_centers(dataset) # dataset_clusters
		# calculate clustering metric
		self.fitness = self.calculate_metric(dataset) # dataset_clusters

	def calculate_metric(self, dataset): # dataset_clusters
		distances = 0
		for k in range(len(dataset)):
			index = self.dataset_clusters[k]
			distances += self.centers[index].calculate_distance_cc(dataset[k])
		return distances
	
	def build_clusters(self, dataset):
		# distances = []
		num_clusters = len(self.centers)
		# dataset_clusters = new int[dataset.length];
		# build the dataset clusters
		
		for i in range(len(dataset)):
			distances = np.zeros(num_clusters)
			for j in range(num_clusters):
				# euclidean distance from a dataset point "i" to each cluster center
				distances[j] = self.centers[j].calculate_distance_cc(dataset[i])
			
			# determine minimum distance
			min_ = distances[0]
			cluster_index = 0
			for j in range(1, num_clusters):
				if (distances[j] < min_):
					min_ = distances[j]
					cluster_index = j
			
			# assign cluster index of the minimum distance to the dataset point "i" 
			self.dataset_clusters[i] = cluster_index
		# return dataset_clusters

	def compute_new_cluster_centers(self, dataset): # int[] dataset_clusters)
		# Random generator = new Random();
		# Calculate total sum for each cluster
		num_clusters = len(self.centers)
		num_points_for_cluster = np.zeros(num_clusters).astype(int)
		point_dimension = len(dataset[0])
		sum_ = np.zeros((num_clusters, point_dimension))
		
		for i in range(len(dataset)):
			index = self.dataset_clusters[i]
			for j in range(point_dimension):
				sum_[index][j] += dataset[i][j]
			num_points_for_cluster[index] += 1
		
		# replace mean dataset as new cluster centers
		new_value = 0
		for i in range (num_clusters):
			for j in range(point_dimension):
				if num_points_for_cluster[i] == 0:
					# when all dataset of cluster have the same value 
					# choose new value from random point at dataset
					pos = np.random.randint(len(dataset))
					new_value = dataset[pos][j]
				else:
					# calculate mean point
					new_value = sum_[i][j] / num_points_for_cluster[i]
				self.centers[i].get_point()[j] = new_value

	# Calculate the dist. between "current" spider and spider "spider"
	def calculate_distance_s(self, spider):
		total_distance = 0
		for k in range(len(self.centers)):
			total_distance += self.centers[k].calculate_distance_cc(spider.centers[k].get_point())		
		return total_distance

	# Difference of the cluster centers of two spiders
	def diff_spiders(self, spider1, spider2, cluster_centers):
		for i in range(len(spider1.centers)):
			cluster_centers[i] = ClusterCenter(len(spider1.centers[0].get_point()))
			for j in range(len(spider1.centers[0].get_point())):
				cluster_centers[i].get_point()[j] = spider1.centers[i].get_point()[j] - spider2.centers[i].get_point()[j]
				# print("w = %f (%f - %f), ", cluster_centers[i].getPoint()[j], spi1.centers[i].getPoint()[j],spi2.centers[i].getPoint()[j]);

	# Sum the cluster centers of a spider with other cluster centers
	def sum_spider(self, cluster_centers, signal):
		for i in range(len(self.centers)):
			for j in range(len(self.centers[0].get_point())):
				self.centers[i].get_point()[j] += signal * cluster_centers[i].get_point()[j]
				# print("\n p %f\n", c[i].getPoint()[j])

	# Multiply cluster centers by a constant
	def mul_cluster_centers_by_constant(self, cluster_centers, const):
		for i in range(len(cluster_centers)):
			for j in range(len(cluster_centers[0].get_point())):
				cluster_centers[i].get_point()[j] *= const

	def show_spider(self):
		print("[")
		for i in range(len(self.centers)):
			print("(")
			for j in range(len(self.centers[0].get_point())):
				print("{}; ".format(centers[i].getPoint()[j]))
			print("),")
		print("] Fitness: {}".format(get_fitness()))

class Population(object):
	"""docstring for Population"""
	number_females = 0
	number_males = 0
	num_clusters = 0
	point_dimension = 0
	radius_mating = 0
	spiders = []
	offspring = []
	median_weight = 0
	index_best = 0 # index of the best result
	index_worst = 0
	# private Random 				generator;//random generator of numbers

	def __init__(self):
		super(Population, self).__init__()
		
		self.number_females = 0
		self.number_males = 0
		self.num_clusters = 0
		self.point_dimension = 0
		self.radius_mating = 0
		self.spiders = []
		self.offspring = []
		self.median_weight = 0
		self.index_best = 0 # Let the first be the best
		self.index_worst = 0 # Let the first be the worst
		# generator = pGenerator;

	def maximum_distance(self, dataset):
		max_distance = 0
		distance = 0
		for i in range(len(dataset) - 1):
			for j in range(i + 1, len(dataset)):
				distance = ClusterCenter(0).calculate_distance_(dataset, i, j)
				if distance > max_distance:
					max_distance = distance
				# printf("l=%d -->dist = %f\n", dataset.length, distance);
		return max_distance

	# Generate initial population: males and females, and calculate radius
	def generate_initial_population(self, lb, ub, population_size, number_clusters, dataset):
		# Random generator = new Random();
		self.number_females = int(np.floor((0.9 - np.random.rand() * 0.25) * population_size))
		self.number_males = population_size - self.number_females
		self.num_clusters = number_clusters;
		self.point_dimension = len(dataset[0])
		self.spiders = np.empty(population_size, dtype=Spider)
		pos = -1 # invalid position
		
		# Generate FEMALES by choosing dataset randomly from dataset
		for i in range(self.number_females):
			self.spiders[i] = Spider(0, self.num_clusters, dataset) # female = 0
			for j in range(self.num_clusters):
				# random integer in range [0 , dataset.length-1]
				# pos = np.random.randint(len(dataset))
				pos = np.random.rand()

				self.spiders[i].get_cluster_centers()[j] = ClusterCenter(self.point_dimension)
				for k in range(self.point_dimension):
					self.spiders[i].get_cluster_centers()[j].get_point()[k] = pos * (ub - lb) + lb # dataset[pos][k]

		# Generate MALES by choosing dataset randomly from dataset
		for i in range(self.number_females, population_size):
			self.spiders[i] = Spider(1, self.num_clusters, dataset) # male = 1
			for j in range(self.num_clusters):
				# random integer in range [0 , dataset.length-1]
				# pos = np.random.randint(len(dataset))
				pos = np.random.rand()
				
				self.spiders[i].get_cluster_centers()[j] = ClusterCenter(self.point_dimension)
				for k in range(self.point_dimension):
					self.spiders[i].get_cluster_centers()[j].get_point()[k] = pos * (ub - lb) + lb # dataset[pos][k]
		
		# Calculate the radius of mating
		d = self.maximum_distance(dataset)
		self.radius_mating = d / 2
		# print("nf = %d\n nm = %d\n radius = %f \n", self.number_females, number_males, radiusMating)
	
	def evaluate_fitness_population(self, dataset):
		for k in range(len(self.spiders)):
			# keeping the best spider the same
			if k == self.index_best:
				continue			
			self.spiders[k].evaluate_fitness(dataset)

	def calculate_best_worst_population(self):
		min_fitness = self.spiders[0].get_fitness()
		max_fitness = self.spiders[0].get_fitness()
		
		for k in range(len(self.spiders)):
			if self.spiders[k].get_fitness() < min_fitness:
				min_fitness = self.spiders[k].get_fitness()
				self.index_best = k;
			
			if self.spiders[k].get_fitness() > max_fitness:
				max_fitness = self.spiders[k].get_fitness()
				self.index_worst = k

	# Calculate the weight of every spider in the population
	def calculate_weight_population(self, dataset):
		# Evaluate the fitness of the population 
		self.evaluate_fitness_population(dataset)
		# Calculate the best and the worst individual of population.
		self.calculate_best_worst_population()
		# Calculate weight of population
		for k in range(len(self.spiders)):
			d = self.spiders[self.index_worst].get_fitness() - self.spiders[self.index_best].get_fitness() # Division by zero
			w = (self.spiders[self.index_worst].get_fitness() - self.spiders[k].get_fitness()) / d
			self.spiders[k].set_weight(w)
			# print("--> fitness: %f, weight: %f\n", spiders[i].getFitness(), spiders[i].getWeight())

	# Return the index of the nearest individual with higher weight compared to individual with index i
	def nearest_with_higher_weight_to(self, i):
		index = 0
		nearest_distance = self.spiders[index].calculate_distance_s(self.spiders[i])
		
		for k in range(len(self.spiders)):
			if self.spiders[k].get_weight() > self.spiders[i].get_weight():
				new_distance = self.spiders[k].calculate_distance_s(self.spiders[i])
				if new_distance < nearest_distance:
					nearest_distance = new_distance
					index = k
		return index

	# Apply the female cooperative operator
	def female_cooperative_operator(self):
		# Random generator = new Random();
		threshold_PF = 0.5 # Test with other values
		w = 0
		d = 0
		d2 = 0
		vibci = 0
		vibbi = 0
		rm = 0
		alpha = 0
		beta = 0
		delta = 0
		rand = 0
		
		for k in range(self.number_females):
			# keeping the best spider the same
			if k == self.index_best:
				continue
			
			# Calculate Vibci: Vib. of ind. nearest with higher weight compared to i
			index = self.nearest_with_higher_weight_to(k)
			w = self.spiders[index].get_weight()
			d = self.spiders[k].calculate_distance_s(self.spiders[index])
			d2 = np.power(d, 2)
			vibci = w / np.power(math.e, d2)
			
			# Calculate Vibbi: vibration of individual with best fitness
			w = self.spiders[self.index_best].get_weight()
			d = self.spiders[k].calculate_distance_s(self.spiders[self.index_best])
			d2 = np.power(d, 2)
			vibbi = w / np.power(math.e, d2)
			
			rm = np.random.rand()
			alpha = np.random.rand()
			beta = np.random.rand()
			delta = np.random.rand()
			rand = np.random.rand()
						
			# Define if the movement is: attraction or repulsion
			if rm < threshold_PF:
				signal = 1 # --> 1 = sum
			else:
				signal = -1 # --> -1 = subtraction
			
			# Sum expression with alpha
			clus_centers = np.empty(self.num_clusters, dtype=ClusterCenter)
			Spider(0, 0, []).diff_spiders(self.spiders[index], self.spiders[k], clus_centers)
			cons = alpha * vibci
			Spider(0, 0, []).mul_cluster_centers_by_constant(clus_centers, cons)
			self.spiders[k].sum_spider(clus_centers, signal)
		
			# Sum expression with beta
			clus_centers = np.empty(self.num_clusters, dtype=ClusterCenter)
			Spider(0, 0, []).diff_spiders(self.spiders[self.index_best], self.spiders[k], clus_centers)
			cons = beta * vibbi
			Spider(0, 0, []).mul_cluster_centers_by_constant(clus_centers, cons)
			self.spiders[k].sum_spider(clus_centers, signal)

			# Sum expression with delta
			clus_centers = np.empty(self.num_clusters, dtype=ClusterCenter)
			for i in range(self.num_clusters):
				clus_centers[i] = ClusterCenter(self.point_dimension)
				for j in range(self.point_dimension):
					clus_centers[i].get_point()[j] = delta * (rand - 0.5)
			self.spiders[k].sum_spider(clus_centers, signal)

	# calculate median weight of males using variant of quicksort
	def calculate_median_weight_of_males(self):
		# Copy the weight of males into males array
		males = np.zeros(self.number_males)
		k = 0
		for i in range(self.number_females, len(self.spiders)):
			males[k] = self.spiders[i].get_weight()
			k += 1
		
		# Calculate the median position
		median_pos = int((self.number_males - 1) / 2)
		
		# Calculate the median value
		begin = 0
		end = self.number_males - 1
		
		while (True):
			p = begin
			r = end
			pivot = males[r]
			i = p - 1
			for j in range(p, r):
				if males[j] < pivot:
					i += 1
					# swap males[i] and males[j]
					tmp = males[i]
					males[i] = males[j]
					males[j] = tmp
			# swap males[i+1] and males[r]
			tmp = males[i + 1]
			males[i + 1] = males[r]
			males[r] = tmp
			
			if i + 1 == median_pos:
				break
			else:
				if median_pos > i + 1:
					# search right
					begin = i + 2
				else:
					# search left
					end = i
		
		# printf("\n --> median = %f, pos = %d \n", males[medianPos], medianPos);
		self.median_weight = males[median_pos]

	def nearest_female_to(self, i):
		index = 0
		nearest_distance = self.spiders[index].calculate_distance_s(self.spiders[i])
		
		for k in range(self.number_females): # females
			new_distance = self.spiders[k].calculate_distance_s(self.spiders[i])
			if new_distance < nearest_distance:
				nearest_distance = new_distance
				index = k
		return index

	def calculate_male_spider_weighted_mean(self, spider):
		# calculate total weight of males
		total_weight = 0
		for k in range(self.number_females, len(self.spiders)):
			total_weight += self.spiders[k].get_weight()
		
		# calculate spiders multiplied by their weights
		spiders_weights = np.empty(self.num_clusters, dtype=ClusterCenter)
		for i in range(self.num_clusters):
			spiders_weights[i] = ClusterCenter(self.point_dimension)
			for j in range(self.point_dimension):
				spiders_weights[i].get_point()[j] = 0
		
		for i in range(self.number_females, len(self.spiders)):
			for j in range(self.num_clusters):
				for k in range(self.point_dimension):
					spiders_weights[j].get_point()[k] += self.spiders[i].get_cluster_centers()[j].get_point()[k] * self.spiders[i].get_weight()
		
		# calculate the weighted mean
		for i in range(self.num_clusters):
			spider.get_cluster_centers()[i] = ClusterCenter(self.point_dimension)
			for j in range(self.point_dimension):
				spider.get_cluster_centers()[i].get_point()[j] = spiders_weights[i].get_point()[j] / total_weight

	# Create Mating Roulette for a male spider
	def create_mating_roulette(self, mating_roulette, mating_group):
		# Sum fitness of mating spiders
		total =  0
		for k in range(len(mating_group)):
			total += self.spiders[mating_group[k]].get_fitness()
		
		# Calculate values of the roulette
		mating_roulette[0] = self.spiders[mating_group[0]].get_fitness() / total
		for k in range(1, len(mating_group)):
			mating_roulette[k] = mating_roulette[k - 1] + self.spiders[mating_group[k]].get_fitness() / total
		
		# debug
		""" for(int i=0; i<mating_group.size(); i++){
			System.out.printf("%f \n", mating_roulette[i]);
		}
		System.out.printf("\n") """

	# Apply the mating operator
	def mating_operator(self, dataset):
		self.offspring = [] # Create a new offspring
		# Begin mating
		for i in range(self.number_females, len(self.spiders)): # males
			# printf("male[%d] weight: %f, medianw:%f\n",i,spiders[i].getWeight(),medianWeight);//-debug
			
			if self.spiders[i].get_weight() > self.median_weight: # male is dominant 
				# Calculate females in the radius of male "i"
				mating_group = [] # indexes  
				mating_group.append(i) # Add male index as first element
				for j in range(self.number_females): # females
					distance = self.spiders[i].calculate_distance_s(self.spiders[j])
					# print("dist: %f \n", distance) # -debug
					if distance < self.radius_mating:
						mating_group.append(j) # Add female index

				if len(mating_group) > 1: # Do mating
					# print("spide male %d, have %d females \n",i,matingGroup.size()-1);//-debug

					# Create mating roulette
					mating_roulette = np.zeros(len(mating_group)) # females +  1 male
					self.create_mating_roulette(mating_roulette, mating_group)
					# Create the new spider using mating roulette
					spider = Spider(0, self.num_clusters, dataset)
					# Random generator = new Random();
					for j in range(self.num_clusters):
						rand = np.random.rand() # 0.0 <= rand < 1.0
						# Go through the mating roulette
						for k in range(len(mating_roulette)):
							if rand < mating_roulette[k]:
								# Copy cluster "j"
								spider.get_cluster_centers()[j] = ClusterCenter(self.point_dimension)
								for h in range(self.point_dimension):
									spider.get_cluster_centers()[j].get_point()[h] = self.spiders[mating_group[k]].get_cluster_centers()[j].get_point()[h]
								break
					# Calculate fitness of new spider and put into offspring
					spider.evaluate_fitness(dataset)
					self.offspring.append(spider)
		# end-mating

	# Using a factor
	def create_replacement_roulette(self, replacement_roulette):
		# Sum fitness of all spiders
		factor = 1
		total = 0
		for k in range(len(self.spiders)):
			if self.spiders[k].get_weight() > self.median_weight:
				total += self.spiders[k].get_fitness()
			else:
				total += self.spiders[k].get_fitness() * factor
		# print("total = {}".format(total))

		# Calculate values of the roulette
		replacement_roulette[0] = self.spiders[0].get_fitness() / total # Division by zero
		for k in range(1, len(self.spiders)):
			if self.spiders[k].get_weight() > self.median_weight:
				replacement_roulette[k] = replacement_roulette[k - 1] + self.spiders[k].get_fitness() / total
			else:
				replacement_roulette[k] = replacement_roulette[k - 1] + (self.spiders[k].get_fitness() * factor) / total
			# print("Prob[%d] = %f \n",i,spiders[i].getFitness()/total)
			# print("Roulette[%d] = %f \n",i,replacement_roulette[i])

	# Apply the male cooperative operator
	def male_cooperative_operator(self, dataset):
		# Calculate the median weight of male population
		self.calculate_median_weight_of_males()
		# Calculate the male spider with weighted mean
		spider_weighted_mean = Spider(1, self.num_clusters, dataset)
		self.calculate_male_spider_weighted_mean(spider_weighted_mean)
		
		# Random generator = new Random()
		w = 0
		d = 0
		d2 = 0
		vibfi = 0
		alpha = 0
		delta = 0
		rand = 0
		cons = 0
		
		for i in range(self.number_females, len(self.spiders)): # males
			# keeping the best spider the same
			if i == self.index_best:
				continue
						
			# Calculate vibfi: vibration of nearest female
			index = self.nearest_female_to(i)
			w = self.spiders[index].get_weight()
			d = self.spiders[i].calculate_distance_s(self.spiders[index])
			d2 = np.power(d, 2)
			vibfi = w / np.power(math.e, d2)
			
			# Define if movement is attraction to females or to the mean
			alpha = np.random.rand()
			delta = np.random.rand()
			rand = np.random.rand()

			if self.spiders[i].get_weight() > self.median_weight: # male is dominant(D)
				# Sum expression with alpha
				clusCenters = np.empty(self.num_clusters, dtype=ClusterCenter)
				Spider(0, 0, []).diff_spiders(self.spiders[index], self.spiders[i], clusCenters)
				cons = alpha * vibfi
				Spider(0, 0, []).mul_cluster_centers_by_constant(clusCenters, cons)
				self.spiders[i].sum_spider(clusCenters, 1) # sum = 1
				
				# Sum expression with delta
				clusCenters = np.empty(self.num_clusters, dtype=ClusterCenter)
				for j in range(self.num_clusters):
					clusCenters[j] = ClusterCenter(self.point_dimension)
					for k in range(self.point_dimension):
						clusCenters[j].get_point()[k] = delta * (rand - 0.5)

				self.spiders[i].sum_spider(clusCenters, 1) # sum = 1
			else: # Male is not dominant(ND)
				# Sum expression with alpha
				clusCenters = np.empty(self.num_clusters, dtype=ClusterCenter)
				Spider(0, 0, []).diff_spiders(spider_weighted_mean, self.spiders[i], clusCenters)
				Spider(0, 0, []).mul_cluster_centers_by_constant(clusCenters, alpha)
				self.spiders[i].sum_spider(clusCenters, 1) # sum = 1

	# Replace offspring into spiders
	def replacement(self):
		# Create replacement roulette for all spiders, giving more prob. to worst spi.
		replacement_roulette = np.zeros(len(self.spiders))
		self.create_replacement_roulette(replacement_roulette)
		# Replace worst spider by offspring by comparing its fitness
		# Random generator = new Random();
		for i in range(len(self.offspring)):
			rand = np.random.rand() # 0.0 <= rand < 1.0
			# Go through the replacement roulette
			for j in range(len(replacement_roulette)):
				if rand < replacement_roulette[j]:
					# Replace spider "j" if it is worst than offspring "i"
					if self.offspring[i].get_fitness() < self.spiders[j].get_fitness():
						for k in range(self.num_clusters):
							for h in range(self.point_dimension):
								self.spiders[j].get_cluster_centers()[k].get_point()[h] = self.offspring[i].get_cluster_centers()[k].get_point()[h]
								
						self.spiders[j].set_fitness(self.offspring[i].get_fitness())
						self.spiders[j].set_weight(0)
						for k in range(len(self.offspring[i].get_dataset_clusters())):
							self.spiders[j].get_dataset_clusters()[k] = self.offspring[i].get_dataset_clusters()[k]					
					break

	# Create Replacement Roulette for all spiders
	""" private void createReplacementRoulette_(double[] replacementRoulette){
		//Sum fitness of all spiders
		double total=0;
		for(int i=0; i<spiders.length;i++){
			total += spiders[i].getFitness();
		}
		//System.out.printf("total = %f\n", total);
		
		//calculate values of the roulette
		replacementRoulette[0] = spiders[0].getFitness() / total;
		for(int i=1; i<spiders.length; i++){
			replacementRoulette[i] = replacementRoulette[i-1] + 
					spiders[i].getFitness() / total;
			//System.out.printf("Prob[%d] = %f \n",i,spiders[i].getFitness()/total);
			//System.out.printf("Roulette[%d] = %f \n",i,replacementRoulette[i]);
		}
		
	} """

	# Show best fitness
	def show_best_fitness(self, number_generation):
		print("Generation: {}, Best Fitness: {}, Worst Fitness: {}".format(number_generation, 
			self.spiders[self.index_best].get_fitness(), self.spiders[self.index_worst].get_fitness()))

	# Show Population
	def show_population(self):
		print("----> POPULATION")
		for i in range(len(self.spiders)):
			print("{}: [".format(i))
			for j in range(self.num_clusters):
				print("(")
				for k in range(self.point_dimension):
					print("{}; ".format(self.spiders[i].get_cluster_centers()[j].get_point()[k]))
				print("),")
			print("] Fitness: {}".format(self.spiders[i].get_fitness()))
		print("*Pop.")

	# Show some parameters
	def show_parameters(self, number_generations):
		print("----------------------------")
		print("----S. Spider Algorithm-----")
		print("----------------------------")
		print("Number of Generations: \t{}".format(number_generations))
		print("Population Size: \t{}".format(len(self.spiders)))
		print("Number of females: \t{}".format(self.number_females))
		print("Number of males: \t{}".format(self.number_males))

	# Show best spider
	def show_best_spider(self):
		print("----------------------------")
		print("Best Spider:")
		print("[")
		for i in range(self.num_clusters):
			print("(")
			for j in range(self.point_dimension):
				print("{}; ".format(self.spiders[self.index_best].get_cluster_centers()[i].get_point()[j]))
			print("),")
		print("] Fitness: {}".format(self.spiders[self.index_best].get_fitness()))

	# Show cluster generated by the best spider
	def show_cluster_best_spider(self, dataset):
		print("----------------------------")
		print("Cluster Generated by Best Spider:")
		for i in range(self.num_clusters):
			print("Cluster {}: ".format(i))
			for j in range(len(self.spiders[self.index_best].get_dataset_clusters())):
				if i == self.spiders[self.index_best].get_dataset_clusters()[j]:
					print("{}, ".format(j))
			print("-->Center {}: (".format(i))
			for j in range(self.point_dimension):
				print("{}; ".format(self.spiders[self.index_best].get_cluster_centers()[i].get_point()[j]))
			print(")")

	def show_metric_best_spider(self):
		print(self.spiders[self.index_best].get_fitness())

def CSSO(objective_function, lb, ub, dimension, population_size, iterations, num_clusters, points, metric, dataset_name):
	# Parameters:
	# points: points
	# SSO Algorithm
	# print(points)

	sol = Solution()
	convergence_curve = []
	print("SSO is optimizing \"" + objective_function.__name__ + "\"")

	timer_start = time.time()
	sol.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

	population = Population()
	population.generate_initial_population(lb, ub, population_size, num_clusters, points)
	population.calculate_weight_population(points)
	# population.show_population()

	for k in range(1, iterations):
		population.female_cooperative_operator()
		population.male_cooperative_operator(points)
		population.mating_operator(points)
		population.replacement()
		population.calculate_weight_population(points)

		# population.show_population()
		population.show_best_fitness(k)

		index_best = population.index_best
		fitness = population.spiders[index_best].get_fitness()
		
		convergence_curve.append(fitness)
		print(["At iteration " + str(k) + " the best fitness is " + str(fitness)])

	# Show results
	""" print("points number of points: {}".format(len(points)))
	population.show_parameters(iterations);
	population.show_best_spider()
	population.show_cluster_best_spider(points) """
	
	# show just best metric
	# population.show_metric_best_spider()
	best_individual = []
	for i in range(population.num_clusters):
		for j in range(population.point_dimension):
				best_individual.append(population.spiders[index_best].get_cluster_centers()[i].get_point()[j])

	timer_end = time.time()
	sol.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
	sol.runtime = timer_end - timer_start
	sol.convergence = convergence_curve
	sol.optimizer = "SSO"
	sol.objf_name = objective_function.__name__
	sol.dataset_name = dataset_name
	sol.labels_pred = np.array(population.spiders[index_best].get_dataset_clusters(), dtype=np.int64)
	sol.best_individual = best_individual
	
	sol.save()
	# return sol