# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:22:53 2019

@author: Raneem
"""

from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

import math
import numpy as np
import sys

def HS(labels_true, labels_pred):
	return float("%0.2f" % metrics.homogeneity_score(labels_true, labels_pred))

def CS(labels_true, labels_pred):
	return float("%0.2f" % metrics.completeness_score(labels_true, labels_pred))

def VM(labels_true, labels_pred):
	return float("%0.2f" % metrics.v_measure_score(labels_true, labels_pred))

def AMI(labels_true, labels_pred):
	return float("%0.2f" % metrics.adjusted_mutual_info_score(labels_true, labels_pred))

def ARI(labels_true, labels_pred):
	return float("%0.2f" % metrics.adjusted_rand_score(labels_true, labels_pred))

def Fmeasure(labels_true, labels_pred):
	return float("%0.2f" % metrics.f1_score(labels_true, labels_pred, average="macro"))

def SC(points, labels_pred):  # Silhouette Coefficient
	if np.unique(labels_pred).size == 1:
		fitness = sys.float_info.max
	else:
		silhouette = float("%0.2f" % metrics.silhouette_score(points, labels_pred, metric="euclidean"))
		silhouette = (silhouette + 1) / 2
		fitness = 1 - silhouette
	return fitness

def accuracy(labels_true, labels_pred):  # Silhouette Coefficient
	# silhouette = metrics.accuracy_score(labels_true, labels_pred, normalize=False)
	return ARI(labels_true, labels_pred)

def delta_fast(ck, cl, distances):
	values = distances[np.where(ck)][:, np.where(cl)]
	values = values[np.nonzero(values)]
	return np.min(values)

def big_delta_fast(ci, distances):
	values = distances[np.where(ci)][:, np.where(ci)]
	# values = values[np.nonzero(values)]
	return np.max(values)

def dunn_fast(points, labels):
	""" Dunn index - FAST (using sklearn pairwise euclidean_distance function)
	Parameters
	----------
	points: np.array
		np.array([N, p]) of all points
	labels: np.array
		np.array([N]) labels of all points
	"""
	distances = euclidean_distances(points)
	ks = np.sort(np.unique(labels))

	deltas = np.ones([len(ks), len(ks)]) * 1000000
	big_deltas = np.zeros([len(ks), 1])

	l_range = list(range(len(ks)))

	for i in l_range:
		for j in (l_range[0:i] + l_range[i + 1:]):
			deltas[i, j] = delta_fast((labels == ks[i]), (labels == ks[j]), distances)

		big_deltas[i] = big_delta_fast((labels == ks[i]), distances)

	di = np.min(deltas) / np.max(big_deltas)
	return di


def DI(points, labels_pred):  # dunn index
	dunn = float("%0.2f" % dunn_fast(points, labels_pred))
	if(dunn < 0):
		dunn = 0
	fitness = 1 - dunn
	return fitness

def DB(points, labels_pred): 
	try:
		return float("%0.2f" % metrics.davies_bouldin_score(points, labels_pred))
	except Exception as e:
		return 0.5
	
def stdev(individual, labels_pred, num_clusters, points):
	std = 0
	distances = []
	f = (int)(len(individual) / num_clusters)
	startpts = np.reshape(individual, (num_clusters, f))

	for k in range(num_clusters):
		index_list = np.where(labels_pred == k)
		distances = np.append(distances, np.linalg.norm(points[index_list] - startpts[k], axis=1))
	std = np.std(distances)

	# stdev = math.sqrt(std) / num_clusters
	# print("stdev:", stdev)
	return std

"""
def SSE(individual, k, points):
	f = (int)(len(individual) / k)
	startpts = np.reshape(individual, (k,f))    
	labels_pred = [-1] * len(points)
	sse = 0
	
	for i in range(len(points)):
		distances = np.linalg.norm(points[i]-startpts, axis = 1)
		sse = sse + np.min(distances)
		clust = np.argmin(distances)
		labels_pred[i] = clust
		
	if np.unique(labels_pred).size < k:
		sse = sys.float_info.max
			   
	print("SSE:",sse)
	return sse
"""

def SSE(individual, labels_pred, num_clusters, points):
	f = (int)(len(individual) / num_clusters)
	startpts = np.reshape(individual, (num_clusters, f))
	fitness = 0

	centroids_for_points = startpts[labels_pred]
	fitness_values = np.linalg.norm(points - centroids_for_points, axis=1) ** 2
	fitness = sum(fitness_values)
	return fitness

def TWCV(individual, labels_pred, num_clusters, points):
	sum_all_features = sum(sum(np.power(points, 2)))
	sum_all_pair_points_cluster = 0
	for cluster_id in range(num_clusters):
		indixes = np.where(np.array(labels_pred) == cluster_id)[0]
		points_in_cluster = points[np.array(indixes)]
		sum_pair_points_cluster = sum(points_in_cluster)
		sum_pair_points_cluster = np.power(sum_pair_points_cluster, 2)
		if len(points_in_cluster) != 0:
			sum_pair_points_cluster = sum(sum_pair_points_cluster)
			sum_pair_points_cluster = sum_pair_points_cluster / len(points_in_cluster)

		sum_all_pair_points_cluster += sum_pair_points_cluster
	fitness = sum_all_features - sum_all_pair_points_cluster
	return fitness

def purity(labels_true, labels_pred):
	# get the set of unique cluster ids
	labels_true = np.asarray(labels_true).astype(int)
	labels_pred = np.asarray(labels_pred).astype(int)

	k = (max(labels_true) + 1).astype(int)

	total_sum = 0

	for i in range(k):
		max_freq = 0
		t1 = np.where(labels_pred == i)

		for j in range(k):
			t2 = np.where(labels_true == j)
			z = np.intersect1d(t1, t2)
			e = np.shape(z)[0]

			if (e >= max_freq):
				max_freq = e
		total_sum = total_sum + max_freq
	purity = total_sum / np.shape(labels_true)[0]

	# print("purity:",purity)
	return purity

def entropy(labels_true, labels_pred):
	# get the set of unique cluster ids
	labels_true = np.asarray(labels_true).astype(int)
	labels_pred = np.asarray(labels_pred).astype(int)

	k = (max(labels_true) + 1).astype(int)
	entropy = 0

	for i in range(k):
		t1 = np.where(labels_pred == i)
		entropy_i = 0

		for j in range(k):
			t2 = np.where(labels_true == j)
			z = np.intersect1d(t1, t2)

			e = np.shape(z)[0]
			if (e != 0):
				entropy_i = entropy_i + (e / np.shape(t1)[1]) * math.log(e / np.shape(t1)[1])

		a = np.shape(t1)[1]
		b = np.shape(labels_true)[0]
		entropy = entropy + ((a / b) * ((-1 / math.log(k)) * entropy_i))

	# print("entropy:",entropy)
	return entropy