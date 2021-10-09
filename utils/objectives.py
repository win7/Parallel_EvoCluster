# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:12:29 2019

@author: Raneem
"""

from sklearn import cluster, metrics
from scipy.spatial.distance import cdist, pdist
# from time import perf_counter

import numpy as np
import sys

def get_labels_pred(startpts, points, num_clusters):
	# labels_pred = [-1] * len(points)
	labels_pred = np.zeros(len(points), dtype="int")
	# labels_pred[:] = -1
	labels_pred.fill(-1)
	
	for k in range(len(points)):
		distances = np.linalg.norm(points[k] - startpts, axis=1)
		labels_pred[k] = np.argmin(distances)

	""" labels_pred = pymp.shared.array(len(points), dtype="uint8")
	labels_pred[:] = -1

	with pymp.Parallel(cores) as p:
		# p.print(p.num_threads, p.thread_num)
		for k in p.range(len(points)):
			distances = np.linalg.norm(points[k] - startpts, axis=1)
			labels_pred[k] = np.argmin(distances) """

	return labels_pred

def SSE(startpts, points, num_clusters, metric):
	labels_pred = get_labels_pred(startpts, points, num_clusters)
	fitness = 0

	if np.unique(labels_pred).size < num_clusters:
		fitness = sys.float_info.max
	else:
		centroids_for_points = startpts[labels_pred]
		fitness = 0
		for k in range(num_clusters):
			indexes = [count for count, value in enumerate(labels_pred) if value == k]
			fit = cdist(points[indexes], centroids_for_points[indexes], metric) ** 2
			fit = sum(fit)[0]
			fitness += fit
	return fitness, labels_pred

def TWCV(startpts, points, num_clusters):
	labels_pred = get_labels_pred(startpts, points, num_clusters)

	if np.unique(labels_pred).size < num_clusters:
		fitness = sys.float_info.max
	else:
		sum_all_features = np.sum(np.sum(np.power(points, 2)))
		sum_all_pair_points_cluster = 0
		for cluster_id in range(num_clusters):
			indexes = np.where(np.array(labels_pred) == cluster_id)[0]
			points_in_cluster = points[np.array(indexes)]
			sum_pair_points_cluster = np.sum(points_in_cluster)
			sum_pair_points_cluster = np.power(sum_pair_points_cluster, 2)
			sum_pair_points_cluster = np.sum(sum_pair_points_cluster)
			sum_pair_points_cluster = sum_pair_points_cluster / len(points_in_cluster)

			sum_all_pair_points_cluster += sum_pair_points_cluster
		fitness = (sum_all_features - sum_all_pair_points_cluster)
	return fitness, labels_pred

def SC(startpts, points, num_clusters, metric):
	labels_pred = get_labels_pred(startpts, points, num_clusters)

	if np.unique(labels_pred).size < num_clusters:
		fitness = sys.float_info.max
	else:
		silhouette = metrics.silhouette_score(points, labels_pred, metric=metric)
		#silhouette = (silhouette - (-1)) / (1 - (-1))
		silhouette = (silhouette + 1) / 2
		fitness = 1 - silhouette
	return fitness, labels_pred

def DB(startpts, points, num_clusters):
	labels_pred = get_labels_pred(startpts, points, num_clusters)
	if np.unique(labels_pred).size < num_clusters:
		fitness = sys.float_info.max
	else:
		fitness = metrics.davies_bouldin_score(points, labels_pred)
	return fitness, labels_pred

def CH(startpts, points, num_clusters):
	labels_pred = get_labels_pred(startpts, points, num_clusters)

	if np.unique(labels_pred).size < num_clusters:
		fitness = sys.float_info.max
	else:
		ch = metrics.calinski_harabaz_score(points, labels_pred)
		fitness = 1 / ch
	return fitness, labels_pred

def delta_fast(ck, cl, distances):
	values = distances[np.where(ck)][:, np.where(cl)]
	values = values[np.nonzero(values)]

	return np.min(values)

def big_delta_fast(ci, distances):
	values = distances[np.where(ci)][:, np.where(ci)]
	#values = values[np.nonzero(values)]

	return np.max(values)

def dunn_fast(points, labels, metric):
	v = pdist(points, metric)
	size_X = len(points)
	X = np.zeros((size_X, size_X))
	X[np.triu_indices(X.shape[0], k=1)] = v
	distances = X + X.T
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

def DI(startpts, points, num_clusters, metric):
	labels_pred = get_labels_pred(startpts, points, num_clusters)

	if np.unique(labels_pred).size < num_clusters:
		fitness = sys.float_info.max
	else:
		dunn = dunn_fast(points, labels_pred, metric)
		if(dunn < 0):
			dunn = 0
		fitness = 1 - dunn
	return fitness, labels_pred

def get_function_details(a):
	# [name, lb, ub]
	param = {
		0: ["SSE", 0, 1],
		1: ["TWCV", 0, 1],
		2: ["SC", 0, 1],
		3: ["DB", 0, 1],
		# 4: ["CH",0,1],
		4: ["DI", 0, 1]
	}
	return param.get(a, "nothing")