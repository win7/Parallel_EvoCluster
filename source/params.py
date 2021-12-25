# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:06:19 2016

@author: Hossam Faris
"""
import pickle

class Params():
	def __init__(self):
		self.algorithm = ""
		self.objetive_name = ""
		self.num_clusters = 0
		self.num_features = 0
		self.population_size = 0
		self.iterations = 0
		self.iteration = 0
		self.points = []
		self.metric = ""
		self.dataset_name = ""
		self.policy = {}
		self.cores = 0

	def save(self):
		with open("temp/params.pkl", "wb") as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def get(self):
		with open("temp/params.pkl", "rb") as input:
			params = pickle.load(input)
			return params
