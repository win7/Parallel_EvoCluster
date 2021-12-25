# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:06:19 2016

@author: Hossam Faris
"""
# from json import JSONEncoder
# import numpy as np
# import json
import pickle

""" class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(object, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, object) """

class Solution():
	def __init__(self):
		self.points = []
		self.labels_pred = []
		self.best_individual = []
		self.convergence = []
		self.fitness = 0
		self.best = 0
		self.optimizer = ""
		self.objf_name = ""
		self.dataset_name = ""
		self.start_time = 0
		self.end_time = 0
		self.runtime = 0
		self.lb = 0
		self.ub = 0
		self.dimension = 0
		self.num_clusters = 2
		self.population_size = 0
		self.policy = {"topology": "-"}

	def save(self):
		with open("temp/{}_{}_{}.pkl".format(self.optimizer, self.objf_name, self.dataset_name), "wb") as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
		""" with open("output/{}_{}_{}.json".format(self.optimizer, self.objf_name, self.dataset_name), "w") as f:
			json.dump(self.__dict__, f, cls=NumpyArrayEncoder) """

	def get(self, file_name):
		with open("temp/{}.pkl".format(file_name), "rb") as input:
			sol = pickle.load(input)
			return sol
		""" with open("output/{}.json".format(file_name), "r") as f:
			data = json.load(f)
			return data """
