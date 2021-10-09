# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:48 2019

@author: Raneem
"""
from optimizer import run

# Select optimizers
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS"
optimizer=["PSO", "GWO"]

# Select objective function
# "SSE","TWCV","SC","DB","DI"
objective_function=["SSE","TWCV"] 

# Select data setsz
#"aggregation","aniso","appendicitis","balance","banknote","blobs","Blood","circles","diagnosis_II","ecoli","flame","glass","heart","ionosphere","iris","iris2D","jain","liver","moons","mouse","pathbased","seeds","smiley","sonar","varied","vary-density","vertebral2","vertebral3","wdbc","wine"
dataset_list = ["iris","aggregation"]

# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
num_runs=3

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {'population_size' : 30, 'iterations' : 50}

#Choose whether to Export the results in different formats
export_flags = {'Export_avg':True, 'Export_details':True, 'Export_details_labels':True, 
'Export_convergence':True, 'Export_boxplot':True}

run(optimizer, objective_function, dataset_list, num_runs, params, export_flags, 
	auto_cluster = False, n_clusters = [3,7], labels_exist = True, metric='cityblock')