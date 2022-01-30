import os
import pandas as pd

def absolute_maximum_scale(series):
    return series / series.abs().max()

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

def z_score_standardization(series):
    return (series - series.mean()) / series.std()

# Only for combine files
""" my_list = sorted(os.listdir('results_'))
# print(my_list)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv("results_/{}/experiment_best_params.csv".format(f)) for f in my_list ])
#export to csv
combined_csv.to_csv("best_params.csv", index=False, encoding="utf-8-sig") """

datasets = ["blood", "flame", "pathbased", "smiley", "vary-density"]
optimizers = ["MPI_SSA", "MPI_PSO", "MPI_GA", "MPI_BAT", "MPI_FFA", "MPI_GWO", "MPI_WOA", "MPI_MVO", "MPI_MFO", "MPI_CS"]
topologies = ["RING", "TREE", "NETA", "NETB", "TORUS", "GRAPH", "SAME", "GOODBAD", "RAND"]

df = pd.read_csv("best_params_config1.csv")
# print(df.head())

group_df = df.groupby(["Dataset", "Optimizer", "Topology"])
# print(group_df.grouper.groups.get(('blood', 'MPI_BAT', 'GOODBAD')))

for optimizer in optimizers:
    for topology in topologies:
        df_rating = pd.DataFrame() 
        for dataset in datasets:
            key = (dataset, optimizer, topology)
            aux = group_df.get_group(key)
            for column in aux[["SSE"]]:
                df_rating[dataset] = list(absolute_maximum_scale(aux[column]))
        df_rating["mean"] = df_rating.mean(axis=1)
        min_index = df_rating["mean"].idxmin()
        print((optimizer, topology))
        print("Min: {}".format(min_index + 1))
        print(df_rating)
        print()

""" for name, group in group_df:
    for column in group[["SSE"]]:
        # group["a"] = (group[column] - group[column].min()) / (group[column].max() - group[column].min())    
        group["x"] = absolute_maximum_scale(group[column])
        group["y"] = min_max_scaling(group[column])
        group["z"] = z_score_standardization(group[column])

    print(name)
    # print(group)
    print(group["x"])
    print() """

# Run
# python best_params.py > z_config1.txt