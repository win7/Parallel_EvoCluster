import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.objectives

def run(results_directory, optimizer, objective_function, dataset_list, ev_measures, iterations):
    plt.ioff()
    
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment_details.csv')
    for d in range(len(dataset_list)):        
        dataset_filename = dataset_list[d] + '.csv' 
        for j in range (0, len(objective_function)):
            for z in range (0, len(ev_measures)):
                
                #Box Plot
                data = []      
                    
                for i in range(len(optimizer)): 
                    objective_name = objective_function[j]
                    optimizer_name = optimizer[i]
                    
                    detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == dataset_list[d]) & (fileResultsDetailsData["Optimizer"] == optimizer_name) & (fileResultsDetailsData["ObjfName"] == objective_name)]
                    detailedData = detailedData[ev_measures[z]]
                    detailedData = np.array(detailedData).T.tolist()
                    data.append(detailedData)

                #, notch=True
                box=plt.boxplot(data,patch_artist=True,labels=optimizer)
                

                colors = ['#5c9eb7','#f77199', '#cf81d2','#4a5e6a','#f45b18',
                '#ffbd35','#6ba5a1','#fcd1a1','#c3ffc1','#68549d',
                '#1c8c44','#a44c40','#404636']
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                 
                plt.legend(handles= box['boxes'], labels=optimizer, 
                    loc="upper right", bbox_to_anchor=(1.2,1.02))
                fig_name = results_directory + "/boxplot-" + dataset_list[d] + "-" + objective_name + "-" + ev_measures[z] + ".png"
                plt.savefig(fig_name, bbox_inches='tight')
                plt.clf()
                #plt.show()
                


