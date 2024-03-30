# mpirun -np 8 python get_matrices_for_different_alphaLevels_parallel.py

import glob
import os
import subprocess
import pickle as cPickle
import ast
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pickle
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import numpy as np
import sklearn
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import pandas as pd
import seaborn as sns
from mpi4py import MPI  # ERROR https://stackoverflow.com/questions/36156822/error-when-starting-open-mpi-in-mpi-init-via-python
import os, sys, time
import re
import math


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    #return [container[_i::count] for i in range(count)]
    return [container[i::count] for i in range(count)]




def get_metric_f1_PCMCIplus(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, ref_graph_matrix, graph_matrix, alpha, 
            tau_min=0, tau_diff=1, same_sign=True):    #only correct for tau_diff=0!!!
    tau_diff=0
    same_sign=True
    N, N, taumaxp1 = val_matrix.shape
    TP = 0
    FP = 0
    FN = 0
    auto = 0
    count = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                for tau in range(tau_min, taumaxp1):
                    if tau==0:
                        if ref_graph_matrix[i,j,tau] == '' and graph_matrix[i,j,tau] != '': #only FP if there is causal link
                            if graph_matrix[i,j,tau] == '<--':
                                FP += 1
                            if graph_matrix[i,j,tau] == '-->':
                                FP += 1
                        elif ref_graph_matrix[i,j,tau] != '' and graph_matrix[i,j,tau] != '': #
                            count +=1
                            if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                if(ref_graph_matrix[i,j,tau] == 'x-x') or (ref_graph_matrix[i,j,tau] == 'o-o'):#no true causal link in refMatrix vs causal link in matrix
                                    if(graph_matrix[i,j,tau] == '-->') or (graph_matrix[i,j,tau] == '<--'): #if matrix has causal link, then its FP. otherwise its ignored
                                        FP += 1 

                                elif(ref_graph_matrix[i,j,tau] == '-->') or (ref_graph_matrix[i,j,tau] == '<--'):
                                    if(graph_matrix[i,j,tau] == 'x-x') or (graph_matrix[i,j,tau] == 'o-o'): #then true causal link is missing
                                        FN += 1 
                                    elif(ref_graph_matrix[i,j,tau] == '-->') and (graph_matrix[i,j,tau] == '-->'):
                                        TP += 1
                                    elif(ref_graph_matrix[i,j,tau] == '-->') and (graph_matrix[i,j,tau] == '<--'): #then true link is missing AND other false link is found
                                        FN += 1
                                        FP += 1
                                    elif(ref_graph_matrix[i,j,tau] == '<--') and (graph_matrix[i,j,tau] == '-->'): #same as above
                                        FN += 1
                                        FP += 1
                                    elif(ref_graph_matrix[i,j,tau] == '<--') and (graph_matrix[i,j,tau] == '<--'):
                                        TP += 1



                            elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]): 
                                if(ref_graph_matrix[i,j,tau] == 'x-x') or (ref_graph_matrix[i,j,tau] == 'o-o'): #no true causal link vs causal link
                                    if(graph_matrix[i,j,tau] == '-->') or (graph_matrix[i,j,tau] == '<--'): #then its always FP, otherwise its ignored
                                        FP += 1 

                                elif(ref_graph_matrix[i,j,tau] == '-->') or (ref_graph_matrix[i,j,tau] == '<--'):
                                    if(graph_matrix[i,j,tau] == 'x-x') or (graph_matrix[i,j,tau] == 'o-o'): #then true causal link is missing
                                        FN += 1 
                                    elif(ref_graph_matrix[i,j,tau] == '-->') and (graph_matrix[i,j,tau] == '-->'): #then its FN since not same sign
                                        FN += 1
                                    elif(ref_graph_matrix[i,j,tau] == '-->') and (graph_matrix[i,j,tau] == '<--'): #then true link is missing AND false positive
                                        FN += 1
                                        FP += 1
                                    elif(ref_graph_matrix[i,j,tau] == '<--') and (graph_matrix[i,j,tau] == '-->'): #same as above
                                        FN += 1
                                        FP += 1
                                    elif(ref_graph_matrix[i,j,tau] == '<--') and (graph_matrix[i,j,tau] == '<--'): #then its FN since not same sign
                                        FN += 1                                        




                            elif same_sign==False:
                                TP += 1
                        elif ref_graph_matrix[i,j,tau] != '' and graph_matrix[i,j,tau] == '': #causal link vs no causal link, then its FN
                            if(ref_graph_matrix[i,j,tau] == '-->'):
                                FN += 1
                            elif(ref_graph_matrix[i,j,tau] == '<--'):
                                FN += 1





                    else:
                        if ref_graph_matrix[i,j,tau] == ''  and graph_matrix[i,j,tau] != '':
                            FP += 1
                        elif ref_graph_matrix[i,j,tau] != '' and np.any(graph_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] != ''): #this part is wrong for taudiff>0! should be adapted when using taudiff>0
                            count +=1
                            if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                TP += 1
                            elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                FN += 1
                            elif same_sign==False:
                                TP += 1
                        elif ref_graph_matrix[i,j,tau] != '' and not(np.any(graph_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] != '')):
                            FN += 1
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    print("precision, recall, TP, FP, FN", precision, recall, TP, FP, FN)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count   


def val_mat_l2_distance_correct1(ref_val_matrix, val_matrix):
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return math.sqrt(sum) 

def val_mat_l2_distance_correct_modified(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix):
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(0, len(ref_val_matrix[0][0])):
                if tau == 0 and i<j: #no double counting for links
                    if (ref_graph_matrix[i][j][tau] == graph_matrix[i][j][tau]) and (ref_graph_matrix[i][j][tau] == '-->' or ref_graph_matrix[i][j][tau]=='<--'): #only when causal link exists
                        sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2
                else:
                    sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return math.sqrt(sum) 



def f1score_modelAll(ref_dataset, val_mat_dict, p_mat_dict, link_mat_dic, alpha_levelTest, f14d):
    df_f1score= {}
    ref_ds=ref_dataset
    val_mat_dic= val_mat_dict
    p_mat_dic=p_mat_dict
    #link_mat_dic=graph_mat_dict
    score_list =[]
    score_list2=[]
    for season in link_mat_dic:
        if season !="global":#drop global
            for dataset in link_mat_dic[season]:               
                for i, ensemble in enumerate(link_mat_dic[season][dataset]):
                    if (dataset!=ref_ds or dataset==ref_ds) and dataset!='ncar':
                        datasetIndex=model_names.index(dataset)
                        refdatasetIndex=model_names.index(ref_dataset)
                        for j, ref_ds_ensemble in enumerate(link_mat_dic[season][ref_ds]):                      
                            ref_p_matrix= p_mat_dic[season][ref_ds][ref_ds_ensemble]
                            ref_val_matrix= val_mat_dic[season][ref_ds][ref_ds_ensemble]
                            ref_graph_matrix= link_mat_dic[season][ref_ds][ref_ds_ensemble]
                            p_matrix= p_mat_dic[season][dataset][ensemble]
                            val_matrix= val_mat_dic[season][dataset][ensemble]
                            graph_matrix= link_mat_dic[season][dataset][ensemble]

                            #precision, recall, TP, FP, FN, score, auto, count = get_metric_f1_PCMCIplus(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, ref_graph_matrix, graph_matrix, alpha_levelTest, 
                            #tau_min=0, tau_diff=0, same_sign=True)
                            score=val_mat_l2_distance_correct_modified(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix)
                            score_list.append([season,dataset,ensemble,score])
                            score_list2.append([season,dataset,ensemble,score,ref_ds_ensemble])
                            if not ((datasetIndex==refdatasetIndex) and (i==j)):                               
                                f14d[refdatasetIndex][datasetIndex][j][i]=score #swap indices e.g. on error

    season,dataset,ensemble,score= [list(a) for a in zip(*score_list)]
    df_f1score_ = pd.DataFrame({"season":season,"model":dataset,"ensemble":ensemble,"F1-score":score})
    #get average F1-score over seasons (not used)
    df_f1score_seasonaveraged = df_f1score_.groupby(["model"])["F1-score"].mean().rename("F1-score",inplace=True).to_frame()
    return score_list2














pc_alpha=0.02  #same as in used pcmci results
n_kept_comp=50 #same as in used pcmci results
selected_components = [] #same as in used pcmci results

for i in range(1, n_kept_comp+1):
    selected_components.append('c' + str(i))

selected_comps_file="./selected_comps_NCEP_jja.csv" #same as in used pcmci results
comps_csv = pd.read_csv(selected_comps_file) #same as in used pcmci results
selected_comps_indices=[] #same as in used pcmci results
for i in range(len(selected_components)):
    selected_comps_indices.append(int(comps_csv["comps"][i]))


model_names=['ACCESS-CM2', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC-ES2L',
             'MPI-ESM1-2-HR', 'UKESM1-0-LL']
season= '[6, 7, 8]' #same as in used pcmci results

alphaLevelList=[1e-1,5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 3e-5, 7e-5, 1e-6, 3e-6, 7e-6, 1e-7, 3e-7, 7e-7, 1e-8, 5e-8, 1e-9, 1e-10] #not used since PCMCI+ has no mci-alpha
pca_res_path="./c-transfer"
pcmci_res_path="./output/1"     #path to used pcmci results                                                            

allResultsGlobal=[]




make_dic=True 
use_CMIP6_data = True

allResults={}




COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

if(COMM.rank==0):
    splitted_jobs=split(alphaLevelList, size)
else:
    splitted_jobs=None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)
print("my jobs are alpha_values ", scattered_jobs)

#get PCMCI result matrices
for alpha_levelTest in scattered_jobs:
    print("calculating for alpha_level = ", alpha_levelTest)
    graph_mat_dict={}
    p_mat_dict={}
    val_mat_dict={}

    resTestPath=pcmci_res_path+"/results_*.bin"
    resTest1=None
    file_nameTest=""
    for res_file in glob.glob(pcmci_res_path+"/results_*.bin"):        
        res = pickle.load(open(res_file,"rb"))
        resTest = res
        resTest1=resTest

        if 'file_name' in resTest1.keys():
            file_nameTest = resTest1['file_name']


        else:
            file_nameTest2=res_file.replace('./output/1/results_', '')
            file_nameTest2=file_nameTest2.replace('_50-VAR_10-LAG_pcmci.bin', '')  #---------!!!---edit so the script can find the associated PCA-varimax dimension reduced file---!!!----------



        name=file_nameTest
        print(f"calculating for {name} with alphaLevel = {alpha_levelTest}")
        file_nameTest = pca_res_path+"/"+ file_nameTest
        info_model= file_nameTest.split("_")
        #print("info model liste ist ", info_model)
        dataset_name = info_model[2]               #on error 2 or 3
        ensemble=""
        if dataset_name != "ncar":
            dataset_name= info_model[2] #on error 2 or 3
            if use_CMIP6_data:
                ensemble= info_model[5]  #on error 5 or 6
            else : ensemble= info_model[7]
        if dataset_name == "GISS-E2-R":
            ensemble= info_model[5]
        if dataset_name == "ERA5":
            ensemble= ""
        season= info_model[-1][7:-4]


        datadictTest = cPickle.load(open(file_nameTest, 'rb'))
        dTest = datadictTest['results']
        time_maskTest = dTest['time_mask']
        dateseriesTest = dTest['time'][:]
        fulldataTest = dTest['ts_unmasked']
        N = fulldataTest.shape[1]
        fulldata_mask = np.repeat(time_maskTest.reshape(len(dTest['time']), 1), N, axis=1)

        fulldataTest = fulldataTest[:, selected_comps_indices]
        fulldata_mask = fulldata_mask[:, selected_comps_indices]
        dataframeTest = pp.DataFrame(fulldataTest, mask=fulldata_mask)
        T, N = dataframeTest.values[0].shape
        CI_params = {       'significance':'analytic', 
                                    'mask_type':['y'],
                                    'recycle_residuals':False,
                                    }
        cond_ind_test = ParCorr(**CI_params)
        pcmci=PCMCI(cond_ind_test=cond_ind_test,dataframe=dataframeTest, verbosity=0)



        graphTest = resTest1['graph']

        valMatrixTest = resTest1['val_matrix']
        
        graph_mat_dict.setdefault(season,{})
        graph_mat_dict[season].setdefault(dataset_name,{})
        graph_mat_dict[season][dataset_name]
        graph_mat_dict[season][dataset_name][ensemble] = graphTest
        val_mat_dict.setdefault(season,{})
        val_mat_dict[season].setdefault(dataset_name,{})
        val_mat_dict[season][dataset_name].setdefault(ensemble,None)
        val_mat_dict[season][dataset_name][ensemble]=valMatrixTest
        p_mat_dict.setdefault(season,{})
        p_mat_dict[season].setdefault(dataset_name,{})
        p_mat_dict[season][dataset_name].setdefault(ensemble,)
        p_mat_dict[season][dataset_name][ensemble]=resTest1['p_matrix']
        allResults.setdefault(alpha_levelTest, {})
        allResults[alpha_levelTest].setdefault('graph_mat_dict', {})
        allResults[alpha_levelTest].setdefault('val_mat_dict', {})
        allResults[alpha_levelTest].setdefault('p_mat_dict', {})       
        allResults[alpha_levelTest]['graph_mat_dict']=graph_mat_dict
        allResults[alpha_levelTest]['val_mat_dict']=val_mat_dict
        allResults[alpha_levelTest]['p_mat_dict']=p_mat_dict


COMM.Barrier()


#calculate F1-scores
resDict={}
for key in allResults.keys():
    print("aktueller key ist: ", key)
    modelRes=allResults[key]
    print(f"process nr. {rank} calculates f1-score for alpha_level = {key}")
    graph_mat_dict=modelRes['graph_mat_dict']
    val_mat_dict=modelRes['val_mat_dict']
    p_mat_dict=modelRes['p_mat_dict']
    f14d=np.empty((len(model_names), len(model_names), 4, 4))
    shape=(len(model_names), len(model_names), 4, 4)
    f14d = np.full(shape, np.nan)

    for element2 in model_names:
        x=f1score_modelAll(element2, val_mat_dict, p_mat_dict, graph_mat_dict, key, f14d)

    resDict[key]=f14d

resDict = MPI.COMM_WORLD.gather(resDict, root=0)

if rank == 0:
    result = {'pc_alpha': pc_alpha,
    'n_kept_comp': n_kept_comp,
    'f1Scores': resDict}
    file_name='./output_f1-scores/f1scores_new1/val_mat_l2_distance_correct_modified_50var_pcAlpha=0.02.bin' #---------------------set output file name---!!!------------
    file = open(file_name, 'wb')
    cPickle.dump(result, file, protocol=-1)
    file.close()

COMM.Barrier()













