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


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    #return [container[_i::count] for i in range(count)]
    return [container[i::count] for i in range(count)]


def get_metric_f1(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, subnet, 
            tau_min=0, tau_diff=1, same_sign=True):
    tau_diff=0
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
                    if (i,j,tau) in subnet:
                        if (len(subnet[(i,j,tau)]) > 0):
        #                     print(np.sum(ref_p_matrix[i,j,tau] < alpha),np.sum(p_matrix[i,j,tau] < alpha))
                            if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                                FP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                                count +=1
                                if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                    TP += 1
                                elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                    FN += 1
                                elif same_sign==False:
                                    TP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                FN += 1
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count

def get_metric_f1_lag0(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, subnet, 
            tau_min=0, tau_diff=1, same_sign=True):
    tau_diff=0
    tau_min=1

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
                    if (i,j,tau) in subnet:
                        if (len(subnet[(i,j,tau)]) > 0):
                            if tau==0:
                                if i<j: #avoid double counting
                                    if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                                        FP += 1
                                    elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                                        count +=1
                                        if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                            TP += 1
                                        elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                            FN += 1
                                        elif same_sign==False:
                                            TP += 1
                                    elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[j,i,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                                        count +=1
                                        if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[j,i,tau]):
                                            TP += 1
                                        elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[j,i,tau]):
                                            FN += 1
                                        elif same_sign==False:
                                            TP += 1
                                    elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                        FN += 1
                                    elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[j,i,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                        FN += 1
                            else:
                                if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                                    FP += 1
                                elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                                    count +=1
                                    if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                        TP += 1
                                    elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                        FN += 1
                                    elif same_sign==False:
                                        TP += 1
                                elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                    FN += 1
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count





def f1score_modelAll(ref_dataset, val_mat_dict, p_mat_dict, link_mat_dic, alpha_levelTest, f14d, subnet):
    df_f1score= {}
    ref_ds=ref_dataset
    val_mat_dic= val_mat_dict
    p_mat_dic=p_mat_dict
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
                            p_matrix= p_mat_dic[season][dataset][ensemble]
                            val_matrix= val_mat_dic[season][dataset][ensemble]
                            precision, recall, TP, FP, FN, score, auto, count = get_metric_f1_lag0(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha_levelTest, subnet,
                            tau_min=0, tau_diff=2, same_sign=True)
                            score_list.append([season,dataset,ensemble,score])
                            score_list2.append([season,dataset,ensemble,score,ref_ds_ensemble])
                            if not ((datasetIndex==refdatasetIndex) and (i==j)):                               
                                f14d[refdatasetIndex][datasetIndex][j][i]=score 

    season,dataset,ensemble,score= [list(a) for a in zip(*score_list)]
    df_f1score_ = pd.DataFrame({"season":season,"model":dataset,"ensemble":ensemble,"F1-score":score})
    #get average F1-score over seasons (not used here)
    df_f1score_seasonaveraged = df_f1score_.groupby(["model"])["F1-score"].mean().rename("F1-score",inplace=True).to_frame()
    return score_list2


def getModelLinks(model_name):
    modelGraphs=[]
    for element in graphMats:
        if element[0]==model_name:
            modelGraphs.append(element[2])
            
    result_array = np.empty_like(modelGraphs[0], dtype='<U3')

    for index in np.ndindex(result_array.shape):
        if all(np.all(arr[index] == modelGraphs[0][index]) for arr in modelGraphs):
            result_array[index] = str(modelGraphs[0][index])
        else:
            result_array[index] = ''
    return result_array

def thresholdModelGraph(graph, model_name, threshold1, threshold2):
    print('tresholding model :', model_name)
    otherGraphs=[]
    relatedGraphs=[]
    for element in related_models:
        if (model_names.index(model_name)) in list(element):
            for index in element:
                relatedGraphs.append(model_names[index])
    print('related models: ', relatedGraphs)
    for element in graphMats:
        if(element[0] != 'ncar'): #skip ncar
            if element[0] not in relatedGraphs:
                otherGraphs.append(element[2])
    print('relatedGraphs for model :', model_name, relatedGraphs)
    
    
    linksDict={}

    for index, element in np.ndenumerate(graph):
        if element != '':
            count=0
            count = sum(1 for arr in otherGraphs if np.all(arr[index] == element))
            if (count >= threshold1) and (count<=threshold2):
                linksDict[index]=[]
                linksDict[index].append((count,model_name,element))

    return linksDict

pcmci_res_path='./output/gridSearch/alpha=3e-20_nVAR=100' #path of results from best performing PCMCI settings
pca_res_path='./c-transfer'
alpha_levelTest=0.0001           #use best performing mci_alpha 
use_CMIP6_data=True




nrEnsembles=0
graphMats=[]
plotDict={}



make_dic=True #
n_kept_comp = 100 #number of kept PCA comp time series from optimal hyperparameter setting
selected_comps_indices=None
var_names=["X_"+str(i) for i in range(0,n_kept_comp)]




resTestPath=pcmci_res_path+"/results_*.bin"
resTest1=None
file_nameTest=""
for res_file in glob.glob(pcmci_res_path+"/results_*.bin"):        
    res = pickle.load(open(res_file,"rb"))
    resTest = res
    resTest1=resTest
    #print(resTest1)
    file_nameTest = resTest1['file_name']
    
    
    
    #selected_comps_indices=resTest1['variables']
    selected_comps_indices=range(0, n_kept_comp)
    
    
    
    name=file_nameTest
    file_nameTest = pca_res_path+"/"+ file_nameTest
    info_model= file_nameTest.split("_")
    dataset_name = info_model[2]               #on error change to 2 or 3
    ensemble=""
    if dataset_name != "ncar":
        dataset_name= info_model[2]
        if use_CMIP6_data:
            ensemble= info_model[5]  #on error change ot 5 or 6
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


#correctedPMatrix=pcmci.get_corrected_pvalues(p_matrix=resTest1['results']['p_matrix'],
 #                                            tau_min=0
    graphTest = pcmci.get_graph_from_pmatrix(p_matrix=resTest1['results']['p_matrix'],
                                     alpha_level=alpha_levelTest,
                                     tau_min=1,
                                     tau_max=10,
                                     link_assumptions=None,)

    graphMats.append((dataset_name, ensemble, graphTest))
    nrEnsembles=nrEnsembles+1
    valMatrixTest = resTest1['results']['val_matrix']
    
    plotDict[file_nameTest]=(graphTest, valMatrixTest, datadictTest, name)


model_names=['ACCESS-CM2', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC-ES2L',
             'MPI-ESM1-2-HR', 'UKESM1-0-LL']
season= '[6, 7, 8]'   #set es used season
#pattern = r"PCalpha=([0-9.]+)_nVAR=(\d+).bin"

related_models=[] #as in gridSearch.ipynb
related_models.append((0, 6, 10))
related_models.append((1,))
related_models.append((2,))
related_models.append((3,))
related_models.append((4,))
related_models.append((5,))
related_models.append((7,))
related_models.append((8,))
related_models.append((9,))





alphaLevelList=[0.0001]                                                                                                                                           #!!!!!!!MCI-ALPHA FROM BEST SETTING!!!!!!!!!
pca_res_path="./c-transfer"
pcmci_res_path="./output/gridSearch"

allResultsGlobal=[]

#list of threshold values to use
tresholdList=[(0,0),(0,3),(0,6),(0,32),(0,23),(0,30),(0,21),(0,32),(0,34),(0,27),(0,10),(0,14),(0,11),(0,31),
              (2,3),(2,6),(2,32),(2,23),(2,30),(2,32),(2,34),(2,27),(2,10),(2,14),(2,11),(2,33),
              (5,6),(5,32),(5,23),(5,32),(5,27),(5,10),(5,14),(5,11),(5,30),(5,31),(5,33),(5,34),
              (8,32),(8,23),(8,33),(8,27),(8,10),(8,14),(8,11),(8,30),(8,31),(8,33),(8,34),
              (12,32),(12,23),(12,33),(12,27),(12,14),(12,30),(12,31),(12,34),
              (17,32),(17,23),(17,33),(17,27),(17,30),(17,31),(17,34)]

toAppend=[(0,9),(0,12),(0,13),(0,15),(0,16),(0,17),(0,18),(2,12),(2,13),(2,14),(5,12),(5,13),(5,14),(8,12),(8,13),(8,14)]
for element in toAppend:
    tresholdList.append(element)

tresholdList=[(0,0)]

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()




if COMM.rank==0:
    splittedList=split(tresholdList, size)
else:
    splittedList=None 
tresholdListSplitted = COMM.scatter(splittedList, root=0)
print(rank, tresholdListSplitted)

#split threshold values among processes
currentLinksListGraph=[]
for (under, upper) in tresholdListSplitted:
    currentLinks = {}
    for model_name in model_names:
        print(f"Process {rank} - Current model: {model_name}", under, upper)
        graph = getModelLinks(model_name)
        newLinks = thresholdModelGraph(graph, model_name, under, upper)
        for key, value in newLinks.items():
            if key not in currentLinks:
                currentLinks[key]=value
            else:
                currentLinks[key].extend(value)
    currentLinksListGraph.append((under, upper, currentLinks))



all_results = COMM.gather(currentLinksListGraph, root=0)
if rank == 0:
    final_result = []
    for result_chunk in all_results:
        final_result.extend(result_chunk)
    print('final_result', len(final_result))
else:
    final_result = None
final_result = COMM.bcast(final_result, root=0)
currentLinksListGraph=final_result



for folder in os.listdir(pcmci_res_path):
    pcmci_res_path_=os.path.join(pcmci_res_path, folder)
    print("calculating for ", pcmci_res_path_)


    pattern = r"alpha=([0-9.eE+-]+)_nVAR=(\d+)" #set pattern as required
    match = re.search(pattern, pcmci_res_path_)
    # just get information from folder name
    if match:
        pc_alpha = float(match.group(1))
        n_kept_comp = int(match.group(2))
        print("file match found!")
    else:
        alpha_value=None
        n_kept_comp=None


    if pc_alpha!=3e-20: #only use best hyperparameter setting!!!!!!!!!!!!!!!!!!!!!!
        continue
    if n_kept_comp!=100:
        continue




    for tresholdLinks_ in currentLinksListGraph:
        tresholdLinks=tresholdLinks_[2]




        file_name='./output_f1-scores_lag1/mapSearchIndices3_tauDiff0/f1Scores_PCalpha=%f_nVAR=%d_range=%s.bin' % (pc_alpha, n_kept_comp, str((tresholdLinks_[0], tresholdLinks_[1])))
        if os.path.isfile(file_name):
            print("FILE ALREADY EXISTS", file_name)
            continue
        print("----------------------------------------------------------------------------------------", pcmci_res_path_)

        make_dic=True 
        use_CMIP6_data = True

        allResults={}



        if(COMM.rank==0):
            splitted_jobs=split(alphaLevelList, size)
        else:
            splitted_jobs=None

        scattered_jobs = COMM.scatter(splitted_jobs, root=0)
        print("my jobs are alpha_values ", scattered_jobs)

        for alpha_levelTest in scattered_jobs:
            print("calculating for alpha_level = ", alpha_levelTest)
            graph_mat_dict={}
            p_mat_dict={}
            val_mat_dict={}
            resTestPath=pcmci_res_path_+"/results_*.bin"
            resTest1=None
            file_nameTest=""
            for res_file in glob.glob(pcmci_res_path_+"/results_*.bin"):       #get the result matrices 
                res = pickle.load(open(res_file,"rb"))
                resTest = res
                resTest1=resTest
                pc_alpha=resTest1['PC_params']['pc_alpha']              
                tau_max=resTest1['PC_params']['tau_max']
                file_nameTest = resTest1['file_name']

                n_kept_comp=len(resTest1['PC_params']['selected_variables'])
                selected_comps_indices=[i for i in range(0,n_kept_comp)]
                var_names=["X_"+str(i) for i in range(0,n_kept_comp)]

                name=file_nameTest
                print(f"calculating for {name} with alphaLevel = {alpha_levelTest}")
                file_nameTest = pca_res_path+"/"+ file_nameTest
                info_model= file_nameTest.split("_")
                #print("info model liste ist ", info_model)
                dataset_name = info_model[2]               #same
                ensemble=""
                if dataset_name != "ncar":
                    dataset_name= info_model[2] #same
                    if use_CMIP6_data:
                        ensemble= info_model[5]  #same
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



                graphTest = pcmci.get_graph_from_pmatrix(p_matrix=resTest1['results']['p_matrix'],
                                                alpha_level=alpha_levelTest,
                                                tau_min=1,
                                                tau_max=10,
                                                link_assumptions=None,)

                valMatrixTest = resTest1['results']['val_matrix']
                
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
                p_mat_dict[season][dataset_name][ensemble]=resTest1['results']['p_matrix']
                allResults.setdefault(alpha_levelTest, {})
                allResults[alpha_levelTest].setdefault('graph_mat_dict', {})
                allResults[alpha_levelTest].setdefault('val_mat_dict', {})
                allResults[alpha_levelTest].setdefault('p_mat_dict', {})       
                allResults[alpha_levelTest]['graph_mat_dict']=graph_mat_dict
                allResults[alpha_levelTest]['val_mat_dict']=val_mat_dict
                allResults[alpha_levelTest]['p_mat_dict']=p_mat_dict


        COMM.Barrier()


        #calculate F1-scores based on subnetwork
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
                x=f1score_modelAll(element2, val_mat_dict, p_mat_dict, graph_mat_dict, key, f14d, tresholdLinks)

            resDict[key]=f14d

        resDict = MPI.COMM_WORLD.gather(resDict, root=0)

        if rank == 0:
            result = {'pc_alpha': pc_alpha, #set result dictonary as required
            'n_kept_comp': n_kept_comp,
            'f1Scores': resDict,
            'selected_comps': range(0,100),
            'treshold': (tresholdLinks_[0], tresholdLinks_[1]),
            'links': tresholdLinks}
            file = open(file_name, 'wb')
            cPickle.dump(result, file, protocol=-1)
            file.close()

        COMM.Barrier()


    










