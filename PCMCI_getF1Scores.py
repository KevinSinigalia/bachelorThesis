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


def get_metric_f1(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, 
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



def f1score_modelAll(ref_dataset, val_mat_dict, p_mat_dict, link_mat_dic, alpha_levelTest, f14d):
    df_f1score= {}
    ref_ds=ref_dataset
    val_mat_dic= val_mat_dict
    p_mat_dic=p_mat_dict
    link_mat_dic=graph_mat_dict
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
                            #von hier
                            ref_p_matrix= p_mat_dic[season][ref_ds][ref_ds_ensemble]
                            ref_val_matrix= val_mat_dic[season][ref_ds][ref_ds_ensemble]
                            #bis hier über alle ref_ds gehen in for loop
                            p_matrix= p_mat_dic[season][dataset][ensemble]
                            val_matrix= val_mat_dic[season][dataset][ensemble]
                            precision, recall, TP, FP, FN, score, auto, count = get_metric_f1(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha_levelTest, 
                            tau_min=0, tau_diff=2, same_sign=True)
                            score_list.append([season,dataset,ensemble,score])
                            score_list2.append([season,dataset,ensemble,score,ref_ds_ensemble])
                            if not ((datasetIndex==refdatasetIndex) and (i==j)):                               
                                f14d[refdatasetIndex][datasetIndex][j][i]=score #on error swap dimensions

    season,dataset,ensemble,score= [list(a) for a in zip(*score_list)]
    df_f1score_ = pd.DataFrame({"season":season,"model":dataset,"ensemble":ensemble,"F1-score":score})
    #get average F1-score over seasons
    df_f1score_seasonaveraged = df_f1score_.groupby(["model"])["F1-score"].mean().rename("F1-score",inplace=True).to_frame()
    return score_list2


















model_names=['ACCESS-CM2', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC-ES2L',
             'MPI-ESM1-2-HR', 'UKESM1-0-LL']
alphaLevel_matrices_path = "./output_matrices/gridSearch"
season= '[6, 7, 8]'
pattern = r"PCalpha=([0-9.]+)_nVAR=(\d+).bin"




alphaLevelList=[1e-1,5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 3e-5, 7e-5, 1e-6, 3e-6, 7e-6, 1e-7, 3e-7, 7e-7, 1e-8, 5e-8, 1e-9, 1e-10] #mci_alpha values to use
pca_res_path="./c-transfer" #path to pca-varimax dimension reduced data
pcmci_res_path="./output/gridSearch" #path to FOLDERS containing PCMCI results

allResultsGlobal=[]

for folder in os.listdir(pcmci_res_path):   #loop over folders
    pcmci_res_path_=os.path.join(pcmci_res_path, folder)


    pattern = r"alpha=([0-9.]+)_nVAR=(\d+)"
    pattern = r"PCalpha=([0-9.eE+-]+)_nVAR=(\d+).bin"
    pattern = r"alpha=([0-9.eE+-]+)_nVAR=(\d+)" #change on requirements
    


    match = re.search(pattern, pcmci_res_path_)
    #get information from folder name
    if match:
        pc_alpha = float(match.group(1))
        n_kept_comp = int(match.group(2))
    else:
        pc_alpha=None
        n_kept_comp=None
        print("not found: ", pcmci_res_path_)

    #just create output folder...change on requirements
    if pc_alpha > 1e-16:
        file_name='./output_f1-scores/gridSearchNew_tauDiff0/f1Scores_PCalpha={:.20f}_nVAR={}.bin'.format(pc_alpha, str(n_kept_comp))
    else:
        file_name='./output_f1-scores/gridSearchNew_tauDiff0/f1Scores_PCalpha={}_nVAR={}.bin'.format(str(pc_alpha), str(n_kept_comp))
    if os.path.isfile(file_name):
        print("FILE ALREADY EXISTS", file_name)
        continue

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

    for alpha_levelTest in scattered_jobs:  #loop over given mci_alpha values
        print("calculating for alpha_level = ", alpha_levelTest)
        graph_mat_dict={}
        p_mat_dict={}
        val_mat_dict={}
        resTestPath=pcmci_res_path_+"/results_*.bin"
        resTest1=None
        file_nameTest=""
        for res_file in glob.glob(pcmci_res_path_+"/results_*.bin"):        #loop over PCMCI results of one folder
            res = pickle.load(open(res_file,"rb"))
            resTest = res
            resTest1=resTest
            pc_alpha=resTest1['PC_params']['pc_alpha']              #depends on resdict keys from PCMCI script
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
            dataset_name = info_model[2]               #on error change to 2 or 3
            ensemble=""
            if dataset_name != "ncar":
                dataset_name= info_model[2] #same
                if use_CMIP6_data:
                    ensemble= info_model[5]  #on error change to 5 or 6
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
                                            link_assumptions=None,) #get graph

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


    #get F1-scores
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

        resDict[key]=f14d #contains F1-scores of the ensembles for given PCMCI setting (folder in first loop)

    resDict = MPI.COMM_WORLD.gather(resDict, root=0)

    if rank == 0:
        result = {'pc_alpha': pc_alpha,
        'n_kept_comp': n_kept_comp,
        'f1Scores': resDict}
        file = open(file_name, 'wb')
        cPickle.dump(result, file, protocol=-1)
        file.close()

    COMM.Barrier()


    










