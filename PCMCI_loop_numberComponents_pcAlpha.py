#path depending on timebin
pca_res_path="./c-transfer"
pcmci_res_path="./pcmci_res"
global_res_path = "./global_res/results.bin"

















#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tigramite causal discovery for time series: Parallization script implementing 
the PCMCI method based on mpi4py. 

Parallelization is done across variables j for both the PC condition-selection
step and the MCI step.
"""


# Author: Jakob Runge <jakobrunge@posteo.de>
# Modification by Kevin Debeire, Kevin Sinigalia
# License: GNU General Public License v3.0
#
import time
import glob
from mpi4py import MPI  # ERROR https://stackoverflow.com/questions/36156822/error-when-starting-open-mpi-in-mpi-init-via-python
import numpy
import os, sys, time
from datetime import datetime, date
import pickle as cPickle
from tigramite import data_processing as pp
from matplotlib import pyplot as plt
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
#from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS 
from tigramite.independence_tests.gpdc import GPDC
import pandas as pd

# Default communicator
COMM = MPI.COMM_WORLD

rank = COMM.Get_rank()
size = COMM.Get_size()

print("Anzahl Prozesse: ", size)
print("Ich bin Prozess Nr. ", rank)


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    #return [container[_i::count] for i in range(count)]
    return [container[i::count] for i in range(count)]

def run_pc_stable_parallel(j, dataframe, cond_ind_test, params):
    """Wrapper around PCMCI.run_pc_stable estimating the parents for a single 
    variable j.

    Parameters
    ----------
    j : int
        Variable index.

    Returns
    -------
    j, pcmci_of_j, parents_of_j : tuple
        Variable index, PCMCI object, and parents of j
    """

    N = dataframe.values[0].shape[1]

    # CondIndTest is initialized globally below
    # Further parameters of PCMCI as described in the documentation can be
    # supplied here:
    pcmci_of_j = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        #selected_variables=[j],
        #var_names=var_names,
        verbosity=verbosity)

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    tempDict = {}


    parents_of_j={}


    #calculate only on links for variable j
    s_l_assumptions=pcmci_of_j._set_link_assumptions(None, params['tau_min'], params['tau_max'])
    filtered_dict = {key: value for key, value in s_l_assumptions.items() if key == j}
    rest_of_keys = [key for key in s_l_assumptions if key != j]
    filtered_dict.update({key: {} for key in rest_of_keys})

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    parents_of_j_temp = pcmci_of_j.run_pc_stable(
        link_assumptions=filtered_dict,
        tau_min=params['tau_min'],
        tau_max=params['tau_max'],
        pc_alpha=params['pc_alpha'],
    )
    parents_of_j[j]=parents_of_j_temp[j]
    

    

    
    

    # We return also the PCMCI object because it may contain pre-computed
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j


def run_mci_parallel(j, pcmci_of_j, all_parents, params, link_assumptions_j):
    """Wrapper around PCMCI.run_mci step.

    Parameters
    ----------
    j : int
        Variable index.

    pcmci_of_j : object
        PCMCI object for variable j. This may contain pre-computed results 
        (such as residuals or null distributions).

    all_parents : dict
        Dictionary of parents for all variables. Needed for MCI independence
        tests.

    Returns
    -------
    j, results_in_j : tuple
        Variable index and results dictionary containing val_matrix, p_matrix,
        and optionally conf_matrix with non-zero entries only for
        matrix[:,j,:].
    """
    #print(params['selected_links'],params['tau_min'],params['tau_max'],params['max_conds_px'])
    parentsdict_of_j={}
    linkassumptionsDict_of_j={}
    parentsdict_of_j[j] = all_parents[j]
    results_in_j = pcmci_of_j.run_mci(
        selected_links=params['selected_links'],
        tau_min=params['tau_min'],
        tau_max=params['tau_max'],
        parents=all_parents,
        max_conds_px=params['max_conds_px'],
        link_assumptions=link_assumptions_j,

        )
   
    print("results_in_j for j = ", j, results_in_j)
    return j, results_in_j


# JAKOB: Based on the full model time available, we chunk up the time axis into
# as many periods of length "length" (in years) we can fit into the full model time
n_comps=100 #number of max components in the PCA-Varimax (different from number of components kept)
n_var_list = [100] #list of number of used components to use
pc_alpha_list=[1e-40, 1e-41, 1e-42, 1e-43, 1e-44, 1e-45] #list of pc_alpha values to use
verbosity = 0 


for n_VAR in n_var_list:
    for pc_alpha in pc_alpha_list:

        model_path = "./c-transfer/" #path to model and reanalysis data

        model_filename_list = sorted([
            os.path.basename(file_path) for file_path in glob.glob(
                os.path.join(model_path,'*.bin'))
        ])

        print(len(model_filename_list))
        #print((sys.argv[1]))
        #print((sys.argv[2]))
        plot=True
        selected_components = []

        for i in range(1, n_VAR+1):
            selected_components.append('c' + str(i))
        print(selected_components)

        if len(sys.argv) > 1 :
            first_model_idx = 0#int(sys.argv[1])
            last_model_idx = 34#int(sys.argv[2])
        else :
            first_model_idx = 0
            last_model_idx = len(model_filename_list)

        for model in model_filename_list[first_model_idx:last_model_idx]:   
            for method_arg in ['pcmci']:

                print("Setup %s %s" % (model, method_arg))
                file_name = model_path + model

                datadict = cPickle.load(open(file_name, 'rb'))

                d = datadict['results']
                time_mask = d['time_mask']
                dateseries = d['time'][:]
                fulldata = d['ts_unmasked']
                N = fulldata.shape[1]
                fulldata_mask = numpy.repeat(time_mask.reshape(len(d['time']), 1),
                                            N,
                                            axis=1)

                print("Fulldata shape = %s" % str(fulldata.shape))
                print("Fulldata masked shape = %s" % str(fulldata_mask.shape))
                print("Unmasked samples %d" % (fulldata_mask[:, 0] == False).sum())


                if n_comps<51:
                    if "[12, 1, 2]" in model:
                        selected_comps_file="./selected_comps_NCEP_djf.csv"
                    elif "[3, 4, 5]" in model:
                        selected_comps_file="PATH TO SELECTED COMPS/selected_comps_NCEP_mam.csv"
                    elif "[6, 7, 8]" in model:
                        selected_comps_file="./selected_comps_NCEP_jja.csv"
                    elif "[9, 10, 11]" in model:
                        selected_comps_file="PATH TO SELECTED COMPS/selected_comps_NCEP_son.csv"
                    else : continue
                    comps_csv = pd.read_csv(selected_comps_file)
                    selected_comps_indices=[]
                    for i in range(len(selected_components)):
                        selected_comps_indices.append(int(comps_csv["comps"][i]))
                else:
                    selected_comps_indices=range(0, n_VAR)

                fulldata = fulldata[:, selected_comps_indices] #first dimension is number of days
                fulldata_mask = fulldata_mask[:, selected_comps_indices]
                time_bin= 3
                print("Aggregating data to time_bin_length=%s" %time_bin)
                fulldata = pp.time_bin_with_mask(fulldata, time_bin_length=time_bin)[0]
                fulldata_mask = pp.time_bin_with_mask(fulldata_mask, time_bin_length=time_bin)[0] > 0.
                dataframe = pp.DataFrame(fulldata, mask=fulldata_mask)
                print("Fulldata shape after binning= %s" % str(dataframe.values[0].shape))
                print("Unmasked samples %d" % (dataframe.mask[0][:, 0] == False).sum())
                T, N = dataframe.values[0].shape
                print("selectedCompsI", selected_comps_indices)
                resdict = {
                    "CI_params": {
                        'significance': 'analytic',
                        'mask_type': ['y'],
                        'recycle_residuals': False,
                    },
                    "PC_params": {
                        'pc_alpha': pc_alpha, #0.2
                        'tau_min': 0,
                        'tau_max': 10,
                        'max_conds_dim': None,
                        # Selected links may be used to restricted estimation to given links.
                        'selected_links': None,
                        'selected_variables': range(N),  #selected_comps_indices,
                        # Optionalonally specify variable names
                        # 'var_names':range(N),
                        'var_names': selected_comps_indices,
                    },
                    "MCI_params": {
                        # Minimum time lag (can also be 0)
                        'tau_min': 0,
                        # Maximum time lag
                        'tau_max': 10,
                        # Maximum number of parents of X to condition on in MCI step, leave this to None
                        # to condition on all estimated parents.
                        'max_conds_px': None,
                        # Selected links may be used to restricted estimation to given links.
                        'selected_links': None,
                        # Alpha level for MCI tests (just used for printing since all p-values are
                        # stored anyway)
                        'alpha_level': 0.05,
                    }
                }
                # Chosen conditional independence test
                cond_ind_test = ParCorr(verbosity=verbosity, **resdict['CI_params'])
                # Store results in file

                folder_name = './output/gridSearchTemp/alpha={}_nVAR={}'.format(resdict['PC_params']['pc_alpha'], n_VAR)  #set output folder name              
                if COMM.rank==0:
                    if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                COMM.barrier()
                file_name = folder_name + '/results_%s_%d-VAR_%d-LAG_%s.bin' % (
                    model, n_VAR, resdict["MCI_params"]["tau_max"], method_arg)
                print("output filename %s" % file_name)
                if os.path.isfile(file_name):#if output file already exists skip current iteration
                    print("Skipping current model %s as result dict already exists" %model)
                    continue

                if COMM.rank == 0:
                    # Only the master node (rank=0) runs this
                    if verbosity > -1:
                        print(
                            "\n##\n## Running Parallelized Tigramite PC algorithm\n##"
                            "\n\nParameters:")
                        print("\nindependence test = %s" % cond_ind_test.measure +
                            "\ntau_min = %d" % resdict['PC_params']['tau_min'] +
                            "\ntau_max = %d" % resdict['PC_params']['tau_max'] +
                            "\npc_alpha = %s" % resdict['PC_params']['pc_alpha'] +
                            "\nmax_conds_dim = %s" %
                            resdict['PC_params']['max_conds_dim'])
                        print("\n")

                    # Split selected_variables into however many cores are available.
                    splitted_jobs = split(resdict['PC_params']['selected_variables'],
                                        COMM.size)
                    if verbosity > -1:
                        print("Splitted selected_variables = "), splitted_jobs
                else:
                    splitted_jobs = None

                ##
                ##  PC algo condition-selection step
                ##
                # Scatter jobs across cores.
                scattered_jobs = COMM.scatter(splitted_jobs, root=0)

                print("\nCPU %d estimates parents of %s" % (COMM.rank, scattered_jobs))

                # Now each rank just does its jobs and collects everything in a results list.
                results = []
                time_start = time.time()
                for j_index, j in enumerate(scattered_jobs):
                    print("Running PC on %d -th Variable" %j)
                    # Estimate conditions
                    (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(
                        j, dataframe, cond_ind_test, params=resdict['PC_params'])

                    results.append((j, pcmci_of_j, parents_of_j))

                    num_here = len(scattered_jobs)
                    current_runtime = (time.time() - time_start) / 3600.
                    current_runtime_hr = int(current_runtime)
                    current_runtime_min = 60. * (current_runtime % 1.)
                    estimated_runtime = current_runtime * num_here / (j_index + 1.)
                    estimated_runtime_hr = int(estimated_runtime)
                    estimated_runtime_min = 60. * (estimated_runtime % 1.)

                # Gather results on rank 0.
                results = MPI.COMM_WORLD.gather(results, root=0)

                if COMM.rank == 0:
                    # Collect all results in dictionaries and send results to workers
                    all_parents = {}
                    pcmci_objects = {}
                    for res in results:
                        for (j, pcmci_of_j, parents_of_j) in res:
                            all_parents[j] = parents_of_j[j]
                            pcmci_objects[j] = pcmci_of_j
                    #print(pcmci_objects[0].__dict__.keys())

                    if verbosity > -1:
                        print(
                            "\n##\n## Running Parallelized Tigramite MCI algorithm\n##"
                            "\n\nParameters:")

                        print("\nindependence test = %s" % cond_ind_test.measure +
                            "\ntau_min = %d" % resdict['MCI_params']['tau_min'] +
                            "\ntau_max = %d" % resdict['MCI_params']['tau_max'] +
                            "\nmax_conds_px = %s" %
                            resdict['MCI_params']['max_conds_px'])
                        print(
                            "Master node: Sending all_parents and pcmci_objects to workers."
                        )

                    for i in range(1, COMM.size):
                        COMM.send((all_parents, pcmci_objects), dest=i)

                else:
                    if verbosity > -1:
                        print(
                            "Slave node %d: Receiving all_parents and pcmci_objects..."
                            "" % COMM.rank)
                        (all_parents, pcmci_objects) = COMM.recv(source=0)

                ##
                ##   MCI step
                ##
                # Scatter jobs again across cores.
                scattered_jobs = COMM.scatter(splitted_jobs, root=0)
                # Now each rank just does its jobs and collects everything in a results list.
                results = []
                for j_index, j in enumerate(scattered_jobs):
                    print("\n\t# Variable %s (%d/%d)" % (selected_comps_indices[j], j_index+1, len(scattered_jobs)))
                    s_l_assumptions=pcmci_objects[j]._set_link_assumptions(None, resdict['MCI_params']['tau_min'], resdict['MCI_params']['tau_max'])
                    filtered_dict = {key: value for key, value in s_l_assumptions.items() if key == j}
                    rest_of_keys = [key for key in s_l_assumptions if key != j]
                    filtered_dict.update({key: {} for key in rest_of_keys})

                    (j, results_in_j) = run_mci_parallel(j,
                                                        pcmci_objects[j],
                                                        all_parents,
                                                        params=resdict['MCI_params'], link_assumptions_j = filtered_dict)
                    results.append((j, results_in_j))

                    num_here = len(scattered_jobs)
                    current_runtime = (time.time() - time_start) / 3600.
                    current_runtime_hr = int(current_runtime)
                    current_runtime_min = 60. * (current_runtime % 1.)
                    estimated_runtime = current_runtime * num_here / (j_index + 1.)
                    estimated_runtime_hr = int(estimated_runtime)
                    estimated_runtime_min = 60. * (estimated_runtime % 1.)

                # Gather results on rank 0.
                results = MPI.COMM_WORLD.gather(results, root=0)

                if COMM.rank == 0:
                    # Collect all results in dictionaries
                    #
                    if verbosity > -1:
                        print("\nCollecting results...")
                    all_results = {}
                    for res in results:
                        for (j, results_in_j) in res:
                            for key in ['p_matrix', 'val_matrix', 'conf_matrix']: #results_in_j.keys():
                                if results_in_j[key] is None:
                                    all_results[key] = None
                                else:
                                    if key not in all_results.keys():
                                        if key == 'p_matrix':
                                            all_results[key] = numpy.ones(
                                                results_in_j[key].shape)
                                        else:
                                            all_results[key] = numpy.zeros(
                                                results_in_j[key].shape)
                                        all_results[key][:, j, :] = results_in_j[
                                            key][:, j, :]
                                    else:
                                        all_results[key][:, j, :] = results_in_j[
                                            key][:, j, :]

                    p_matrix = all_results['p_matrix']
                    val_matrix = all_results['val_matrix']
                    conf_matrix = all_results['conf_matrix']

                    sig_links = (p_matrix <= resdict['MCI_params']['alpha_level'])  #not relevant here

                    if verbosity > -1:
                        print("\n## Significant links at alpha = %s:" %
                            resdict['MCI_params']['alpha_level'])
                        for j in resdict['PC_params']['selected_variables']:

                            links = dict([
                                ((p[0], -p[1]),
                                numpy.abs(val_matrix[p[0], j, abs(p[1])]))
                                for p in zip(*numpy.where(sig_links[:, j, :]))
                            ])

                            # Sort by value
                            sorted_links = sorted(links, key=links.get, reverse=True)

                            n_links = len(links)

                            string = ""
                            string = ("\n    Variable %s has %d "
                                    "link(s):" %
                                    (resdict['PC_params']['var_names'][j], n_links))
                            for p in sorted_links:
                                string += ("\n        (%s %d): pval = %.5f" %
                                        (resdict['PC_params']['var_names'][p[0]],
                                            p[1], p_matrix[p[0], j, abs(p[1])]))

                                string += " | val = %.3f" % (
                                    val_matrix[p[0], j, abs(p[1])])

                                if conf_matrix is not None:
                                    string += " | conf = (%.3f, %.3f)" % (
                                        conf_matrix[p[0], j, abs(p[1])][0],
                                        conf_matrix[p[0], j, abs(p[1])][1])

                                    print(string)

                    if verbosity > -1:
                        print("Pickling to %s" %file_name)
                    resdict['file_name']= model
                    resdict['results'] = all_results
                    file = open(file_name, 'wb')
                    cPickle.dump(resdict, file, protocol=-1)
                    file.close()
                    