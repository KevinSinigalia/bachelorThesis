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
#all code is copied from Tigramite run_pcmciplus, to parallelize some parts (pc_step officially paralellized, further parallelization of the collider phase (unshielded triples part only) by Kevin Sinigalia)
import time
import glob
from mpi4py import MPI  # ERROR https://stackoverflow.com/questions/36156822/error-when-starting-open-mpi-in-mpi-init-via-python #https://github.com/theislab/cellrank/issues/864
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
from copy import deepcopy
import itertools



# Default communicator
COMM = MPI.COMM_WORLD

rank = COMM.Get_rank()
size = COMM.Get_size()

# Alle Prozesse geben ihre Rangnummer und die Gesamtzahl der Prozesse aus
print("Anzahl Prozesse: ", size)
print("Ich bin Prozess Nr. ", rank)

#exit() #WICHTIG!!! ENTFERNEN, NUR ZUM TESTEN

def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    #return [container[_i::count] for i in range(count)]
    return [container[i::count] for i in range(count)]

def run_pc_stable_parallel(j):
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

    # CondIndTest is initialized globally below
    # Further parameters of PCMCI as described in the documentation can be
    # supplied here:
    pcmci_of_j = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    #LINK ASSUMPTIONS NONE GGF. ANPASSEN!!!
    s_l_assumptions=pcmci_of_j._set_link_assumptions(None, tau_min, tau_max)
    filtered_dict = {key: value for key, value in s_l_assumptions.items() if key == j}
    rest_of_keys = [key for key in s_l_assumptions if key != j]
    filtered_dict.update({key: {} for key in rest_of_keys})

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    parents_of_j = pcmci_of_j.run_pc_stable(
        link_assumptions=filtered_dict,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
    )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    #print("run pc stable parallel val_matrix", j, pcmci_of_j.val_matrix)
    return j, pcmci_of_j, parents_of_j










# JAKOB: Based on the full model time available, we chunk up the time axis into
# as many periods of length "length" (in years) we can fit into the full model time
n_comps = 100 #number of max components in the PCA-Varimax (different from number of components kept) #möglicher fehler
verbosity = 0

model_path = "./c-transfer/" #path to model and reanalysis data

model_filename_list = sorted([
    os.path.basename(file_path) for file_path in glob.glob(
        os.path.join(model_path,'*.bin'))
])

#print(len(model_filename_list))
#print((sys.argv[1]))
#print((sys.argv[2]))
plot=True
selected_components = []

n_VAR = 50# number of PCA-Variamx components to keep = normal 50
for i in range(1, n_VAR+1):
    selected_components.append('c' + str(i))
#print(selected_components)

if len(sys.argv) > 1 :
    first_model_idx = 0#int(sys.argv[1])
    last_model_idx = 34#int(sys.argv[2])
else :
    first_model_idx = 0
    last_model_idx = len(model_filename_list)

for model in model_filename_list[first_model_idx:last_model_idx]:
    for method_arg in ['pcmci']:
        start = time.time()
        startTemp=time.time()


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

        #recover selected_comps from csv file
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

        fulldata = fulldata[:, selected_comps_indices]
        fulldata_mask = fulldata_mask[:, selected_comps_indices]
        time_bin= 3
        print("Aggregating data to time_bin_length=%s" %time_bin)
        fulldata = pp.time_bin_with_mask(fulldata, time_bin_length=time_bin)[0]
        fulldata_mask = pp.time_bin_with_mask(fulldata_mask, time_bin_length=time_bin)[0] > 0.
        dataframe = pp.DataFrame(fulldata, mask=fulldata_mask)
        print("Fulldata shape after binning= %s" % str(dataframe.values[0].shape))
        print("Unmasked samples %d" % (dataframe.mask[0][:, 0] == False).sum())
        T, N = dataframe.values[0].shape
        #print("selectedCompsI", selected_comps_indices)
        
        
        
        # Significance level in condition-selection step.
        # In this parallelized version it only supports a float,
        # not a list or None. But you can can run this script
        # for different pc_alpha and then choose the optimal
        # pc_alpha as done in "_optimize_pcmciplus_alpha"
        pc_alpha = 0.02

        # Maximum time lag
        tau_max = 10

        # Optional minimum time lag
        tau_min = 0

        # PCMCIplus specific parameters (see docs)
        contemp_collider_rule='majority'
        conflict_resolution=True
        reset_lagged_links=False

        # Maximum cardinality of conditions in PC condition-selection step. The
        # recommended default choice is None to leave it unrestricted.
        max_conds_dim = None

        # Maximum number of parents of X/Y to condition on in MCI step, leave this to None
        # to condition on all estimated parents.
        max_conds_px = None
        max_conds_py = None
        max_conds_px_lagged = None

        # Selected links may be used to restricted estimation to given links.
        selected_links = None

        # Verbosity level. Note that slaves will ouput on top of each other.
        verbosity = 0

        # Chosen conditional independence test
        cond_ind_test = ParCorr()  #confidence='analytic')

        # FDR control applied to resulting p_matrix
        fdr_method = 'none'
        
        # Store results in file
        file_name = os.path.expanduser('~') + '/test_results.dat'

        # Create master PCMCI object
        pcmci_master = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0)

        _int_sel_links = pcmci_master._set_link_assumptions(selected_links, tau_min, tau_max)

        

        # Store results in file
        file_name = './output/results_%s_%d-VAR_%d-LAG_%s.bin' % (
            model, n_VAR, tau_max, method_arg)
        print("output filename %s" % file_name)
        if os.path.isfile(file_name):#if output file already exists skip current iteration
            print("Skipping current model %s as result dict already exists" %model)
            continue

        #
        #  Start of the script
        #
        if COMM.rank == 0:
        # Only the master node (rank=0) runs this
            if verbosity > -1:
                pcmci_master._print_pc_params(selected_links, tau_min, tau_max,
                            pc_alpha, max_conds_dim,
                            1)

            # Split selected_variables into however many cores are available.
            splitted_jobs = split(list(range(N)), COMM.size)
            if verbosity > -1:
                print("Splitted selected_variables = ", splitted_jobs)
        else:
                splitted_jobs = None


        ##
        #Step 1: Get a superset of lagged parents from run_pc_stable  
        ##
        # Scatter jobs across cores.
        scattered_jobs = COMM.scatter(splitted_jobs, root=0)

        # Now each rank just does its jobs and collects everything in a results list.
        results = []
        for j in scattered_jobs:
         # Estimate conditions
            (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(j)
            #print("for j in scattered_jobs", j, pcmci_of_j, parents_of_j)
            results.append((j, pcmci_of_j, parents_of_j))

        # Gather results on rank 0.
        results = MPI.COMM_WORLD.gather(results, root=0)
        results = MPI.COMM_WORLD.bcast(results, root=0)
        COMM.Barrier()
        #if COMM.rank == 0:
            # Collect all results in dictionaries and
        lagged_parents = {}
        p_matrix = numpy.ones((N, N, tau_max + 1))
        val_matrix = numpy.zeros((N, N, tau_max + 1))
            # graph = numpy.zeros((N, N, tau_max + 1), dtype='<U3')
            # graph[:] = ""
        for res in results:
            for (j, pcmci_of_j, parents_of_j) in res:
                #print("lagged_parents Festlegung ", j, parents_of_j[j])
                lagged_parents[j] = parents_of_j[j]
                p_matrix[:, j, :] = pcmci_of_j.p_matrix[:, j, :]
                val_matrix[:, j, :] = pcmci_of_j.val_matrix[:, j, :]

        if verbosity > -1:
            print("\n\n## Resulting lagged condition sets:")
            for j in [var for var in lagged_parents.keys()]:
                pcmci_master._print_parents_single(j, lagged_parents[j],
                                                   None,
                                                   None)
            #print("val Matrix line 327", val_matrix)


        if COMM.rank==0:
            if verbosity > -1:
                print("\n##\n## Step 2: PC algorithm with contemp. conditions "
                    "and MCI tests\n##"
                    "\n\nParameters:")
                if selected_links is not None:
                    print("\nselected_links = %s" % _int_sel_links)
                print("\nindependence test = %s" % cond_ind_test.measure
                    + "\ntau_min = %d" % tau_min
                    + "\ntau_max = %d" % tau_max
                    + "\npc_alpha = %s" % pc_alpha
                    + "\ncontemp_collider_rule = %s" % contemp_collider_rule
                    + "\nconflict_resolution = %s" % conflict_resolution
                    + "\nreset_lagged_links = %s" % reset_lagged_links
                    + "\nmax_conds_dim = %s" % max_conds_dim
                    + "\nmax_conds_py = %s" % max_conds_py
                    + "\nmax_conds_px = %s" % max_conds_px
                    + "\nmax_conds_px_lagged = %s" % max_conds_px_lagged
                    + "\nfdr_method = %s" % fdr_method
                        )

            # lagged_parents = all_results['lagged_parents']
            # p_matrix = all_results['p_matrix']
            # val_matrix = all_results['val_matrix']
            # graph = all_results['graph']
            # if verbosity > -1:
            #     print(all_results['graph'])


            # Set the maximum condition dimension for Y and X
        max_conds_py = pcmci_master._set_max_condition_dim(max_conds_py,
                                                tau_min, tau_max)
        max_conds_px = pcmci_master._set_max_condition_dim(max_conds_px,
                                                tau_min, tau_max)
       
       #Step2 #Step2 einfach die Methoden rausgeschrieben aber NICHT parallelisiert
        COMM.Barrier()


        
        links_for_pc=None
        if COMM.rank==0:
            endTemp=time.time()
            print('step 1 finished in {:5.3f}s'.format(endTemp-startTemp))




            startTemp=time.time()
            _int_link_assumptions = pcmci_master._set_link_assumptions(None, tau_min, tau_max) ##Link assumptions None ggf. anpassen
            if reset_lagged_links:
                # Run PCalg on full graph, ignoring that some lagged links
                # were determined as non-significant in PC1 step
                links_for_pc = deepcopy(_int_link_assumptions)
            else:
                # Run PCalg only on lagged parents found with PC1 
                # plus all contemporaneous links
                links_for_pc = {}  #deepcopy(lagged_parents)
                for j in range(n_VAR):
                    links_for_pc[j] = {}
                    for parent in lagged_parents[j]:
                        if _int_link_assumptions[j][parent] in ['-?>', '-->']:
                            links_for_pc[j][parent] = _int_link_assumptions[j][parent]

                    # Add contemporaneous links
                    for link in _int_link_assumptions[j]:
                        i, tau = link
                        link_type = _int_link_assumptions[j][link]
                        if abs(tau) == 0:
                            links_for_pc[j][(i, 0)] = link_type

            if max_conds_dim is None:
                max_conds_dim = pcmci_master.N

            max_combinations=None
            if max_combinations is None:
                max_combinations = numpy.inf

            initial_graph = pcmci_master._dict_to_graph(links_for_pc, tau_max=tau_max)

            skeleton_results = pcmci_master._pcalg_skeleton(
                initial_graph=initial_graph,
                lagged_parents=lagged_parents,
                mode='contemp_conds',
                pc_alpha=pc_alpha,
                tau_min=tau_min,
                tau_max=tau_max,
                max_conds_dim=max_conds_dim,
                max_combinations=max_combinations,
                max_conds_py=max_conds_py,
                max_conds_px=max_conds_px,
                max_conds_px_lagged=max_conds_px_lagged,
                )

            graph=skeleton_results['graph']
            # Symmetrize p_matrix and val_matrix coming from skeleton
            symmetrized_results = pcmci_master.symmetrize_p_and_val_matrix(
                                p_matrix=skeleton_results['p_matrix'], 
                                val_matrix=skeleton_results['val_matrix'], 
                                link_assumptions=links_for_pc,
                                conf_matrix=None)

            # Update p_matrix and val_matrix with values from skeleton phase
            # Contemporaneous entries (not filled in run_pc_stable lagged phase)
            p_matrix[:, :, 0] = symmetrized_results['p_matrix'][:, :, 0]
            val_matrix[:, :, 0] = symmetrized_results['val_matrix'][:, :, 0]

            # Update all entries computed in the MCI step 
            # (these are in links_for_pc); values for entries
            # that were removed in the lagged-condition phase are kept from before
            for j in range(pcmci_master.N):
                for link in links_for_pc[j]:
                    i, tau = link
                    if links_for_pc[j][link] not in ['<--', '<?-']:
                        p_matrix[i, j, abs(tau)] = symmetrized_results['p_matrix'][i, j, abs(tau)]
                        val_matrix[i, j, abs(tau)] = symmetrized_results['val_matrix'][i, j, 
                                                                    abs(tau)]

            # Optionally correct the p_matrix
            if fdr_method != 'none':
                p_matrix = pcmci_master.get_corrected_pvalues(p_matrix=p_matrix, tau_min=tau_min, 
                                                    tau_max=tau_max, 
                                                    link_assumptions=link_assumptions,
                                                    fdr_method=fdr_method)
            sepsets=skeleton_results['sepsets']
            endTemp=time.time()
            print('-------------------------------------SKELETON STEP FINISHED-----------------------------------in {:5.3f}s'.format(endTemp-startTemp))
        else:
            graph=None
            sepsets=None

        graph = MPI.COMM_WORLD.bcast(graph, root=0)
        p_matrix = MPI.COMM_WORLD.bcast(p_matrix, root=0)
        val_matrix = MPI.COMM_WORLD.bcast(val_matrix, root=0)
        sepsets = MPI.COMM_WORLD.bcast(sepsets, root=0)
        pcmci_master = MPI.COMM_WORLD.bcast(pcmci_master, root=0)            
        COMM.Barrier()

        startTemp=time.time()


       


        # Start collider phase
        # Set the maximum condition dimension for Y and X
        max_conds_py = pcmci_master._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = pcmci_master._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)

        # Now change assumed links marks
        graph[graph=='o?o'] = 'o-o'
        graph[graph=='-?>'] = '-->'
        graph[graph=='<?-'] = '<--'
        mode='contemp_conds'
        if COMM.rank==0:
            if pcmci_master.verbosity > 1:
                print("\n----------------------------")
                print("Collider orientation phase")
                print("----------------------------")
                print("\ncontemp_collider_rule = %s" % contemp_collider_rule)
                print("conflict_resolution = %s\n" % conflict_resolution)

            # Check that no middle mark '?' exists
            for (i, j, tau) in zip(*numpy.where(graph!='')):
                if graph[i,j,tau][1] != '-':
                    raise ValueError("Middle mark '?' exists!")

            # Find unshielded triples
            triples = pcmci_master._find_unshielded_triples(graph)
            

            triples_to_scatter=split(triples, COMM.size)
        else:
            triples=[]
            triples_to_scatter=None
        #scatter  unshielded triples
        v_structures = []
        ambiguous_triples = []           
        triples = COMM.scatter(triples_to_scatter, root=0)
        # Apply 'majority' or 'conservative' rule to orient colliders          
            # Compute all (contemp) subsets of potential parents of i and all 
            # subsets of potential parents of j that make i and j independent
        def subsets(s):
            if len(s) == 0: return []
            subsets = []
            for cardinality in range(len(s) + 1):
                subsets += list(itertools.combinations(s, cardinality))
            subsets = [list(sub) for sub in list(set(subsets))]
            return subsets

        # We only consider contemporaneous adjacencies because only these
        # can include the (contemp) k. Furthermore, next to adjacencies of j,
        # we only need to check adjacencies of i for tau=0
        if mode == 'contemp_conds':
            adjt = pcmci_master._get_adj_time_series_contemp(graph)
        elif mode == 'standard':
            adjt = pcmci_master._get_adj_time_series(graph)

        n_triples = len(triples)



        for ir, itaukj in enumerate(triples):
            (i, tau), k, j = itaukj

            if pcmci_master.verbosity > 1:
                pcmci_master._print_triple_info(itaukj, ir, n_triples)

            neighbor_subsets_tmp = subsets(
                [(l, taul) for (l, taul) in adjt[j]
                if not (l == i and tau == taul)])
            if tau == 0:
                # Furthermore, we only need to check contemp. adjacencies
                # of i for tau=0
                neighbor_subsets_tmp += subsets(
                    [(l, taul) for (l, taul) in adjt[i]
                    if not (l == j and taul == 0)])

            # Make unique
            neighbor_subsets = []
            for subset in neighbor_subsets_tmp:
                if subset not in neighbor_subsets:
                    neighbor_subsets.append(subset)

            n_neighbors = len(neighbor_subsets)

            if pcmci_master.verbosity > 1:
                print(
                    "    Iterate through %d condition subset(s) of "
                    "neighbors: " % n_neighbors)
                if lagged_parents is not None:
                    pcmci_master._print_pcmciplus_conditions(lagged_parents, i, j,
                                    abs(tau), max_conds_py, max_conds_px,
                                    max_conds_px_lagged)

            # Test which neighbor subsets separate i and j
            neighbor_sepsets = []
            for iss, S in enumerate(neighbor_subsets):
                val, pval, Z, dependent = pcmci_master._run_pcalg_test(graph=graph,
                        i=i, abstau=abs(tau), j=j, S=S, lagged_parents=lagged_parents, 
                        max_conds_py=max_conds_py,
                        max_conds_px=max_conds_px, max_conds_px_lagged=max_conds_px_lagged,
                        tau_max=tau_max, alpha_or_thres=pc_alpha)

                if pcmci_master.verbosity > 1:
                    pcmci_master._print_cond_info(Z=S, comb_index=iss, pval=pval,
                                        val=val)

                if not dependent: #pval > pc_alpha:
                    neighbor_sepsets += [S]

            if len(neighbor_sepsets) > 0:
                fraction = numpy.sum(
                    [(k, 0) in S for S in neighbor_sepsets]) / float(
                    len(neighbor_sepsets))

            

            if contemp_collider_rule == 'majority':                  #only works for majority!!!

                if len(neighbor_sepsets) == 0:
                    if pcmci_master.verbosity > 1:
                        print(
                            "    No separating subsets --> ambiguous "
                            "triple found")
                    ambiguous_triples.append(itaukj)
                else:
                    if fraction == 0.5:
                        if pcmci_master.verbosity > 1:
                            print(
                                "    Fraction of separating subsets "
                                "containing (%s 0) is = 0.5 --> ambiguous "
                                "triple found" % pcmci_master.var_names[k])
                        ambiguous_triples.append(itaukj)
                    elif fraction < 0.5:
                        v_structures.append(itaukj)
                        if pcmci_master.verbosity > 1:
                            print(
                                "    Fraction of separating subsets "
                                "containing (%s 0) is < 0.5 "
                                "--> collider found" % pcmci_master.var_names[k])
                        # Also delete (k, 0) from sepsets (if present)
                        if (k, 0) in sepsets[((i, tau), j)]:
                            sepsets[((i, tau), j)].remove((k, 0))
                        if tau == 0:
                            if (k, 0) in sepsets[((j, tau), i)]:
                                sepsets[((j, tau), i)].remove((k, 0))
                    elif fraction > 0.5:
                        if pcmci_master.verbosity > 1:
                            print(
                                "    Fraction of separating subsets "
                                "containing (%s 0) is > 0.5 "
                                "--> non-collider found" %
                                pcmci_master.var_names[k])
                        # Also add (k, 0) to sepsets (if not present)
                        if (k, 0) not in sepsets[((i, tau), j)]:
                            sepsets[((i, tau), j)].append((k, 0))
                        if tau == 0:
                            if (k, 0) not in sepsets[((j, tau), i)]:
                                sepsets[((j, tau), i)].append((k, 0)) # Sepsets wird nicht gegathered




        v_structures = MPI.COMM_WORLD.gather(v_structures, root=0)
        ambiguous_triples = MPI.COMM_WORLD.gather(ambiguous_triples, root=0)
        #now again sequential
        if COMM.rank==0:

            v_structures_flat = [j for sub in v_structures for j in sub]
            ambiguous_triples_flat = [j for sub in ambiguous_triples for j in sub]

            v_structures=v_structures_flat
            ambiguous=ambiguous_triples_flat



            if pcmci_master.verbosity > 1 and len(v_structures) > 0:
                print("\nOrienting links among colliders:")

            link_marker = {True:"o-o", False:"-->"}

            # Now go through list of v-structures and (optionally) detect conflicts
            oriented_links = []
            for itaukj in v_structures:
                (i, tau), k, j = itaukj

                if pcmci_master.verbosity > 1:
                    print("\n    Collider (%s % d) %s %s o-o %s:" % (
                        pcmci_master.var_names[i], tau, link_marker[
                            tau==0], pcmci_master.var_names[k],
                        pcmci_master.var_names[j]))

                if (k, j) not in oriented_links and (j, k) not in oriented_links:
                    if pcmci_master.verbosity > 1:
                        print("      Orient %s o-o %s as %s --> %s " % (
                            pcmci_master.var_names[j], pcmci_master.var_names[k], pcmci_master.var_names[j],
                            pcmci_master.var_names[k]))
                    # graph[k, j, 0] = 0
                    graph[k, j, 0] = "<--" #0
                    graph[j, k, 0] = "-->"

                    oriented_links.append((j, k))
                else:
                    if conflict_resolution is False and pcmci_master.verbosity > 1:
                        print("      Already oriented")

                if conflict_resolution:
                    if (k, j) in oriented_links:
                        if pcmci_master.verbosity > 1:
                            print(
                                "        Conflict since %s <-- %s already "
                                "oriented: Mark link as `2` in graph" % (
                                    pcmci_master.var_names[j], pcmci_master.var_names[k]))
                        graph[j, k, 0] = graph[k, j, 0] = "x-x" #2

                if tau == 0:
                    if (i, k) not in oriented_links and (
                            k, i) not in oriented_links:
                        if pcmci_master.verbosity > 1:
                            print("      Orient %s o-o %s as %s --> %s " % (
                                pcmci_master.var_names[i], pcmci_master.var_names[k],
                                pcmci_master.var_names[i], pcmci_master.var_names[k]))
                        graph[k, i, 0] = "<--" #0
                        graph[i, k, 0] = "-->"

                        oriented_links.append((i, k))
                    else:
                        if conflict_resolution is False and pcmci_master.verbosity > 1:
                            print("      Already oriented")

                    if conflict_resolution:
                        if (k, i) in oriented_links:
                            if pcmci_master.verbosity > 1:
                                print(
                                    "        Conflict since %s <-- %s already "
                                    "oriented: Mark link as `2` in graph" % (
                                        pcmci_master.var_names[i], pcmci_master.var_names[k]))
                            graph[i, k, 0] = graph[k, i, 0] = "x-x"  #2

            if pcmci_master.verbosity > 1:
                adjt = pcmci_master._get_adj_time_series(graph)
                print("\nUpdated adjacencies:")
                pcmci_master._print_parents(all_parents=adjt, val_min=None, pval_max=None)   
            endTemp=time.time()
            print('-------------------------------------COLLIDER STEP FINISHED-----------------------------------in {:5.3f}s'.format(endTemp-startTemp))

            final_graph = pcmci_master._pcmciplus_rule_orientation_phase(
                                collider_graph=graph,
                                ambiguous_triples=ambiguous_triples, 
                                conflict_resolution=conflict_resolution)

            # Store the parents in the pcmci member
            pcmci_master.all_lagged_parents = lagged_parents

            return_dict = {
                'graph': final_graph,
                'p_matrix': p_matrix,
                'val_matrix': val_matrix,
                'pc_alpha': pc_alpha,
                'tau_max': tau_max,
                'sepsets': sepsets,
                'ambiguous_triples': ambiguous_triples,
                }

            # No confidence interval estimation here
            return_dict['conf_matrix'] = None

            # Print the results
            if pcmci_master.verbosity > 0:
                pcmci_master.print_results(return_dict, alpha_level=pc_alpha)
            
            # Return the dictionary
            pcmci_master.results = return_dict     
 

 	    # Print the results
            if verbosity > -1:
                #pcmci_master.print_results(return_dict, alpha_level=pc_alpha) 
                print('')      
            # Save the dictionary
            if verbosity > -1:
                print("Pickling to ", file_name)
            file = open(file_name, 'wb')
            cPickle.dump(return_dict, file, protocol=-1)
            file.close()



            end=time.time()
            print('-------------------------------------TOTAL TIME----------------------------------in {:5.3f}s'.format(end-start))
        COMM.Barrier()     



        #------------------ONLY WORKS HERE FOR CONTEMP COLLIDER RULE 0 MAJORITY!!!-----------------