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
import networkx as nx
from scipy.stats import entropy
import numpy.linalg as LA





def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    #return [container[_i::count] for i in range(count)]
    return [container[i::count] for i in range(count)]


def get_metric_f1_v1(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, 
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
                            FP += 3
                        elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                            count +=1
                            if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                TP += 1
                            elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                FN += 3
                            elif same_sign==False:
                                TP += 1
                        elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                            FN += 3
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    print("precision, recall, TP, FP, FN", precision, recall, TP, FP, FN)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count



def get_metric_f1_v1_penalty(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, penalty,
            tau_min=0, tau_diff=1, same_sign=True): #F1 score with penalty term
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
                    if tau==0:
                        if i<j:
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
                            FP += penalty
                        elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                            count +=1
                            if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                TP += 1
                            elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                FN += penalty
                            elif same_sign==False:
                                TP += 1
                        elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                            FN += penalty
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    #print("precision, recall, TP, FP, FN", precision, recall, TP, FP, FN)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count

def get_metric_f1_v1_square(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, 
            tau_min=0, tau_diff=1, same_sign=True): #return squared F1
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
                    if tau==0:
                        if i<j:
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
    #print("precision, recall, TP, FP, FN", precision, recall, TP, FP, FN)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1**2, auto, count

def get_metric_f1_v2(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, 
            tau_min=0, tau_diff=1, same_sign=True): #modified TP,FN,FP
    tau_min=1
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
                    if tau==0: #not used and thus not modified
                        if i<j:   #math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau]))   #abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2
                            if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                                FP += math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau]))
                            elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                                count +=1
                                if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                    TP += 1-abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2 #likeRelu1(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])   #modifed 1-sigmoid(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])), 1-f(sigmoid(g(x)) bzw. streckung+verschiebung
                                elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]): 
                                    FN += math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])) #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[i,j,tau]) #gleich wie oben, aber ohne 1-
                                elif same_sign==False:
                                    TP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[j,i,max(0,tau-tau_diff):tau+tau_diff+1] < alpha): 
                                count +=1
                                if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[j,i,tau]):
                                    TP += abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2 #likeRelu1(ref_val_matrix[i,j,tau], val_matrix[j,i,tau])
                                elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[j,i,tau]):
                                    FN +=  math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])) #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[j,i,tau])
                                elif same_sign==False:
                                    TP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                FN += math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau]))
                            elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[j,i,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                FN += math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau]))
                    else:
                        if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                            FP +=  abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2
                        elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                            count +=1
                            if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                TP += 1-abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2# likeRelu1(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])
                            elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                FN +=  abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2 #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])
                            elif same_sign==False:
                                TP += 1
                        elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                            FN +=  abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2 #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count


def get_metric_f1_v2_strict(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, 
            tau_min=0, tau_diff=1, same_sign=True): #modified TP
    tau_min=1
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
                    if tau==0: #same as above
                        if i<j:   #math.sqrt(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau]))   #abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2
                            if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                                FP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha): 
                                count +=1
                                if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                    TP += 1-abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2 #likeRelu1(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])   #modifed 1-sigmoid(abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])), 1-f(sigmoid(g(x)) bzw. streckung+verschiebung
                                elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]): 
                                    FN += 1 #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[i,j,tau]) #gleich wie oben, aber ohne 1-
                                elif same_sign==False:
                                    TP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[j,i,max(0,tau-tau_diff):tau+tau_diff+1] < alpha): 
                                count +=1
                                if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[j,i,tau]):
                                    TP += abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2 #likeRelu1(ref_val_matrix[i,j,tau], val_matrix[j,i,tau])
                                elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[j,i,tau]):
                                    FN += 1 #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[j,i,tau])
                                elif same_sign==False:
                                    TP += 1
                            elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                FN += 1
                            elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[j,i,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                                FN += 1
                    else:
                        if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                            FP +=  1
                        elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                            count +=1
                            if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                                TP += 1-abs(ref_val_matrix[i,j,tau] - val_matrix[i,j,tau])**2# likeRelu1(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])
                            elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                                FN +=  1 #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])
                            elif same_sign==False:
                                TP += 1
                        elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                            FN +=  1 #likeRelu2(ref_val_matrix[i,j,tau], val_matrix[i,j,tau])
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count

def edgeKernel_getVector(refGraph_mat, refVal_mat, graph_mat, val_mat):
    refGraph_mat_np = np.array(refGraph_mat)
    refVal_mat_np=np.array(refVal_mat)
    refGraph_mat_flat=refGraph_mat_np.flatten()
    refVal_mat_flat=refVal_mat_np.flatten()
    refVal_mat_flat[refGraph_mat_flat == ''] = 0.0


    graph_mat_np = np.array(graph_mat)
    val_mat_np=np.array(val_mat)
    graph_mat_flat=graph_mat_np.flatten()
    val_mat_flat=val_mat_np.flatten()
    val_mat_flat[graph_mat_flat == ''] = 0.0



    dot_product = np.dot(refVal_mat_flat, val_mat_flat)
    
    norm_a = np.linalg.norm(refVal_mat_flat)
    norm_b = np.linalg.norm(val_mat_flat)
    
    cos_theta = dot_product / (norm_a * norm_b)
    
    return cos_theta


def val_mat_sum1_2(ref_val_mat, ref_graph_mat, val_mat, graph_mat, pnlty):
    penalty = 0
    sum = 0
    for i in range(0,100):
        for j in range(0,100):
            for tau in range(0,11): #here tauMin=0
                if i<j:
                    if ref_graph_mat[i][j][tau] != graph_mat[i][j][tau]:        
                        penalty += pnlty
                    else:
                        sum += (2-abs(ref_val_mat[i][j][tau] - val_mat[i][j][tau])) 
                                                                                 
    return sum - penalty


def p_mat_l2_distance(ref_p_matrix, p_matrix): #p_mat l2 distance without sqrt
    sum=0
    for i in range(len(ref_p_matrix)):
        for j in range(len(ref_p_matrix[0])):
            for tau in range(1, len(ref_p_matrix[0][0])):
                sum += 1-(ref_p_matrix[i][j][tau]-p_matrix[i][j][tau])**2

    return sum


def val_mat_l2_distance(ref_val_matrix, val_matrix): #val_mat l2 distance without sqrt
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])): #here tauMin=1
                sum += 1-(ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return sum 


def val_mat_l2_distance_correct1(ref_val_matrix, val_matrix): #val_mat l2 distance
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return math.sqrt(sum) 

def val_mat_l2_distance_correct1xgraphMat(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix): #val_mat x graph_mat L2 distance
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                if ref_graph_matrix[i][j][tau] != graph_matrix[i][j][tau]:  #if no same link entry then maximum distance
                    sum +=4
                else:
                    sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return math.sqrt(sum) 

def val_mat_l2_distance_correct1xgraphMatv2(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix, penalty):
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                if ref_graph_matrix[i][j][tau] != graph_matrix[i][j][tau]:
                    sum +=penalty #same as above but with penalty term which can be varied
                else:
                    sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return math.sqrt(sum) 

def val_mat_l2_distance_correct1xgraphMatv3(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix, penalty):
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                if ref_graph_matrix[i][j][tau] != graph_matrix[i][j][tau]:
                    sum +=penalty
                elif ref_graph_matrix[i][j][tau] == '-->' and (ref_graph_matrix[i][j][tau] == graph_matrix[i][j][tau]): #same as above but only calculate on LINKS and NOT LINK ENTRIES
                    sum += (ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2

    return math.sqrt(sum) 


def val_mat_l2_distance_correct2(ref_val_matrix, val_matrix): #some other L2 norm interpretation...
    sum1=0
    sum2=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                sum1 += (ref_val_matrix[i][j][tau])**2
                sum2 += (val_matrix[i][j][tau])**2

    return math.sqrt(sum1)+math.sqrt(sum2)





def val_matxgraph_mat_l2_distance(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix): #modified L2 without sqrt
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                if ref_graph_matrix[i][j][tau] == graph_matrix[i][j][tau]:
                    sum += 1-(ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2   #large difference also acts like penalty
                else:
                    sum -= 1


    return sum 

def val_matxgraph_mat_l2_distance_noPenalty(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix): #modified sqrt without sqrt
    sum=0
    for i in range(len(ref_val_matrix)):
        for j in range(len(ref_val_matrix[0])):
            for tau in range(1, len(ref_val_matrix[0][0])):
                if ref_graph_matrix[i][j][tau] == graph_matrix[i][j][tau]:
                    sum += 1-(ref_val_matrix[i][j][tau]-val_matrix[i][j][tau])**2   #large difference also acts like penalty


    return sum 




def val_mat_sum2(ref_val_mat, ref_graph_mat, val_mat, graph_mat): 
    sum = 0
    for i in range(0,100):
        for j in range(0,100):
            for tau in range(0,11):
                if ref_graph_mat[i][j][tau] != graph_mat[i][j][tau]:
                    sum += (1-abs(ref_val_mat[i][j][tau] - val_mat[i][j][tau]))  

    return sum

def l1_distance(ref_val_mat, ref_graph_mat, val_mat, graph_mat): 
    sum = 0
    for i in range(0,100):
        for j in range(0,100):
            for tau in range(1,11):
                sum += (1-abs(ref_val_mat[i][j][tau] - val_mat[i][j][tau]))  

    return sum


def val_mat_sum_L2(ref_val_mat, ref_graph_mat, val_mat, graph_mat): #l2 norm with taumin=0
    sum = 0
    for i in range(0,100):
        for j in range(0,100):
            for tau in range(0,11):
                sum+=math.sqrt(ref_val_mat[i][j][tau] - val_mat[i][j][tau])      
    return sum   


def val_mat_sum4(ref_val_mat, ref_graph_mat, val_mat, graph_mat):
    sum = 0
    for i in range(0,100):
        for j in range(0,100):
            for tau in range(0,11):
                if ref_graph_mat[i][j][tau] != '':
                    sum += (1-abs(ref_val_mat[i][j][tau] - val_mat[i][j][tau]))     
    return sum


def val_mat_sum5(ref_val_mat, ref_graph_mat, val_mat, graph_mat):
    sum = 0
    for i in range(0,100):
        for j in range(0,100):
            for tau in range(0,11):
                if graph_mat[i][j][tau] != '':
                    sum += (1-abs(ref_val_mat[i][j][tau] - val_mat[i][j][tau])) 
    return sum





    
def maximumCommonSubgraph(refGraph, graph): #invers formulieren "maximum subgraph verschieden"
    refGraphNx = getNxGraph()
    graphNx = getNxGraph()
    for (i,j,k) in np.ndindex(refGraph.shape):
        if refGraph[i,j,k] != '':
            refGraphNx.add_edge((i,0), (j,k))
        if graph[i,j,k] != '':
            graphNx.add_edge((i,0), (j,k))

    print("number of edges start", refGraphNx.number_of_edges(), graphNx.number_of_edges())

    G1=refGraphNx
    G2=graphNx
    mcs = nx.DiGraph()

    for n1 in G1.nodes():
        for n2 in G2.nodes():
            if G1.out_degree(n1) == G2.out_degree(n2):
                if all(G1.has_edge(n1, succ1) == G2.has_edge(n2, succ2) 
                       for succ1, succ2 in zip(G1.successors(n1), G2.successors(n2))):
                    mcs.add_node(n1)
                    mcs.add_node(n2)
                    for succ1, succ2 in zip(G1.successors(n1), G2.successors(n2)):
                        if G1.has_edge(n1, succ1):
                            mcs.add_edge(n1, succ1)
                            mcs.add_edge(n2, succ2)
    print("final number of edges", mcs.number_of_edges())
    return mcs.number_of_edges()


def degree_centrality(refGraph, graph):
    refGraphNx = getNxGraph()
    graphNx = getNxGraph()
    for (i,j,k) in np.ndindex(refGraph.shape):
        if refGraph[i,j,k] != '':
            refGraphNx.add_edge((i,0), (j,k))
        if graph[i,j,k] != '':
            refGraphNx.add_edge((i,0), (j,k))
    G1=refGraphNx
    G2=graphNx

    degree_centrality_G1 = nx.degree_centrality(G1)
    degree_centrality_G2 = nx.degree_centrality(G2)

    differences = []
    for node in G1.nodes():
        differences.append(abs(degree_centrality_G1[node] - degree_centrality_G2.get(node, 0)))

    average_difference = sum(differences) / len(differences)

    return average_difference


def between_centrality(refGraph, graph):
    refGraphNx = getNxGraph()
    graphNx = getNxGraph()
    for (i,j,k) in np.ndindex(refGraph.shape):
        if refGraph[i,j,k] != '':
            refGraphNx.add_edge((i,0), (j,k))
        if graph[i,j,k] != '':
            refGraphNx.add_edge((i,0), (j,k))
    G1=refGraphNx
    G2=graphNx


    between_centrality_G1 = nx.betweenness_centrality(G1)
    between_centrality_G2 = nx.betweenness_centrality(G2)

    differences = []
    for node in G1.nodes():
        differences.append(abs(between_centrality_G1[node] - between_centrality_G2.get(node, 0)))

    average_difference = sum(differences) / len(differences)

    return average_difference






#helper functions
def getNxGraph():
    graph=nx.DiGraph()
    for node in range(0,100):
        for tau in range(0,11):
            graph.add_node((node, tau))
    
    return graph


def sigmoid1(a,b): #for TP
    x=abs(a-b)
    y=1/(1+pow(math.e, -5*(x-0.5)))
    return 1-y

def sigmoid2(a,b): #for FN
    x=abs(a-b)
    y=1/(1+pow(math.e, -5*(x-0.5)))
    return y


def likeRelu1(a,b):
    x=abs(a-b)
    knick=0.05 #first 0.2
    if x < knick:
        return 1
    else:
        return 1-(1/1.8)*(x-knick)

def likeRelu2(a,b):
    x=abs(a-b)
    knick=0.05

    if x < knick:
        return 1
    else:
        return (1/1.8)*(x-knick)

def linear1(a,b):
    x=abs(a-b)
    return 1-0.5*x

def linear2(a,b):
    x=abs(a-b)
    return 0.5*x




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
                            ref_graph_matrix= link_mat_dic[season][ref_ds][ref_ds_ensemble]
                            #bis hier über alle ref_ds gehen in for loop
                            p_matrix= p_mat_dic[season][dataset][ensemble]
                            val_matrix= val_mat_dic[season][dataset][ensemble]
                            graph_matrix = link_mat_dic[season][dataset][ensemble]

                            #set desired distance measurement!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! examples below
                            #precision, recall, TP, FP, FN, score, auto, count = get_metric_f1_v1_penalty(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha_levelTest, penalty,
                            #tau_min=0, tau_diff=2, same_sign=True)
                            score=val_mat_l2_distance_correct1xgraphMatv3(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix, penalty)
                            #score=val_mat_sum1_2(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix, penalty)
                            #score=kullbackLeibler2(ref_val_matrix, ref_graph_matrix, val_matrix, graph_matrix)

                            score_list.append([season,dataset,ensemble,score])
                            score_list2.append([season,dataset,ensemble,score,ref_ds_ensemble])
                            if not ((datasetIndex==refdatasetIndex) and (i==j)):                               
                                f14d[refdatasetIndex][datasetIndex][j][i]=score #kann sein dass hier die indizes vertauscht sind

    return score_list2


















model_names=['ACCESS-CM2', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC-ES2L',
             'MPI-ESM1-2-HR', 'UKESM1-0-LL']
alphaLevel_matrices_path = "./output_matrices/gridSearch"
season= '[6, 7, 8]'




alphaLevelList=[1e-1,5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 3e-5, 7e-5, 1e-6, 3e-6, 7e-6, 1e-7, 3e-7, 7e-7, 1e-8, 5e-8, 1e-9, 1e-10, 1e-12, 1e-15, 1e-20] #mci_alpha values to use
#alphaLevelList=[1e-8, 5e-8, 1e-9, 5e-9, 1e-10, 5e-10, 1e-11, 5e-11, 1e-12, 5e-12, 1e-13, 5e-13, 1e-14, 5e-14, 1e-15, 5e-15, 1e-16, 5e-16, 1e-17, 1e-18- 1e-19, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26, 1e-27, 1e-28, 1e-29]
#alphaLevelList=[1e-20, 1e-23, 1e-24, 1e-26, 1e-27, 1e-28, 1e-29, 1e-30, 1e-31, 1e-32, 1e-33, 1e-34, 1e-35, 1e-36, 1e-37, 1e-40, 1e-45, 1e-50, 1e-55, 1e-60]
#penaltyList=[3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.15, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
#penaltyList=[0.75, 1.25, 2, 2.5, 2.75, 3.5, 4, 4.5]
penaltyList=[10, 2, 1, 0.5, 0.1, 0.01] #penalty terms to use 
pca_res_path="./c-transfer" #path to PCA varimax dimension reduced results (used data)
pcmci_res_path="./output/gridSearch" #path to FOLDERS containing PCMCI results



COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()


for penalty in penaltyList:
    for folder in os.listdir(pcmci_res_path):
        pcmci_res_path_=os.path.join(pcmci_res_path, folder)


        #pattern = r"PCalpha=([0-9.eE+-]+)_nVAR=(\d+).bin"
        pattern = r"alpha=([0-9.eE+-]+)_nVAR=(\d+)" #set pattern as required

        match = re.search(pattern, pcmci_res_path_)
        # just get information from folder name
        if match:
            pc_alpha = float(match.group(1))
            n_kept_comp = int(match.group(2))
        else:
            alpha_value=None
            n_kept_comp=None
            print(pcmci_res_path_)

        if n_kept_comp < 100:
            continue
        if pc_alpha<1e-50:
            continue
        if pc_alpha > 1e-16:
            file_name='./output_f1-scores_lag1/val_mat_l2_distance_correct1xgraphMatv3_penalty={}/PCalpha={:.20f}_nVAR={}.bin'.format(str(penalty), pc_alpha, str(n_kept_comp)) #set output folder name as required
        else:
            file_name='./output_f1-scores_lag1/val_mat_l2_distance_correct1xgraphMatv3_penalty={}/PCalpha={}_nVAR={}.bin'.format(str(penalty), str(pc_alpha), str(n_kept_comp))
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
            for res_file in glob.glob(pcmci_res_path_+"/results_*.bin"):       
                res = pickle.load(open(res_file,"rb"))
                resTest = res
                resTest1=resTest
                pc_alpha=resTest1['PC_params']['pc_alpha']              #use result keys as defined in PCMCI script
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


        #calculate F1-scores or "metric scores"
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

        COMM.Barrier()
        resDict = MPI.COMM_WORLD.gather(resDict, root=0)
        print("saving file at ", file_name)

        if COMM.rank == 0:
            result = {'pc_alpha': pc_alpha,
            'n_kept_comp': n_kept_comp,
            'f1Scores': resDict}
            file = open(file_name, 'wb')
            cPickle.dump(result, file, protocol=-1)
            file.close()

        COMM.Barrier()


        










