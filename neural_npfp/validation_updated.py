import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import random
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from sim_search_validation import *

from sklearn.metrics.pairwise import cosine_similarity, distance_metrics
from scipy.spatial.distance import pdist, jaccard
from scipy.special import expit
import torch
import os
import seaborn as sns


def task1_validation(best_model):
    #Asses the predictive accuracy of the model itslef at differentiating between NPs and synthetic
    pred_clf = []
    true_clf = []
    task1_data = pd.read_csv("../data/validation_sets/clean_task1.csv")
    task1_input=torch.tensor(task1_data.iloc[:,0:-1].values, dtype=torch.float)
    pred_clf = best_model(task1_input.cuda())[1][:,0].cpu().detach().numpy()
    true_clf = np.array(task1_data["target"].astype(int))
    pred_clf = expit(pred_clf)
    model_auc=roc_auc_score( true_clf, pred_clf)
    # Asses how well the fingerprint is good differentiating (this is how it is done in the original paper)
    nnfp  = best_model(task1_input.cuda())[2].cpu().detach().numpy()
    auc_ll = []
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(nnfp):
        sim_search=cosine_similarity(nnfp[test_index,:],nnfp[train_index,:])
        y_pred=task1_data["target"][train_index[np.argmax(sim_search,axis=1)]]   
        auc_ll.append(roc_auc_score( true_clf[test_index], y_pred))    
    fp_auc=np.mean(auc_ll)

    # Calulate AUC  from Numbers based on the paper NC-MFP
    TPR = 183/(183+14)
    FPR = 87/(113+87) 
    nc_mfp_auc=np.trapz([0,TPR,1], x=[0,FPR,1])
    return pd.DataFrame({"Model AUC":model_auc, "NNFP AUC": fp_auc, "NC_MFP AUC":nc_mfp_auc }, index =["Task1 AUC"])

#%%
def task2_validation(best_model):
    results  = []    
    task2_data = pd.read_csv("../data/validation_sets/clean_task2.csv")
    for protein in os.listdir("../data/validation_sets/clean_task2"):
        np_mfp = pd.read_csv("../data/validation_sets/clean_task2/"+protein)
        nnfp_model=best_model(torch.tensor(task2_data.iloc[:,:-2][task2_data.protein_target==protein.split(".")[0]].values,dtype=torch.float).cuda())[2].cpu().detach().numpy()
    
        nnfp ={"np_mfp":np_mfp.iloc[:,1:-2], "nnfp":pd.DataFrame(nnfp_model)}
        test_data = [np_mfp.Activity,[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2,nBits=2048) for x in task2_data.smiles[task2_data.protein_target==protein.split(".")[0]]]]
        results.append(np.mean(compare_fp(nnfp,test_data),axis=0))
        
    return results



def task2_mcc(best_model):
    mean_mcc_og = list()
    mean_mcc_nnfp = list()
    task2_data = pd.read_csv("Data/validation_sets/clean_task2.csv")
    for protein in os.listdir("Data/validation_sets/clean_task2"):
        subset_npfp = pd.read_csv("Data/validation_sets/clean_task2/"+ protein)
        target_var = np.array(subset_npfp.Activity)
        npfp = np.array(subset_npfp.iloc[:,1:-1])
        
        nnfp = best_model(torch.tensor(task2_data.iloc[:,:-2][task2_data.protein_target==protein.split(".")[0]].values,dtype=torch.float).cuda())[2].cpu().detach().numpy()
        
        
        #ogfp
        mcc = []
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in cv.split(npfp):        
            y_pred = []
            for idx in test_index:
                hit = []
                for tidx in train_index: 
                    hit.append(1-jaccard(npfp[idx], npfp[tidx]))
            
                y_pred.append(target_var[train_index[np.argmax(hit)]])
            mcc.append(matthews_corrcoef(target_var[test_index],y_pred))
        mean_mcc_og.append(np.mean(mcc))
        #nnfp
        mcc = []
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in cv.split(nnfp):        
            sim_search=cosine_similarity(nnfp[test_index,:],nnfp[train_index,:])
            y_pred=target_var[train_index[np.argmax(sim_search,axis=1)]]            
            mcc.append(matthews_corrcoef(target_var[test_index],y_pred))
        mean_mcc_nnfp.append(np.mean(mcc))
    return mean_mcc_og, mean_mcc_nnfp


