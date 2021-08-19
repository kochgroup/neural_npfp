from FPSim2 import FPSim2CudaEngine
import pandas as pd
import numpy as np
import os
import time
import seaborn as sns
from rdkit import Chem
from utils import *
#%%% LOAD DATA
fpce = FPSim2CudaEngine("../data/zinc_fp.h5")
db = pd.read_pickle("../data/zinc_smiles_clean.pkl")
coconut_data=pd.read_csv("../data/coconut_smiles_clean.csv")

#coconut_data.smiles=[Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in coconut_data.smiles]
#coconut_data.to_csv("Data/coconut_smiles_clean.csv", index =False)


#%%% SETUP SIMSEARCH
num_to_process=coconut_data.shape[0]
results = pd.DataFrame(np.zeros((10,3*num_to_process)))
results=results.replace(0, np.NaN)

results.columns = ["coeff", "smiles", "npl"]*num_to_process

#%% Perform simsearch 
count=0
start = time.time()
for i in range(num_to_process):
    query = coconut_data.iloc[i,0]
    sim_search_results = pd.DataFrame(fpce.similarity(query, 0.50))
    sim_search_results.mol_id= sim_search_results.mol_id-1 #minus 1 because of different starting points f ids
    sim_search_results=pd.concat([sim_search_results,db.iloc[sim_search_results.mol_id,:].reset_index(drop=True)],axis=1)
    found_hits =sim_search_results.head(10).reset_index(drop=True).iloc[:,1:4]
    results.iloc[:found_hits.shape[0],count:count+3]=found_hits
    if i%1000==0:
        print(i)
        print(start-time.time())
    count +=3
end = time.time()
results.to_pickle("../data/sim_search_nocutoff.pkl")
#%%%

results_static_original   = pd.read_pickle("../data/sim_search_results.pkl")
results_static_nocut   = pd.read_pickle("../data/sim_search_nocutoff.pkl")

no_cut = list()
original = list()
for i in range(0,results_static_nocut.shape[1],3):
    original.append(np.sum(results_static_original.iloc[:,i].notna()))
    no_cut.append(np.sum(results_static_nocut.iloc[:,i].notna()))
np.mean(np.array(no_cut) >= np.array(original))

results_static_original.iloc[:,91*3:(91*3)+3]    
results_static_nocut.iloc[:,91*3:(91*3)+3]    

mean_sim = []
mean_npl = []
for i in range(0,results.shape[1],3):
    mean_sim.append(np.nanmean(results_static_nocut.iloc[:,i]).astype(float))
    mean_npl.append(np.nanmean(results_static_nocut.iloc[:,i+2].astype(float)))


results_static=pd.DataFrame({"mean_sim":mean_sim, "mean_npl": mean_npl})

#%% 
# % Molecules found 
#np.sum(results_dynamic.mean_sim.isna())/coconut_data.shape[0]
np.sum(results_static.mean_sim.isna())/coconut_data.shape[0]


np.sum(results_dynamic.mean_sim.notna())

np.sum(results_static.mean_sim.notna())

# Average NPL for which we found molecules 
np.mean(coconut_data.iloc[results_dynamic[results_dynamic.mean_sim.isna()].index,1])
np.mean(coconut_data.iloc[results_dynamic[results_dynamic.mean_sim.notna()].index,1])
# Average NPL for which we found molecules
np.mean(coconut_data.iloc[results_static[results_static.mean_sim.isna()].index,1])
np.mean(coconut_data.iloc[results_static[results_static.mean_sim.notna()].index,1])

# How many decoys above 1
np.sum(coconut_data.iloc[results_dynamic[results_dynamic.mean_sim.notna()].index,1]>1)
np.sum(coconut_data.iloc[results_static[results_static.mean_sim.notna()].index,1]>1)


# Plot distribution of NPL scores
import seaborn as sns
sns.kdeplot(coconut_data.iloc[results_dynamic[results_dynamic.mean_sim.notna()].index,1])
sns.kdeplot(coconut_data.iloc[results_static[results_static.mean_sim.notna()].index,1])
sns.kdeplot(coconut_data.npl)


#%%
decoys_static=list(results_static[results_static.mean_sim.notna()].index)

smiles_list_static= []
for idx in decoys_static:
    smiles_list_static.append(results_static_nocut.iloc[:, (idx*3)+1].dropna().tolist())

smiles_list_static = [x for l in smiles_list_static for x in l]

smiles_list_static= list(set(smiles_list_static))
decoy_smiles_static =  [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in smiles_list_static]
decoy_smiles_static = pd.Series(decoy_smiles_static).drop_duplicates().tolist()
len(decoy_smiles_static)
#%%
coconut_smiles = coconut_data.smiles
no_np_decoys=list(set(decoy_smiles_static) - (set(decoy_smiles_static).intersection(set(coconut_data.smiles))))
len(no_np_decoys)

#%%fps = pd.read_csv("coconut_decoy.csv").iloc[:,-1]

db_check=db
db_check.index = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in db_check.smiles]
decoys_found=db_check.loc[no_np_decoys,["smiles","npl"]]
decoys_found.index = list(range(decoys_found.shape[0]))
biogenic=pd.read_csv("../data/biogenic.txt", sep="\t", header = None)

biogenic_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x.replace("@",""))) for x in biogenic.iloc[:,1]]
biogenic_smiles = pd.Series(biogenic_smiles).drop_duplicates().tolist()

decoys_found.shape
union = list(set(decoys_found.smiles).intersection(set(biogenic_smiles)))
decoys_found=decoys_found[~decoys_found.smiles.isin(union)].reset_index(drop=True)
decoys_found.to_csv("../data/decoys_notcut.csv", index = False)

del biogenic_smiles, biogenic
#%%
coconut_fingerprints["npl"] = coconut_data.npl
coconut_fingerprints["is_np"] = 1
coconut_fingerprints.to_csv("../data/coconut_fp.csv", index = False)
#%%
decoy_fingerprints = get_fingerprints(decoys_found)
decoy_fingerprints["npl"] = decoys_found.npl
decoy_fingerprints["is_np"] = 0
decoy_fingerprints.columns =[str(x) for x in decoy_fingerprints.columns]


coconut_fingerprints = pd.read_csv("../data/coconut_decoy.csv")
coconut_fingerprints = coconut_fingerprints[coconut_fingerprints.is_np==1] 


coconut_fingerprints.append(decoy_fingerprints).to_csv("../data/coconut_decoy_nocut.csv", index = False)

coconut_data.append(decoys_found).to_csv("../data/complete_smiles_nocut.csv", index =False) 
