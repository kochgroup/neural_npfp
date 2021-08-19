import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools, AllChem
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint, GetMorganFingerprintAsBitVect
from rdkit import DataStructs
import os
from scipy.stats import lognorm
import time
#%%% Read in the InStock i chunks

db = pd.DataFrame(np.zeros([0,2]), columns= ["smiles","npl"])
num_to_read=int(1e+5)
for i in range(0,10000000,num_to_read):
    temp =pd.read_csv("../data/original/in-stock_NPL.smi", sep ="\t",header =None,names =["smiles", "npl"],skiprows= i, nrows =num_to_read )
    db=pd.concat([db,temp], axis=0)
    print(i)
#%%% Prepare data

#reset_index
db.reset_index(inplace=True, drop=True)

# get indices of molecules with missing npl
missing_npl_idx=db[db.npl.isna()].index

# remove molecules with missing npl
db.drop(missing_npl_idx, axis=0, inplace =True)

# reset indices
db.reset_index(inplace =True, drop=True)

# remove stereoinformation
db["smiles"] = [x.replace("@","") for x in db.smiles] 
db.drop_duplicates(subset="smiles", inplace =True)
db.reset_index(inplace =True, drop=True)
db.to_pickle("../data/zinc_smiles_clean.pkl")
#%% Same process for coconut
coconut_data = pd.read_csv("../data/original/COCONUT_DB_NPL.smi", sep="\t", header = None, names= ["smiles", "npl"])
coconut_data.dropna(inplace=True)
coconut_data.drop_duplicates("smiles")

# canonicalize
coc_smiles = []
for x in coconut_data.smiles:
    try:
        coc_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    except:
        coc_smiles.append(None)

coconut_data["smiles"] = coc_smiles
coconut_data.dropna(inplace=True)

coconut_data=coconut_data.drop_duplicates("smiles")
coconut_data.reset_index(inplace=True, drop=True)
coconut_data.to_pickle("../data/coconut_smiles_clean.pkl")



