import pandas as pd
import numpy as np
from rdkit import Chem
import pickle
smiles_clean  = pd.read_csv("../data/complete_smiles_nocut.csv")
npass = pd.read_csv("/nfs/home/jmenke2/Downloads/npass.txt", sep="\t")



npass_smiles = list()
for smile in npass.canonical_smiles:
    try:
        npass_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile.replace("@", ""))))
    except:
        npass_smiles.append(None)


npass["canonical_smiles"] = npass_smiles
npass=npass.dropna()
npass.reset_index(drop=True, inplace=True)
npass=npass.drop_duplicates("canonical_smiles")
npass.reset_index(drop=True, inplace=True)

dupes_droped = pd.concat([npass.canonical_smiles, smiles_clean.smiles]).drop_duplicates()
left_in_dataset=smiles_clean.iloc[dupes_droped.iloc[npass.shape[0]:].index.values.tolist(),:]
#%%
task1 = pd.read_csv("../data/validation_sets/classification_part_1.txt", sep="\t")
task1["SMARTS"] = [Chem.MolToSmiles(Chem.MolFromSmarts(x)) for x in task1["SMARTS"]] 
task2 = pd.read_csv("../data/validation_sets/clean_task2.csv")
tasks_combined = pd.concat([task1["SMARTS"], task2.smiles]).drop_duplicates()
tasks_combined = pd.Series([Chem.MolToSmiles(Chem.MolFromSmiles(smile.replace("@",""))) for smile in tasks_combined])
tasks_combined = tasks_combined.drop_duplicates()
dupes_droped = pd.concat([tasks_combined, left_in_dataset.smiles]).drop_duplicates()[tasks_combined.shape[0]:]
left_in_dataset = left_in_dataset.loc[dupes_droped.index.values.tolist(),:]


#%%

for i in range(14):
    test = pd.read_csv("../data/validation_sets/masse_data/smiles_target"+str(i)+".csv")
    dupes_droped = pd.concat([test.smiles, left_in_dataset.smiles]).drop_duplicates()
    if dupes_droped.iloc[test.shape[0]:].index.values.tolist()[0]==0:
        left_in_dataset=left_in_dataset.loc[dupes_droped.iloc[test.shape[0]:].index.values.tolist(),:]
    else: 
        print("ERROR " + str(i))
        break

#%%    
index_to_keep = left_in_dataset.index.values.tolist()    
open_file=open("../data/to_keep_molecules_nocut.pkl", "wb")
pickle.dump(index_to_keep, open_file)
open_file.close()
#%%

desc = comp_descriptors(smiles_clean.smiles)
desc.to_csv("../data/descriptors_nocut.csv",index = False)
