import pandas as pd
import numpy as np
import torch
from model import *
import seaborn as sns
from matplotlib import pyplot as plt
from validation_updated import *
from prettytable import PrettyTable
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit import Chem
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.metrics.pairwise import cosine_similarity
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeteroatoms, CalcFractionCSP3
from utils import *
import yaml
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind, ttest_rel
from rdkit.Chem.Draw import MolsToGridImage
import argparse

#%%  
model_desc =  MLP(settings["aux_model"]["layers"],1 ,settings["aux_model"]["dropout"])
model_desc.load_state_dict(torch.load("../data/trained_models/npl_nonorm_64/aux_cv1.pt"))
model_desc.cuda()
model_desc.eval()


wald = pd.read_csv("../../Data/wald_compounds.smi", sep ="\t")
wald_fps = get_fingerprints(wald)

nnfp_model_desc     = (model_desc(torch.tensor(wald_fps.values, dtype =torch.float).cuda())[1].cpu().detach().flatten().numpy())
wald["ours"] = nnfp_model_desc 
np.corrcoef(wald.npl, nnfp_model_desc)

label_ll = list()
for i in range(wald.shape[0]):
     label_ll.append("Ertls Score: "+  str(np.round(wald.iloc[i,1],3)) +"\n \n" + "NN Score: " + str(np.round(wald.iloc[i,2],3)) )  

MolsToGridImage([Chem.MolFromSmiles(x) for x in wald.smiles], legends = label_ll,useSVG=True )

#%%%
model_desc =  MLP(settings["aux_model"]["layers"],1 ,settings["aux_model"]["dropout"])
model_desc.load_state_dict(torch.load("../data/trained_models/npl_nonorm_64/aux_cv1.pt"))
model_desc.cuda()
model_desc.eval()


zinc=pd.read_pickle("../data/zinc_smiles_clean.pkl")
wald = pd.read_csv("../../Data/wald_compounds.smi", sep ="\t")
wald_fps = get_fingerprints(wald)
pseudo_nnfp = (model_desc(torch.tensor(wald_fps.values, dtype =torch.float).cuda())[2].cpu().detach().clone().numpy())

sim_ll = list()
model_desc.cpu()
for i in range(0,zinc.shape[0],100000):
    print(i/100000)
    zincfp = get_fingerprints(zinc.iloc[i:(i+100000),:])  
    zincfp = model_desc(torch.tensor(zincfp.values, dtype=torch.float))[2].detach().clone().numpy()    
    sim_ll.append(cosine_similarity(pseudo_nnfp, zincfp))    

sim_zinc_pseudo_np=np.hstack(sim_ll).transpose()
sim_zinc_pseudo_np=pd.DataFrame(sim_zinc_pseudo_np)
sim_zinc_pseudo_np["smiles"] = zinc.smiles
sim_zinc_pseudo_np["npl"] = zinc.npl

sim_zinc_pseudo_np.to_csv("../results/wald_simsearch_nnfp.csv", index=False)
sim_zinc_pseudo_np["npl"] = zinc.npl
#%%
from rdkit.Chem import Draw


zinc_fph5 = FPSim2CudaEngine("../data/zinc_fp.h5")
coc_fph5 = FPSim2CudaEngine("Data/coconut.h5")

to_draw = list()
labels  = list()
for i in range(15):
    
    to_draw.append(Chem.MolFromSmiles(wald.smiles.iloc[i]))
    labels.append("Ertl NPL: "+str(wald.npl.iloc[i]))
    
    #ZINC
    found_zinc_nnfp = sim_zinc_pseudo_np.sort_values(i, ascending =False).iloc[:5,:]
    found_zinc_ecfp=zinc.iloc[(pd.DataFrame(zinc_fph5.similarity(wald.smiles.iloc[i], 0.1)).mol_id.iloc[:5].values-1),:]
    similarities_zinc=pd.DataFrame(zinc_fph5.similarity(wald.smiles.iloc[i], 0.1)).coeff.iloc[:5]
    
# =============================================================================
#     #COCONUT
#     found_coc_nnfp = sim_coc_pseudo_np.sort_values(i, ascending =False).iloc[:5,:]
#     found_coc_ecfp=coconut_smiles.iloc[(pd.DataFrame(coc_fph5.similarity(wald.smiles.iloc[i], 0.1)).mol_id.iloc[:5].values-1),:]
#     similarities_coc=pd.DataFrame(coc_fph5.similarity(wald.smiles.iloc[i], 0.1)).coeff.iloc[:5]
# =============================================================================

    for k in range(5):
        to_draw.append(Chem.MolFromSmiles(found_zinc_nnfp.smiles.iloc[k]))
        labels.append("Ertl NPL: "+str(np.round(found_zinc_nnfp.npl.iloc[k],3))+"\nCosine Similarity: "+str(np.round(found_zinc_nnfp.iloc[k,i],4)))
    to_draw.append(Chem.MolFromSmiles("CC"))
    labels.append("x")
    for k in range(5):
        to_draw.append(Chem.MolFromSmiles(found_zinc_ecfp.smiles.iloc[k]))
        labels.append("Ertl NPL: "+str(np.round(found_zinc_ecfp.npl.iloc[k],3))+"\nTanimoto Similarity: "+str(np.round(similarities_zinc.iloc[k],4)))
   
    
        
# =============================================================================
#     for k in range(5):
#         to_draw.append(Chem.MolFromSmiles(found_coc_nnfp.smiles.iloc[k]))
#         labels.append("Ertl NPL: "+str(np.round(found_coc_nnfp.npl.iloc[k],3))+"\nCosine Similarity: "+str(np.round(found_coc_nnfp.iloc[k,i],4)))
#     to_draw.append(Chem.MolFromSmiles("CC"))
#     labels.append("x")
#     for k in range(5):
#         to_draw.append(Chem.MolFromSmiles(found_coc_ecfp.smiles.iloc[k]))
#         labels.append("Ertl NPL: "+str(np.round(found_coc_ecfp.npl.iloc[k],3))+"\nTanimoto Similarity: "+str(np.round(similarities_coc.iloc[k],4)))
# 
# =============================================================================
Draw.MolsToGridImage(to_draw, molsPerRow=6, maxMols=400, legends=labels,useSVG=True)
#%%

zinc_npl = data.npl
coconut_npl = pd.read_csv("../data/precomputed_fingerprints.csv", usecols=["is_np", "npl"  ])
remove_val_mol = pd.read_pickle("../data/to_keep_molecules.pkl")
coconut_npl=coconut_npl.loc[remove_val_mol,:]
coconut_npl.reset_index(inplace =True, drop=True)
np.sum(coconut_npl.is_np==1)

sns.kdeplot(zinc_npl, shade=True)
sns.kdeplot(coconut_npl.npl[coconut_npl.is_np==0],shade=True, clip = (-10,0))
sns.kdeplot(coconut_npl.npl[coconut_npl.is_np==1],shade=True)
sns.despine()
plt.legend(labels=['ZINC - InStock', 'Decoys Included', 'Coconut'],frameon=False)
plt.xlabel("Natural Product Likeness")
plt.savefig("../results/plots/density_npl.pdf",format="pdf", dpi =300, bbox_inches='tight')
