import pandas as pd
import numpy as np
from torch import optim
from rdkit import Chem
import copy
from rdkit.Chem import AllChem
import random
from sklearn.preprocessing import StandardScaler
from utils import *
from model import *
import yaml
import argparse
import os
#%% load the data
parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument("--input",default = "../settings/settings.yml",const = "../settings/settings.yml",nargs ="?",type = str,help="Path to the settings file.")

args = parser.parse_args()
manual_seeds = [42,2344,87865,784,677]
settings = yaml.safe_load(open(args.input, "r"))

if not os.path.exists(settings["path_to_save"]):
    os.makedirs(settings["path_to_save"])

print("Loading data ...")
fps = pd.read_csv(settings["data"]["fp_file"])
remove_val_mol = pd.read_pickle(settings["data"]["remove_val_file"])
fps = fps.loc[remove_val_mol,:].reset_index(drop = True )

print("Please stand by ...")
desc = pd.read_csv(settings["data"]["descriptor_file"])
desc = desc.loc[remove_val_mol,:].reset_index(drop = True )


#%%

settings["aux_model"]["layers"][-1] +=1 if settings["aux_model"]["with_npl"] else 0
settings["baseline_model"]["layers"][-1] +=1 if settings["baseline_model"]["with_npl"] else 0

#%%
idx_list = list(range(fps.shape[0]))
random.seed(42)
random.shuffle(idx_list)
cv_chunks = np.array_split(idx_list,5)

val_chunks = []
train_chunks = []
for i in range(5):
    val_chunks.append( cv_chunks[i])
    train_chunks.append(np.concatenate(cv_chunks[:i]+cv_chunks[i+1:]))


for cv_fold in range(5):
    clf_targets= fps.is_np
    scaler_std = StandardScaler()
    
    # add npl socres to the regression targets 
    if (settings["aux_model"]["with_npl"]) | (settings["baseline_model"]["with_npl"]) | (settings["ae_model"]["with_npl"]):
        print("Training With NPL")
        desc["npl"] = fps.npl

    scaler_std.fit(desc.iloc[train_chunks[cv_fold],:])
    reg_targets=scaler_std.transform(desc)

    # create Datalaoder
    data=FPDataset(fps.iloc[:,:2048], reg_targets, clf_targets.values.reshape(clf_targets.shape[0],1))
    train_loader =DataLoader(torch.utils.data.Subset(data,train_chunks[cv_fold]),batch_size=settings["data"]["batch_size"])
    val_loader =DataLoader(torch.utils.data.Subset(data,val_chunks[cv_fold]), batch_size=settings["data"]["batch_size"])
    data_dict = {"train": train_loader, "val":val_loader}    


    print("Train AUX Model")
    torch.manual_seed(manual_seeds[cv_fold])
    model_to_train = train_model(model = MLP(settings["aux_model"]["layers"],
                                             1,
                                             settings["aux_model"]["dropout"]),
                                 seed=manual_seeds[cv_fold],
                                 with_npl = settings["aux_model"]["with_npl"],
                                 norm = settings["aux_model"]["norm"])
    
    best_model_desc = model_to_train.train(data=data_dict,
                                           lr=settings["aux_model"]["lr"],
                                           epochs=settings["aux_model"]["epochs"],
                                           baseline=settings["aux_model"]["baseline"],
                                           scaler_std= scaler_std)
    model_to_train.save(settings["path_to_save"]+"aux_cv"+str(cv_fold)+".pt")
    
    print("Train Baseline Model")
    torch.manual_seed(manual_seeds[cv_fold])
    model_to_train = train_model(model = MLP(settings["baseline_model"]["layers"],
                                             1,
                                             settings["baseline_model"]["dropout"]),
                                 seed = manual_seeds[cv_fold],
                                 with_npl = settings["baseline_model"]["with_npl"],
                                 norm = settings["baseline_model"]["norm"])
    
    best_model_baseline = model_to_train.train(data=data_dict,
                                               lr=settings["baseline_model"]["lr"],
                                               epochs=settings["baseline_model"]["epochs"],
                                               baseline=settings["baseline_model"]["baseline"],
                                               scaler_std= scaler_std)
    model_to_train.save(settings["path_to_save"]+"baseline_cv"+str(cv_fold)+".pt")
    
  

    data=FPAutoencoder_Dataset(fps.iloc[:,:2048], clf_targets, reg_targets[:,-1])
    train_loader =DataLoader(torch.utils.data.Subset(data,train_chunks[cv_fold]),batch_size = settings["data"]["batch_size"])
    val_loader =DataLoader(torch.utils.data.Subset(data,val_chunks[cv_fold]),batch_size = settings["data"]["batch_size"])
    data_dict = {"train": train_loader,
                 "val": val_loader}

    torch.manual_seed(manual_seeds[cv_fold])
    model_to_train = train_ae(FP_AE(settings["ae_model"]["layers"],
                                    1+settings["ae_model"]["with_npl"],
                                    settings["ae_model"]["dropout"]),
                              seed = manual_seeds[cv_fold],
                              with_npl = settings["ae_model"]["with_npl"],
                              norm = settings["ae_model"]["norm"])
    
    model_to_train.train(data_dict,
                     settings["ae_model"]["lr"],
                     epochs = settings["ae_model"]["epochs"])
    model_to_train.save(settings["path_to_save"]+"ae_cv"+str(cv_fold)+".pt")
    
with open(settings["path_to_save"]+'settings.yml', 'w') as outfile:
    yaml.dump(settings, outfile, default_flow_style=False)
    

    
    
    
    
    
    
    
    
