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
#%% Colors

parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument("--input",default = "../data/trained_models/npl_nonorm_64/",const = "../data/trained_models/npl_nonorm_64/",nargs ="?",type = str,help="Path to the trained Models")

args = parser.parse_args()

my_white = "#f7f7f7"
my_grey = "#969696"
my_black = "#252525"


mfp_patch = mpatches.Patch(edgecolor = "black", facecolor = my_white,hatch ="xx", label='NC_MFP')
aux_patch = mpatches.Patch(edgecolor = "black", facecolor = my_grey, label='NP_AUX')
ae_patch = mpatches.Patch(edgecolor = "black", facecolor = my_white,hatch ="///", label='NP_AE')
baseline_patch = mpatches.Patch(edgecolor = "black", facecolor = my_white, label='Baseline')
ecfp_patch = mpatches.Patch(edgecolor = "black", facecolor = my_black, label='ECFP4')
#%% Load Data
print("Load Data \n")
fps = pd.read_csv("../data/precomputed_fingerprints.csv")
remove_val_mol = pd.read_pickle("../data/to_keep_molecules.pkl")
fps=fps.loc[remove_val_mol,:]
fps.reset_index(inplace =True, drop=True)

idx_list = list(range(fps.shape[0]))
random.seed(42)
random.shuffle(idx_list)
cv_chunks = np.array_split(idx_list,5)

val_chunks = []
train_chunks = []
for i in range(5):
    val_chunks.append( cv_chunks[i])
    train_chunks.append(np.concatenate(cv_chunks[:i]+cv_chunks[i+1:]))


#%% Load Trained Models 

model_path = args.input
settings = yaml.safe_load(open(model_path+"settings.yml", "r"))

model_aux = MLP(settings["aux_model"]["layers"],1 ,settings["aux_model"]["dropout"])
model_baseline = MLP(settings["baseline_model"]["layers"],1 ,settings["baseline_model"]["dropout"])
model_ae =FP_AE(settings["ae_model"]["layers"],1+settings["ae_model"]["with_npl"],settings["ae_model"]["dropout"])
#%% Evaluate Model performance on Validation Sets
print("Evaluate model performance...\n")
results = np.zeros([5,6])
for i in range(5):
    model_baseline.load_state_dict(torch.load(model_path+"baseline_cv"+str(i)+".pt"))
    model_baseline.eval()
    model_aux.load_state_dict(torch.load(model_path+"aux_cv"+str(i)+".pt"))
    model_aux.eval()
    model_ae.load_state_dict(torch.load(model_path+"ae_cv"+str(i)+".pt"))
    model_ae.eval()
    
    validation_set =  fps.iloc[val_chunks[i],:2048]
    np_val = fps.iloc[val_chunks[i],2048:]

    pred, nprod ,_ = model_baseline(torch.tensor(validation_set.values, dtype = torch.float))
    perc = expit(nprod.detach().clone().numpy())
    results[i,0] = np.round(roc_auc_score(np_val.iloc[:,1], perc.reshape(-1)),4)
    results[i,1]= np.round(roc_auc_score(np_val.iloc[:,1][np_val.iloc[:,0]<0], perc.reshape(-1)[np_val.iloc[:,0]<0]),4)

    pred, nprod ,_ = model_aux(torch.tensor(validation_set.values, dtype = torch.float))
    perc = expit(nprod.detach().clone().numpy())
    results[i,2] = np.round(roc_auc_score(np_val.iloc[:,1], perc.reshape(-1)),4)
    results[i,3]= np.round(roc_auc_score(np_val.iloc[:,1][np_val.iloc[:,0]<0], perc.reshape(-1)[np_val.iloc[:,0]<0]),4)

    pred, nprod ,_ = model_ae(torch.tensor(validation_set.values, dtype = torch.float))
    perc = expit(nprod[:,0].detach().clone().numpy())
    results[i,4] = np.round(roc_auc_score(np_val.iloc[:,1], perc.reshape(-1)),4)
    results[i,5]= np.round(roc_auc_score(np_val.iloc[:,1][np_val.iloc[:,0]<0], perc.reshape(-1)[np_val.iloc[:,0]<0]),4)

mean = np.round(np.mean(results,axis=0),4)
sd = np.round(np.std(results, axis=0),4)
to_print = pd.DataFrame(np.zeros([3,2]))
to_print.index = ["Baseline", "NP_AUX", "NP_AE"]
to_print.columns = ["AUC (SD)", "AUC NPL < 0 (SD)"]

for k in range(3):
    to_print.iloc[k,0] = str(mean[2*k]) + " (" + str(sd[2*k]) +")"
    to_print.iloc[k,1] = str(mean[2*k+1]) + " (" + str(sd[2*k+1]) +")"



print(to_print.to_latex())

#%% NP Validation
print("\nNP Identification Task")

res = list()
for i in range(5):
    model_baseline.load_state_dict(torch.load(model_path+"baseline_cv"+str(i)+".pt"))
    model_aux.load_state_dict(torch.load(model_path+"aux_cv"+str(i)+".pt"))
    model_ae.load_state_dict(torch.load(model_path+"ae_cv"+str(i)+".pt"))

    model_ae.cuda()
    model_ae.eval()
    model_aux.cuda()
    model_aux.eval()
    model_baseline.cuda()
    model_baseline.eval()

    aux = task1_validation(model_aux).values
    ae = task1_validation(model_ae).values
    baseline = task1_validation(model_baseline).values

    res.append(np.vstack((aux,ae,baseline)))


ttest_ind(np.stack(res)[:,2,1], np.stack(res)[:,1,1])


mean_t1 = np.mean(np.stack(res),axis=0).round(3)
sd_t1 = np.std(np.stack(res),axis=0).round(3)
to_print = pd.DataFrame(np.zeros((4,2)))
to_print.index = ["NC_MFP", "NP_AUX", "NP_AE", "Baseline"]
to_print.columns = ["Model AUC (SD)", "Fingerprint AUC (SD)"] 

for i in range(3):
    to_print.iloc[i+1,0] = str(mean_t1[i,0]) + " (" +str(sd_t1[i,0]) + ")" 
    to_print.iloc[i+1,1] = str(mean_t1[i,1]) + " (" +str(sd_t1[i,1]) + ")" 
to_print.iloc[0,0] = "-"
to_print.iloc[0,1] = mean_t1[0,2] 
print(to_print.to_latex())  
#%% Target Identification


print("\nTarget Identification Task")
auc = []
ef = []
auc_rank = []
ef_rank = []
for i in range(5):
    model_baseline.load_state_dict(torch.load(model_path+"baseline_cv"+str(i)+".pt"))
    model_aux.load_state_dict(torch.load(model_path+"aux_cv"+str(i)+".pt"))
    model_ae.load_state_dict(torch.load(model_path+"ae_cv"+str(i)+".pt"))

    model_ae.cuda()
    model_ae.eval()
    model_aux.cuda()
    model_aux.eval()
    model_baseline.cuda()
    model_baseline.eval()
    
    task2_results_baseline = task2_validation(model_baseline)
    task2_results_aux = task2_validation(model_aux)
    task2_results_ae = task2_validation(model_ae)
    
    results = np.stack(task2_results_ae)
    results_2  =np.stack(task2_results_aux)
    results_3 = np.stack(task2_results_baseline)
    
    results=np.hstack([results, results_2[:,4:8], results_3[:,4:8]])
    auc_rank.append(pd.DataFrame(results[:,[0,12,4,16,8]]).rank(axis=1, ascending = False))
    ef_rank.append(pd.DataFrame(results[:,[1,13,5,17,9]]).rank(axis=1, ascending = False))
    auc.append(pd.DataFrame(results[:,[0,12,4,16,8]]))
    ef.append(pd.DataFrame(results[:,[1,13,5,17,9]]))

auc_rank = np.stack(auc_rank)
ef_rank = np.stack(ef_rank)
mean_auc_rank = np.round(np.mean(np.mean(auc_rank,axis=0),axis=0),3)
mean_ef_rank = np.round(np.mean(np.mean(ef_rank,axis=0),axis=0),3)
sd_auc_rank = np.round(np.std(np.mean(auc_rank,axis=1),axis=0),3)
sd_ef_rank = np.round(np.std(np.mean(ef_rank,axis=1),axis=0),3)

to_print = pd.DataFrame(np.zeros([2,5]))
to_print.columns = ["NC_MFP", "NP_AUX", "NP_AE", "Baseline", "ECFP4"]
to_print.index = ["AUC (SD)", "EF 1% (SD)"]

for k in range(5):
    to_print.iloc[0,k] = str(mean_auc_rank[k]) + " (" + str(sd_auc_rank[k]) +")"
    to_print.iloc[1,k] = str(mean_ef_rank[k]) + " (" + str(sd_ef_rank[k]) +")"

print(to_print.to_latex())


auc = np.stack(auc)
ef = np.stack(ef)

mean_auc = np.mean(auc, axis=0)
se_auc =np.std(auc,axis=0)
#h_auc = se_auc * stats.t.ppf((1 + 0.95) / 2., 4)
h_auc = se_auc[:,1:-1]

mean_ef = np.mean(ef, axis=0)
np.median(mean_ef,axis=0)

se_ef =np.std(ef,axis=0)
h_ef = se_ef[:,1:-1]


mean_ef_print = np.mean(mean_ef,axis = 0 ).round(3)
mean_auc_print = np.mean(mean_auc,axis = 0 ).round(3)

sd_ef_print = np.std(mean_ef,axis = 0 ).round(3)
sd_auc_print = np.std(mean_auc,axis = 0 ).round(3)



to_print = pd.DataFrame(np.zeros([5,2]))
to_print.index = ["NC_MFP", "NP_AUX", "NP_AE", "Baseline", "ECFP4"]
to_print.columns = ["AUC (SD)", "EF 1% (SD)"]


for k in range(5):
    to_print.iloc[k,0] = str(mean_auc_print[k]) + " (" + str(sd_auc_print[k]) +")"
    to_print.iloc[k,1] = str(mean_ef_print[k]) + " (" + str(sd_ef_print[k]) +")"

print(to_print.to_latex())



#%%%

n_groups=7
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8



fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,7) )

ax[0].bar(index, mean_auc[:,0],bar_width,color = my_white,hatch ="xx", edgecolor="black")
ax[0].bar(index+bar_width*1, mean_auc[:,1],bar_width, edgecolor="black", color = my_grey,yerr = h_auc[:,0],capsize=3)
ax[0].bar(index+bar_width*2, mean_auc[:,2],bar_width, edgecolor="black", color = my_white,hatch ="///", yerr = h_auc[:,1],capsize=3)
ax[0].bar(index+bar_width*3, mean_auc[:,3],bar_width, edgecolor="black", color = my_white, yerr = h_auc[:,2],capsize=3)
ax[0].bar(index+bar_width*4, mean_auc[:,4],bar_width, edgecolor="black", color = my_black)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_ylabel("AUC")
ax[0].xaxis.set_ticks_position('bottom')
ax[0].set_xticks(index + 2*bar_width)

ax[1].bar(index, mean_ef[:,0],bar_width,color = my_white,hatch ="xx", edgecolor="black")
ax[1].bar(index+bar_width*1, mean_ef[:,1],bar_width,edgecolor="black",color = my_grey, yerr = h_ef[:,0],capsize=3)
ax[1].bar(index+bar_width*2, mean_ef[:,2],bar_width, edgecolor="black",color = my_white,hatch ="///", yerr = h_ef[:,1],capsize=3)
ax[1].bar(index+bar_width*3, mean_ef[:,3],bar_width, edgecolor="black", color = my_white, yerr = h_ef[:,2],capsize=3)
ax[1].bar(index+bar_width*4, mean_ef[:,4],bar_width, edgecolor="black", color = my_black)
ax[1].set_ylabel("EF 1%")
ax[1].xaxis.set_ticks_position('bottom')
ax[1].set_xticks(index + 2*bar_width)
ax[1].set_xticklabels( [str(x) for x in range(1,8)])
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_xlabel('Target')


ax[1].legend(handles=[mfp_patch,aux_patch,ae_patch, baseline_patch, ecfp_patch], loc="upper center", bbox_to_anchor=(.45, -0.17), ncol=5, fancybox=False, frameon=False)
fig.tight_layout(pad=2) 
plt.savefig("../results/plots/target_identification.pdf",format="pdf", dpi =300, bbox_inches='tight')

#%% t-tests task 2

np.mean(mean_ef, axis=0)
ttest_rel(mean_auc[:,3], mean_auc[:,0])


 #%% NP and Target Identification
import warnings
from FPSim2 import FPSim2CudaEngine

warnings.filterwarnings("ignore")

print("NP and Target Identification")
print("This will take some time...")


if not os.path.exists("../results/np+target/"+model_path.split("/")[-2]):
    os.makedirs("../results/np+target/"+model_path.split("/")[-2])


fp_nobinary = list()
for i in range(14):
    aux_data = pd.read_csv("../data/validation_sets/np_target_identification/smiles_target" +str(i)+".csv") 
    to_drop = aux_data[(aux_data.npl>1) & ( aux_data.np==1)].index.tolist()
    aux_data = aux_data.drop(to_drop,axis=0).reset_index(drop=True)
            
    fp_nobinary.append([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile),2,nBits=2048) for smile in aux_data.smiles])

for cv in range(5):    
    model_baseline.load_state_dict(torch.load(model_path+"baseline_cv"+str(cv)+".pt"))
    model_aux.load_state_dict(torch.load(model_path+"aux_cv"+str(cv)+".pt"))
    model_ae.load_state_dict(torch.load(model_path+"ae_cv"+str(cv)+".pt"))

    model_ae.cuda()
    model_ae.eval()
    model_aux.cuda()
    model_aux.eval()
    model_baseline.cuda()
    model_baseline.eval()
    
    
    model_ll = [model_ae, model_aux, model_baseline]
    label_ll = ["ae", "aux", "baseline"]
    
    

    for k in range(3):    
        for i in range(14):
            aux_data = pd.read_csv("../data/validation_sets/np_target_identification/smiles_target" +str(i)+".csv") 
            to_drop = aux_data[(aux_data.npl>1) & ( aux_data.np==1)].index.tolist()
            aux_data = aux_data.drop(to_drop,axis=0).reset_index(drop=True)
            fps_data = pd.read_csv("../data/validation_sets/np_target_identification/fps_target" +str(i)+".csv") 
            fps_data = fps_data.drop(to_drop,axis=0).reset_index(drop=True)
             
            nnfp_model = model_ll[k](torch.tensor(fps_data.values, dtype =torch.float).cuda())[2]
            nnfp ={"nnfp":pd.DataFrame(nnfp_model)}
            npass_active = np.where((aux_data.active==1) & (aux_data.npass==1))[0]
            results = pd.DataFrame(np.zeros([18,len(npass_active)]))
            
            true_pos_rate = list() 
            for x in range(npass_active.shape[0]):
                out=evaluate_fp(nnfp,fp_nobinary[i], aux_data.active,x)
                out=pd.concat([out, aux_data.loc[out.index].drop("smiles",axis=1)],axis=1)
            
                ordered_nnfp=out.sort_values("nnfp", ascending =False)
                true_pos_rate.append(np.sum((ordered_nnfp.target==1)& (ordered_nnfp.np==1).iloc[:ordered_nnfp.shape[0]//100])/  np.sum((ordered_nnfp.np==1).iloc[:ordered_nnfp.shape[0]//100]))
                target_list=[[x] for x in ordered_nnfp.target]
                results.iloc[0,x]=CalcAUC(target_list,0)
                results.iloc[1:3,x]=CalcEnrichment(target_list,0,[0.01, 0.025])
                results.iloc[3,x]=np.sum(np.sum(ordered_nnfp.iloc[:ordered_nnfp.shape[0]//100,[5,6]],axis=1)==2)
                results.iloc[4,x]=np.sum(ordered_nnfp.iloc[:ordered_nnfp.shape[0]//100,[6]]).values
                target_list=[[x] for x in ((ordered_nnfp.target==1) & (ordered_nnfp.np==1))]
                results.iloc[5,x]=CalcAUC(target_list,0)
                results.iloc[6:8,x]=CalcEnrichment(target_list,0,[0.01, 0.025])
                results.iloc[8,x] = np.mean(ordered_nnfp.npl.iloc[:ordered_nnfp.shape[0]//100])
                
                ordered_ecfp=out.sort_values("ECFP", ascending =False)
                target_list=[[x] for x in ordered_ecfp.target]
                results.iloc[9,x]=CalcAUC(target_list,0)
                results.iloc[10:12,x]=CalcEnrichment(target_list,0,[0.01, 0.025]) 
                results.iloc[12,x]=np.sum(np.sum(ordered_ecfp.iloc[:ordered_ecfp.shape[0]//100,[5,6]],axis=1)==2)
                results.iloc[13,x]=np.sum(ordered_ecfp.iloc[:ordered_ecfp.shape[0]//100,[6]]).values
                
                target_list=[[x] for x in ((ordered_ecfp.target==1) & (ordered_ecfp.np==1))]
                results.iloc[14,x]=CalcAUC(target_list,0)
                results.iloc[15:17,x]=CalcEnrichment(target_list,0,[0.01, 0.025])
                results.iloc[17,x] = np.mean(ordered_ecfp.npl.iloc[:ordered_ecfp.shape[0]//100])
                
            results.to_csv("../results/np+target/"+model_path.split("/")[-2]+"/"+str(label_ll[k])+"_"+str(i)+"_cv"+str(cv)+".csv",index=False)
    
warnings.filterwarnings("default")
#%%
out_aux= pd.DataFrame(np.zeros([18,14]))
out_aux.index= ["AUC", "EF1", "EF2.5", "ActiveNP", "NP", "NP AUC", "EF1 NP", "EP2.5 NP", "Mean NPL"]*2
cv_out_aux = np.zeros([5,18,14])
for cv in range(5):
    for i in range(14):
        results = pd.read_csv("../results/np+target/"+model_path.split("/")[-2]+ "/aux_"+str(i)+"_cv"+str(cv)+".csv")    
        cv_out_aux[cv,:,i]=np.mean(results,axis=1)
out_aux.iloc[:,:]=np.mean(cv_out_aux, axis=0)


out_ae=pd.DataFrame(np.zeros([18,14]))
out_ae.index= ["AUC", "EF1", "EF2.5", "ActiveNP", "NP", "NP AUC", "EF1 NP", "EP2.5 NP", "Mean NPL"]*2
cv_out_ae = np.zeros([5,18,14])
for cv in range(5):
    for i in range(14):
        results = pd.read_csv("../results/np+target/"+model_path.split("/")[-2]+ "/ae_"+str(i)+"_cv"+str(cv)+".csv")   
        cv_out_ae[cv,:,i]=np.mean(results,axis=1)
out_ae.iloc[:,:]=np.mean(cv_out_ae, axis=0)

out_baseline=pd.DataFrame(np.zeros([18,14]))
out_baseline.index= ["AUC", "EF1", "EF2.5", "ActiveNP", "NP", "NP AUC", "EF1 NP", "EP2.5 NP", "Mean NPL"]*2
cv_out_baseline= np.zeros([5,18,14])
for cv in range(5):
    for i in range(14):
        results = pd.read_csv("../results/np+target/"+model_path.split("/")[-2]+ "/baseline_"+str(i)+"_cv"+str(cv)+".csv")        
        cv_out_baseline[cv,:,i]=np.mean(results,axis=1)
out_baseline.iloc[:,:]=np.mean(cv_out_baseline, axis=0)


auc = np.stack([cv_out_aux[:,5,:], cv_out_ae[:,5,:], cv_out_baseline[:,5,:], cv_out_aux[:,14,:]])

mean_auc = np.mean(auc, axis=1).transpose()
se_auc =np.std(auc,axis=1).transpose()
h_auc = se_auc * stats.t.ppf((1 + 0.95) / 2., 4)
h_auc = se_auc[:,:-1]

ef = np.stack([cv_out_aux[:,6,:], cv_out_ae[:,6,:], cv_out_baseline[:,6,:], cv_out_aux[:,15,:]])

np.mean(np.mean(ef,axis=2), axis=1)

mean_ef = np.mean(ef, axis=1).transpose()
se_ef = np.std(ef,axis=1).transpose()

h_ef = se_ef[:,:-1]

mean_ef_print = np.mean(mean_ef,axis = 0 ).round(3)
mean_auc_print = np.mean(mean_auc,axis = 0 ).round(3)

sd_ef_print = np.std(mean_ef,axis = 0 ).round(3)
sd_auc_print = np.std(mean_auc,axis = 0 ).round(3)


to_print = pd.DataFrame(np.zeros([4,2]))
to_print.index = [ "NP_AUX", "NP_AE", "Baseline", "ECFP4"]
to_print.columns = ["AUC (SD)", "EF 1% (SD)"]


for k in range(4):
    to_print.iloc[k,0] = str(mean_auc_print[k]) + " (" + str(sd_auc_print[k]) +")"
    to_print.iloc[k,1] = str(mean_ef_print[k]) + " (" + str(sd_ef_print[k]) +")"

print(to_print.to_latex())

#%%
n_groups=14
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8



fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,7) )

ax[0].bar(index, mean_auc[:,0],bar_width, edgecolor="black", color = my_grey,yerr = h_auc[:,0],capsize=3)
ax[0].bar(index+bar_width*1, mean_auc[:,1],bar_width, edgecolor="black", color = my_white,hatch ="///", yerr = h_auc[:,1],capsize=3)
ax[0].bar(index+bar_width*2, mean_auc[:,2],bar_width, edgecolor="black", color = my_white, yerr = h_auc[:,2],capsize=3)
ax[0].bar(index+bar_width*3, mean_auc[:,3],bar_width, edgecolor="black", color = my_black)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_ylabel("AUC")
ax[0].xaxis.set_ticks_position('bottom')
ax[0].set_xticks(index + 2*bar_width)

ax[1].bar(index+bar_width*0, mean_ef[:,0],bar_width,edgecolor="black",color = my_grey, yerr = h_ef[:,0],capsize=3)
ax[1].bar(index+bar_width*1, mean_ef[:,1],bar_width, edgecolor="black",color = my_white,hatch ="///", yerr = h_ef[:,1],capsize=3)
ax[1].bar(index+bar_width*2, mean_ef[:,2],bar_width, edgecolor="black", color = my_white, yerr = h_ef[:,2],capsize=3)
ax[1].bar(index+bar_width*3, mean_ef[:,3],bar_width, edgecolor="black", color = my_black)
ax[1].set_ylabel("EF 1%")
ax[1].xaxis.set_ticks_position('bottom')
ax[1].set_xticks(index + 2*bar_width)
ax[1].set_xticklabels( [str(x) for x in range(1,15)])
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_xlabel('Target')


ax[1].legend(handles=[aux_patch,ae_patch, baseline_patch, ecfp_patch], loc="upper center", bbox_to_anchor=(.5, -0.17), ncol=5, fancybox=False, frameon=False)
fig.tight_layout(pad=2) 
plt.savefig("../results/plots/np+target_identification.pdf",format="pdf", dpi =300, bbox_inches='tight')



#%% Create Plot Comparing plot pre vs post training

i= 1 # second cv
model_baseline.load_state_dict(torch.load(model_path+"baseline_cv"+str(i)+".pt"))
model_baseline.eval()
model_baseline.cuda()
model_aux.load_state_dict(torch.load(model_path+"aux_cv"+str(i)+".pt"))
model_aux.eval()
model_aux.cuda()
model_ae.load_state_dict(torch.load(model_path+"ae_cv"+str(i)+".pt"))
model_ae.eval()


#model_untrained = MLP(settings["aux_model"]["layers"],1 ,settings["aux_model"]["dropout"])
model_untrained = FP_AE(settings["ae_model"]["layers"],1+settings["ae_model"]["with_npl"],settings["ae_model"]["dropout"])
model_untrained.cuda()
model_untrained.eval()

i=0 # first target
aux_data = pd.read_csv("../data/validation_sets/np_target_identification/smiles_target" +str(i)+".csv") 
to_drop = aux_data[(aux_data.npl>1) & ( aux_data.np==1)].index.tolist()
aux_data = aux_data.drop(to_drop,axis=0).reset_index(drop=True)
fps_data = pd.read_csv("../data/validation_sets/np_target_identification/fps_target" +str(i)+".csv") 
fps_data = fps_data.drop(to_drop,axis=0).reset_index(drop=True)
    
nnfp_model_desc = model_aux(torch.tensor(fps_data.values, dtype =torch.float).cuda())[2]
nnfp_model_base = model_baseline(torch.tensor(fps_data.values, dtype =torch.float).cuda())[2]
nnfp_untrained = model_untrained(torch.tensor(fps_data.values, dtype =torch.float).cuda())[2]

nnfp_data_desc = nnfp_model_desc.cpu().detach().numpy()
nnfp_data_base = nnfp_model_base.cpu().detach().numpy() 
nnfp_data_untrained= nnfp_untrained.cpu().detach().numpy()

ecfp=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2,nBits=2048) for x in aux_data.smiles]

ecfp_sim = [
        DataStructs.FingerprintSimilarity(ecfp[0], x) for x in ecfp
    ]

nnfp_sim_cos_desc = [cosine_similarity(nnfp_data_desc[0].reshape(1,-1), nnfp_data_desc[x].reshape(1,-1))[0][0] for x in range(nnfp_data_desc.shape[0])]
nnfp_sim_cos_base = [cosine_similarity(nnfp_data_base[0].reshape(1,-1), nnfp_data_base[x].reshape(1,-1))[0][0] for x in range(nnfp_data_base.shape[0])]
nnfp_sim_cos_untrained= [cosine_similarity(nnfp_data_untrained[0].reshape(1,-1), nnfp_data_untrained[x].reshape(1,-1))[0][0] for x in range(nnfp_data_untrained.shape[0])]

activities =np.array(aux_data.active)
nps =np.array(aux_data.np)
#%% Correlation Analysis
fig, ax =plt.subplots(2,2, figsize=(10,5), sharex= "col")


ax[0][0].plot(np.array(nnfp_sim_cos_untrained[1:])[np.where(nps[1:]==0)[0].tolist()], np.array(ecfp_sim[1:])[np.where(nps[1:]==0)[0].tolist()],"o",color =sns.color_palette("pastel")[7], markersize =4)
ax[0][0].plot(np.array(nnfp_sim_cos_untrained[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()],"o", mfc='none', color = sns.color_palette()[1] , markersize= 4)
ax[0][0].plot(np.array(nnfp_sim_cos_untrained[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()],"o", color = sns.color_palette()[2] , markersize= 4)
ax[0][0].set_xlabel("Similarity of NP_AUX Fingerprint")
ax[0][0].xaxis.set_tick_params(which='both', labelbottom=True)
ax[0][1].xaxis.set_tick_params(which='both', labelbottom=True)

ax[0][1].plot(np.array(nnfp_sim_cos_desc[1:])[np.where(nps[1:]==0)[0].tolist()], np.array(ecfp_sim[1:])[np.where(nps[1:]==0)[0].tolist()],"o",color =sns.color_palette("pastel")[7], markersize =4)
ax[0][1].plot(np.array(nnfp_sim_cos_desc[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()],"o",  mfc='none',color = sns.color_palette()[1] , markersize= 4)
ax[0][1].plot(np.array(nnfp_sim_cos_desc[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()],"o", color = sns.color_palette()[2] , markersize= 4)
ax[1][1].set_xlabel("Similarity of NP_AUX Fingerprint")

ax[1][0].plot(np.array(nnfp_sim_cos_untrained[1:])[np.where(nps[1:]==0)[0].tolist()], np.array(ecfp_sim[1:])[np.where(nps[1:]==0)[0].tolist()],"o",color =sns.color_palette("pastel")[7], markersize =4)
ax[1][0].plot(np.array(nnfp_sim_cos_untrained[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()],"o", mfc='none',color = sns.color_palette()[1] , markersize= 4)
ax[1][0].plot(np.array(nnfp_sim_cos_untrained[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()],"o", color = sns.color_palette()[2] , markersize= 4)

ax[1][1].plot(np.array(nnfp_sim_cos_base[1:])[np.where(nps[1:]==0)[0].tolist()], np.array(ecfp_sim[1:])[np.where(nps[1:]==0)[0].tolist()],"o",color =sns.color_palette("pastel")[7], markersize =4)
ax[1][1].plot(np.array(nnfp_sim_cos_base[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==0))[0].tolist()],"o", mfc='none', color = sns.color_palette()[1] , markersize= 4)
ax[1][1].plot(np.array(nnfp_sim_cos_base[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()], np.array(ecfp_sim[1:])[np.where((nps[1:]==1)& (activities[1:]==1))[0].tolist()],"o",color = sns.color_palette()[2] , markersize= 4)




ax[0][0].set_ylabel("ECFP Similarity")
ax[0][1].set_xlabel("Similarity of NP_AUX Fingerprint")
#ax[1].set_ylabel("Similarity of ECFP")
ax[1][0].set_xlabel("Similarity of Baseline Fingerprint")
ax[1][1].set_xlabel("Similarity of Baseline Fingerprint")

ax[1][0].set_ylabel("ECFP Similarity")

ax[1][1].set_ylabel("ECFP Similarity")

ax[0][1].set_ylabel("ECFP Similarity")
plt.tight_layout()

ax[0][0].set_title("Before Training")
ax[0][1].set_title("After Training")


plt.legend(labels=['Synthetic', "Inactive NP", "Active NP"],frameon=False, bbox_to_anchor=(-0.1, -0.48), fancybox=True,ncol=4,loc='lower center', prop={'size': 12})
plt.savefig("../results/plots/correlation_trainedVSuntrained.pdf",format="pdf", dpi =300, bbox_inches='tight')





#%%
print("Our Vs Ertls score on the ROR-Gamma Subset")

i=0 #first target
aux_data = pd.read_csv("../data/validation_sets/np_target_identification/smiles_target" +str(i)+".csv") 
fps_data = pd.read_csv("../data/validation_sets/np_target_identification/fps_target" +str(i)+".csv") 
model_desc =  MLP(settings["aux_model"]["layers"],1 ,settings["aux_model"]["dropout"])
model_desc.load_state_dict(torch.load("../data/trained_models/npl_nonorm_64/aux_cv0.pt"))
model_desc.cuda()
model_desc.eval()


nnfp_model_desc = (model_desc(torch.tensor(fps_data.values, dtype =torch.float).cuda())[1].cpu().detach().flatten().numpy())
#nnfp_model_desc = abs(nnfp_model_desc)-x_bar
#sns.kdeplot(nnfp_model_desc)
plt.legend(["NN Score", "Ertl Score"])

sns.scatterplot(x=aux_data.npl[aux_data.np ==0], y=nnfp_model_desc[aux_data.np ==0]  )
sns.scatterplot(x= aux_data.npl[aux_data.np ==1], y=nnfp_model_desc[aux_data.np ==1]  )
plt.xlabel("Ertl Score")
plt.ylabel("NN Score")
plt.legend(["Synthetic", "NP"])
plt.savefig("../results/plots/ror_gamma_np_ertelvsours.svg",format="svg", bbox_inches='tight')

#Compute Correlation


molwt = [ExactMolWt(Chem.MolFromSmiles(x)) for x in aux_data.smiles] 
numhetero = [CalcNumHeteroatoms(Chem.MolFromSmiles(x))/Chem.MolFromSmiles(x).GetNumAtoms() for x in aux_data.smiles] 
sp3_fraction = [CalcFractionCSP3(Chem.MolFromSmiles(x)) for x in aux_data.smiles] 
np.corrcoef(nnfp_model_desc,aux_data.npl)
correlation_comparison=pd.DataFrame(np.array([[np.corrcoef(nnfp_model_desc, molwt)[0,1],
np.corrcoef(nnfp_model_desc, numhetero)[0,1],
np.corrcoef(nnfp_model_desc, sp3_fraction)[0,1]],
[np.corrcoef(aux_data.npl, molwt)[0,1],
np.corrcoef(aux_data.npl, numhetero)[0,1],
np.corrcoef(aux_data.npl, sp3_fraction)[0,1]]]).transpose())

correlation_comparison.index= ["Molecular Weight", "Num. Heteroatoms", "Ratio SP3 Carbon"]
correlation_comparison.columns = ["Ours", "Ertl et. al."]

out_t = PrettyTable()
out_t.field_names = ["Property","Ours", "Ertl et. al."]
out_t.add_row(["Molecular Weight"]+[str(np.round(x,3)) for x in correlation_comparison.iloc[0,:]])
out_t.add_row(["Ratio Heteroatoms"]+[str(np.round(x,3)) for x in correlation_comparison.iloc[1,:]])
out_t.add_row(["Ratio SP3 Carbon"]+[str(np.round(x,3)) for x in correlation_comparison.iloc[2,:]])


print("Correlation between Properties and Natural Product Scores")
print(out_t)
to_save = out_t.get_string()
to_save= to_save.encode(encoding='UTF-8')



