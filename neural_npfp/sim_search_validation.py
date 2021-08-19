from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve,roc_auc_score
from rdkit.ML.Scoring.Scoring import CalcAUC 
from rdkit.ML.Scoring.Scoring import CalcEnrichment
import pandas as pd
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

def evaluate_fp(nnfp, ecfp, target_test, querry_number):
    """
    Performs a single similarity search for a query
    """
    distance_df = pd.DataFrame(
        np.zeros([nnfp[list(nnfp)[0]].shape[0] - 1, len(nnfp) + 2]),
        index=nnfp[list(nnfp)[0]].drop(nnfp[list(nnfp)[0]].index[querry_number]).index,
        columns=list(nnfp) + ["ECFP", "target"],
    )
    for model in nnfp:
        querry = pd.DataFrame(nnfp[model].iloc[querry_number, :])
        test_compounds = nnfp[model].drop(nnfp[model].index[querry_number])
        distance_df[model] = cosine_similarity(querry.T, test_compounds)[0, :]

    fp_test = ecfp[:]
    querry_fp = fp_test[querry_number]
    fp_test.remove(querry_fp)
    distance_df["ECFP"] = [
        DataStructs.FingerprintSimilarity(querry_fp, x) for x in fp_test
    ]
    distance_df["target"] = np.array(target_test.drop(target_test.index[querry_number]))
    return distance_df

def compare_fp(data, test_data):
    
    "Evaluates the fingerprints for a given set of queries"
    input_querries=np.where(test_data[0]==1)[0].tolist()
    out_df = pd.DataFrame(np.zeros([len(input_querries), 4*(len(data)+1)]))
    names= []
    for name in data:
        names.append("AUC "+ str(name) )
        names.append("EF1 "+ str(name) )
        names.append("EF2.5 "+ str(name) )
        names.append("Similarity "+ str(name) )
    names=names+ ["AUC ECFP", "EF1 ECFP", "EF2.5 ECFP","Similarity ECFP"]
    out_df.columns= names
    for i in range(len(input_querries)):
        querry_target= input_querries[i]
        out=evaluate_fp(data, test_data[1],test_data[0],input_querries[i])  
        for col in range(len(out.columns)-1):
            ordered=out.sort_values([out.columns[col]], ascending =False)
            
            target_list=[[x] for x in ordered.target]
            out_df.iloc[i,0+(4*col)]= CalcAUC(target_list,0)
            out_df.iloc[i,(1+(4*col)):(3+(4*col))]=CalcEnrichment(target_list,0,[0.01, 0.025])
            
            
            top1=ordered.ECFP.iloc[:10][ordered.target.iloc[:10]==1]
            out_df.iloc[i,(3+(4*col))]=top1.mean()


    return(out_df)




def counting_scaff(molecules):
    x = molecules

    #Check 
    if type(x[0]) == str:
        M_Scaffs = [Chem.MolFromSmiles(smiles) for smiles in x]
        M_Scaffs =[ MurckoScaffold.GetScaffoldForMol(mols) for mols in M_Scaffs]
        
    elif type(x[0]) == (Chem.rdchem.Mol):
        #Umwandlung in Scaffolds(Murcko)
        M_Scaffs =[ MurckoScaffold.GetScaffoldForMol(mols) for mols in x]
        # else:
        print("Non readable date format, please use mol or Smiles(str)")
    #Filter, sodass jeder MurckoScaffold nur einmal vorkommt 
    M_Scaffs =[ Chem.MolToSmiles(mols) for mols in M_Scaffs]
   
    #print(M_Scaffs)
    return (len(set(M_Scaffs)))
