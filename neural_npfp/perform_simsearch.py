from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import time

#%%
parser = argparse.ArgumentParser(description="Generate NNFP")
parser.add_argument("input",nargs = "?",type = str,help="Path to the input file.")
parser.add_argument("-q", "--query", type= int,nargs='+',help="Index of Query to peform similarity search for.")
parser.add_argument("-d", "--drop" ,default = False,action='store_const', const=True, help="To not include the other queries in the Similarity Search  use -d")

args = parser.parse_args()


path_to_save = "/".join(args.input.split("/")[:-1])+'/simsearch_results_'+ time.strftime("%Y%m%d-%H%M%S")
os.makedirs(path_to_save)

data = pd.read_csv(args.input)
fps =data.iloc[:,2:]
for query in args.query:
    if np.sum(fps.iloc[query,:].isna())>0:
        print("Query "+str(query)+  " does not have a valid fingerpint")
        break
    
    simsearch_results=cosine_similarity(fps.iloc[query,:].values.reshape(1,-1),fps.dropna())
    simsearch_results=pd.DataFrame({"smiles": data.dropna().smiles,"nn_naturalproductscore": data.dropna().nn_naturalproductscore , "similarity":simsearch_results.reshape(-1)}).sort_values("similarity", ascending=False)
    if args.drop:
        simsearch_results.drop(args.query, axis=0).to_csv(path_to_save+"/query_"+str(query)+ ".csv")
    else:simsearch_results.drop(query, axis=0).to_csv(path_to_save+"/query_"+str(query)+ ".csv")