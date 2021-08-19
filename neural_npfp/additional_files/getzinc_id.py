import pandas as pd
import sys
import sqlite3
import urllib
import os
from multiprocessing import Pool
import numpy as np
data = pd.read_csv("../data/complete_smiles.csv")
data["is_np"] =  pd.read_csv("../data/coconut_decoy.csv", usecols=["is_np"])
data = data[data.is_np==0]

zinc_id = list()
for i, smiles in enumerate(data.smiles):
    print(i)
    try:
        zinc_id.append(
            get_zincid_from_smile(smiles)[0]
            )
    except:
        zinc_id.append(None)


zinc_fast(data.smiles.iloc[0])

a_pool = Pool(30)
result = a_pool.map(zinc_fast,data.smiles.tolist())
result = np.array(result)


np.sum(result==None)

data.loc[data.is_np==0,["ZINC_ID"]] = result
data.to_csv("../data/publish_data.csv", index =False)

check  = data.loc[data["is_np"]==0, ["smiles","ZINC_ID"]]
()
np.sum(result== None)
check[check.ZINC_ID.isna()].iloc[1,0]


a_pool = Pool(30)
result2 = a_pool.map(zinc_fast,check[check.ZINC_ID.isna()].smiles.tolist())
result2 = np.array(result2)
check.loc[check.ZINC_ID.isna(),"ZINC_ID"] =result2


np.sum(result2==None)
get_zincid_from_smile("CC1=CC(=O)C(=C(O1)C2=C(N=C(N2)C3=CC=C(O3)C4=CC=CC=C4C(=O)O)C5=CC(=CC=C5)OC)O" )

data.loc[data.is_np==0,["ZINC_ID"]] = check["ZINC_ID"]
data.iloc[425456,[0,-2]]
data.iloc[425456,0]
#%%
def zinc_fast(smile):
    try:
        a = get_zincid_from_smile(smile)[0]
            
    except:
        a = None
    return a

def get_zincid_from_smile(smile_str, backend='zinc15'):
   
    
    if backend not in {'zinc12', 'zinc15'}:
        raise ValueError("backend must be 'zinc12' or 'zinc15'")

    stripped_smile = smile_str.strip()
    encoded_smile = urllib.parse.quote(stripped_smile)

    if backend == 'zinc12':
        url_part1 = 'http://zinc12.docking.org/results?structure.smiles='
        url_part3 = '&structure.similarity=1.0'
    elif backend == 'zinc15':
        url_part1 = 'http://zinc.docking.org/substances/search/?q='
        url_part3 = ''
    else:
        raise ValueError("Backend must be 'zinc12' or 'zinc15'. "
                         "Got %s" % (backend))

    zinc_ids = []

    try:
        if sys.version_info[0] == 3:
            #smile_url = urllib.request.pathname2url(encoded_smile)
            response = urllib.request.urlopen('{}{}{}'
                                              .format(url_part1,
                                                      encoded_smile,
                                                      url_part3))
        else:
            #smile_url = urllib.pathname2url(encoded_smile)
            response = urllib.urlopen('{}{}{}'
                                      .format(url_part1,
                                              encoded_smile,
                                              url_part3))
    except urllib.error.HTTPError:
        print('Invalid SMILE string {}'.format(smile_str))
        response = []
    for line in response:
        line = line.decode(encoding='UTF-8').strip()

        if backend == 'zinc15':
            if line.startswith('<a href="/substances/ZINC'):
                line = line.split('/')[-2]
                if sys.version_info[0] == 3:
                    zinc_id = urllib.parse.unquote(line)
                else:
                    zinc_id = urllib.unquote(line)
                zinc_ids.append(str(zinc_id))
        else:
            if line.startswith('<a href="//zinc.docking.org/substance/'):
                line = line.split('</a>')[-2].split('>')[-1]
                if sys.version_info[0] == 3:
                    zinc_id = urllib.parse.unquote(line)
                else:
                    zinc_id = urllib.unquote(line)
                zinc_id = 'ZINC' + (8-len(zinc_id)) * '0' + zinc_id
                zinc_ids.append(str(zinc_id))
    return zinc_ids