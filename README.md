# Neural Natural Product Fingerprint

This is the accompanying Repository for our work on "Natural Product Scores and Fingerprints Extracted from Artificial Neural Networks." https://doi.org/10.1016/j.csbj.2021.07.032

The code can be used to reproduce our results, retrain the models and compute fingerprints for your own `SMILES`-string.

# Installation

1. Download the Repository
1. Download the data at ...(only requiered for reproduction of experiments) 

   *we are still dealing with the University Hosting Service, in the mean time send an e-mail to janosch.menke [at] wwu.de* We will provide an personalized Download link
   1. extract the data.zip file into the same repository
     ```
    |- data
    |- neural_npfp
    |- results
    |- settings
    ```

1. Create and Activate Conda Environment:
    Navigate to the Folder containing the environment.yml
    ```
    conda env create -f environment.yml
    conda activate neural_npfp_env
    ```   
    Please install Pyorch with you appropriate choice of cuda.
    ```
    conda install pytorch==1.7.0 cudatoolkit=*Your Version* -c pytorch
    ```
# Data    

The `data` folder contains multiple datasets in case you are interested in just the dataset consisting of the NP and synthetic compounds from ZINC please use the `data/coconut_synthetic.csv` 
The validation data collected by us can be found in `data/validation_sets/np_target_identification/`
* `fps_targets` contain the precomputed ECFP for each target
* `smiles_targets` contains the SMILES.

The `clean_task1.csv` and `clean_task2.csv` were not created by us, we did not include the SMILES for those compounds. If you are interested in the actual compounds we refer to the original publication.

Seo, M.; Shin, H.K.; Myung, Y. et al. Development of Natural Compound Molecular Fingerprint (NC-MFP) with the Dictionary of Natural Products (DNP) for natural product-based drug development. _Journal of Cheminformatics_. **2020**, 12(6) https://doi.org/10.1186/s13321-020-0410-3


### The data was collect with the help of:

* Sterling, T.; Irwin, J. J. ZINC 15–ligand Discovery for Everyone. _Journal of Chemical Information and Modeling._ **2015**, 55, 2324–2337.

* Sorokina, M.; Merseburger, P.; Rajan, K.; Yirik, M. A.; Stein-beck, C. COCONUT online: Collection of Open Natural Products Database. _Journal of Cheminformatics_ **2021**, 13, 1–13.

* Mendez,  D.  et  al.  ChEMBL:  Towards  Direct  Deposition  of Bioassay Data. _Nucleic Acids Research_. **2018**, 47, D930–D940.

* Zeng,  X.;  Zhang,  P.;  He,  W.;  Qin,  C.;  Chen,  S.;  Tao,  L.;Wang, Y.;  Tan, Y.;  Gao, D.;  Wang, B.;  Chen, Z.;  Chen, W.;Jiang, Y. Y.; Chen, Y. Z. NPASS: Natural Product Activity and Species Source Database for Natural Product Research, Discovery and Tool Development. _Nucleic Acids Research_. **2018**, 46, D1217–D1222.

# Experiments

To reproduce the results, please run:

```
python experiment.py
```
This will train the models again. The hyperparameters used can be found in `settings/settings.yml`
You can also change the hyperparameters. We recommend creating a new `settings.yml`
If you want to use your own `settings.yml` please run.
The trained models will be saved in `data/trained_models/*folder name specified in settings.yml*`

```
python experiment.py --input *path to your settings.yml*
```
After training the `results` script will perfrom the similarity search and reproduce some of the graphics used.
```
python results.py
```
In `results/plots` the plots and some addtionall files will be saved.
In case you want to evaluate models that you trained with `experiment.py` but a different `settings.yml`
Run:
```
python results --input *path to the folder containing the models*
```
    
# Generate Fingerprints
You can use a csv file containing a column with SMILES strings as input to our model.
Naviagte to `*your path*/neural_npfp/neural_npfp` and run:

```
python get_fp.py ../data/testdata.csv -s smiles
```
`smiles` refers to the name of the column containing the SMILES strings.

By default the fingerprints using the NP_AUX model. If you want to use a different model use the flag `-m` followed by either `ae`, `aux` or `base`.

```
python get_fp.py ../data/testdata.csv -s smiles
```

You can also provide the Index of the column containing the SMILES. 
```
python get_fp.py ../data/testdata.csv -s 0
```
If you do not have a header add the `-n` flag

```
python get_fp.py ../data/testdata.csv -s 0 -n
```
# Perform a Similarity Seach based on the produced Fingerprint
If you want to use the NNFPs for a similarity search make sure that the query is in the same file as the molecules you want to screen before you generate the Fingerprints.

With `python simsearch.py *path of fingerprintfile* -q *index of the query*` the similairty search can be performed

Given you generated fingerprints for the `testdata.csv`. The following code will perform a similarity search for the query with index 0 

```
python simsearch.py ../data/aux_npfp_*date*.csv -q 0
```
You can also perform a similarity search for multiple queries by adding addtional indices.

```
python simsearch.py ../data/aux_npfp.csv -q 0 15 8 1 84
```
A new folder will be generated containing the results of the similairty search
Like in the original paper, the cosine similarity is used for the search.

You can add the `-d` to not include the other queries in the similarity search.

