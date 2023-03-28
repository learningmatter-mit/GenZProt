# GenZProt : Chemically Transferable Generative Backmapping of Coarse-Grained Proteins

![Overview](./overview.png)  

[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a repo accompanying our paper "Chemically Transferable Generative Backmapping of Coarse-Grained Proteins" ([arxiv link](https://arxiv.org/abs/2303.01569)). 

We propose a geometric generative model for backmapping all-atom coordinates from (coarse-grained) C_alpha traces of proteins.

### Download PED data
```
cd data
wget https://zenodo.org/record/7683192/files/genzprot_pedfiles.tar.gz
tar -xvf genzprot_pedfiles.tar.gz
```

### Install packages  

You can install the requirements via conda. 
pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0
```
conda env create -f environment.yml
```

Download and install this package
```
git clone https://github.com/learningmatter-mit/GenZProt
cd GenZProt
conda activate genzprot
pip install -e . 
```

### Backmapping C_alpha traces into all-atom structures  

Saved checkpoint of GenZProt is located in './ckpt/'. 
Save your C_alpha traces in .pdb format and pass its path to 'ca_trace_path' argument.
We need an all-atom pdb file (at least one model/frame) to get the topology and C_alpha mapping. Pass the path to 'topology_path' argument.   
```
cd scripts
MPATH=../ckpt/model_seed_12345
ca_trace_name=PED00055_CA_trace
ca_trace_path=../data/${ca_trace_name}.pdb
top_path=../data/PED00055.pdb
python inference.py -load_model_path $MPATH -ca_trace_path $ca_trace_path -topology_path $top_path
```
The results are saved in both ```.npy``` ( shape = ( 10,n_cg_samples,n_atoms_truncated,3 ) ) and ```.pdb``` format, in a directory named ```result_{MPATH}_{ca_trace_name}```. Because our algorithm requires i-1th and i+1th C_alpha positions to locate the atoms of the ith residue, it does not backmap the first and the last residue. Hence, ```n_atom_truncated = n_atom - (n_atom_first_res + n_atom_last_res)```.     


### Training your own GenZProt
```
cd script
python train_model.py -load_json modelparams/multi.json
```

### Test script
```
cd script
MPATH=../ckpt/model_seed_12345
test_data_path=../data/PED00055e000.pdb
python test_model.py -load_model_path $MPATH -test_data_path $test_data_path
```
