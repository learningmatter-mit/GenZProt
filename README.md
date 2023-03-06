# GenZProt : Chemically Transferable Generative Backmapping of Coarse-Grained Proteins

This is a repo accompanying our paper "Chemically Transferable Generative Backmapping of Coarse-Grained Proteins" ([arxiv link](https://arxiv.org/abs/2303.01569)). 

We propose a geometric generative model for backmmaping fine-grained coordinates from coarse-grained coordinates. It is essentially performing the geometric super-resolution task for molecular geometries. 

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
cd script
MPATH=../ckpt/model_seed_12345
ca_trace_path=../data/PED00055_CA_trace.pdb
top_path=../data/PED00055.pdb
python inference.py -load_model_path $MPATH -ca_trace_path $ca_trace_path -topology_path $top_path
```

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
