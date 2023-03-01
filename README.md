# GenZProt : Chemically Transferable Generative Backmapping of Coarse-Grained Proteins

<object data="http://github.com/SoojungYang/GenZProt/overview.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://github.com/SoojungYang/GenZProt/overview.pdf">
    </embed>
</object>

This is a repo accompanying our paper "Chemically Transferable Generative Backmapping of Coarse-Grained Proteins" ([arxiv link](https://arxiv.org/abs/)). 

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
git clone https://github.com/SoojungYang/GenZProt.git
cd GenZProt
conda activate genzprot
pip install -e . 
```

### Backmapping C_alpha traces into all-atom structures  

Saved checkpoint of GenZProt is located in './models/'. 
Save your C_alpha traces in .pdb format and place them in data_dir.
```
ped_id=00055
MPATH=./ckpt/model_seed_12345.pt
test_data_path=./data/PED00055_CA_trace.pdb
python inference.py -load_model_path $MPATH -test_data $test_data_path
```

### Training your own GenZProt
```
python train_model.py -load_json modelparams/multi.json
```

### Test script
```
ped_id=00055
MPATH=./ckpt/model_seed_12345.pt
python test_model.py -load_model_path $MPATH -test_data $ped_id
```