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
python inference.py -load_model '' -data_dir ''
```


### Training your own GenZProt

### Test script
```
python test_model.py -load_model '' -data_dir ''
```