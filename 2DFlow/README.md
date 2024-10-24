## 2D Flow Prediction
This code was adapted from the [Rose-STL-Lab/Approximately-Equivariant-Nets](https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets/tree/master) repository

## Installation
- To install the requirements for these experiments through pip run
```
pip install -r requirements.txt
```


## Dataset and Preprocessing
- Install PhiFlow First 
```
git clone -b 2.0.1 --single-branch https://github.com/tum-pbs/PhiFlow.git
```

- Move data_prep.ipynb to the PhiFlow folder and run data_prep.ipynb to generate approximate translation, rotation and scaling symmetry smoke plume datasets.

- Or you can directly download preprocessed smoke plume dataset from [here](https://drive.google.com/drive/folders/1P3bPhBio3Zj0GP5rd1kuWbOQ7YI1CFxQ?usp=sharing). 

## Experiments
Train the relaxed rotational equivariant network using the following:
```
python3 run_model.py --dataset=PhiFlow --relaxed_symmetry=Rotation --hidden_dim=92 --num_layers=5 --out_length=6 --alpha=1e-5 --batch_size=16 --learning_rate=0.001 --decay_rate=0.95 --logdir exp_rot
```
and the relaxed scale equivariant network using:
```
 python3 run_model.py --dataset=PhiFlow --relaxed_symmetry=Scale --hidden_dim=64 --num_layers=5 --out_length=6 --alpha=1e-6 --batch_size=8 --learning_rate=0.0001 --decay_rate=0.95 --logdir=exp_scale
```
