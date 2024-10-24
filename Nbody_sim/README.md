# Nbody Simulation
Code adapted from [RobDHess/Steerable-E3-GNN](https://github.com/RobDHess/Steerable-E3-GNN/tree/main) repository.

## Dependencies
Install the dependencies of this experiment through pip by running
```
pip install -r requirements.txt
```

#### Creating N-Body data
To recreate the datasets used in this work, navigate to ```nbody/dataset/``` and run 
```bash
python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
```

#### N-Body (charged):
To run SEGNN using our proposed optimization algorithm use: 
```bash
python3 main.py --dataset=nbody --epochs=1000 --max_samples=3000 --model=segnn --lmax_h=1 --lmax_attr=1 --layers=4 --hidden_features=64 --subspace_type=weightbalanced --norm=none --batch_size=100 --gpu=1 --weight_decay=1e-12 --augmentations --cycle --exp_name nbody_charged --log True
```

