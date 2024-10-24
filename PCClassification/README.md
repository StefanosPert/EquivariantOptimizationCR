# ModelNet40 Classification
Code adapted from the [FlyingGiraffe/vnn-pc](https://github.com/FlyingGiraffe/vnn-pc/tree/master/vn-pointnet) repository
## Installation
You can install the dependencies for this experiment by running
```
pip install -r requirements.txt
```
## Data
 Download and unzip the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).


## Optimization of VNN using the constraint relaxation
To train the relaxed vnn on point cloud classification 

```
python train_cls.py --model_mode=pointnet_equi --model=pointnet_cls_dual --log_dir=exp1 --rot=so3 --num_point=300 --resume -1 --data_dir=<DATA_DIR>
```
