## Improving Equivariant Model Training via Constraint Relaxation

🔴 **Please refer to [StefanosPert/Equivariant_Optimization_CR](https://github.com/StefanosPert/Equivariant_Optimization_CR) for the most up to date version of this repository.**

This repository provides an implementation for the paper: 

__Stefanos Pertigkiozoglou\*, Evangelos Chatzipantazis\*, Shubhendu Trivedi and Kostas Daniilidis, "Improving Equivariant Model Training via Constraint Relaxation" (Neurips 2024). [[link](https://openreview.net/forum?id=tWkL7k1u5v&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions))]__

<p align="center">
<img src="https://github.com/StefanosPert/Equivariant_Optimization_CR/blob/main/assets/ApproximateEquivarianceOptimization.jpg" width=60%>
</p>

### Abstract
Equivariant neural networks have been widely used in a variety of applications due to their ability to generalize well in tasks where the underlying data symmetries are known. Despite their successes, such networks can be difficult to optimize and require careful hyperparameter tuning to train successfully. In this work, we propose a novel framework for improving the optimization of such models by relaxing the hard equivariance constraint during training: We relax the equivariance constraint of the network's intermediate layers by introducing an additional non-equivariance term that we progressively constrain until we arrive at an equivariant solution. By controlling the magnitude of the activation of the additional relaxation term, we allow the model to optimize over a larger hypothesis space containing approximate equivariant networks and converge back to an equivariant solution at the end of training. We provide experimental results on different state-of-the-art network architectures, demonstrating how this training framework can result in equivariant models with improved generalization performance. 

## Experiments
Please see the following directories for the individual experiments where we applied our proposed optimization method
- [Point Cloud Classification](https://github.com/StefanosPert/Equivariant_Optimization_CR/tree/main/PCClassification)
- [Nbody Simulation](https://github.com/StefanosPert/Equivariant_Optimization_CR/tree/main/Nbody_sim)
- [Flow Simulation](https://github.com/StefanosPert/Equivariant_Optimization_CR/tree/main/2DFlow)

## Cite
```
@inproceedings{NEURIPS2024_98082e6b,
 author = {Pertigkiozoglou, Stefanos and Chatzipantazis, Evangelos and Trivedi, Shubhendu and Daniilidis, Kostas},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {83497--83520},
 publisher = {Curran Associates, Inc.},
 title = {Improving Equivariant Model Training via Constraint Relaxation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/98082e6b4b97ab7d3af1134a5013304e-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
