## Improving Equivariant Model Training via Constraint Relaxation
This repository is the official implementation of the paper:

__Stefanos Pertigkiozoglou\*, Evangelos Chatzipantazis\*, Shubhendu Trivedi and Kostas Daniilidis, "Improving Equivariant Model Training via Constraint Relaxation" (Neurips 2024). [[arXiv](https://arxiv.org/pdf/2408.13242)]__

<p align="center">
<img src="https://github.com/StefanosPert/setup_Equivariant_Optimization_CR/blob/main/assets/ApproximateEquivarianceOptimization.jpg" width=60%>
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
@inproceedings{equivOptimCR2024,
 title = {Improving Equivariant Model Training via Constraint Relaxation},
 author = {Stefanos Pertigkiozoglou and Evangelos Chatzipantazis and Shubhendu Trivedi and Kostas Daniilidis},
 booktitle = {Advances in Neural Information Processing Systems},
 year = {2024}
}
```
