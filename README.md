# Adversarially Robust Few-Shot Learning:  A Meta-Learning Approach

This repository contains PyTorch code for adversarial querying with [ProtoNet](https://arxiv.org/abs/1703.05175), [R2-D2](https://arxiv.org/abs/1805.08136), and [MetaOptNet](https://arxiv.org/pdf/1904.03758.pdf).  Adversarial querying is an algorithm for producing robust meta-learners. We adapt models and data loading from [here](https://github.com/kjunelee/MetaOptNet).

## Prerequisites:
* Python2
* PyTorch
* CUDA

## Results:
| Model                        | A_nat           | A_adv           |
|------------------------------|-----------------|-----------------|
| Naturally Trained R2-D2      | 73.01% +/- 0.13 | 0.00% +/- 0.13  |
| AT Transfer (R2-D2 backbone) | 39.13% +/- 0.13 | 25.33% +/- 0.13 |
| ADML                         | 47.75% +/- 0.13 | 18.49% +/- 0.13 |
| AQ R2-D2 (Ours)              | 57.87% +/- 0.13 | 31.52% +/- 0.13 |

A comparison of robustness method on 5-shot Mini-ImageNet. Natural accuracy is denoted A_nat, robust accuracy A_adv. A_adv is computed w.r.t. a 20-step PGD attack with epsilon bound 8/255. 

The result above can be reproduced by training a network with the following command: 
```
train.py --val_shot 5 --attack_embedding
```
