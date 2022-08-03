# SepEns: Ensembling Neural Networks for Improved Prediction and Privacy in Early Diagnosis of Depsis

The basic neural network model is based on a word-level Language Model using an RNN (see [https://github.com/pytorch/examples](https://github.com/pytorch/examples)).

## Requirements

* [PyTorch](http://pytorch.org/) version >= 1.7.0
* numpy version >= 1.19.4
* scipy version >= 1.6.0
* Python version >= 3.6
* You will also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

These are the versions the code was tested with. It might be possible to use versions not listed above. 

## Overview

The procedure generate an esemble for sepsis prediction and reproduce the experiments from the paper is as follows:
* Prepare data
* Generate fully trained and patient specific models
* Make predictions on dev
* Grow ensemble
* Make predictions on test
* Calculate metrics

## Prepare code and data

Clone the repository and download the data from the [StatNLP web site](https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/). Extract the data to the code directory:

``` bash
git clone https://github.com/statnlp/sepens
cd sepens
wget https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/SepsisExp.tar.gz
tar zxvf SepsisExp.tar.gz
```

Make train/dev/test:
``` bash
. ./make_data.sh 0  # 0..3: split number for cross-validation
```

## Generate fully trained and patient specific models

To generate the model that is trained on all data ('full model'):
``` bash
. ./make_full_model.sh
```

To generate the patient specific models:
``` bash
. ./make_models_perpat.sh
```
You might want to parallelize this step as each model is trained independently from the others.

## Make predictions on dev set

Generate predictions for all patient specific (pool) models:
``` bash
. ./inference_poolmodels.sh
```

## Grow ensemble

Based on the mean suqared error and the correlation to existing ensemble members, grow an ensemble of patient specific models:

``` bash
. ./grow_ensemble_perrone.py 0 | tee logs/grow_ensemble.log   # 0..3: split number for cross-validation
tail -n1 logs/grow_ensemble.log > new_ensemle.py
```

## Make predictions on test set

Generate predictions for the fully trained model:
``` bash
. ./inference_fullmodel.sh
```

Generate predictions for the uniform and the weighted ensemble:
``` bash
. ./inference_ensemble.sh
```

## Calculate metrics

Generate AUROC for fully trained and ensemble models for different time intervals:
``` bash
. ./python calc_auroc.py
```

Generate AUROC for fully trained and ensemble models for different time intervals and various privacy budgets:
``` bash
. ./python calc_auroc.py
```
Calculates AUROC and accuracy loss.

## Membership attack

Apply a membership attack on the fully trained model for various privacy budgets.
``` bash
. ./python membership_fullmodel_epsilon_1k.py
```

Apply a membership attack on the uniform ensemble model for various privacy budgets.
``` bash
. ./python membership_ensemble_epsilon_alltrain_1k.py
```

## Citation

If you use the data or the code, please cite as:

``` bibtex
@inproceedings{schamoni2022,
  author = {Schamoni, Shigehiko and Hagmann, Michael and Riezler, Stefan},
  title = {Ensembling Neural Networks for Improved Prediction and Privacy in Early Diagnosis of Sepsis},
  booktitle = {Proceedings of the 6th Machine Learning for Healthcare Conference},
  year = {2022},
  city = {Durham, NC},
  country = {USA},
  volume = {182},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://www.cl.uni-heidelberg.de/~schamoni/publications/dl/MLHC2022_Ensembling.pdf}
}
```
