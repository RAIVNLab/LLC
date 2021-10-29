# [LLC: Accurate, Multi-purpose Learnt Low-dimensional Binary Codes](https://arxiv.org/abs/2106.01487)
[Aditya Kusupati](http://www.adityakusupati.com/), [Matthew Wallingford](https://mattwallingford.github.io/), [Vivek Ramanujan](https://vkramanuj.github.io/), [Raghav Somani](http://raghavsomani.github.io/), [Jae Sung Park](https://homes.cs.washington.edu/~jspark96/), [Krishna Pillutla](https://krishnap25.github.io/), [Prateek Jain](http://www.prateekjain.org/), [Sham Kakade](https://homes.cs.washington.edu/~sham/) and [Ali Farhadi](https://homes.cs.washington.edu/~ali/)

This repository contains code for the ImageNet-1K classification experiments presented in the NeurIPS 2021 [paper](https://arxiv.org/abs/2106.01487) along with more functionalities.

This code base is built upon the [hidden-networks repository](https://github.com/allenai/hidden-networks).

The Image retrieval experiments presented in the paper were done using the models obtained for classification and further adapted to ImageNet-100.

## Set Up
0. Clone this repository.
1. Using `Python 3.6`, create a `venv` with  `python -m venv myenv` and run `source myenv/bin/activate`. You can also use `conda` to create a virtual environment.
2. Install requirements with `pip install -r requirements.txt` for `venv` and appropriate `conda` commands for `conda` environment.
3. Create a **data directory** `<data-dir>`.
To run the ImageNet experiments there must be a folder `<data-dir>/imagenet`
that contains the ImageNet `train` and `val` folders that contains images of each class in a seperate folder.

## Training
This codebase contains model architecture for [ResNet50](models/resnet.py#L190) and support to train them on ImageNet-1K (other model architectures can be added to the same file for ease of utilization). We have provided some `config` files for training ResNet50 which can be modified for other architectures and datasets. To support more datasets, please add new dataloaders to [`data`](data/) folder.

Training across multiple GPUs is supported, however, the user should check the minimum number of GPUs required to scale ImageNet-1K. 

### Codebook Learning:

We support naive codebook learning with ``num_bits`` long codes for ``num_classes`` classes  and also support warmstarting them. Please check the ``config`` file for more potential parameters.

Base: ```python main.py --config configs/largescale/baselines/resnet50-llc-codebook-learning.yaml --multigpu 0,1,2,3 --save-codebook <path-to-save-codebook>```.

Warmstarting codebook: ```python main.py --config configs/largescale/baselines/resnet50-llc-codebook-learning.yaml --multigpu 0,1,2,3 --load-codebook <path-to-warmstart-codebook> --save-codebook <path-to-save-codebook>```.

### Instance Code Learning:

For a given codebook and (pretrained) backbone, instance code learning of ``num_bits`` long bnary codes. Please check the ``config`` file for more potential parameters.

Base: ```python main.py --config configs/largescale/baselines/resnet50-llc-instance-code-learning.yaml --multigpu 0,1,2,3 --instance-code --load-codebook <path-to-saved-codebook> --pretrained <path-to-pretrained-model>```.

While the default decoding scheme is set ot ``mhd``, one can change the scheme to ``ed`` by using ``--decode`` flag in the command line.

## Models and Logging
The saved models are compatible with the traditional dense models for simple evaluation and usage as transfer learning backbones. However, the final layer is decomposed that might result in unforseen issues, so caution advised. 

Every experiment creates a directory inside `runs` folder (which will be created automatically) along with the tensorboard logs, initial model state and best model (`model_best.pth`) - pretrained model for instance code learning.

The `runs` folder also has dumps of the csv with final and best accuracies. The code checkpoints after every epoch giving a chance to resume training when pre-empted, the extra functionalities can be explored through ```python main.py -h```. 

### Evaluating models on ImageNet-1K:

If you want to evaluate a [pretrained](#pretrained-models) LLC model provided below, you can use the model as is along with the codebook. 

Class Code Evaluation: ```python main.py --config configs/largescale/baselines/resnet50-llc-codebook-learning.yaml --multigpu 0,1,2,3 --load-codebook <path-to-saved-codebook> --pretrained <path-to-pretrained-model> --evaluate```

Instance Code Evaluation: ```python main.py --config configs/largescale/baselines/resnet50-llc-instance-code-learning.yaml --multigpu 0,1,2,3 --load-codebook <path-to-saved-codebook> --pretrained <path-to-pretrained-model> --evaluate --instance-code --decode <mhd/ed>```

## Pretrained Models
We provide the ``20-bit`` model trained with ResNet50 backbon on ImageNet-1K according to the settings in the [paper](https://arxiv.org/abs/2106.01487). 

The Class Code Evaluation of the provided pretrained model and codebook should give a top-1 accuracy of ``75.5%``. While the Instance Code Evaluation using ``MHD`` (Minimum Hamming Distance) gives ``74.5%`` and using ``ED`` (Exact Decoding) gives ``68.9%``.

``ResNet50 pretrained backbone projecting to 20 dim space`` - [R50 Pretrained Backbone](https://drive.google.com/file/d/1iAhEKsiT542QXjs3CAxnpE7GhhTvTbfA/view?usp=sharing).

``Learnt 20-bit codebook`` - [20-bit Codebook](https://drive.google.com/file/d/1s0ezOsdMCfZUvjgp4oCvxtgVQbn5YSST/view?usp=sharing).

Note that the codebook is stored with the underlying real-values. We recommend people store it this way and binarize the codebook to ``+1/-1`` for usage in LLC.


## Citation

If you find this project useful in your research, please consider citing:

```
@InProceedings{Kusupati21
  author    = {Kusupati, Aditya and Wallingford, Matthew and Ramanujan, Vivek and Somani, Raghav and Park, Jae Sung and Pillutla, Krishna and Jain, Prateek and Kakade, Sham and Farhadi, Ali},
  title     = {LLC: Accurate, Multi-purpose Learnt Low-dimensional Binary Codes},
  booktitle = {Advances in Neural Information Processing Systems},
  month     = {December},
  year      = {2021},
}
```

A portion of the code has been refactored for ease of usage by [William Howard-Snyder](https://github.com/williamhowardsnyder) with assitance from [Gary Geng](https://github.com/evilnose). Please contact [Aditya Kusupati](http://www.adityakusupati.com/) for comments or questions.
