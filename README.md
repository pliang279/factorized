# Learning Factorized Multimodal Representations

> Pytorch implementation for learning factorized multimodal representations using deep generative models.

Correspondence to: 
  - Paul Liang (pliang@andrew.cmu.edu)
  - Yao-Hung Hubert Tsai (yaohungt@cs.cmu.edu)
  
## Paper

[**Learning Factorized Multimodal Representations**](https://arxiv.org/abs/1806.06176)<br>
[Yao-Hung Hubert Tsai](https://yaohungt.github.io)* , [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/)* , [Amir Zadeh](https://www.amir-zadeh.com/), [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>
ICLR, 2019. (*equal contribution)

## Installation

First check that the requirements are satisfied:</br>
Python 2.7</br>
PyTorch 0.4.0</br>
numpy 1.13.3</br>
sklearn 0.20.0

The next step is to clone the repository:
```bash
git clone https://github.com/pliang279/factorized.git
```

## Dataset

Please download the latest version of the CMU-MOSI, CMU-MOSEI, POM, and IEMOCAP datasets which can be found at https://github.com/A2Zadeh/CMU-MultimodalSDK/

## Scripts

Please run
```bash
python mfm_test_mosi.py
```
in the command line.

If you use this code, please cite our paper:

```bash
@inproceedings{DBLP:journals/corr/abs-1806-06176,
  title     = {Learning Factorized Multimodal Representations},
  author    = {Yao{-}Hung Hubert Tsai and
               Paul Pu Liang and
               Amir Zadeh and
               Louis{-}Philippe Morency and
               Ruslan Salakhutdinov},
  booktitle={ICLR},
  year={2019}
}
```

Related papers and repositories building upon these datasets:
Memory Fusion Network (https://arxiv.org/abs/1802.00927, https://github.com/pliang279/MFN)</br>
Multi-Attention Recurrent Network (https://arxiv.org/abs/1802.00923, https://github.com/A2Zadeh/CMU-MultimodalSDK/)</br>
Graph-MFN (http://aclweb.org/anthology/P18-1208, https://github.com/A2Zadeh/CMU-MultimodalSDK/)</br>
Multimodal Transformer (https://arxiv.org/abs/1906.00295, https://github.com/yaohungt/Multimodal-Transformer)
