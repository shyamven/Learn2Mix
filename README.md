# Learn2Mix

## Overview
This repository contains the implementation of the **Learn2Mix** training mechanism, which accelerates model convergence by adaptively adjusting the proportion of classes within batches, across training epochs, using real-time class-wise error rates. We provide empirical results on various benchmark datasets, covering classification, regression, and reconstruction tasks. The code in this repository demonstrates how **Learn2Mix** can be integrated into common training workflows.

## Requirements
- Python >= 3.7  
- PyTorch >= 1.7  
- torchvision >= 0.8  
- numpy >= 1.19  
- scikit-learn >= 0.23  
- imbalanced-learn (for SMOTE) >= 0.7  
- matplotlib >= 3.3  
- scipy >= 1.5  

## Note
For the **IMDB dataset**, please download the CSV from [this Kaggle link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Once downloaded, place it in the `./data` directory  so it can be accessed by the dataset-loading utilities provided in this repository.

## Citation: 

Plain Text:
```
Shyam Venkatasubramanian, Vahid Tarokh. Learn2Mix: Training Neural Networks Using Adaptive Data Integration. arXiv preprint arXiv:2412.16482
```
BibTeX:
```
@misc{shyam2024learn2mix,
      title={Learn2Mix: Training Neural Networks Using Adaptive Data Integration}, 
      author={Shyam Venkatasubramanian and Vahid Tarokh},
      year={2024},
      eprint={2412.16482},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
