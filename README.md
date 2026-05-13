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

## Repository Structure
- `main.py`: unified entrypoint to run any experiment from one CLI.
- `learn2mix/experiments/`: unified experiment engines (`classification_engine.py`, `l2m_regression_engine.py`), registry, and runner.
- `learn2mix/models/`: reusable model definitions (CNN, MobileNet/ResNet, Transformer).
- `learn2mix/utils/`: shared utility components (sampling helpers, losses).

## Unified Runner Usage
Run any experiment through the new `main.py`:

```bash
python main.py --experiment mnist --method L2M
```

Regression/reconstruction Learn2Mix-vs-classical experiments do not use `--method`:

```bash
python main.py --experiment mean-estimation
```

List all runnable experiments and methods:

```bash
python main.py --list
```

Supported experiments:
- `mnist`
- `fashion-mnist`
- `cifar10`
- `cifar10-mobile`
- `cifar100`
- `cifar100-mobile`
- `imdb`
- `imagenette`
- `mean-estimation`
- `mnist-reconstruction`
- `fashion-mnist-reconstruction`
- `cifar10-reconstruction`
- `california-housing`
- `wine-quality`

Supported methods:
- `L2M`
- `CBL`
- `SMOTE`
- `IS`
- `CL`
- `classical`

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
