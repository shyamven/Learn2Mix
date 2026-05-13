SUPPORTED_METHODS = ("L2M", "CBL", "SMOTE", "IS", "CL", "classical")

EXPERIMENTS = {
    "mnist": {"kind": "classification"},
    "fashion-mnist": {"kind": "classification"},
    "cifar10": {"kind": "classification"},
    "cifar10-mobile": {"kind": "classification"},
    "cifar100": {"kind": "classification"},
    "cifar100-mobile": {"kind": "classification"},
    "imdb": {"kind": "classification"},
    "imagenette": {"kind": "classification"},
    "mean-estimation": {
        "kind": "notebook",
        "url": "https://raw.githubusercontent.com/shyamven/Learn2Mix/main/Learn2Mix_Mean_Estimation.ipynb",
    },
    "mnist-reconstruction": {
        "kind": "notebook",
        "url": "https://raw.githubusercontent.com/shyamven/Learn2Mix/main/Learn2Mix_MNIST_Reconstruction.ipynb",
    },
    "fashion-mnist-reconstruction": {
        "kind": "notebook",
        "url": "https://raw.githubusercontent.com/shyamven/Learn2Mix/main/Learn2Mix_Fashion_MNIST_Reconstruction.ipynb",
    },
    "cifar10-reconstruction": {
        "kind": "notebook",
        "url": "https://raw.githubusercontent.com/shyamven/Learn2Mix/main/Learn2Mix_CIFAR-10_Reconstruction.ipynb",
    },
    "california-housing": {
        "kind": "notebook",
        "url": "https://raw.githubusercontent.com/shyamven/Learn2Mix/main/Learn2Mix_California_Housing.ipynb",
    },
    "wine-quality": {
        "kind": "notebook",
        "url": "https://raw.githubusercontent.com/shyamven/Learn2Mix/main/Learn2Mix_Wine_Quality.ipynb",
    },
}

