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
    "mean-estimation": {"kind": "l2m_regression"},
    "mnist-reconstruction": {"kind": "l2m_regression"},
    "fashion-mnist-reconstruction": {"kind": "l2m_regression"},
    "cifar10-reconstruction": {"kind": "l2m_regression"},
    "california-housing": {"kind": "l2m_regression"},
    "wine-quality": {"kind": "l2m_regression"},
}

