from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from learn2mix.models import (
    LeNet5,
    LeNetCIFAR,
    MobileNetV3Classifier,
    ResNetImagenette,
    TransformerNN,
)
from learn2mix.utils import FocalLoss, compute_class_counts, shuffle_class_data


@dataclass
class ClassificationConfig:
    name: str
    output_suffix: str
    batch_size: int
    num_classes: int
    learning_rate: float
    epochs_by_method: dict[str, int]
    iterations: int
    model_factory: Callable[[], nn.Module]
    train_dataset_builder: Callable[[], tuple[TensorDataset, TensorDataset, list[TensorDataset]]]


def _build_mnist() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return _build_tensor_datasets_from_classification_dataset(train_ds, test_ds, 10)


def _build_fashion_mnist() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return _build_tensor_datasets_from_classification_dataset(train_ds, test_ds, 10)


def _build_cifar10() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=1.0, inplace=False),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    return _build_tensor_datasets_from_classification_dataset(train_ds, test_ds, 10)


def _build_cifar10_mobile() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform_train = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    return _build_tensor_datasets_from_classification_dataset(train_ds, test_ds, 10)


def _build_cifar100() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))]
    )
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    return _build_unbalanced_image_dataset(train_ds, test_ds, 100, np.logspace(np.log10(1.0), np.log10(0.1), base=40.0, num=100))


def _build_cifar100_mobile() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform_train = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    return _build_unbalanced_image_dataset(train_ds, test_ds, 100, np.logspace(np.log10(1.0), np.log10(0.1), base=40.0, num=100))


def _build_imagenette() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_ds = datasets.Imagenette(root="./data", split="train", download=False, transform=transform)
    test_ds = datasets.Imagenette(root="./data", split="val", download=False, transform=transform)
    factors = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    return _build_unbalanced_image_dataset(train_ds, test_ds, 10, factors)


def _tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text.lower())


def _build_vocab(tokenized_texts: list[list[str]], min_freq: int) -> dict[str, int]:
    counter: dict[str, int] = {}
    for tokens in tokenized_texts:
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def _preprocess_imdb(df: pd.DataFrame, vocab: dict[str, int], max_seq_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    texts = []
    labels = []
    for _, row in df.iterrows():
        token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in _tokenize(row["review"])]
        if len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]
        else:
            token_ids += [vocab["<pad>"]] * (max_seq_length - len(token_ids))
        texts.append(token_ids)
        labels.append(int(row["sentiment"]))
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _build_imdb() -> tuple[TensorDataset, TensorDataset, list[TensorDataset]]:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    data = pd.read_csv("./data/IMDB.csv")
    data["sentiment"] = data["sentiment"].map({"positive": 1, "negative": 0})
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["sentiment"])
    tokenized = [_tokenize(text) for text in train_df["review"]]
    vocab = _build_vocab(tokenized, min_freq=2)
    train_x, train_y = _preprocess_imdb(train_df, vocab, max_seq_length=500)
    test_x, test_y = _preprocess_imdb(test_df, vocab, max_seq_length=500)

    class_data = [[], []]
    for x, y in zip(train_x, train_y):
        class_data[y.item()].append(x)
    unbalance_factors = [0.3, 1.0]
    unbalanced = []
    for idx in range(2):
        keep = int(len(class_data[idx]) * unbalance_factors[idx])
        selected = random.sample(class_data[idx], keep)
        unbalanced.append(torch.stack(selected))
    tensor_datasets = [TensorDataset(data_, torch.full((data_.size(0),), label, dtype=torch.long)) for label, data_ in enumerate(unbalanced)]
    train_concat = torch.utils.data.ConcatDataset(tensor_datasets)
    train_list_x, train_list_y = [], []
    for seq, lbl in train_concat:
        train_list_x.append(seq)
        train_list_y.append(lbl)
    train_tensor = TensorDataset(torch.stack(train_list_x), torch.tensor(train_list_y, dtype=torch.long))
    test_tensor = TensorDataset(test_x, test_y)
    return train_tensor, test_tensor, tensor_datasets


def _build_tensor_datasets_from_classification_dataset(train_ds, test_ds, num_classes: int):
    train_x, train_y = [], []
    for img, label in train_ds:
        train_x.append(img)
        train_y.append(label)
    train_tensor = TensorDataset(torch.stack(train_x), torch.tensor(train_y, dtype=torch.long))

    test_x, test_y = [], []
    for img, label in test_ds:
        test_x.append(img)
        test_y.append(label)
    test_tensor = TensorDataset(torch.stack(test_x), torch.tensor(test_y, dtype=torch.long))

    class_data = [[] for _ in range(num_classes)]
    for img, label in train_ds:
        class_data[label].append(img)
    class_data = [torch.stack(data_) for data_ in class_data]
    tensor_datasets = [
        TensorDataset(data_, torch.full((data_.size(0),), label, dtype=torch.long))
        for label, data_ in enumerate(class_data)
    ]
    return train_tensor, test_tensor, tensor_datasets


def _build_unbalanced_image_dataset(train_ds, test_ds, num_classes: int, factors):
    class_data = [[] for _ in range(num_classes)]
    for img, lbl in train_ds:
        class_data[lbl].append(img)
    unbalanced = []
    for idx in range(num_classes):
        keep = int(len(class_data[idx]) * factors[idx])
        selected = random.sample(class_data[idx], keep)
        unbalanced.append(torch.stack(selected))
    tensor_datasets = [TensorDataset(data_, torch.full((data_.size(0),), label, dtype=torch.long)) for label, data_ in enumerate(unbalanced)]
    train_concat = torch.utils.data.ConcatDataset(tensor_datasets)

    train_x, train_y = [], []
    for img, lbl in train_concat:
        train_x.append(img)
        train_y.append(lbl)
    train_tensor = TensorDataset(torch.stack(train_x), torch.tensor(train_y, dtype=torch.long))

    test_x, test_y = [], []
    for img, lbl in test_ds:
        test_x.append(img)
        test_y.append(lbl)
    test_tensor = TensorDataset(torch.stack(test_x), torch.tensor(test_y, dtype=torch.long))
    return train_tensor, test_tensor, tensor_datasets


CLASSIFICATION_CONFIGS: dict[str, ClassificationConfig] = {
    "mnist": ClassificationConfig(
        name="mnist",
        output_suffix="mnist",
        batch_size=1000,
        num_classes=10,
        learning_rate=5e-3,
        epochs_by_method={"L2M": 60, "CBL": 45, "SMOTE": 50, "IS": 45, "CL": 62, "classical": 45},
        iterations=5,
        model_factory=LeNet5,
        train_dataset_builder=_build_mnist,
    ),
    "fashion-mnist": ClassificationConfig(
        name="fashion-mnist",
        output_suffix="fashion_mnist",
        batch_size=1000,
        num_classes=10,
        learning_rate=5e-3,
        epochs_by_method={"L2M": 70, "CBL": 55, "SMOTE": 53, "IS": 50, "CL": 63, "classical": 56},
        iterations=5,
        model_factory=LeNet5,
        train_dataset_builder=_build_fashion_mnist,
    ),
    "cifar10": ClassificationConfig(
        name="cifar10",
        output_suffix="cifar10",
        batch_size=1000,
        num_classes=10,
        learning_rate=7e-5,
        epochs_by_method={"L2M": 125, "CBL": 123, "SMOTE": 113, "IS": 105, "CL": 125, "classical": 125},
        iterations=5,
        model_factory=lambda: LeNetCIFAR(num_classes=10),
        train_dataset_builder=_build_cifar10,
    ),
    "cifar10-mobile": ClassificationConfig(
        name="cifar10-mobile",
        output_suffix="cifar10",
        batch_size=500,
        num_classes=10,
        learning_rate=1e-5,
        epochs_by_method={"L2M": 40, "CBL": 40, "SMOTE": 40, "IS": 40, "CL": 54, "classical": 40},
        iterations=3,
        model_factory=lambda: MobileNetV3Classifier(num_classes=10),
        train_dataset_builder=_build_cifar10_mobile,
    ),
    "cifar100": ClassificationConfig(
        name="cifar100",
        output_suffix="cifar100",
        batch_size=5000,
        num_classes=100,
        learning_rate=1e-4,
        epochs_by_method={"L2M": 135, "CBL": 130, "SMOTE": 130, "IS": 120, "CL": 137, "classical": 125},
        iterations=5,
        model_factory=lambda: LeNetCIFAR(num_classes=100),
        train_dataset_builder=_build_cifar100,
    ),
    "cifar100-mobile": ClassificationConfig(
        name="cifar100-mobile",
        output_suffix="cifar100",
        batch_size=1000,
        num_classes=100,
        learning_rate=1e-5,
        epochs_by_method={"L2M": 40, "CBL": 40, "SMOTE": 40, "IS": 40, "CL": 54, "classical": 40},
        iterations=3,
        model_factory=lambda: MobileNetV3Classifier(num_classes=100),
        train_dataset_builder=_build_cifar100_mobile,
    ),
    "imagenette": ClassificationConfig(
        name="imagenette",
        output_suffix="imagenette",
        batch_size=250,
        num_classes=10,
        learning_rate=1e-6,
        epochs_by_method={"L2M": 60, "CBL": 60, "SMOTE": 60, "IS": 67, "CL": 74, "classical": 60},
        iterations=5,
        model_factory=lambda: ResNetImagenette(num_classes=10),
        train_dataset_builder=_build_imagenette,
    ),
    "imdb": ClassificationConfig(
        name="imdb",
        output_suffix="imdb",
        batch_size=500,
        num_classes=2,
        learning_rate=1e-4,
        epochs_by_method={"L2M": 40, "CBL": 40, "SMOTE": 40, "IS": 48, "CL": 53, "classical": 40},
        iterations=5,
        model_factory=lambda: TransformerNN(vocab_size=50000, max_seq_length=500),
        train_dataset_builder=_build_imdb,
    ),
}


def _save_outputs(method: str, suffix: str, train_arr: np.ndarray, test_arr: np.ndarray, time_arr: np.ndarray):
    prefix_map = {
        "L2M": ("train_errors", "test_errors", "time_l2m"),
        "CBL": ("balanced_errors", "balanced_test_errors", "time_cbl"),
        "SMOTE": ("smote_errors", "smote_test_errors", "time_smote"),
        "IS": ("is_train_errors", "is_test_errors", "time_is"),
        "CL": ("cl_train_errors", "cl_test_errors", "time_cl"),
        "classical": ("classical_errors", "classical_test_errors", "time_classical"),
    }
    train_name, test_name, time_name = prefix_map[method]
    np.savetxt(f"{train_name}_{suffix}.csv", train_arr, delimiter=",")
    np.savetxt(f"{test_name}_{suffix}.csv", test_arr, delimiter=",")
    np.savetxt(f"{time_name}_{suffix}.csv", time_arr, delimiter=",")


def run_classification_experiment(experiment: str, method: str) -> int:
    if experiment not in CLASSIFICATION_CONFIGS:
        raise ValueError(f"Unknown classification experiment '{experiment}'")
    cfg = CLASSIFICATION_CONFIGS[experiment]
    epochs = cfg.epochs_by_method[method]
    iterations = cfg.iterations

    train_results = np.zeros((epochs, iterations))
    test_results = np.zeros((epochs, iterations))
    time_results = np.zeros((epochs, iterations))

    for iter_idx in range(iterations):
        train_err, test_err, times = _run_single_iteration(cfg, method, epochs, iter_idx)
        train_results[:, iter_idx] = train_err
        test_results[:, iter_idx] = test_err
        time_results[:, iter_idx] = times

    _save_outputs(method, cfg.output_suffix, train_results, test_results, time_results)
    return 0


def _run_single_iteration(cfg: ClassificationConfig, method: str, epochs: int, iter_idx: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, test_dataset, tensor_datasets = cfg.train_dataset_builder()
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    n_total = sum(len(ds) for ds in tensor_datasets)
    m_batches = int(np.floor(n_total / cfg.batch_size))
    if cfg.name == "imdb":
        vocab_size = int(train_dataset.tensors[0].max().item()) + 1
        model = TransformerNN(vocab_size=vocab_size, max_seq_length=500).to(device)
    else:
        model = cfg.model_factory().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if method == "L2M":
        alpha = torch.tensor([len(ds) / n_total for ds in tensor_datasets], dtype=torch.float32).to(device)
        alpha_lr = 1e-1
    elif method == "CBL":
        class_counts = compute_class_counts(tensor_datasets)
        classes = sorted(class_counts.keys())
        counts = np.array([class_counts[c] for c in classes], dtype=np.float32)
        class_weights = 1.0 / (counts + 1e-6)
        class_weights = class_weights / np.sum(class_weights) * len(classes)
        focal_criterion = FocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float32).to(device), gamma=2, reduction="mean").to(device)
    elif method == "SMOTE":
        all_x, all_y = [], []
        for batch_x, batch_y in train_loader:
            all_x.append(batch_x)
            all_y.append(batch_y)
        x_train = torch.cat(all_x, dim=0)
        y_train = torch.cat(all_y, dim=0)
        x_flat = x_train.view(len(x_train), -1)
        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(x_flat.numpy(), y_train.numpy())
        if cfg.name == "imdb":
            x_tensor = torch.tensor(x_smote, dtype=torch.long)
        else:
            c, h, w = x_train.shape[1], x_train.shape[2], x_train.shape[3]
            x_tensor = torch.tensor(x_smote, dtype=torch.float32).view(-1, c, h, w)
        y_tensor = torch.tensor(y_smote, dtype=torch.long)
        smote_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=cfg.batch_size, shuffle=True)
    elif method == "IS":
        n_subset = cfg.batch_size // 2
        is_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    elif method == "CL":
        warmup_epochs = 50
        for _ in range(warmup_epochs):
            model.train()
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_train), y_train)
                loss.backward()
                optimizer.step()
        model.eval()
        all_confidences = []
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                probs = torch.softmax(model(imgs), dim=1)
                conf = probs[range(len(labels)), labels]
                all_confidences.append(conf.cpu())
        scores = 1 - torch.cat(all_confidences).numpy()
        sorted_indices = np.argsort(scores)
        if cfg.name == "imdb":
            vocab_size = int(train_dataset.tensors[0].max().item()) + 1
            model = TransformerNN(vocab_size=vocab_size, max_seq_length=500).to(device)
        else:
            model = cfg.model_factory().to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        starting_percent = 0.5
        inc = 1.2
        step_length = 10

    train_err = np.zeros(epochs)
    test_err = np.zeros(epochs)
    time_arr = np.zeros(epochs)
    total_time = 0.0

    for epoch in range(epochs):
        correct_train = correct_test = total_train = total_test = 0
        start = time.time()
        model.train()

        if method == "L2M":
            shuffled_indices = shuffle_class_data(tensor_datasets)
            prev_indices = np.zeros(cfg.num_classes, dtype=int)
            tracking_error = np.zeros(cfg.num_classes, dtype=float)
            for _ in range(m_batches):
                combined_data, combined_labels = [], []
                for i in range(cfg.num_classes):
                    num_samples = max(int(alpha[i].item() * cfg.batch_size), 0)
                    data_list, label_list = [], []
                    remaining = num_samples
                    while remaining > 0:
                        available = len(tensor_datasets[i]) - prev_indices[i]
                        take = min(available, remaining)
                        idx_slice = shuffled_indices[i][prev_indices[i] : prev_indices[i] + take]
                        data_list.append(tensor_datasets[i].tensors[0][idx_slice])
                        label_list.append(tensor_datasets[i].tensors[1][idx_slice])
                        remaining -= take
                        prev_indices[i] += take
                        if prev_indices[i] >= len(tensor_datasets[i]):
                            prev_indices[i] = 0
                    combined_data.append(torch.cat(data_list, dim=0))
                    combined_labels.append(torch.cat(label_list, dim=0))
                optimizer.zero_grad()
                x_mixed = torch.cat(combined_data, dim=0).to(device)
                y_mixed = torch.cat(combined_labels, dim=0).to(device)
                loss = criterion(model(x_mixed), y_mixed)
                loss.backward()
                optimizer.step()
            start_idx = 0
            z_mixed = model(x_mixed)
            for i in range(cfg.num_classes):
                num_samples = max(int(alpha[i].item() * cfg.batch_size), 0)
                if num_samples > 0:
                    tracking_error[i] = criterion(z_mixed[start_idx : start_idx + num_samples], y_mixed[start_idx : start_idx + num_samples]).item()
                start_idx += num_samples
            rewards = (tracking_error / np.sum(tracking_error)) if np.sum(tracking_error) > 0 else np.ones(cfg.num_classes) / cfg.num_classes
            alpha += 1e-1 * (torch.tensor(rewards, dtype=torch.float32).to(device) - alpha)
        elif method == "CBL":
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                loss = focal_criterion(model(x_train), y_train)
                loss.backward()
                optimizer.step()
        elif method == "SMOTE":
            smote_it = iter(smote_loader)
            for _ in range(m_batches):
                try:
                    x_train, y_train = next(smote_it)
                except StopIteration:
                    smote_it = iter(smote_loader)
                    x_train, y_train = next(smote_it)
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_train), y_train)
                loss.backward()
                optimizer.step()
        elif method == "IS":
            for x_train, y_train in is_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                model.eval()
                with torch.no_grad():
                    losses = nn.functional.cross_entropy(model(x_train), y_train, reduction="none")
                    p = losses / losses.sum()
                    subset_indices = torch.multinomial(p, cfg.batch_size // 2, replacement=False)
                model.train()
                x_subset = x_train[subset_indices]
                y_subset = y_train[subset_indices]
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(x_subset), y_subset, reduction="mean")
                loss.backward()
                optimizer.step()
        elif method == "CL":
            subset_frac = min(starting_percent * (inc ** (epoch // step_length)), 1.0)
            subset_size = int(subset_frac * len(train_dataset))
            curriculum_subset = Subset(train_dataset, sorted_indices[:subset_size])
            curriculum_loader = DataLoader(curriculum_subset, batch_size=cfg.batch_size, shuffle=True)
            for x_train, y_train in curriculum_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_train), y_train)
                loss.backward()
                optimizer.step()
        else:
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_train), y_train)
                loss.backward()
                optimizer.step()

        total_time += time.time() - start
        time_arr[epoch] = total_time

        with torch.no_grad():
            model.eval()
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                pred = torch.argmax(model(x_train), dim=1)
                total_train += y_train.size(0)
                correct_train += (pred == y_train).sum().item()
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                pred = torch.argmax(model(x_test), dim=1)
                total_test += y_test.size(0)
                correct_test += (pred == y_test).sum().item()

        train_err[epoch] = 100.0 * correct_train / total_train
        test_err[epoch] = 100.0 * correct_test / total_test
        print(
            f"Iter {iter_idx}, Epoch {epoch}, {method} train/test: "
            f"{train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}"
        )

    return train_err, test_err, time_arr

