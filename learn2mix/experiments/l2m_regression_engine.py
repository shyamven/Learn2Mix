from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _shuffle_class_data(tensor_datasets: list[TensorDataset]) -> list[torch.Tensor]:
    shuffled_indices = []
    for dataset in tensor_datasets:
        shuffled_indices.append(torch.randperm(dataset.tensors[0].size(0)))
    return shuffled_indices


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Autoencoder(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        in_dim = channels * height * width
        self.channels = channels
        self.height = height
        self.width = width
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(self.encoder(x))
        return decoded.view(-1, self.channels, self.height, self.width)


@dataclass
class L2MRegressionConfig:
    name: str
    epochs: int
    iterations: int
    batch_size: int
    alpha_lr: float
    lr: float
    model_factory: Callable[[], nn.Module]
    build_data: Callable[[], tuple[list[TensorDataset], DataLoader, DataLoader]]
    optimizer_kind: str = "adam"
    noise_std: float = 0.0
    classical_mode: str = "mixed"  # mixed or full_loader


def _generate_synthetic_data(size: int, distribution: str) -> tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    if distribution == "normal":
        for _ in range(size):
            mean = np.random.uniform(0, 1, 1)
            data.append(np.random.normal(mean, np.sqrt(1.0), size=(1, 10)))
            labels.append(mean)
    elif distribution == "exponential":
        for _ in range(size):
            mean = np.random.uniform(0, 1, 1)
            data.append(np.random.exponential(mean, size=(1, 10)))
            labels.append(mean)
    elif distribution == "chi_squared":
        for _ in range(size):
            mean = np.random.uniform(0, 1, 1)
            data.append(np.random.chisquare(df=mean, size=(1, 10)))
            labels.append(mean)
    elif distribution == "uniform":
        for _ in range(size):
            mean = np.random.uniform(20, 50, 1)
            data.append(np.random.uniform(mean - 10, mean + 10, size=(1, 10)))
            labels.append(mean)
    return np.concatenate(data, axis=0), np.array(labels).reshape(-1, 1)


def _build_mean_estimation_data():
    distributions = ["normal", "exponential", "chi_squared", "uniform"]
    train_sizes = [1000, 1000, 800, 200]
    test_sizes = [1000, 1000, 1000, 1000]

    train_data_parts = []
    train_label_parts = []
    tensor_datasets = []
    for dist, size in zip(distributions, train_sizes):
        data, labels = _generate_synthetic_data(size, dist)
        train_data_parts.append(data)
        train_label_parts.append(labels)
        tensor_datasets.append(
            TensorDataset(
                torch.tensor(data, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32),
            )
        )

    train_data = np.concatenate(train_data_parts, axis=0)
    train_labels = np.concatenate(train_label_parts, axis=0)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32),
        ),
        batch_size=500,
        shuffle=True,
    )

    test_data_parts = []
    test_label_parts = []
    for dist, size in zip(distributions, test_sizes):
        data, labels = _generate_synthetic_data(size, dist)
        test_data_parts.append(data)
        test_label_parts.append(labels)
    test_data = np.concatenate(test_data_parts, axis=0)
    test_labels = np.concatenate(test_label_parts, axis=0)
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.float32),
        ),
        batch_size=500,
        shuffle=False,
    )
    return tensor_datasets, train_loader, test_loader


def _build_wine_quality_data():
    red = pd.read_csv("./data/winequality-red.csv", sep=";")
    white = pd.read_csv("./data/winequality-white.csv", sep=";")
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    df = pd.concat([red, white], axis=0).dropna()
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce").astype(float)

    x = df.iloc[:, :11].values
    y = df.iloc[:, 11].values
    wine_type = df.iloc[:, 12]
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test, type_train, _ = train_test_split(
        x, y, wine_type, test_size=0.5, random_state=42, stratify=wine_type
    )

    proportions = {"white": 0.1, "red": 1.0}
    subset_list = []
    for category, proportion in proportions.items():
        mask = type_train == category
        cx, cy, ct = x_train[mask], y_train[mask], type_train[mask]
        n = int(len(cx) * proportion)
        subset_list.append((cx[:n], cy[:n], ct[:n]))

    x_train = np.vstack([s[0] for s in subset_list])
    y_train = np.hstack([s[1] for s in subset_list])
    type_train = np.hstack([s[2] for s in subset_list])

    tensor_datasets = []
    for category in list(set(type_train)):
        mask = type_train == category
        tensor_datasets.append(
            TensorDataset(
                torch.tensor(x_train[mask], dtype=torch.float32),
                torch.tensor(y_train[mask], dtype=torch.float32).view(-1, 1),
            )
        )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=100,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=100,
        shuffle=False,
    )
    return tensor_datasets, train_loader, test_loader


def _build_california_housing_data():
    df = pd.read_csv("./data/housing.csv").dropna()
    df = df[df["ocean_proximity"] != "ISLAND"]
    x = df.iloc[:, :8].values
    y = df.iloc[:, 8].values
    ocean = df.iloc[:, 9]
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test, ocean_train, _ = train_test_split(
        x, y, ocean, test_size=0.5, random_state=42
    )

    proportions = {"<1H OCEAN": 0.05, "INLAND": 1.0, "NEAR BAY": 0.05, "NEAR OCEAN": 0.05}
    subset_list = []
    for category, proportion in proportions.items():
        mask = ocean_train == category
        cx, cy, co = x_train[mask], y_train[mask], ocean_train[mask]
        n = int(len(cx) * proportion)
        subset_list.append((cx[:n], cy[:n], co[:n]))

    x_train = np.vstack([s[0] for s in subset_list])
    y_train = np.hstack([s[1] for s in subset_list])
    ocean_train = np.hstack([s[2] for s in subset_list])

    tensor_datasets = []
    for category in list(set(ocean_train)):
        mask = ocean_train == category
        tensor_datasets.append(
            TensorDataset(
                torch.tensor(x_train[mask], dtype=torch.float32),
                torch.tensor(y_train[mask], dtype=torch.float32).view(-1, 1),
            )
        )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=1000,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=1000,
        shuffle=False,
    )
    return tensor_datasets, train_loader, test_loader


def _build_reconstruction_data(dataset_name: str):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist-reconstruction":
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        factors = [1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2]
    elif dataset_name == "fashion-mnist-reconstruction":
        train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        factors = [1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2]
    else:
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        factors = [0.1, 0.1, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1]

    k = 10
    class_data = [[] for _ in range(k)]
    for data, label in train_ds:
        class_data[label].append(data)
    unbalanced = []
    for i in range(k):
        keep = int(len(class_data[i]) * factors[i])
        unbalanced.append(torch.stack(random.sample(class_data[i], keep)))
    tensor_datasets = [
        TensorDataset(data, torch.full((data.size(0),), label, dtype=torch.long))
        for label, data in enumerate(unbalanced)
    ]
    train_concat = torch.utils.data.ConcatDataset(tensor_datasets)
    train_loader = DataLoader(train_concat, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    return tensor_datasets, train_loader, test_loader


EXPERIMENTS: dict[str, L2MRegressionConfig] = {
    "mean-estimation": L2MRegressionConfig(
        name="mean-estimation",
        epochs=500,
        iterations=5,
        batch_size=500,
        alpha_lr=1e-2,
        lr=5e-5,
        model_factory=lambda: SimpleNN(10),
        build_data=_build_mean_estimation_data,
    ),
    "wine-quality": L2MRegressionConfig(
        name="wine-quality",
        epochs=300,
        iterations=5,
        batch_size=100,
        alpha_lr=5e-2,
        lr=1e-4,
        model_factory=lambda: SimpleNN(11),
        build_data=_build_wine_quality_data,
    ),
    "california-housing": L2MRegressionConfig(
        name="california-housing",
        epochs=1200,
        iterations=5,
        batch_size=1000,
        alpha_lr=5e-2,
        lr=5e-5,
        model_factory=lambda: SimpleNN(8),
        build_data=_build_california_housing_data,
    ),
    "mnist-reconstruction": L2MRegressionConfig(
        name="mnist-reconstruction",
        epochs=40,
        iterations=5,
        batch_size=1000,
        alpha_lr=1e-1,
        lr=5e-4,
        model_factory=lambda: Autoencoder(1, 28, 28),
        build_data=lambda: _build_reconstruction_data("mnist-reconstruction"),
        optimizer_kind="adamw",
        noise_std=5e-2,
        classical_mode="mixed",
    ),
    "fashion-mnist-reconstruction": L2MRegressionConfig(
        name="fashion-mnist-reconstruction",
        epochs=70,
        iterations=5,
        batch_size=1000,
        alpha_lr=1e-1,
        lr=1e-5,
        model_factory=lambda: Autoencoder(1, 28, 28),
        build_data=lambda: _build_reconstruction_data("fashion-mnist-reconstruction"),
        optimizer_kind="adam",
        noise_std=7e-3,
        classical_mode="full_loader",
    ),
    "cifar10-reconstruction": L2MRegressionConfig(
        name="cifar10-reconstruction",
        epochs=110,
        iterations=5,
        batch_size=1000,
        alpha_lr=1e-1,
        lr=1e-5,
        model_factory=lambda: Autoencoder(3, 32, 32),
        build_data=lambda: _build_reconstruction_data("cifar10-reconstruction"),
        optimizer_kind="adam",
        noise_std=5e-2,
        classical_mode="full_loader",
    ),
}


def _make_optimizer(params, kind: str, lr: float):
    if kind == "adamw":
        return optim.AdamW(params, lr=lr)
    return optim.Adam(params, lr=lr)


def _compute_mean_ci(data: np.ndarray):
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    sem = scipy.stats.sem(data, axis=0)
    ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2.0, data.shape[0] - 1)
    return mean, ci


def run_l2m_regression_experiment(experiment: str) -> int:
    cfg = EXPERIMENTS[experiment]
    tensor_datasets, train_loader, test_loader = cfg.build_data()
    criterion = nn.MSELoss()
    n_total = sum(len(ds) for ds in tensor_datasets)
    k = len(tensor_datasets)
    m_batches = int(np.floor(n_total / cfg.batch_size))

    train_errors = np.zeros((cfg.epochs, cfg.iterations))
    classical_errors = np.zeros((cfg.epochs, cfg.iterations))
    test_errors = np.zeros((cfg.epochs, cfg.iterations))
    classical_test_errors = np.zeros((cfg.epochs, cfg.iterations))

    for iters in range(cfg.iterations):
        model = cfg.model_factory()
        if cfg.noise_std > 0:
            torch.manual_seed(42)
            base = cfg.model_factory()
            model = copy.deepcopy(base)
            for param, base_param in zip(model.parameters(), base.parameters()):
                param.data = base_param.data + torch.randn_like(base_param) * cfg.noise_std
        model_l2m = copy.deepcopy(model)
        optimizer = _make_optimizer(model.parameters(), cfg.optimizer_kind, cfg.lr)
        optimizer_l2m = _make_optimizer(model_l2m.parameters(), cfg.optimizer_kind, cfg.lr)

        alpha = torch.tensor([len(ds) / n_total for ds in tensor_datasets], dtype=torch.float32)
        classical_alpha = torch.tensor([len(ds) / n_total for ds in tensor_datasets], dtype=torch.float32)

        for epoch in range(cfg.epochs):
            prev_indices = np.zeros(k, dtype=int)
            tracking_error = np.zeros(k, dtype=float)
            shuffled_indices = _shuffle_class_data(tensor_datasets)

            model_l2m.train()
            for _ in range(m_batches):
                combined_data = []
                combined_labels = []
                for i in range(k):
                    num_samples = max(int(alpha[i].item() * cfg.batch_size), 0)
                    data_list, labels_list = [], []
                    remaining = num_samples
                    while remaining > 0:
                        available = len(tensor_datasets[i]) - prev_indices[i]
                        take = min(available, remaining)
                        idx = shuffled_indices[i][prev_indices[i] : prev_indices[i] + take]
                        data_list.append(tensor_datasets[i].tensors[0][idx])
                        labels_list.append(tensor_datasets[i].tensors[1][idx])
                        remaining -= take
                        prev_indices[i] += take
                        if prev_indices[i] >= len(tensor_datasets[i]):
                            prev_indices[i] = 0
                    combined_data.append(torch.cat(data_list, dim=0))
                    combined_labels.append(torch.cat(labels_list, dim=0))

                optimizer_l2m.zero_grad()
                x_mixed = torch.cat(combined_data, dim=0)
                y_mixed = torch.cat(combined_labels, dim=0)
                z_mixed = model_l2m(x_mixed)
                loss = criterion(z_mixed, x_mixed if cfg.name.endswith("reconstruction") else y_mixed)
                loss.backward()
                optimizer_l2m.step()

                start_idx = 0
                for i in range(k):
                    num_samples = max(int(alpha[i].item() * cfg.batch_size), 0)
                    target = x_mixed if cfg.name.endswith("reconstruction") else y_mixed
                    tracking_error[i] = criterion(
                        z_mixed[start_idx : start_idx + num_samples],
                        target[start_idx : start_idx + num_samples],
                    ).item()
                    start_idx += num_samples

            rewards = tracking_error / np.sum(tracking_error)
            alpha += cfg.alpha_lr * (torch.tensor(rewards, dtype=torch.float32) - alpha)

            model.train()
            if cfg.classical_mode == "mixed":
                for _ in range(m_batches):
                    combined_data = []
                    combined_labels = []
                    for i in range(k):
                        num_samples = max(int(classical_alpha[i].item() * cfg.batch_size), 0)
                        data_list, labels_list = [], []
                        remaining = num_samples
                        while remaining > 0:
                            available = len(tensor_datasets[i]) - prev_indices[i]
                            take = min(available, remaining)
                            idx = shuffled_indices[i][prev_indices[i] : prev_indices[i] + take]
                            data_list.append(tensor_datasets[i].tensors[0][idx])
                            labels_list.append(tensor_datasets[i].tensors[1][idx])
                            remaining -= take
                            prev_indices[i] += take
                            if prev_indices[i] >= len(tensor_datasets[i]):
                                prev_indices[i] = 0
                        combined_data.append(torch.cat(data_list, dim=0))
                        combined_labels.append(torch.cat(labels_list, dim=0))
                    optimizer.zero_grad()
                    x_mixed = torch.cat(combined_data, dim=0)
                    y_mixed = torch.cat(combined_labels, dim=0)
                    z_mixed = model(x_mixed)
                    loss = criterion(z_mixed, x_mixed if cfg.name.endswith("reconstruction") else y_mixed)
                    loss.backward()
                    optimizer.step()
            else:
                for x_train, _ in train_loader:
                    optimizer.zero_grad()
                    outputs = model(x_train)
                    loss = criterion(outputs, x_train)
                    loss.backward()
                    optimizer.step()

            l2m_train_loss = classical_train_loss = 0.0
            l2m_test_loss = classical_test_loss = 0.0
            total = total_test = 0
            model.eval()
            model_l2m.eval()
            with torch.no_grad():
                for x_train, y_train in train_loader:
                    l2m_out = model_l2m(x_train)
                    cls_out = model(x_train)
                    total += x_train.size(0)
                    target = x_train if cfg.name.endswith("reconstruction") else y_train
                    l2m_train_loss += F.mse_loss(l2m_out, target, reduction="sum").item()
                    classical_train_loss += F.mse_loss(cls_out, target, reduction="sum").item()
                for x_test, y_test in test_loader:
                    l2m_out = model_l2m(x_test)
                    cls_out = model(x_test)
                    total_test += x_test.size(0)
                    target = x_test if cfg.name.endswith("reconstruction") else y_test
                    l2m_test_loss += F.mse_loss(l2m_out, target, reduction="sum").item()
                    classical_test_loss += F.mse_loss(cls_out, target, reduction="sum").item()

            train_errors[epoch, iters] = l2m_train_loss / total
            classical_errors[epoch, iters] = classical_train_loss / total
            test_errors[epoch, iters] = l2m_test_loss / total_test
            classical_test_errors[epoch, iters] = classical_test_loss / total_test
            print(
                f"Iteration {iters}, Epoch {epoch}, Cls Error: {classical_test_errors[epoch, iters]:.4f}, "
                f"L2M Error: {test_errors[epoch, iters]:.4f}, Alpha: {np.round(alpha.detach().numpy(),4)}"
            )

    mean_l2m_test, _ = _compute_mean_ci(test_errors.T)
    print(f"Mean L2M: {mean_l2m_test[-1].round(4)}")

    np.savetxt(f"l2m_train_errors_{cfg.name}.csv", train_errors, delimiter=",")
    np.savetxt(f"classical_train_errors_{cfg.name}.csv", classical_errors, delimiter=",")
    np.savetxt(f"l2m_test_errors_{cfg.name}.csv", test_errors, delimiter=",")
    np.savetxt(f"classical_test_errors_{cfg.name}.csv", classical_test_errors, delimiter=",")
    return 0

