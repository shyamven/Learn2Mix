import torch


def shuffle_class_data(tensor_datasets):
    shuffled_indices = []
    for dataset in tensor_datasets:
        num_samples = dataset.tensors[0].size(0)
        shuffled_indices.append(torch.randperm(num_samples))
    return shuffled_indices


def compute_class_counts(tensor_datasets):
    class_counts = {}
    for dataset in tensor_datasets:
        _, labels = dataset.tensors
        for label in labels.numpy():
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

