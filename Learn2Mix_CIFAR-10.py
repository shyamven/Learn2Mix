import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
from torchvision import datasets, transforms, models
import math
from imblearn.over_sampling import SMOTE

# Load CIFAR-10 dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15), transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5,scale=(0.02, 0.1),value=1.0, inplace=False)])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
batch_size = 1000
k = 10  # Number of classes

# Split CIFAR-10 by class
class_data = [[] for _ in range(k)]
for data, label in cifar10_train:
    class_data[label].append(data)
class_data = [torch.stack(data) for data in class_data]

# Create TensorDatasets for each class
tensor_datasets = [TensorDataset(data, torch.full((data.size(0),), label, dtype=torch.long)) for label, data in enumerate(class_data)]

# Combine all training data for a single DataLoader
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)

# Create a DataLoader for the test data directly
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

# Create a Dataloader for SMOTE
def extract_data_labels(concat_dataset):
    data_list = []
    labels_list = []
    for data, label in concat_dataset:
        data_list.append(data)
        labels_list.append(label)
    X = np.array(torch.cat(data_list, dim=0))
    y = np.array(torch.cat(labels_list, dim=0))
    return X, y

X_train_flat, y_train_flat = extract_data_labels(train_loader)
X_train_flat = X_train_flat.reshape(len(X_train_flat), -1)
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train_flat, y_train_flat)
X_smote_tensor = torch.tensor(X_smote, dtype=torch.float32).view(-1, 3, 32, 32)
y_smote_tensor = torch.tensor(y_smote, dtype=torch.long)
smote_dataset = TensorDataset(X_smote_tensor, y_smote_tensor)
smote_loader = DataLoader(smote_dataset, batch_size=batch_size, shuffle=True)

# Shuffle indices at the start of each epoch
def shuffle_class_data(tensor_datasets):
    shuffled_indices = []
    for dataset in tensor_datasets:
        num_samples = dataset.tensors[0].size(0)
        # Shuffle indices for the current class
        shuffled_indices.append(torch.randperm(num_samples))
    return shuffled_indices

# Define the neural network model for classification
class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
      self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
      self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
      self.fc1 = nn.Linear(4*4*64, 500)
      self.dropout1 = nn.Dropout(0.5)
      self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*64)
      x = F.relu(self.fc1(x))
      x = self.dropout1(x)
      x = self.fc2(x)
      return x

# Define Focal Loss as per the original paper
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t is the probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Compute class frequencies across all datasets
def compute_class_counts(tensor_datasets):
    class_counts = {}
    for dataset in tensor_datasets:
        _, labels = dataset.tensors
        labels = labels.numpy()
        for label in labels: class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

class_counts = compute_class_counts(tensor_datasets)
num_classes = len(class_counts)
classes = sorted(class_counts.keys())

# Calculate class weights (alpha) inversely proportional to class frequency
def compute_class_weights(class_counts):
    counts = np.array([class_counts[c] for c in classes], dtype=np.float32)
    weights = 1.0 / (counts + 1e-6) # Inverse frequency
    weights = weights / np.sum(weights) * num_classes # Normalize weights
    return torch.tensor(weights, dtype=torch.float32).to(device)

class_weights = compute_class_weights(class_counts)

# Initialize Focal Loss with computed class weights and gamma
gamma = 2  # As per the original paper
focal_criterion = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean').to(device)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
N_total = sum(len(dataset) for dataset in tensor_datasets)

# Initialize training and test losses
epochs = 125
iterations = 5
M = int(np.floor(N_total / batch_size))  # Number of training batches
train_errors = np.zeros((epochs, iterations))
classical_errors = np.zeros((epochs, iterations))
balanced_errors = np.zeros((epochs, iterations))
smote_errors = np.zeros((epochs, iterations))
test_errors = np.zeros((epochs, iterations))
classical_test_errors = np.zeros((epochs, iterations))
balanced_test_errors = np.zeros((epochs, iterations))
smote_test_errors = np.zeros((epochs, iterations))

for iters in range(iterations):
    # Initialize the model and optimizer
    model = LeNet().to(device)
    model_l2m = LeNet().to(device)
    model_cbl = LeNet().to(device)
    model_smote = LeNet().to(device)
    model.load_state_dict(model_l2m.state_dict())
    model_cbl.load_state_dict(model_l2m.state_dict())
    model_smote.load_state_dict(model_l2m.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=7e-5)
    optimizer_l2m = optim.Adam(model_l2m.parameters(), lr=7e-5)
    optimizer_cbl = optim.Adam(model_cbl.parameters(), lr=7e-5)
    optimizer_smote = optim.Adam(model_smote.parameters(), lr=7e-5)
    alpha = torch.tensor([len(dataset) / N_total for dataset in tensor_datasets], dtype=torch.float32).to(device)
    classical_alpha = torch.tensor([len(dataset) / N_total for dataset in tensor_datasets], dtype=torch.float32).to(device)
    alpha_lr = 1e-1

    # Training loop
    for epoch in range(epochs):
        correct_old = 0; correct_l2m = 0; correct_cbl = 0; correct_smote = 0; total = 0; 
        correct_test_old = 0; correct_test_l2m = 0; correct_test_cbl = 0; correct_test_smote = 0; total_test = 0
        prev_indices = np.zeros(k, dtype=int)
        tracking_error = np.zeros(k, dtype=float)
        tracking_error_cbl = np.zeros(k, dtype=float)
        shuffled_indices = shuffle_class_data(tensor_datasets)
        
        # Learn2Mix Training
        model_l2m.train()
        for j in range(M):  # For each batch in an epoch
            combined_data = []; combined_labels = []
            for i in range(k):  # For each dataset
                num_samples = max(int(alpha[i].item() * batch_size), 0)  # Number of samples from dataset 'i'
                data_list = []; labels_list = []; remaining = num_samples
                while remaining > 0:
                    available = len(tensor_datasets[i]) - prev_indices[i]
                    take = min(available, remaining)
                    indices = shuffled_indices[i][prev_indices[i]:prev_indices[i] + take]
                    data_list.append(tensor_datasets[i].tensors[0][indices].to(device))
                    labels_list.append(tensor_datasets[i].tensors[1][indices].to(device))
                    
                    # Update the remaining samples and the previous index, and wrap around if the end of the dataset is reached
                    remaining -= take; prev_indices[i] += take
                    if prev_indices[i] >= len(tensor_datasets[i]): prev_indices[i] = 0  # Reset to start
                
                # Concatenate all collected data and labels
                combined_data.append(torch.cat(data_list, dim=0))
                combined_labels.append(torch.cat(labels_list, dim=0))
                
            # Inner optimization step: update model parameters
            optimizer_l2m.zero_grad()
            x_mixed = torch.cat(combined_data, dim=0)
            y_mixed = torch.cat(combined_labels, dim=0)
            z_mixed = model_l2m(x_mixed)
            loss = criterion(z_mixed, y_mixed)      
            loss.backward()
            optimizer_l2m.step()

            # Compute class-wise losses
            start_idx = 0
            for i in range(k):
                num_samples = max(int(alpha[i].item() * batch_size), 0)  # Number of samples from dataset 'i'
                tracking_error[i] = criterion(z_mixed[start_idx:start_idx + num_samples], y_mixed[start_idx:start_idx + num_samples]).item()
                start_idx += num_samples
        
            # Outer optimization step: update alpha based on dataset-specific losses
            rewards = tracking_error / np.sum(tracking_error) if np.sum(tracking_error) > 0 else np.ones(k) / k
            alpha += alpha_lr * (torch.tensor(rewards, dtype=torch.float32).to(device) - alpha)

        # Class-balanced focal loss training
        model_cbl.train()
        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            # Forward pass and optimization with Focal Loss
            optimizer_cbl.zero_grad()
            z_cbl = model_cbl(X_train)
            loss_cbl = focal_criterion(z_cbl, y_train)
            loss_cbl.backward()
            optimizer_cbl.step()

        # SMOTE-based NN Training
        model_smote.train()
        smote_iterator = iter(smote_loader)
        for j in range(M):  # Ensure same number of batches
            try: X_train, y_train = next(smote_iterator)
            except StopIteration:
                smote_iterator = iter(smote_loader)
                X_train, y_train = next(smote_iterator)
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            optimizer_smote.zero_grad()
            outputs_smote = model_smote(X_train)
            loss_smote = criterion(outputs_smote, y_train)
            loss_smote.backward()
            optimizer_smote.step()

        # Classical NN Training
        model.train()
        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            classical_outputs = model(X_train)
            loss = criterion(classical_outputs, y_train)
            loss.backward()
            optimizer.step()

        # Computing Train Accuracy
        model.eval(); model_l2m.eval(); model_cbl.eval(); model_smote.eval()
        with torch.no_grad():
            for X_train, y_train in train_loader:
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                train_outputs = model_l2m(X_train)
                classical_outputs = model(X_train)
                balanced_outputs = model_cbl(X_train)
                smote_outputs = model_smote(X_train)

                _, predicted_l2m = torch.max(train_outputs.data, 1)
                _, predicted_old = torch.max(classical_outputs.data, 1)
                _, predicted_cbl = torch.max(balanced_outputs.data, 1)
                _, predicted_smote = torch.max(smote_outputs.data, 1)

                total += y_train.size(0)
                correct_l2m += (predicted_l2m == y_train).sum().item()
                correct_old += (predicted_old == y_train).sum().item()
                correct_cbl += (predicted_cbl == y_train).sum().item()
                correct_smote += (predicted_smote == y_train).sum().item()

        # Computing Test Accuracy
        model.eval(); model_l2m.eval(); model_cbl.eval(); model_smote.eval()
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                test_outputs = model_l2m(X_test)
                classical_outputs = model(X_test)
                balanced_outputs = model_cbl(X_test)
                smote_outputs = model_smote(X_test)

                _, predicted_l2m = torch.max(test_outputs.data, 1)
                _, predicted_old = torch.max(classical_outputs.data, 1)
                _, predicted_cbl = torch.max(balanced_outputs.data, 1)
                _, predicted_smote = torch.max(smote_outputs.data, 1)

                total_test += y_test.size(0)
                correct_test_l2m += (predicted_l2m == y_test).sum().item()
                correct_test_old += (predicted_old == y_test).sum().item()
                correct_test_cbl += (predicted_cbl == y_test).sum().item()
                correct_test_smote += (predicted_smote == y_test).sum().item()

        train_errors[epoch, iters] = 100 * correct_l2m / total
        classical_errors[epoch, iters] = 100 * correct_old / total
        balanced_errors[epoch, iters] = 100 * correct_cbl / total
        smote_errors[epoch, iters] = 100 * correct_smote / total
        test_errors[epoch, iters] = 100 * correct_test_l2m / total_test
        classical_test_errors[epoch, iters] = 100 * correct_test_old / total_test
        balanced_test_errors[epoch, iters] = 100 * correct_test_cbl / total_test
        smote_test_errors[epoch, iters] = 100 * correct_test_smote / total_test

        if epoch % 1 == 0:
            print(f"Iter {iters}, Epoch {epoch}, Cls Acc: {classical_test_errors[epoch, iters]:.3f}, L2M Acc: {test_errors[epoch, iters]:.3f}, CBL Acc: {balanced_test_errors[epoch, iters]:.3f}, SMOTE Acc: {smote_test_errors[epoch, iters]:.3f}, Alpha: {np.round(alpha.detach().cpu().numpy(),3)}")

# Step 2: Compute means and 95% confidence intervals
def compute_mean_and_CI(data):
    mean = np.mean(data, axis=0)
    std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])  # Standard error of the mean
    stdev = np.std(data, axis=0)
    ci = 1.96 * std_error  # 95% CI for a normal distribution
    return mean, ci, stdev

A = classical_errors.T[:,:]; B = train_errors.T[:,:]; C = balanced_errors.T[:,:]; D = smote_errors.T[:,:];
mean_A, ci_A, stdev_A = compute_mean_and_CI(A)
mean_B, ci_B, stdev_B = compute_mean_and_CI(B)
mean_C, ci_C, stdev_C = compute_mean_and_CI(C)
mean_D, ci_D, stdev_D = compute_mean_and_CI(D)

X = classical_test_errors.T[:,:]; Z = test_errors.T[:,:]; Y = balanced_test_errors.T[:,:]; W = smote_test_errors.T[:,:];
mean_X, ci_X, stdev_X = compute_mean_and_CI(X)
mean_Z, ci_Z, stdev_Z = compute_mean_and_CI(Z)
mean_Y, ci_Y, stdev_Y = compute_mean_and_CI(Y)
mean_W, ci_W, stdev_W = compute_mean_and_CI(W)

print(f'Mean DNN t = E:       {mean_X[len(mean_X)-1].round(4)}, STDEV DNN t = E:      {stdev_X[len(stdev_X)-1].round(4)}')
print(f'Mean L2M t = E:       {mean_Z[len(mean_Z)-1].round(4)}, STDEV L2M t = E:      {stdev_Z[len(stdev_Z)-1].round(4)}')
print(f'Mean CBL t = E:       {mean_Y[len(mean_Y)-1].round(4)}, STDEV CBL t = E:      {stdev_Y[len(stdev_Y)-1].round(4)}')
print(f'Mean SMT t = E:       {mean_W[len(mean_W)-1].round(4)}, STDEV SMT t = E:      {stdev_W[len(stdev_W)-1].round(4)}\n')
print(f'Mean DNN t = 0.5E:       {mean_X[int(0.5*len(mean_X))-1].round(4)}, STDEV DNN t = 0.5E:      {stdev_X[int(0.5*len(stdev_X))-1].round(4)}')
print(f'Mean L2M t = 0.5E:       {mean_Z[int(0.5*len(mean_Z))-1].round(4)}, STDEV L2M t = 0.5E:      {stdev_Z[int(0.5*len(stdev_Z))-1].round(4)}')
print(f'Mean CBL t = 0.5E:       {mean_Y[int(0.5*len(mean_Y))-1].round(4)}, STDEV CBL t = 0.5E:      {stdev_Y[int(0.5*len(stdev_Y))-1].round(4)}')
print(f'Mean SMT t = 0.5E:       {mean_W[int(0.5*len(mean_W))-1].round(4)}, STDEV SMT t = 0.5E:      {stdev_W[int(0.5*len(mean_W))-1].round(4)}\n')
print(f'Mean DNN t = 0.25E:       {mean_X[int(0.25*len(mean_X))-1].round(4)}, STDEV DNN t = 0.25E:      {stdev_X[int(0.25*len(stdev_X))-1].round(4)}')
print(f'Mean L2M t = 0.25E:       {mean_Z[int(0.25*len(mean_Z))-1].round(4)}, STDEV L2M t = 0.25E:      {stdev_Z[int(0.25*len(stdev_Z))-1].round(4)}')
print(f'Mean CBL t = 0.25E:       {mean_Y[int(0.25*len(mean_Y))-1].round(4)}, STDEV CBL t = 0.25E:      {stdev_Y[int(0.25*len(stdev_Y))-1].round(4)}')
print(f'Mean SMT t = 0.25E:       {mean_W[int(0.25*len(mean_W))-1].round(4)}, STDEV SMT t = 0.25E:      {stdev_W[int(0.25*len(mean_W))-1].round(4)}\n')

# Step 3: Plot the results
epochs_all = list(range(1, X.shape[1] + 1))
plt.figure(figsize=(10, 6))

# Plotting for A
plt.fill_between(epochs_all, mean_A - ci_A, mean_A + ci_A, color='green', alpha=0.1, zorder=0)
plt.plot(epochs_all, mean_A, 'g-', label="Classical Train Accuracy", zorder=20)

# Plotting for X
plt.fill_between(epochs_all, mean_X - ci_X, mean_X + ci_X, color='blue', alpha=0.1, zorder=0)
plt.plot(epochs_all, mean_X, 'b-', label="Classical Test Accuracy", zorder=20)

# Plotting for C
plt.fill_between(epochs_all, mean_C - ci_C, mean_C + ci_C, color='pink', alpha=0.2, zorder=0)
plt.plot(epochs_all, mean_C, '-', color='pink', label="Focal Train Accuracy", zorder=20)

# Plotting for Y
plt.fill_between(epochs_all, mean_Y - ci_Y, mean_Y + ci_Y, color='purple', alpha=0.1, zorder=0)
plt.plot(epochs_all, mean_Y, '-', color='purple', label="Focal Test Accuracy", zorder=20)

# Plotting for D
plt.fill_between(epochs_all, mean_D - ci_D, mean_D + ci_D, color='skyblue', alpha=0.2, zorder=0)
plt.plot(epochs_all, mean_D, '-', color='skyblue', label="SMOTE Train Accuracy", zorder=20)

# Plotting for W
plt.fill_between(epochs_all, mean_W - ci_W, mean_W + ci_W, color='cyan', alpha=0.1, zorder=0)
plt.plot(epochs_all, mean_W, '-', color='cyan', label="SMOTE Test Accuracy", zorder=20)

# Plotting for B
plt.fill_between(epochs_all, mean_B - ci_B, mean_B + ci_B, color='orange', alpha=0.3, zorder=10)
plt.plot(epochs_all, mean_B, '-', color='orange', label="Learn2Mix Train Accuracy", zorder=30)

# Plotting for Z
plt.fill_between(epochs_all, mean_Z - ci_Z, mean_Z + ci_Z, color='red', alpha=0.3, zorder=10)
plt.plot(epochs_all, mean_Z, 'r-', label="Learn2Mix Test Accuracy", zorder=30)

# Additional plot settings
plt.xlabel("Number of Epochs", fontsize=20)
plt.ylabel("Accuracy (%)", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(prop={'size': 14},loc='lower right',framealpha=0.7).set_zorder(50)
plt.grid(True)
plt.savefig('test_cifar10.png', bbox_inches='tight')

np.savetxt('train_errors_cifar10.csv', train_errors, delimiter=',')
np.savetxt('classical_errors_cifar10.csv', classical_errors, delimiter=',')
np.savetxt('balanced_errors_cifar10.csv', balanced_errors, delimiter=',')
np.savetxt('smote_errors_cifar10.csv', smote_errors, delimiter=',')
np.savetxt('test_errors_cifar10.csv', test_errors, delimiter=',')
np.savetxt('classical_test_errors_cifar10.csv', classical_test_errors, delimiter=',')
np.savetxt('balanced_test_errors_cifar10.csv', balanced_test_errors, delimiter=',')
np.savetxt('smote_test_errors_cifar10.csv', smote_test_errors, delimiter=',')