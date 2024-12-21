import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import nltk
from typing import List
from imblearn.over_sampling import SMOTE
nltk.download('punkt_tab')
nltk.download('punkt')  # Download tokenizer data

# Load IMDB dataset
data = pd.read_csv('./data/IMDB.csv')
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])
max_seq_length = 500  # Maximum sequence length for padding/truncating
vocab_min_freq = 2  # Minimum frequency for a token to be included in the vocabulary
batch_size = 500
k = 2  # Number of classes for sentiment analysis
unbalance_factors = [0.3, 1]

# Tokenizer
def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())

# Build vocabulary from training data
def build_vocab(tokenized_texts: List[List[str]], min_freq: int) -> dict:
    counter = Counter()
    for tokens in tokenized_texts: counter.update(tokens)
    # Include tokens that meet the minimum frequency
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    # Create token to index mapping
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, token in enumerate(tokens, start=2): vocab[token] = idx
    return vocab

# Preprocess data: tokenize, convert to indices, pad/truncate
def preprocess_data(df: pd.DataFrame, vocab: dict, max_seq_length: int) -> (torch.Tensor, torch.Tensor):
    texts = []; labels = []
    for _, row in df.iterrows():
        tokens = tokenize(row['review'])
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        # Truncate or pad sequences
        if len(token_ids) > max_seq_length: token_ids = token_ids[:max_seq_length]
        else: token_ids += [vocab["<pad>"]] * (max_seq_length - len(token_ids))
        texts.append(token_ids)
        labels.append(row['sentiment'])
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# Tokenize training texts
tokenized_train_texts = [tokenize(text) for text in train_df['review']]

# Build vocabulary
vocab = build_vocab(tokenized_train_texts, vocab_min_freq)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Preprocess training and testing data
train_texts, train_labels = preprocess_data(train_df, vocab, max_seq_length)
test_texts, test_labels = preprocess_data(test_df, vocab, max_seq_length)

# Split training data by class
class_data = [[] for _ in range(k)]
for text, label in zip(train_texts, train_labels):
    class_data[label].append(text)

# Make the classes unbalanced based on the defined unbalance factors
unbalanced_class_data = []
for i in range(k):
    total_samples = len(class_data[i])
    # Calculate the number of samples to keep for the class based on the unbalance factor
    samples_to_keep = int(total_samples * unbalance_factors[i])
    # Randomly select the subset of data
    selected_data = random.sample(class_data[i], samples_to_keep)
    unbalanced_class_data.append(torch.stack(selected_data))

# Create TensorDatasets for each class
tensor_datasets = [TensorDataset(data, torch.full((data.size(0),), label, dtype=torch.long)) for label, data in enumerate(unbalanced_class_data)]
train_dataset = torch.utils.data.ConcatDataset(tensor_datasets)

# Combine all training data for a single DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a DataLoader for the test data directly
test_dataset = TensorDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
X_smote_tensor = torch.tensor(X_smote).view(-1, 500)
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

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1: pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else: pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# Define the transformer-based neural network model for classification
class TransformerNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=1, num_classes=2, max_seq_length=500):
        super(TransformerNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),  # Reduced from 64 to 32
            nn.ReLU(),
            nn.Linear(32, num_classes)  # Output layer for 2 classes
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        embedded = self.pos_encoder(embedded)  # (batch_size, seq_length, embed_dim)
        transformer_out = self.transformer_encoder(embedded)  # (batch_size, seq_length, embed_dim)
        # Aggregate transformer outputs (e.g., take mean over the sequence)
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, embed_dim)
        pooled = self.dropout(pooled)
        output = self.fc(pooled)  # (batch_size, num_classes)
        return output

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
epochs = 40
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
    model = TransformerNN(vocab_size=vocab_size).to(device)
    model_l2m = TransformerNN(vocab_size=vocab_size).to(device)
    model_cbl = TransformerNN(vocab_size=vocab_size).to(device)
    model_smote = TransformerNN(vocab_size=vocab_size).to(device)
    model.load_state_dict(model_l2m.state_dict())
    model_cbl.load_state_dict(model_l2m.state_dict())
    model_smote.load_state_dict(model_l2m.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer_l2m = optim.Adam(model_l2m.parameters(), lr=1e-4)
    optimizer_cbl = optim.Adam(model_cbl.parameters(), lr=1e-4)
    optimizer_smote = optim.Adam(model_smote.parameters(), lr=1e-4)
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
plt.savefig('test_imdb.png', bbox_inches='tight')

np.savetxt('train_errors_imdb.csv', train_errors, delimiter=',')
np.savetxt('classical_errors_imdb.csv', classical_errors, delimiter=',')
np.savetxt('balanced_errors_imdb.csv', balanced_errors, delimiter=',')
np.savetxt('smote_errors_imdb.csv', smote_errors, delimiter=',')
np.savetxt('test_errors_imdb.csv', test_errors, delimiter=',')
np.savetxt('classical_test_errors_imdb.csv', classical_test_errors, delimiter=',')
np.savetxt('balanced_test_errors_imdb.csv', balanced_test_errors, delimiter=',')
np.savetxt('smote_test_errors_imdb.csv', smote_test_errors, delimiter=',')