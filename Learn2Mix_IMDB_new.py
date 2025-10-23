import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import time
import random
import math
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from typing import List
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Download NLTK tokenizers
nltk.download('punkt_tab')
nltk.download('punkt')

######################################################################
# Global helper definitions (these remain outside the main block)
######################################################################
# ------- 1) L2M helper -------
def shuffle_class_data(tensor_datasets):
    shuffled_indices = []
    for ds in tensor_datasets:
        num_samples = ds.tensors[0].size(0)
        shuffled_indices.append(torch.randperm(num_samples))
    return shuffled_indices

def compute_class_counts(tensor_datasets):
    class_counts_ = {}
    for dataset_ in tensor_datasets:
        _, labels_ = dataset_.tensors
        labels_ = labels_.numpy()
        for l_ in labels_:
            class_counts_[l_] = class_counts_.get(l_, 0) + 1
    return class_counts_

# ------- 2) CBL (Class-balanced Focal Loss) -------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# ------- 3) Transformer-based Model for IMDB (instead of LeNet) -------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=1, num_classes=2, max_seq_length=500, pad_idx=0):
        super(TransformerNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        embedded = self.pos_encoder(embedded)  # (batch_size, seq_length, embed_dim)
        transformer_out = self.transformer_encoder(embedded)  # (batch_size, seq_length, embed_dim)
        pooled = torch.mean(transformer_out, dim=1)  # aggregate over sequence length
        pooled = self.dropout(pooled)
        output = self.fc(pooled)  # (batch_size, num_classes)
        return output

# ------- 4) Tokenization and Preprocessing functions for IMDB -------
def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())

def build_vocab(tokenized_texts: List[List[str]], min_freq: int) -> dict:
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, token in enumerate(tokens, start=2):
        vocab[token] = idx
    return vocab

def preprocess_data(df: pd.DataFrame, vocab: dict, max_seq_length: int) -> (torch.Tensor, torch.Tensor):
    texts = []
    labels = []
    for _, row in df.iterrows():
        tokens = tokenize(row['review'])
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
        if len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]
        else:
            token_ids += [vocab["<pad>"]] * (max_seq_length - len(token_ids))
        texts.append(token_ids)
        labels.append(row['sentiment'])
    return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


######################################################################
# Worker function: All initialization and training for one iteration occurs here.
######################################################################
def run_iteration(iter_idx, epochs, chosen_method, return_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #########################
    # IMDB Dataset Loading  #
    #########################
    # Parameters for IMDB
    max_seq_length = 500
    vocab_min_freq = 2
    batch_size = 500
    k = 2  # number of classes (negative and positive)
    unbalance_factors = [0.3, 1]  # unbalance factors for the two classes

    # Load IMDB dataset
    data = pd.read_csv('./data/IMDB.csv')
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])
    
    # Tokenize and build vocabulary using training data
    tokenized_train_texts = [tokenize(text) for text in train_df['review']]
    vocab = build_vocab(tokenized_train_texts, vocab_min_freq)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Preprocess training and testing data
    train_texts, train_labels = preprocess_data(train_df, vocab, max_seq_length)
    test_texts, test_labels = preprocess_data(test_df, vocab, max_seq_length)
    
    # Split training data by class
    class_data = [[] for _ in range(k)]
    for text, label in zip(train_texts, train_labels):
        class_data[label.item()].append(text)
    
    # Make the classes unbalanced based on the defined unbalance factors
    unbalanced_class_data = []
    for i in range(k):
        total_samples = len(class_data[i])
        samples_to_keep = int(total_samples * unbalance_factors[i])
        selected_data = random.sample(class_data[i], samples_to_keep)
        unbalanced_class_data.append(torch.stack(selected_data))
    
    # Create TensorDatasets for each class
    tensor_datasets = [TensorDataset(data, torch.full((data.size(0),), label, dtype=torch.long)) for label, data in enumerate(unbalanced_class_data)]
    
    # Create train and test datasets/loaders
    train_dataset = torch.utils.data.ConcatDataset(tensor_datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine number of batches per epoch.
    N_total = sum(len(ds) for ds in tensor_datasets)
    M = int(np.floor(N_total / batch_size))
    lr = 1e-4
    model = TransformerNN(vocab_size=vocab_size, max_seq_length=max_seq_length, pad_idx=vocab["<pad>"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Method-specific initialization:
    if chosen_method == "L2M":
        alpha = torch.tensor([len(ds) / N_total for ds in tensor_datasets], dtype=torch.float32).to(device)
        alpha_lr = 5e-1
        
    elif chosen_method == "CBL":
        class_counts = compute_class_counts(tensor_datasets)
        classes = sorted(class_counts.keys())
        def compute_class_weights(class_counts_):
            counts = np.array([class_counts_[c] for c in classes], dtype=np.float32)
            weights = 1.0 / (counts + 1e-6)
            weights = weights / np.sum(weights) * len(classes)
            return torch.tensor(weights, dtype=torch.float32).to(device)
        class_weights = compute_class_weights(class_counts)
        gamma = 2
        focal_criterion = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean').to(device)
        
    elif chosen_method == "SMOTE":
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
        
    elif chosen_method == "IS":
        N_IS = batch_size
        N_subset = batch_size // 2
        IS_loader = DataLoader(train_dataset, batch_size=N_IS, shuffle=True)
        
    elif chosen_method == "CL":
        # Self-taught Curriculum Learning
        warmup_epochs = 50  # can be tuned
        print("Self-taught scoring function")
        for warmup_ep in range(warmup_epochs):
            model.train()
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            print(f"[Warmup Epoch: {warmup_ep+1}/{warmup_epochs}], Loss: {loss.item():.4f}")

        print("Computing self-taught scores for curriculum ordering...")
        model.eval()
        all_confidences = []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                conf = probs[range(len(lbls)), lbls]
                all_confidences.append(conf.cpu())
        all_confidences = torch.cat(all_confidences)
        self_taught_scores = 1 - all_confidences.numpy()
        sorted_indices = np.argsort(self_taught_scores)
        model = TransformerNN(vocab_size=vocab_size, max_seq_length=max_seq_length, pad_idx=vocab["<pad>"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        starting_percent = 0.5
        inc = 1.2
        step_length = 10
        
    elif chosen_method == "classical": pass

    # Preallocate local results arrays.
    train_err = np.zeros(epochs)
    test_err = np.zeros(epochs)
    time_arr = np.zeros(epochs)
    total_time = 0.0

    # ----------------------------------------------------------------
    # Training loop: iterate over epochs.
    for epoch in range(epochs):
        correct_train = 0; correct_test = 0; total_train = 0; total_test = 0

        if chosen_method == "L2M":
            start = time.time()
            model.train()
            shuffled_indices = shuffle_class_data(tensor_datasets)
            prev_indices = np.zeros(k, dtype=int)
            tracking_error = np.zeros(k, dtype=float)
            for j in range(M):
                combined_data = []; combined_labels = []
                for i in range(k):
                    num_samples = max(int(alpha[i].item() * batch_size), 0)
                    data_list = []; labels_list = []
                    remaining = num_samples
                    while remaining > 0:
                        available = len(tensor_datasets[i]) - prev_indices[i]
                        take = min(available, remaining)
                        indices = shuffled_indices[i][prev_indices[i]:prev_indices[i] + take]
                        data_list.append(tensor_datasets[i].tensors[0][indices])
                        labels_list.append(tensor_datasets[i].tensors[1][indices])
                        remaining -= take
                        prev_indices[i] += take
                        if prev_indices[i] >= len(tensor_datasets[i]):
                            prev_indices[i] = 0
                    combined_data.append(torch.cat(data_list, dim=0))
                    combined_labels.append(torch.cat(labels_list, dim=0))
                optimizer.zero_grad()
                x_mixed = torch.cat(combined_data, dim=0).to(device)
                y_mixed = torch.cat(combined_labels, dim=0).to(device)
                z_mixed = model(x_mixed)
                loss = criterion(z_mixed, y_mixed)
                loss.backward()
                optimizer.step()

            start_idx = 0
            for i in range(k):
                num_samples = max(int(alpha[i].item() * batch_size), 0)
                tracking_error[i] = criterion(z_mixed[start_idx:start_idx + num_samples], y_mixed[start_idx:start_idx + num_samples]).item()
                start_idx += num_samples

            rewards = (tracking_error / np.sum(tracking_error)) if np.sum(tracking_error) > 0 else np.ones(k)/k
            alpha += alpha_lr * (torch.tensor(rewards, dtype=torch.float32).to(device) - alpha)
            
            diff = time.time() - start
            total_time += diff
            time_arr[epoch] = total_time

        elif chosen_method == "CBL":
            start = time.time()
            model.train()
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = focal_criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            diff = time.time() - start
            total_time += diff
            time_arr[epoch] = total_time

        elif chosen_method == "SMOTE":
            start = time.time()
            model.train()
            smote_iterator = iter(smote_loader)
            for j in range(M):
                try:
                    X_train, y_train = next(smote_iterator)
                except StopIteration:
                    smote_iterator = iter(smote_loader)
                    X_train, y_train = next(smote_iterator)
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            diff = time.time() - start
            total_time += diff
            time_arr[epoch] = total_time

        elif chosen_method == "IS":
            start = time.time()
            model.train()
            for X_train, y_train in IS_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                model.eval()
                with torch.no_grad():
                    logits_large = model(X_train)
                    losses_large = nn.functional.cross_entropy(logits_large, y_train, reduction='none')
                    p_large = losses_large / losses_large.sum()
                    subset_indices = torch.multinomial(p_large, N_subset, replacement=False)
                model.train()
                X_subset = X_train[subset_indices]
                Y_subset = y_train[subset_indices]
                outputs_subset = model(X_subset)
                loss_subset = nn.functional.cross_entropy(outputs_subset, Y_subset, reduction='mean')
                optimizer.zero_grad()
                loss_subset.backward()
                optimizer.step()
            diff = time.time() - start
            total_time += diff
            time_arr[epoch] = total_time

        elif chosen_method == "CL":
            start = time.time()
            model.train()
            subset_frac = min(starting_percent * (inc ** (epoch // step_length)), 1.0)
            subset_size = int(subset_frac * len(train_dataset))
            curriculum_subset = Subset(train_dataset, sorted_indices[:subset_size])
            curriculum_loader = DataLoader(curriculum_subset, batch_size=batch_size, shuffle=True)
            for X_train, y_train in curriculum_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            diff = time.time() - start
            total_time += diff
            time_arr[epoch] = total_time

        elif chosen_method == "classical":
            start = time.time()
            model.train()
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            diff = time.time() - start
            total_time += diff
            time_arr[epoch] = total_time

        # Evaluate training and test accuracy.
        with torch.no_grad():
            model.eval()
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                total_train += y_train.size(0)
                out_train = model(X_train)
                _, predicted = torch.max(out_train, 1)
                correct_train += (predicted == y_train).sum().item()
                
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                total_test += y_test.size(0)
                out_test = model(X_test)
                _, pred_test = torch.max(out_test, 1)
                correct_test += (pred_test == y_test).sum().item()
                
        train_err[epoch] = 100.0 * correct_train / total_train
        test_err[epoch]  = 100.0 * correct_test / total_test

        if chosen_method == "L2M":
            print(f"Iter {iter_idx}, Epoch {epoch}, L2M train/test: {train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}, Alpha: {np.round(alpha.detach().cpu().numpy(),3)}")
        elif chosen_method == "CBL":
            print(f"Iter {iter_idx}, Epoch {epoch}, CBL train/test: {train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}")
        elif chosen_method == "SMOTE":
            print(f"Iter {iter_idx}, Epoch {epoch}, SMOTE train/test: {train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}")
        elif chosen_method == "IS":
            print(f"Iter {iter_idx}, Epoch {epoch}, IS train/test: {train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}")
        elif chosen_method == "CL":
            print(f"Iter {iter_idx}, Epoch {epoch}, CL train/test: {train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}")
        elif chosen_method == "classical":
            print(f"Iter {iter_idx}, Epoch {epoch}, Classical train/test: {train_err[epoch]:.2f}/{test_err[epoch]:.2f}, Total time: {time_arr[epoch]:.4f}")

    return_dict[iter_idx] = (train_err, test_err, time_arr)


######################################################################
# MAIN BLOCK: Only allocate the final result arrays.
######################################################################
if __name__ == '__main__':
    chosen_method = input("Choose method (L2M, CBL, SMOTE, IS, CL, classical): ")

    if chosen_method == "L2M":
        epochs = 40; iterations = 5
        train_errors = np.zeros((epochs, iterations))
        test_errors  = np.zeros((epochs, iterations))
        time_l2m     = np.zeros((epochs, iterations))
    elif chosen_method == "CBL":
        epochs = 40; iterations = 5
        balanced_errors = np.zeros((epochs, iterations))
        balanced_test_errors = np.zeros((epochs, iterations))
        time_cbl = np.zeros((epochs, iterations))
    elif chosen_method == "SMOTE":
        epochs = 40; iterations = 5
        smote_errors = np.zeros((epochs, iterations))
        smote_test_errors = np.zeros((epochs, iterations))
        time_smote = np.zeros((epochs, iterations))
    elif chosen_method == "IS":
        epochs = 48; iterations = 5
        is_train_errors = np.zeros((epochs, iterations))
        is_test_errors = np.zeros((epochs, iterations))
        time_is = np.zeros((epochs, iterations))
    elif chosen_method == "CL":
        epochs = 53; iterations = 5
        cl_train_errors = np.zeros((epochs, iterations))
        cl_test_errors = np.zeros((epochs, iterations))
        time_cl = np.zeros((epochs, iterations))
    elif chosen_method == "classical":
        epochs = 40; iterations = 5
        classical_errors = np.zeros((epochs, iterations))
        classical_test_errors = np.zeros((epochs, iterations))
        time_classical = np.zeros((epochs, iterations))

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []
    for iters_ in range(iterations):
        p = multiprocessing.Process(target=run_iteration, args=(iters_, epochs, chosen_method, return_dict))
        procs.append(p)
        p.start()
        p.join()  # run sequentially to free memory between iterations

    if chosen_method == "L2M":
        for iters_ in range(iterations):
            train_errors[:, iters_], test_errors[:, iters_], time_l2m[:, iters_] = return_dict[iters_]
        np.savetxt('train_errors_imdb.csv', train_errors, delimiter=',')
        np.savetxt('test_errors_imdb.csv', test_errors, delimiter=',')
        np.savetxt('time_l2m_imdb.csv', time_l2m, delimiter=',')
    elif chosen_method == "CBL":
        for iters_ in range(iterations):
            balanced_errors[:, iters_], balanced_test_errors[:, iters_], time_cbl[:, iters_] = return_dict[iters_]
        np.savetxt('balanced_errors_imdb.csv', balanced_errors, delimiter=',')
        np.savetxt('balanced_test_errors_imdb.csv', balanced_test_errors, delimiter=',')
        np.savetxt('time_cbl_imdb.csv', time_cbl, delimiter=',')
    elif chosen_method == "SMOTE":
        for iters_ in range(iterations):
            smote_errors[:, iters_], smote_test_errors[:, iters_], time_smote[:, iters_] = return_dict[iters_]
        np.savetxt('smote_errors_imdb.csv', smote_errors, delimiter=',')
        np.savetxt('smote_test_errors_imdb.csv', smote_test_errors, delimiter=',')
        np.savetxt('time_smote_imdb.csv', time_smote, delimiter=',')
    elif chosen_method == "IS":
        for iters_ in range(iterations):
            is_train_errors[:, iters_], is_test_errors[:, iters_], time_is[:, iters_] = return_dict[iters_]
        np.savetxt('is_train_errors_imdb.csv', is_train_errors, delimiter=',')
        np.savetxt('is_test_errors_imdb.csv', is_test_errors, delimiter=',')
        np.savetxt('time_is_imdb.csv', time_is, delimiter=',')
    elif chosen_method == "CL":
        for iters_ in range(iterations):
            cl_train_errors[:, iters_], cl_test_errors[:, iters_], time_cl[:, iters_] = return_dict[iters_]
        np.savetxt('cl_train_errors_imdb.csv', cl_train_errors, delimiter=',')
        np.savetxt('cl_test_errors_imdb.csv', cl_test_errors, delimiter=',')
        np.savetxt('time_cl_imdb.csv', time_cl, delimiter=',')
    elif chosen_method == "classical":
        for iters_ in range(iterations):
            classical_errors[:, iters_], classical_test_errors[:, iters_], time_classical[:, iters_] = return_dict[iters_]
        np.savetxt('classical_errors_imdb.csv', classical_errors, delimiter=',')
        np.savetxt('classical_test_errors_imdb.csv', classical_test_errors, delimiter=',')
        np.savetxt('time_classical_imdb.csv', time_classical, delimiter=',')