import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms, models
import numpy as np
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import multiprocessing

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

# ------- 6) LeNet-5 (common model) -------
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

######################################################################
# Worker function: All initialization and training for one iteration occurs here.
######################################################################
def run_iteration(iter_idx, epochs, chosen_method, return_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ----------------------------------------------------------------
    # Initialization that is done separately in each worker process.
    # ----------------------------------------------------------------
    # Load cifar10 dataset with transform.
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15), transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5,scale=(0.02, 0.1),value=1.0, inplace=False)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    batch_size = 1000
    k = 10  # number of classes
    total_time = 0.0

    # Build TensorDatasets for train and test.
    train_data_list = []; train_labels_list = []
    for img, label in cifar10_train:
        train_data_list.append(img)
        train_labels_list.append(label)
    train_data_tensor = torch.stack(train_data_list)
    train_labels_tensor = torch.tensor(train_labels_list, dtype=torch.long)
    train_tensor_dataset = TensorDataset(train_data_tensor, train_labels_tensor)

    test_data_list = []; test_labels_list = []
    for img, label in cifar10_test:
        test_data_list.append(img)
        test_labels_list.append(label)
    test_data_tensor = torch.stack(test_data_list)
    test_labels_tensor = torch.tensor(test_labels_list, dtype=torch.long)
    test_tensor_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    # Create class-specific datasets.
    class_data = [[] for _ in range(k)]
    for data_img, label in cifar10_train:
        class_data[label].append(data_img)
    class_data = [torch.stack(data_) for data_ in class_data]
    tensor_datasets = [TensorDataset(data_, torch.full((data_.size(0),), label, dtype=torch.long)) for label, data_ in enumerate(class_data)]

    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    # Determine number of batches per epoch.
    N_total = sum(len(ds) for ds in tensor_datasets)
    M = int(np.floor(N_total / batch_size))
    lr = 7e-5 # 1e-5
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Method-specific initialization:
    if chosen_method == "L2M":
        alpha = torch.tensor([len(ds) / N_total for ds in tensor_datasets], dtype=torch.float32).to(device)
        alpha_lr = 1e-1
        
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
        # Build SMOTE dataset.
        def extract_data_labels(dataloader):
            data_list, labels_list = [], []
            for data_batch, label_batch in dataloader:
                data_list.append(data_batch)
                labels_list.append(label_batch)
            X = torch.cat(data_list, dim=0)
            y = torch.cat(labels_list, dim=0)
            return X, y
        X_train_flat, y_train_flat = extract_data_labels(train_loader)
        X_train_flat = X_train_flat.view(len(X_train_flat), -1)
        smote = SMOTE()
        X_smote, y_smote = smote.fit_resample(X_train_flat.numpy(), y_train_flat.numpy())
        X_smote_tensor = torch.tensor(X_smote, dtype=torch.float32).view(-1, 3, 32, 32)
        y_smote_tensor = torch.tensor(y_smote, dtype=torch.long)
        smote_dataset = TensorDataset(X_smote_tensor, y_smote_tensor)
        smote_loader = DataLoader(smote_dataset, batch_size=batch_size, shuffle=True)
        
    elif chosen_method == "IS":
        N_IS = batch_size
        N_subset = batch_size // 2
        IS_loader = DataLoader(train_tensor_dataset, batch_size=N_IS, shuffle=True)
        
    elif chosen_method == "CL":
        # Self-taught Curriculum Learning
        warmup_epochs = 50  # number of warm-up epochs (can be tuned)
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

        # Now, compute the self-taught scoring function, computed as 1 - confidence.
        print("Computing self-taught scores for curriculum ordering...")
        model.eval()
        all_confidences = []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                # get the predicted probability of the true label for each example
                conf = probs[range(len(lbls)), lbls]
                all_confidences.append(conf.cpu())
                
        all_confidences = torch.cat(all_confidences)
        self_taught_scores = 1 - all_confidences.numpy() # Define self-taught scores: lower values mean easier examples.
        sorted_indices = np.argsort(self_taught_scores)  # Sort indices so that the easiest examples (highest confidence) come first.
        
        # Reset model and optimizer now that ordering has been determined
        model = LeNet().to(device)
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
            # --- Learn2Mix training: updated version ---
            start = time.time()
            model.train()
            shuffled_indices = shuffle_class_data(tensor_datasets)
            prev_indices = np.zeros(k, dtype=int)
            tracking_error = np.zeros(k, dtype=float)
            # For each batch in an epoch
            for j in range(M):
                combined_data = []
                combined_labels = []
                for i in range(k):
                    num_samples = max(int(alpha[i].item() * batch_size), 0)
                    data_list = []
                    labels_list = []
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
                    
                # Inner optimization step: update model parameters.
                optimizer.zero_grad()
                x_mixed = torch.cat(combined_data, dim=0)
                y_mixed = torch.cat(combined_labels, dim=0)
                x_mixed = x_mixed.to(device)
                y_mixed = y_mixed.to(device)
                z_mixed = model(x_mixed)
                loss = criterion(z_mixed, y_mixed)
                loss.backward()
                optimizer.step()

            # Compute class-wise losses.
            start_idx = 0
            for i in range(k):
                num_samples = max(int(alpha[i].item() * batch_size), 0)
                tracking_error[i] = criterion(z_mixed[start_idx:start_idx + num_samples], y_mixed[start_idx:start_idx + num_samples]).item()
                start_idx += num_samples

            # Outer optimization step
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
            subset_size = int(subset_frac * len(train_tensor_dataset))
            curriculum_subset = Subset(train_tensor_dataset, sorted_indices[:subset_size])
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

        # Evaluate training accuracy.
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

    # Return the results for this iteration.
    return_dict[iter_idx] = (train_err, test_err, time_arr)

######################################################################
# MAIN BLOCK: Only allocate the final result arrays.
######################################################################
if __name__ == '__main__':
    
    chosen_method = input("Choose method (L2M, CBL, SMOTE, IS, CL, classical): ")

    # Allocate final results arrays according to the chosen method.
    if chosen_method == "L2M":
        epochs = 125; iterations = 5
        train_errors = np.zeros((epochs, iterations))
        test_errors  = np.zeros((epochs, iterations))
        time_l2m     = np.zeros((epochs, iterations))
    elif chosen_method == "CBL":
        epochs = 123; iterations = 5
        balanced_errors = np.zeros((epochs, iterations))
        balanced_test_errors = np.zeros((epochs, iterations))
        time_cbl = np.zeros((epochs, iterations))
    elif chosen_method == "SMOTE":
        epochs = 113; iterations = 5
        smote_errors = np.zeros((epochs, iterations))
        smote_test_errors = np.zeros((epochs, iterations))
        time_smote = np.zeros((epochs, iterations))
    elif chosen_method == "IS":
        epochs = 105; iterations = 5
        is_train_errors = np.zeros((epochs, iterations))
        is_test_errors = np.zeros((epochs, iterations))
        time_is = np.zeros((epochs, iterations))
    elif chosen_method == "CL":
        epochs = 125; iterations = 5
        cl_train_errors = np.zeros((epochs, iterations))
        cl_test_errors = np.zeros((epochs, iterations))
        time_cl = np.zeros((epochs, iterations))
    elif chosen_method == "classical":
        epochs = 125; iterations = 5
        classical_errors = np.zeros((epochs, iterations))
        classical_test_errors = np.zeros((epochs, iterations))
        time_classical = np.zeros((epochs, iterations))

    # Launch a separate process for each iteration.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []
    for iters_ in range(iterations):
        p = multiprocessing.Process(target=run_iteration, args=(iters_, epochs, chosen_method, return_dict))
        procs.append(p)
        p.start()
        p.join()  # run sequentially to free memory between iterations

    # Collect and save the results.
    if chosen_method == "L2M":
        for iters_ in range(iterations):
            train_errors[:, iters_], test_errors[:, iters_], time_l2m[:, iters_] = return_dict[iters_]
        np.savetxt('train_errors_cifar10.csv', train_errors, delimiter=',')
        np.savetxt('test_errors_cifar10.csv', test_errors, delimiter=',')
        np.savetxt('time_l2m_cifar10.csv', time_l2m, delimiter=',')
    elif chosen_method == "CBL":
        for iters_ in range(iterations):
            balanced_errors[:, iters_], balanced_test_errors[:, iters_], time_cbl[:, iters_] = return_dict[iters_]
        np.savetxt('balanced_errors_cifar10.csv', balanced_errors, delimiter=',')
        np.savetxt('balanced_test_errors_cifar10.csv', balanced_test_errors, delimiter=',')
        np.savetxt('time_cbl_cifar10.csv', time_cbl, delimiter=',')
    elif chosen_method == "SMOTE":
        for iters_ in range(iterations):
            smote_errors[:, iters_], smote_test_errors[:, iters_], time_smote[:, iters_] = return_dict[iters_]
        np.savetxt('smote_errors_cifar10.csv', smote_errors, delimiter=',')
        np.savetxt('smote_test_errors_cifar10.csv', smote_test_errors, delimiter=',')
        np.savetxt('time_smote_cifar10.csv', time_smote, delimiter=',')
    elif chosen_method == "IS":
        for iters_ in range(iterations):
            is_train_errors[:, iters_], is_test_errors[:, iters_], time_is[:, iters_] = return_dict[iters_]
        np.savetxt('is_train_errors_cifar10.csv', is_train_errors, delimiter=',')
        np.savetxt('is_test_errors_cifar10.csv', is_test_errors, delimiter=',')
        np.savetxt('time_is_cifar10.csv', time_is, delimiter=',')
    elif chosen_method == "CL":
        for iters_ in range(iterations):
            cl_train_errors[:, iters_], cl_test_errors[:, iters_], time_cl[:, iters_] = return_dict[iters_]
        np.savetxt('cl_train_errors_cifar10.csv', cl_train_errors, delimiter=',')
        np.savetxt('cl_test_errors_cifar10.csv', cl_test_errors, delimiter=',')
        np.savetxt('time_cl_cifar10.csv', time_cl, delimiter=',')
    elif chosen_method == "classical":
        for iters_ in range(iterations):
            classical_errors[:, iters_], classical_test_errors[:, iters_], time_classical[:, iters_] = return_dict[iters_]
        np.savetxt('classical_errors_cifar10.csv', classical_errors, delimiter=',')
        np.savetxt('classical_test_errors_cifar10.csv', classical_test_errors, delimiter=',')
        np.savetxt('time_classical_cifar10.csv', time_classical, delimiter=',')