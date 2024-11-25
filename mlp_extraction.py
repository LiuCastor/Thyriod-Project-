import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                            roc_curve, auc, precision_recall_curve, log_loss, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# data loading
print('Loading training data...')
data = pd.read_excel(r'D:\research\research\others\figures\data_with_pnatietID_retained.xlsx')
X = data.drop(columns=[ '病理结果编码']).to_numpy()
y = data['病理结果编码'].to_numpy()

# normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# convert to tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# set the random seed
torch.manual_seed(42)
np.random.seed(42)

# create TensorDataset 和 DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
X_train, y_train, X_val, y_val = train_test_split(X,y,test_size=0.3,train_size=0.7, random_state=42)

train_dataset, val_dataset = torch.utils.data.random_split(dataset,
    lengths=[train_size, val_size],
    generator=torch.Generator().manual_seed(42))

# 64 64 32
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64,shuffle=False)
full_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# # cross validation
# skf = StratifiedKFold(n_splits=5)
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
#     val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#     full_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#     break

# enhanced MLP
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedMLP, self).__init__()
        # decreasing dropout
        self.layers = nn.Sequential(
            #（64，756）
            nn.Linear(input_dim, 256),
            #（64，256）
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            #（64，128）
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #（64，64）
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):

        # for layer in self.layers:
        #     x = layer(x)
        #     print(f"Layer output shape: {x.shape}")

        return self.layers(x)



# initialize the model
input_dim = X.shape[1]  # input dimension
output_dim = len(np.unique(y))  # categorical output dimension
model = EnhancedMLP(input_dim=input_dim, output_dim=output_dim).to(device) # move to the device

# extract features from the MLP
def extract_mlp_features(model, dataloader, device):

    model.eval()
    features = []
    with torch.no_grad():
        for arrays, _ in dataloader:
            arrays = arrays.to(device)
            hidden_output = model.layers[:-1](arrays)

            # Move to CPU and convert to numpy array
            features.append(hidden_output.cpu().numpy())
    # vstack all the features
    return np.vstack(features)

# loss function and optimizers
loss_fn = nn.CrossEntropyLoss().to(device)
# these are the best parameters for the loss function and optimizer
# lr = 1e-4 decay = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)


best_model_path = r"D:\research\research\others\hantong\thyroid_mlp_best.pth"
writer = SummaryWriter("cataract_mlp_logs")
epochs = 1000
best_accuracy = 7.3

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        imgs, targets = batch
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate accuracy
    avg_train_loss = total_train_loss / len(train_dataloader)
    writer.add_scalar("Training Loss", avg_train_loss, epoch)
    print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}")

    # validation
    model.eval()
    total_val_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in val_dataloader:
            imgs, targets = batch
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy += accuracy


    # loss and accuracy
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = total_accuracy / len(val_dataset)
    writer.add_scalar("Validation Loss", avg_val_loss, epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

    print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # lr scheduler
    scheduler.step(val_accuracy)

    # save the best model to disk
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {val_accuracy:.4f}")

# evaluate on test set
y_true = []
y_pred = []
y_prob = []
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        imgs, targets = batch
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_prob.extend(outputs.softmax(1).cpu().numpy()[:, 1])

    # targets
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        logloss = log_loss(y_true, y_prob)
        error_rate = 1 - accuracy
        specificity = recall_score(y_true, y_pred, pos_label=0, average='weighted')
        roc_auc = roc_auc_score(y_true, y_prob)

# confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title(f'Fold {fold + 1} Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# ROC curve
# fpr, tpr, _ = roc_curve(y_true, y_prob)
# plt.plot(fpr, tpr, label=f'Fold {fold + 1} AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='best')
# plt.show()

# PR curve
# precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
# plt.plot(recall_vals, precision_vals, label=f'Fold {fold + 1}')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='best')
# plt.show()

# print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, \
#         Recall: {recall:.4f}, F1: {f1:.4f}, Log Loss: {logloss:.4f}, \
#         Error Rate: {error_rate:.4f}, Specificity: {specificity:.4f}")

        print(f"Epoch [{epoch + 1}/{epochs}] - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

model.load_state_dict(torch.load(best_model_path))

mlp_features = extract_mlp_features(model, full_dataloader, device)
np.save(r'D:\research\research\others\hantong\mlp_features.npy', mlp_features)

writer.close()
print(f"The best model has an accuracy of: {accuracy:.4f}")
