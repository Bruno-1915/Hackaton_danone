# %% tags=["parameters"]
upstream = None
product = None
# %%
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
# %%
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torchmetrics import F1Score

warnings.filterwarnings('ignore')
# %%
cols = ['is_beverage', 'non_recyclable_and_non_biodegradable_materials_count', 
        'est_co2_agriculture', 'est_co2_consumption', 
        'est_co2_distribution', 'est_co2_packaging', 'est_co2_processing', 
        'est_co2_transportation']
target_col = 'ecoscore_grade'

# %%
train = pd.read_csv(upstream['Preprocess features']['train_csv'], usecols=cols + [target_col])
test = pd.read_csv(upstream['Preprocess features']['test_csv'], usecols=cols)

# %%
X_train, y_train = train[cols], train[target_col]
X_test = test[cols]
for i in cols:
    X_train[i] = X_train[i].astype(float)
    X_test[i] = X_test[i].astype(float)
    
# %%
batch_size = 32 * 2
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), 
                  torch.tensor(y_train.values, dtype=torch.float32)), 
    batch_size=batch_size, shuffle=True,
)

class Net(nn.Module):
    activation_function: nn.modules.activation = None

    def __init__(self, input_shape, output_shape, activation_function, hidden_layers: list = [160], dropout=0.6):
        super(Net, self).__init__()

        activation_function = getattr(nn, activation_function)
        layers = []
        hidden_layers.insert(0, input_shape)
        for i, n in enumerate(hidden_layers[0:-1]):
            m = int(hidden_layers[i + 1])
            layers.append(nn.Linear(n, m))
            layers.append(nn.BatchNorm1d(m))
            layers.append(nn.Dropout(dropout))
            layers.append(activation_function())
        layers.append(nn.Linear(hidden_layers[-1], output_shape))
        layers.append(nn.Sigmoid())
        layers = nn.Sequential(*layers)
        self.layers = layers
    def forward(self, x):
        x = self.layers(x)
        return x

    def predict(self, x):
        x = torch.from_numpy(x).float().to("cpu")
        outputs = self(x)
        return outputs
    
# %%
metric_f1 = F1Score(task='multiclass',num_classes=len(y_train.unique()), average="weighted", multiclass=True)
network = Net(X_train.shape[1], len(y_train.unique()), 'ReLU6', [128, 64, 32, 16])
optimizer = optim.Adam(network.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.9)
epochs = 100
# %%
for epoch in range(epochs):
    running_loss = 0
    f1 = 0
    
    for (data, target) in train_loader:
        optimizer.zero_grad()
        output = network(data)

        target = nn.functional.one_hot(target.to(torch.int64), num_classes=len(y_train.unique()))
        loss = criterion(output.float(), target.float())

        ## Do backward
        loss.backward()
        optimizer.step()

        f1 += metric_f1(torch.argmax(output, axis=1), torch.argmax(target.int(), axis=1)).item()
        running_loss += loss.item()
    f1 /= len(train_loader)
    running_loss /= len(train_loader)
    # scheduler.step(f1)
    print(f'Epoch: {epoch} | F1: {f1} | Loss: {running_loss}')

# %%
network.eval()
pred = np.argmax(network.predict(X_test.values).detach().numpy(), axis=1)


Path(product['result'])\
    .write_text(json.dumps(
        {'target': {index: int(i) 
                    for index, i in enumerate(pred)}
         }))