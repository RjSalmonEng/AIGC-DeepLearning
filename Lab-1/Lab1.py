# %%
import torch
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv("Salary_dataset.csv")
df.head()
# %%
df.info()
# %%
df[df.isna().any(axis=1)]
# %%
df = df.dropna(subset=['Salary'], axis=0)
# %%
df
# %%



X = df.drop(columns=['Salary'])
y = df[['Salary']]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessing = ColumnTransformer(transformers=[
    ('num', numeric_transformer, ['YearsExperience'])
])
X_train = preprocessing.fit_transform(X_train)
X_test = preprocessing.transform(X_test)
standard_scaler = StandardScaler()
y_train = standard_scaler.fit_transform(y_train)
y_test = standard_scaler.transform(y_test)
# %%
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)
# %%
y_test
# %%
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=8, shuffle=True)
# %%
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_features:int = 1, output_features:int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, output_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            result =  self.forward(x)
        return result
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

epochs = 100
losses = []
criterion = nn.MSELoss()
torch.manual_seed(42)
model = SimpleLinearRegression().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x,batch_y in train_dl:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = criterion(y_pred,batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss/len(train_dl)
    losses.append(avg_loss)
    print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

"""
for epoch in range(num_epochs):
    optimizer.zero_grad() # 1. clear old gradients
    preds = model(x) # 2. forward pass
    loss = criterion(preds, y) # 3. compute the loss
    loss.backward() # 4. backward pass
    optimizer.step() # 5. update parameters
"""
# %%
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Baseline Training Loss')
plt.show()
# %%
updated_losses = []
torch.manual_seed(42)
updated_model = SimpleLinearRegression().to(device)
updated_optimizer = torch.optim.Adam(updated_model.parameters(), lr=0.01)

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x,batch_y in train_dl:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        updated_optimizer.zero_grad()
        y_pred = updated_model(batch_x)
        loss = criterion(y_pred,batch_y)
        loss.backward()
        updated_optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss/len(train_dl)
    updated_losses.append(avg_loss)
    print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")
# %%
plt.plot(losses, label='Baseline learning rate = 0.001')
plt.plot(updated_losses, label='Updated learning rate = 0.01')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Baseline and Updated Training Loss')
plt.legend()
plt.show()
# %% [markdown]
# ### Comparison
# 
# I changed exactly one thing: the learning rate was increased from `0.001` to `0.01`. The higher learning rate makes the optimizer take larger steps when updating the model weights, so the updated model should reduce its loss faster. If the learning rate were increased too much, those larger steps could overshoot the best values and make training unstable.