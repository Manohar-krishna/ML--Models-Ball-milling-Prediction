# -*- coding: utf-8 -*-
"""Ann Classification.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/Ball Milling.csv', encoding = 'latin1')
# or try 'cp1252' if 'latin1' doesn't work
df

df_analysis = df.iloc[:,0:3]
df_analysis

df_analysis = df.iloc[:, 7:8]
df_analysis

X = df.iloc[:,0:3].values
y = df.iloc[:, 7:8].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class BallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze() # Squeeze to remove extra dimensions

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BallDataset(X_train, y_train)
test_dataset = BallDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # input_size should match the number of features in your input data
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Example usage (adjust input_size, hidden_size, output_size)
input_size = 3  # Number of features in your input data (X_train.shape[1])
hidden_size = 128
output_size = len(df.iloc[:, 7].unique())  # Number of classes in your labels
model = ANN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()  # Use appropriate loss for your task
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15  # Adjust as needed
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model_weights = model.state_dict()

for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test data: {100 * correct / total:.2f}%')

# Assuming you want to predict on new data, you'll need to prepare it similarly to how you prepared your test data.

# Example new data (replace with your actual new data)
new_data = [[300, 60, 0.83], [300, 100, 0.83], [300, 40, 0.50]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit the scaler on your original data (you should have this data from your training process)
# Replace 'X' with your original feature data
# X = ...  # Load your original feature data here
# scaler.fit(X)

# For this example, I'll re-use the X_train data that you've already defined.
# This assumes you have already executed the previous cells where X_train is defined.
scaler.fit(X_train)

# Preprocess the new data
new_data_scaled = scaler.transform(new_data)

# Convert the new data to a tensor
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)


# Make predictions
with torch.no_grad():
  outputs = model(new_data_tensor)
  _, predicted = torch.max(outputs.data, 1)

print(f"Predicted classes for new data: {predicted.tolist()}")

# To get probabilities instead of classes:

with torch.no_grad():
  outputs = model(new_data_tensor)
  probabilities = torch.softmax(outputs, dim=1) # Apply softmax to get probabilities
  predicted_probabilities = probabilities.max(dim=1)[0]
  _, predicted_classes = torch.max(outputs.data, 1)


print(f"Predicted probabilities for new data:\n{predicted_probabilities}")
print(f"Predicted classes for new data: {predicted_classes.tolist()}")
