import binascii
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('Car Hacking.csv')
df['Data'] = df['Data'].str.replace(" ", "")
df['Class'] = df['Class'].map({'Attack': 1, 'Normal': 0})
df = df.rename(columns = {'Class':'Labels'})

dataframe = df[['Data', 'Labels']]

#drop nan values
dataframe = dataframe.dropna()  
dataframe = dataframe.drop_duplicates(subset=['Data'])
dataframe['Labels'].sum()

# Shuffle the Dataset.
shuffled_df = dataframe.sample(frac=1,random_state=4)

# Put all the attack class in a separate dataset.
attack_df = shuffled_df.loc[shuffled_df['Labels'] == 1]

#Randomly select n observations from the normal behavior 
normal_df = shuffled_df.loc[shuffled_df['Labels'] == 0].sample(n=30003, random_state=42)

# Concatenate both dataframes again
clean_df = pd.concat([attack_df, normal_df])



#Balanced List of data
col_list = list(clean_df["Data"])

#Balanced labels
labels =  list(clean_df["Labels"])

# Create list of list (Decimal to Hex)
x = [] #Creates Empty list
for i in range(len(col_list)):  
   try:
       x.append(list(binascii.unhexlify(col_list[i].split('\n')[0])))
   except:
       print(col_list[i])

# Max length and threshold of values in list
max_length=0
threshold=10
for i in x:
    if len(i) < threshold:
        if len(i) > max_length:
            max_length = len(i)
    
#Creates a vector of 0 with length 256
lst = [0]*256

#List of list of list
onehot_encoded=[]      
for i in x:
    if len(i) < threshold:
        new_list = []
        for j in i:
            number = [0.0 for _ in range(len(lst))]   #assigns 0 for position of indexed value
            number[j] = 1.0   #assigns a 1
            new_list.append(np.array(number, dtype=float))
        for c in range(len(new_list), max_length):
            new_list.append(np.array(lst, dtype=float))
        onehot_encoded.append(np.array(new_list, dtype=float))

onehot_encoded2 = np.array(onehot_encoded)
       
#Labels as Array
labels1 = np.array(labels, dtype=float)

# Train, validation, and test split
X_trainval, X_test, y_trainval, y_test = train_test_split(onehot_encoded2, labels1, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

my_x = torch.Tensor(X_train)
my_y = torch.Tensor(y_train)
val_x = torch.Tensor(X_val)
val_y = torch.Tensor(y_val)
test_x = torch.Tensor(X_test)
test_y = torch.Tensor(y_test)

# Create Datasets
my_dataset = TensorDataset(my_x, my_y)
val_dataset = TensorDataset(val_x, val_y)
my_testset = TensorDataset(test_x, test_y)
my_dataloader = DataLoader(my_dataset, batch_size=16)
val_dataloader = DataLoader(val_dataset, batch_size=16)
my_testloader = DataLoader(my_testset, batch_size=16)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters
input_size = 256
hidden_size = 128
num_layers = 2
num_classes = 2
dropout_prob = 0.5
num_epochs = 25
learning_rate = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, and optimizer
model = RNN(input_size, hidden_size, num_layers, num_classes, dropout_prob).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(my_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / (i + 1)
    train_acc = correct / total

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / (i + 1)
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_acc*100:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, "
          f"Validation Accuracy: {val_acc*100:.2f}%")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

# Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in my_testloader:
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = correct / total
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot losses and accuracies
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs. Epoch")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs. Epoch")

plt.show()

    