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

training_acc = []
testing_acc = []
confusion = []
train_losses = []
test_losses = []

for iteration in range(10):
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
    
    #Train test Split
    X_train, X_test, y_train, y_test = train_test_split(onehot_encoded2, labels1, test_size=0.33, random_state = 42)
    my_x = torch.Tensor(X_train)
    my_y = torch.Tensor(y_train)
    test_x = torch.Tensor(X_test)
    test_y = torch.Tensor(y_test)
    
    
    #Create Datasets
    my_dataset = TensorDataset(my_x, my_y)
    my_testset = TensorDataset(test_x, test_y)
    my_dataloader = DataLoader(my_dataset, batch_size=16) 
    my_testloader = DataLoader(my_testset, batch_size=16)
    
    
    
    
    #Set Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #Class Object for RNN
    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNNModel, self).__init__()
    
            self.hidden_size = hidden_size
    
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 10), 
                nn.ReLU(), 
                nn.Linear(10, hidden_size), 
                nn.ReLU())
            self.output = nn.Linear(hidden_size, output_size)
            self.rnn = nn.RNN(hidden_size, hidden_size, 1, batch_first=True)
            
    
        def forward(self, input):
            embed = self.encoder(input)
            hidden = torch.zeros(1, input.size()[0], self.hidden_size)
            #print("embed", embed.size()) #embedded size
            #print("hidden", hidden.size()) #hidden size
            output1, hidden1 = self.rnn(embed, hidden) 
            #print("output1 size", output1.size()) 
            output2 = self.output(output1[:,-1,:])
            return output2
    
        def initHidden(self):
            return torch.zeros(2, 16, self.hidden_size)
    
    # Create RNN
    input_size = 256    # input dimension
    hidden_size = 10  # hidden layer dimension (relative)
    output_size = 1   # output dimension
    
    model = RNNModel(input_size, hidden_size, output_size)
    print(model)
    
    # Cross Entropy Loss 
    criterion = nn.BCEWithLogitsLoss()
    
    # SGD Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.9)
    
    
    # Training Loop: 
    for epoch in range(3):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_test_loss = 0.0
        for i, data in enumerate(my_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs[:,0], labels)
            loss.backward()
            running_loss+=loss.detach().item()
            optimizer.step()

        train_losses.append(running_loss / len(my_dataloader))
    
        # Calculate test loss
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(my_testloader, 0):
                inputs, labels = data
                outputs = model(inputs)
                test_loss = criterion(outputs[:,0], labels)
                running_test_loss += test_loss.detach().item()
                
            test_losses.append(running_test_loss / len(my_testloader))

        lr_scheduler.step()    
        print('Epoch ', epoch, ' ----> Train Loss:', train_losses[-1], ' Test Loss:', test_losses[-1])

    
    y_pred_training = []
    model.eval()
    for i, data in enumerate(my_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = model(inputs)
        y_pred_training.append(torch.sigmoid(outputs[:,0]).detach().numpy())
    
    
    
    y_pred_test = []
    for i, data in enumerate(my_testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = model(inputs)
        y_pred_test.append(torch.sigmoid(outputs[:,0]).detach().numpy())
    
    
    
    y_pred_training = np.concatenate(y_pred_training) #predicted values
    y_pred_binary = []
    for i in y_pred_training: 
        if i > 0.5:
            res = 1
        else: 
            res = 0
        y_pred_binary.append(np.array(res))
    
    y_true = np.array(my_y) #for train set
    
    
    print("Training Accuracy", accuracy_score(y_true, y_pred_binary))
    print(precision_recall_fscore_support(y_true, y_pred_binary, average='binary'))
    training_acc.append(accuracy_score(y_true, y_pred_binary))

                    
    y_pred_test = np.concatenate(y_pred_test) #predicted values
    y_pred_binary1 = []
    for i in y_pred_test: 
        if i > 0.5:
            res = 1
        else: 
            res = 0
        y_pred_binary1.append(np.array(res))
    
    y_true1 = np.array(test_y) #for test set
    
    print("Test Accuracy", accuracy_score(y_true1, y_pred_binary1))
    print(precision_recall_fscore_support(y_true1, y_pred_binary1, average='binary'))
    testing_acc.append(accuracy_score(y_true1, y_pred_binary1))
    
    cm = confusion_matrix(y_true1, y_pred_binary1)
    confusion.append(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot()
    plt.show()

# Plot Loss vs Accuracy curve
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Epoch")
plt.show()

print("Mean Training Accuracy", np.mean(training_acc))
print("Mean Testing Accuracy", np.mean(testing_acc))    
print("Mean Training Accuracy", np.mean(training_acc))
print("Mean Testing Accuracy", np.mean(testing_acc))    

