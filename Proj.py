import numpy as np
import torch
from torch import nn
import pandas
import os



class model(nn.Module):
    # Initialize the model
    def __init__(self):
        super(model, self).__init__()
        
        # Alpha (learning rate) hyperparameter of the model
        self.alpha = 0.0001
        
        # Parameters of the model
        self.c1 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c2 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c3 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c4 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c5 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    
    # Get a prediction from the model
    # Inputs:
    #   x - The inputs into the model
    def forward(self, x):
        return (self.c1)/torch.pow((self.c2+self.c3*torch.exp(-x)), self.c4)+self.c5

    # Update the parameters given the derivatives of those values
    # Inputs:
    #   loss - The loss from the model
    def update(self, loss):
        # Compute the gradients
        loss.backward()
        
        # Update the model parameters
        self.c1 = torch.tensor(self.c1 - self.alpha*self.c1.grad, requires_grad=True)
        self.c2 = torch.tensor(self.c2 - self.alpha*self.c2.grad, requires_grad=True)
        self.c3 = torch.tensor(self.c3 - self.alpha*self.c3.grad, requires_grad=True)
        self.c4 = torch.tensor(self.c4 - self.alpha*self.c4.grad, requires_grad=True)
        self.c5 = torch.tensor(self.c5 - self.alpha*self.c5.grad, requires_grad=True)

    # The loss function used to evaluate the model
    # Inputs:
    #   preds - The predictions from the model
    #   labels - The true values we want the model to predict
    def getLoss(self, preds, labels):
        return torch.sqrt(torch.sum((labels-preds)**2))
    

    # Calculate the derivatives of the constants
    def getParams(self):
        return self.c1, self.c2, self.c3, self.c4, self.c5



# Train the model
def train():
    # The number of times to train the model
    updates = 1000
    
    
    
    # Load in the data
    data = pandas.read_csv(os.path.join("data", "data-clean.csv"))
    
    # Get the World data from the dataset
    data = data.iloc[257]
    data = data.drop("Country Name")
    
    # Get the values and year from the data
    # Note that years is the input and the output is the values
    years = torch.from_numpy(np.array(data.axes[0].values, dtype=np.int))
    values = torch.from_numpy(np.array(data.values, dtype=np.float64))
    
    # Divide the values by 1 billion to help the model learn
    values = values/1000000000

    # Initialize the model
    m = model()

    # Iterate "updates" number of times and update the model
    for i in range(0, updates):
        # Get some predictions from the model
        preds = m(years)

        # Calculate the loss
        loss = m.getLoss(preds, values)

        # Update the model
        m.update(loss)

        print(f"Loss: {loss}")
    
    
    # Get the parameters of the model to see them
    c1, c2, c3, c4, c5 = m.getParams()
    print(f"Parameters: c1={c1}, c2={c2}, c3={c3}, c4={c4}, c5={c5}")
    
    # Make the prediction
    print(f"Model prediction for 2122: {m(torch.tensor(2122))} billion people")


train()
