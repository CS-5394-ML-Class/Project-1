import numpy as np
import torch
from torch import nn



class model(nn.Module):
    # Initialize the model
    def __init__(self):
        super(model, self).__init__()
        
        # Alpha (learning rate) hyperparameter of the model
        self.alpha = 0.0001
        
        # Parameters of the model
        self.c1 = torch.tensor(1.0, requires_grad=True)
        self.c2 = torch.tensor(1.0, requires_grad=True)
        self.c3 = torch.tensor(1.0, requires_grad=True)
        self.c4 = torch.tensor(1.0, requires_grad=True)
    
    # Get a prediction from the model
    def forward(self, x):
        return (self.c1)/torch.pow((self.c2+self.c3*torch.exp(-x)), self.c4)

    # Get a prediction from the model
    def predict(self, x):
        return (self.c1)/torch.pow((self.c2+self.c3*torch.exp(-x)), self.c4)

    # Update the parameters given the derivatives of those values
    def update(self, loss):
        loss.backward()
        self.c1 = torch.tensor(self.c1 - self.alpha*self.c1.grad, requires_grad=True)
        self.c2 = torch.tensor(self.c2 - self.alpha*self.c2.grad, requires_grad=True)
        self.c3 = torch.tensor(self.c3 - self.alpha*self.c3.grad, requires_grad=True)
        self.c4 = torch.tensor(self.c4 - self.alpha*self.c4.grad, requires_grad=True)
        #print(self.c1.grad)
        #self.c1 = self.c1 - self.alpha*self.dc1
        #self.c2 = self.c2 - self.alpha*self.dc2
        #self.c3 = self.c3 - self.alpha*self.dc3
        #self.c4 = self.c4 - self.alpha*self.dc4

    # The loss function used to evaluate the model
    def getLoss(self, preds, labels):
        return torch.sum((labels-preds)**2)
    

    # Calculate the derivatives of the constants
    def getParams(self):
        return self.c1, self.c2, self.c3, self.c4



# Train the model
def train():
    # The number of times to train the model
    updates = 1000

    # The data to train the model
    x = torch.tensor([-1, 2, 1])
    y = torch.tensor([0.25, 0.5, 0.75])

    # Initialize the model
    m = model()

    # Iterate "updates" number of times and update the model
    for i in range(0, updates):
        # Get some predictions from the model
        preds = m(x)

        # Calculate the loss
        loss = m.getLoss(preds, y)

        # Calculate the derivatives of the loss
        #m.calculateDerivatives(x, preds, y)

        # Update the model
        m.update(loss)

        print(f"Loss: {loss}")
        print(m.predict(x))
    
    
    # Get the parameters of the model to see them
    c1, c2, c3, c4 = m.getParams()
    print(f"Parameters: c1={c1}, c2={c2}, c3={c3}, c4={c4}")


train()
