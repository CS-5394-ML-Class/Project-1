from importlib.metadata import requires
import numpy as np
import torch
from torch import nn
from torch import optim
import pandas
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

torch.autograd.set_detect_anomaly(True)



class model_sigmoid(nn.Module):
    # Initialize the model
    def __init__(self):
        super(model_sigmoid, self).__init__()
        
        # Alpha (learning rate) hyperparameter of the model
        self.alpha_start = 0.001
        self.alpha = self.alpha_start
        
        # Parameters of the model
        self.c1 = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
        self.c2 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        self.c3 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c4 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c5 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c6 = torch.tensor(-22.0, dtype=torch.float64, requires_grad=True)
        self.c7 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.c8 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    
    # Get a prediction from the model
    # Inputs:
    #   x - The inputs into the model
    def forward(self, x):
        return (self.c1)/torch.pow((self.c2+self.c3*torch.exp(-self.c4*(self.c5*x + self.c6))), self.c7)+self.c8

    # Update the model parameters
    # Inputs:
    #   loss - The loss from the model
    #   numIters - Number of time to update the model
    #   currIter - The current update iteration
    def update(self, loss, numIter, currIter):
        # Compute the gradients
        loss.backward()
        
        # Update the model parameters
        self.c1 = torch.tensor(self.c1 - self.alpha*self.c1.grad, requires_grad=True)
        self.c2 = torch.tensor(self.c2 - self.alpha*self.c2.grad, requires_grad=True)
        self.c3 = torch.tensor(self.c3 - self.alpha*self.c3.grad, requires_grad=True)
        self.c4 = torch.tensor(self.c4 - self.alpha*self.c4.grad, requires_grad=True)
        self.c5 = torch.tensor(self.c5 - self.alpha*self.c5.grad, requires_grad=True)
        self.c6 = torch.tensor(self.c6 - self.alpha*self.c6.grad, requires_grad=True)
        self.c7 = torch.tensor(self.c7 - self.alpha*self.c7.grad, requires_grad=True)
        self.c8 = torch.tensor(self.c8 - self.alpha*self.c8.grad, requires_grad=True)
        
        # Decrease the learning rate
        self.alpha = self.alpha_start*(1-(currIter/numIter))

    # The loss function used to evaluate the model. This loss
    # function is a modified form of the MSE loss where
    # 2*(1-self.c2)**2 is added to the loss value. This term
    # convinced the function to not have a massive range and to
    # keep the c2 term around 1.
    # Inputs:
    #   preds - The predictions from the model
    #   labels - The true values we want the model to predict
    def getLoss(self, preds, labels):
        return torch.sum((labels-preds)**2)/(preds.shape[0])+2*(1-self.c2)**2
    

    # Get the parameters from the model
    def getParams(self):
        return self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7
    
    
    



class model_nn(nn.Module):
    # Initialize the model
    def __init__(self, input_shape, output_shape):
        super(model_nn, self).__init__()
        
        # Alpha (learning rate) hyperparameter of the model
        self.alpha_start = 0.0001
        self.alpha = self.alpha_start
        
        # Model Parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # The model to optimize
        self.model = nn.Sequential(
            nn.Linear(self.input_shape, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, self.output_shape),
        )
        
        # The optimizer for the model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
    
    # Get a prediction from the model
    # Inputs:
    #   x - The inputs into the model
    def forward(self, x):
        try:
            return self.model(x.reshape(x.shape[0], 1).float()).reshape(1, x.shape[0])
        except(IndexError):
            return self.model(torch.tensor([x]).float())[0]

    # Update the model parameters
    # Inputs:
    #   loss - The loss from the model
    #   numIters - Number of time to update the model
    #   currIter - The current update iteration
    def update(self, loss, numIter, currIter):
        # Compute the gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Update the model
        self.optimizer.step()
        
        # Decrease the learning rate
        #self.alpha = self.alpha_start*(1-(currIter/numIter))
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    # The loss function used to evaluate the model
    # Inputs:
    #   preds - The predictions from the model
    #   labels - The true values we want the model to predict
    def getLoss(self, preds, labels):
        return torch.sum((labels-preds)**2)/preds.shape[0]



# Train the model
def train_sigmoid():
    # The number of times to train the model
    updates = 50000
    
    # True if the graph should be shown while training. False otherwise
    showGraph = True
    
    
    
    # Load in the data
    data = pandas.read_csv(os.path.join("data", "data2.csv"))
    
    # Get the World data from the dataset
    years = torch.tensor(data["year"].values[11300:].astype(np.float64), requires_grad=False)
    values = torch.tensor(data["World Population"].values[11300:].astype(np.float64), requires_grad=False)
    
    # Divide the values by 1 billion to help the model learn
    values = values/1000000000
    
    # Divide the years by 100 to help the model learn
    years = years/100

    # Initialize the model
    m = model_sigmoid()

    # Iterate "updates" number of times and update the model
    for i in range(0, updates):
        # Get some predictions from the model
        preds = m(years)

        # Calculate the loss
        loss = m.getLoss(preds, values)

        # Update the model
        m.update(loss, updates, i)
        
        
        # Plot the graph with some data and show the loss every so often
        if i % 50 == 0:
            print(f"Iteration #{i}, Loss: {loss}")
            if showGraph == True:
                testX = torch.from_numpy(np.linspace(13,25.22,100))
                testY = m(testX).detach()
                plt.cla()
                plt.plot(years, values, c="blue", label="Real Data")
                plt.plot(testX, testY, c="red", label="Fitted curve")
                plt.xlabel("Year (divided by 100)")
                plt.ylabel("Population (in Billions)")
                plt.title("Data vs. Prediction")
                plt.legend(loc="upper left")
                plt.pause(0.0001)
    
    
    # Get the parameters of the model to save them
    c1, c2, c3, c4, c5, c6, c7 = m.getParams()
    print(f"Parameters: c1={c1}, c2={c2}, c3={c3}, c4={c4}, c5={c5}, c6={c6}, c7={c7}")
    
    # Make the prediction
    print(f"Model prediction for 2122: {m(torch.tensor(2122))} billion people")
    
    
    
    
    
    
    ### Graph creation ###
    # Create some data to test in the function
    testX = torch.from_numpy(np.linspace(13.00,25.22,100))
    
    # Input the values into the function
    testY = m(testX).detach()
    
    # Create the graph
    plt.cla()
    plt.plot(years, values, c="blue", label="Real Data")
    plt.plot(testX, testY, c="red", label="Fitted curve")
    plt.xlabel("Year (divided by 100)")
    plt.ylabel("Population (in Billions)")
    plt.title("Data vs. Prediction")
    plt.legend(loc="upper left")
    plt.show()






# Train the model
def train_nn():
    # The number of times to train the model
    updates = 50000
    
    # True if the graph should be shown while training. False otherwise
    showGraph = True
    
    
    
    # Load in the data
    data = pandas.read_csv(os.path.join("data", "data2.csv"))
    
    # Get the World data from the dataset
    years = torch.tensor(data["year"].values[11300:].astype(np.float64), requires_grad=False)
    values = torch.tensor(data["World Population"].values[11300:].astype(np.float64), requires_grad=False)
    
    # Divide the values by 1 billion to help the model learn
    values = values/10000000
    
    # Divide the years by 100 to help the model learn
    years = years/100

    # Initialize the model
    m = model_nn(1, 1)

    # Iterate "updates" number of times and update the model
    for i in range(0, updates):
        # Get some predictions from the model
        preds = m(years)

        # Calculate the loss
        loss = m.getLoss(preds, values)

        # Update the model
        m.update(loss, updates, i)

        
        # Plot the graph with some data and show the loss every so often
        if i % 50 == 0:
            print(f"Iteration #{i}, Loss: {loss}")
            if showGraph == True:
                testX = torch.from_numpy(np.linspace(13.00,25.22,100))
                testY = m(testX).detach().reshape(testX.shape)
                plt.cla()
                plt.plot(years, values/100, c="blue", label="Real Data")
                plt.plot(testX, testY/100, c="red", label="Fitted curve")
                plt.xlabel("Year (divided by 100)")
                plt.ylabel("Population (in Billions)")
                plt.title("Data vs. Prediction")
                plt.legend(loc="upper left")
                plt.pause(0.0001)
    
    
    # Make the prediction
    print(f"Model prediction for 2122: {m(torch.tensor(2122))/100} billion people")
    
    
    
    
    
    
    ### Graph creation ###
    # Create some data to test in the function
    testX = torch.from_numpy(np.linspace(13.00,25.22,100))
    
    # Input the values into the function
    testY = m(testX).detach().reshape(testX.shape)
    
    # Create the graph
    plt.cla()
    plt.plot(years, values/100, c="blue", label="Real Data")
    plt.plot(testX, testY/100, c="red", label="Fitted curve")
    plt.xlabel("Year (divided by 100)")
    plt.ylabel("Population (in Billions)")
    plt.title("Data vs. Prediction")
    plt.legend(loc="upper left")
    plt.show()




train_sigmoid()