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
    def __init__(self, c1=10.5, c2=3, c3=1.0, c4=1.0, c5=1.0, c6=-22.0, c7=1.0, c8=1.0):
        super(model_sigmoid, self).__init__()
        
        # Alpha (learning rate) hyperparameter of the model
        self.alpha_start = 0.001
        self.alpha = self.alpha_start
        
        # Parameters of the model
        self.c1 = torch.tensor(c1, dtype=torch.float64, requires_grad=True)
        self.c2 = torch.tensor(c2, dtype=torch.float64, requires_grad=True)
        self.c3 = torch.tensor(c3, dtype=torch.float64, requires_grad=True)
        self.c4 = torch.tensor(c4, dtype=torch.float64, requires_grad=True)
        self.c5 = torch.tensor(c5, dtype=torch.float64, requires_grad=True)
        self.c6 = torch.tensor(c6, dtype=torch.float64, requires_grad=True)
        self.c7 = torch.tensor(c7, dtype=torch.float64, requires_grad=True)
        self.c8 = torch.tensor(c8, dtype=torch.float64, requires_grad=True)
    
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
        return self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8



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
                plt.plot(years*100, values, c="blue", label="Real Data")
                plt.plot(testX*100, testY, c="red", label="Fitted curve")
                plt.plot(21.22*100, m(21.22).detach(), 'o', c="purple") 
                plt.text(21.00*100, 0.5, f"Current 2122 Prediction: \n{m(21.22).detach().numpy().round(3)} Billion People", c="purple")
                plt.xlabel("Year")
                plt.ylabel("Population (in Billions)")
                plt.title("Data vs. Prediction")
                plt.legend(loc="upper left")
                plt.pause(0.0001)
                #plt.savefig(f'gif/img{i/50}.png')
    
    
    # Get the parameters of the model to save them
    c1, c2, c3, c4, c5, c6, c7, c8 = m.getParams()
    print(f"Parameters: c1={c1}, c2={c2}, c3={c3}, c4={c4}, c5={c5}, c6={c6}, c7={c7}, c8={c8}")
    
    # Make the prediction
    print(f"Model prediction for 2122: {m(torch.tensor(21.22))} billion people")
    
    
    
    
    
    
    ### Graph creation ###
    # Create some data to test in the function
    testX = torch.from_numpy(np.linspace(13.00,25.22,100))
    
    # Input the values into the function
    testY = m(testX).detach()
    
    # Create the graph
    plt.cla()
    plt.plot(years*100, values, c="blue", label="Real Data")
    plt.plot(testX*100, testY, c="red", label="Fitted curve")
    plt.plot(21.22*100, m(21.22).detach(), 'o', c="purple") 
    plt.text(12.75*100, 8, f"Final 2122 Prediction: \n{m(21.22).detach().numpy().round(3)} Billion People", c="purple")
    plt.xlabel("Year")
    plt.ylabel("Population (in Billions)")
    plt.title("Data vs. Prediction")
    plt.legend(loc="upper left")
    plt.show()





# Get the prediction for the sigmoid function
def run_sigmoid():
    c1 = 10.654135216823821
    c2 = 0.9864140912245566
    c3 = 1.0385540355655845
    c4 = 1.9698143047910193
    c5 = 1.104389410774113
    c6 = -22.060388036261674
    c7 = 1.0657278995848072
    c8 = 0.5712219176728186
    
    
    # Initialize the model
    m = model_sigmoid(c1, c2, c3, c4, c5, c6, c7, c8)
    
    # Year to predict on
    predYear = 21.22

    # Make the prediction
    print(f"Model prediction for 2122: {m(torch.tensor(predYear))} billion people")
    
    
    
    
    
    
    ### Graph creation ###
    # Load in the data
    data = pandas.read_csv(os.path.join("data", "data2.csv"))
    
    # Get the World data from the dataset
    years = torch.tensor(data["year"].values[11300:].astype(np.float64), requires_grad=False)
    values = torch.tensor(data["World Population"].values[11300:].astype(np.float64), requires_grad=False)
    
    # Divide the values by 1 billion to help the model learn
    values = values/1000000000
    
    # Divide the years by 100 to help the model learn
    years = years/100
    
    # Create some data to test in the function
    testX = torch.from_numpy(np.linspace(13.00,21.22,100))
    
    # Input the values into the function
    testY = m(testX).detach()
    
    # Create the graph
    plt.cla()
    plt.plot(years*100, values, c="blue", label="Real Data")
    plt.plot(testX*100, testY, c="red", label="Fitted curve")
    plt.plot(21.22*100, m(21.22).detach(), 'o', c="purple") 
    plt.text(12.75*100, 8, f"Final 2122 Prediction: \n{m(21.22).detach().numpy().round(3)} Billion People", c="purple")
    plt.xlabel("Year")
    plt.ylabel("Population (in Billions)")
    plt.title("Data vs. Prediction")
    plt.legend(loc="upper left")
    plt.show()





train_sigmoid()