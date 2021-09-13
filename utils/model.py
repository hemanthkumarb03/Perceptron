import numpy as np
import pandas as pd

class perceptron:

  def __init__(self,eta,epochs):
    np.random.seed(42)
    self.weights = np.random.randn(3) * 1e-4
    print(f"initial weights before training: \n{self.weights}")
    self.eta = eta
    self.epochs = epochs

  def activationfunction(self,inputs,weights):
    z = np.dot(inputs, weights)
    return np.where(z>0,1,0)

  def fit(self,x,y):
    self.x=x
    self.y=y
    x_with_bais = np.c_[self.x, -np.ones((len(self.x),1))]
    print(f"x with bias : \n{x_with_bais}")

    for epoch in range(self.epochs):
      print("__"*15)
      print(f"for epoch : \n{epoch}")
      print("__"*15)

      y_hat = self.activationfunction(x_with_bais,self.weights)
      print(f"predicted output after forward pass : \n{y_hat}")
      self.error = self.y - y_hat
      print(f"error: \n{self.error}")
      self.weights = self.weights + self.eta * np.dot(x_with_bais.T,self.error)
      print(f"updated weights after epoch {epoch} is \n{self.weights}")
      print("###"*15)

  def predict(self,x):
    x_with_bais = np.c_[x,-np.ones((len(x),1))]
    return self.activationfunction(x_with_bais,self.weights)

  def totalloss(self):
    total_loss = np.sum(self.error)
    print(f"total loss is {total_loss}")
    return total_loss
