

***Back Propogation OR gate***
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([[0,0,1,1],[0,1,0,1]])
w=np.random.rand(1,2)
y=np.array([[0,1,1,1]])
losses=[]
lr=0.19
print(w)

print(x.shape)

def sigmoid(z):
    z=1/(1+np.exp(-z))
    return z

def forward_pass(w,x):
    z=np.matmul(w,x)
    a=sigmoid(z)
    return z,a

def backward_pass(w,x,y,a):
  dz=(a-y)*a*(1-a)
  dw=np.dot(dz,x.T)
  return dz,dw

epochs = 1000

for i in range (epochs):
  z,a=forward_pass(w,x)
  loss=(0.5)*np.square(a-y)
  loss=np.mean(loss)
  losses.append(loss)
  dz,dw=backward_pass(w,x,y,a)
  w=w-lr*dw

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss Value")

"""***Testing the network***"""

def predict(w,test):
  z,a=forward_pass(w,test)
  print(z)
  print(a)
  if a>=0.5:
    display("1")
  else:
    display("0")

test=np.array([[0],[1]])
predict(w,test)

test=np.array([[0.9],[0.2]])
predict(w,test)
