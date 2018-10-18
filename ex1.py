import numpy as np
import matplotlib.pyplot as plt
from myMLlib import *


#####################Plotting####################
#We can plot the data
print('Plotting Data...\n')
data = np.loadtxt("ex1data1.txt",delimiter=",")

X = data[:,0]
y = data[:,1]
m = len(y) #number of training examples
plt.scatter(X,y)

#uncomment below to show plot
#plt.show()


##################Gradient Descent###############
print('Running Gradient Descent...\n')
temp = np.empty([2,m])
temp[0] = np.ones(m)
temp[1] = X
X = temp
theta = np.zeros([2,1])

#Some gradient descent settings
iterations = 1500
alpha = 0.01

#compute and display initial cost
print(computeCost(X, y, theta))
#run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

#print theta to screen
print('Theta found by gradient descent: ')
print(theta[0])
print(theta[1])

#Plot the linear fit
plt.plot(X[1,:],np.matmul(np.transpose(X),theta))
plt.show()
