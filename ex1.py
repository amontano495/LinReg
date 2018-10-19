import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from myMLlib import *


#####################Plotting####################
#We can plot the data
print('Plotting Data...\n')
data = np.loadtxt("ex1data1.txt",delimiter=",")

X = data[:,0]
y = data[:,1]
m = len(y) #number of training examples
#plt.scatter(X,y)

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
print(np.asscalar(computeCost(X, y, theta)))
#run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

#print theta to screen
print('Theta found by gradient descent:')
print(theta[0])
print(theta[1])

#Plot the linear fit
#plt.plot(X[1,:],np.matmul(np.transpose(X),theta))
#plt.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul(np.array([1, 3.5])[np.newaxis],theta)
print('For population = 35,000, we predict a profit of')
print(np.asscalar(predict1*10000))

predict2 = np.matmul(np.array([1, 7])[np.newaxis],theta)
print('For population = 70,000, we predict a profit of')
print(np.asscalar(predict2*10000))

#########Visualizing J(theta_0, theta_1)#########
print('Visualizing J(theta_0, theta_1) ...')

#Grid over which we will calculate J
theta0_vals = np.linspace(-10,10,num=100)
theta1_vals = np.linspace(-1,4,num=100)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

#Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.empty([2,1])
        t[0] = theta0_vals[i]
        t[1] = theta1_vals[j]
        J_vals[j,i] = computeCost(X,y,t)

#setup the mesh
theta0_vals,theta1_vals = np.meshgrid(theta0_vals,theta1_vals)
J_vals = np.transpose(J_vals)

#mpl 3d projection setup
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(theta0_vals,theta1_vals,J_vals, rstride=1,cstride=1,cmap='hot',linewidth=0,antialiased=False)
#fig.colorbar(surf,shrink=0.5,aspect=5)

#mpl contour plot
plt.contour(theta0_vals,theta1_vals,J_vals,colors='black')
plt.show()

