import numpy as np 

def computeCost(X, y, theta):
    m = len(y)
    J = 0

    for i in range(m):
        #proper matrix multiplication of theta and X
        #results in scalar value (1x2 * 2x1)
        thetaX = np.matmul(
                    np.transpose(theta),
                    np.transpose(X[:,i][np.newaxis])) 

        J = J + (thetaX - y[i])**2

    J = J/(2*m)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        err = np.matmul(np.transpose(X),theta) - np.transpose(y[np.newaxis])
        gradient = (alpha/m) * np.matmul(X,err)
        theta = theta - gradient

        J_history.append(computeCost(X,y,theta))


    return theta

