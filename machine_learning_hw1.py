import numpy as np
import matplotlib.pyplot as plt

N = 20 #mutable to anything
d = 2 #mutable to anything

## Generate random training data
X = np.random.uniform(-1,1,size=(N,d+1))
X[:,0] = 1

## Generate random target function: f(x) = w^T x
w = np.random.uniform(-1,1,size=(d+1))

## Compute true labels for the training data
Y = np.sign(np.dot(X,w))
ind_pos = np.where(Y == 1)[0] ## positive examples RED
ind_neg = np.where(Y == -1)[0] ## negative examples BLUE
## If there are two few positive or negative examples, then repeat w = np.random... until you find good values for w. Or do something even better.


## Plot points
plt.clf()
plt.plot(X[ind_pos,1],X[ind_pos,2],'ro', label="Positive Data")
plt.plot(X[ind_neg,1],X[ind_neg,2],'bx', label="Negative Data")

#map target function
x_tar_val = np.linspace(-1,1)
y_tar_val = ((-(w[1]/ w[2]) * x_tar_val - (w[0]/w[2])))
plt.plot(x_tar_val,y_tar_val, "-g", label="Target function")

#create perceptron
iterations = 0 #NUMBER OF ITERATIONS BEFORE SUCCESS

g = np.random.uniform(-1,1,size=(d+1)) #create random hypothesis

breakpoint = 0
while breakpoint != 1: #runs loop until dot product of hyp. function and X is equal to dot product of target function and X as seen in line 42

    g_Y = np.sign(np.dot(X,g))#IMPORTANT TO CALCULATE IN LOOP SO THAT VALUES CHANGE
    randomint = np.random.randint(len(X)-1)#RANDOM INDEX TO TEST

    if np.array_equal(g_Y, Y):#checks if arrays are elementwise equivalent
        breakpoint = 1 #correct seperator found

    if np.sign(np.dot(X[randomint],g)) != np.sign(np.dot(X[randomint], w)):
        g = g + (X[randomint] * Y[randomint]) #otherwise, fix seperator

    iterations += 1


x_hyp_val = np.linspace(-1,1) #create x and y values for g
y_hyp_val = ((-(g[1]/ g[2]) * x_hyp_val - (g[0]/g[2])))

plt.plot(x_hyp_val, y_hyp_val, "orange",label="Hypothesis function")

print("Number of iterations: ", iterations)



#---- formatting
title = "For N = " + str(N) + " and d = " + str(d)
plt.title(title)
plt.legend(loc="best")
plt.show()
