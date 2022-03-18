import numpy as np
import matplotlib.pyplot as plt
import math
from math import *

def f(x1,x2):
    return 0.5*x1**2 + (5/2)*x2**2 - x1*x2 -2*(x1+x2)

def grad_f(x1,x2):
    return np.array([x1-x2-2,-x1+5*x2-2])

def norm(matrix1x2):
        nLine =matrix1x2.shape[0]
        N = 0
        for i in range(nLine):
            N += matrix1x2[i]**2
        return math.sqrt(N)


def gradientDescent(start, learningRate, min_end):
    x1,x2 = start
    gradx1x2 = grad_f(x1,x2)
    print(x1,x2,gradx1x2)
    n_grad = norm(gradx1x2)
    print(n_grad)
    
    history = [[x1,x2]]
    i=1
    while n_grad > min_end:
        diff = -gradx1x2
        x1 = x1 + learningRate*diff[0]
        x2 = x2 + learningRate*diff[1]
        history = np.vstack((history,[x1,x2]))
        gradx1x2 = grad_f(x1,x2)
        n_grad = norm(gradx1x2)
        i+=1
    
    return history

his = gradientDescent((2,1),0.1,pow(10,-6))
his_x1 = his[:,0]
his_x2 = his[:,1]


x1 = np.linspace(2, 3.5, 150)
x2 = np.linspace(0.25, 1.75, 150)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)
fig = plt.figure(figsize = (10,7))
contours = plt.contour(X1, X2, Z, 20)

plt.clabel(contours, inline = True, fontsize = 10)
plt.title("$f(x_1,x_2)=\\frac{1}{2}x_1^2+\\frac{5}{2}x_2^2 - x_1x_2 - 2(x_1+x_2)$")
plt.plot(his_x1, his_x2)
plt.plot(his_x1, his_x2, 'o', label = "Cost function")
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar()
plt.legend(loc = "upper right")
plt.show()