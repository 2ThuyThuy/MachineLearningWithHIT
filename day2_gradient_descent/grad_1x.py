import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**4 - 2*x**3+2

def grad_f(x):
    return 4*x**3 - 6*x**2


def gradientDescent(start, learningRate, grad_F, loop, min_end):
    x = start
    His=[x]
    for i in range(loop):
        xNew = learningRate*grad_F(x)
        if np.abs(xNew) < min_end:
            x = x - xNew
            His.append(x)
            break
        x = x - xNew
        His.append(x)
    return His    
    

his = gradientDescent(-0.5,0.3,grad_f,100,0.0001)

#draw
x = np.arange(-1, 2.5, 0.1)
fig, ax = plt.subplots()
ax.plot(x, f(x))

#draw lines
#for index in range(1,len(his)):
#    ax.annotate('',xy=(his[index],f(his[index])),xytext=(his[index-1],f(his[index-1])),arrowprops={'arrowstyle': '->', 'color': 'g', 'lw': 1},
#                   va='center', ha='center')

#draw points
for item in his:
    plt.scatter(item,f(item),c="g") 
plt.scatter(his[-1],f(his[-1]),c='red')


#ax.set_xlim(-1,3)
ax.set_title('$f(x) = x^4 - 2x^3+2$')
plt.ylabel('f(x)')
plt.xlabel('x')
print(his[-1])
plt.show()
