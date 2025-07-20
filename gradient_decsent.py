# Gradient Descent

'''
Gradient descent is an algorithm to minimize the cost function. It works by reducing the 
paramter with learning_rate * gradient after every single step where the gradient is the slope of the cost function.

In case when cost function is dependent of more than one parameter, each parameter is minimized 
and their gradient at each step is the partial derivative with respect to the parameter.
Gradient descent algorithm also depends on the initial or starting point as they might 
lead to different local minima.

The learning rate controls how big each step is. If it's too large, the algorithm may overshoot 
the minimum and if too small, convergence will be slow.
'''

# We are taking the given function as cost function and computing the gradient decsent for it

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(2 * np.pi * x) + x ** 2

def diff_x(x):
    return -2 * np.pi * np.sin(2 * np.pi * x) + x*2

def gradient_descent(initial_x, learning_rate, iterations):
    x_values = [initial_x]
    for i in range(iterations):
        grad = diff_x(x_values[-1])
        new_x = x_values[-1] - learning_rate * grad
        x_values.append(new_x)
    return x_values

initial_x = 1.0
iterations = 200
learning_rate = 0.05

x_path = gradient_descent(initial_x, learning_rate, iterations)
y_path = [f(i) for i in x_path]

print("Vaue of x for y is minimum: ", x_path[-1])
print("Minimum of y: ", y_path[-1])
print("Slope at the minimum point: ", diff_x(x_path[-1]))


# Code for graph
x = np.linspace(-2, 2, 500)
y = f(x)
plt.plot(x, y, label = "f(x) = cos(2πx) + x^2")
plt.plot(x_path, y_path, 'ro--', label='Gradient Descent Path')
plt.plot(x_path[-1], y_path[-1], 'go', label='Final Point')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent on f(x) = cos(2πx) + x²')
plt.legend()
plt.grid(True)
plt.show()


