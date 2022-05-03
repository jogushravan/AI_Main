import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 30, 3)
print("X points are: \n", x)
y = x**2
print("Y points are: \n", y)
plt.scatter(x, y)
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.plot(x, y)
p = np.polyfit(x, y, 2)  # Last argument is degree of polynomial

print("Coeeficient values:\n", p)

pred = np.poly1d(p)
print(pred)
x_test = 15
print("\nGiven x_test value is: ", x_test)
y_pred = pred(x_test)
print("\nPredicted value of y_pred for given x_test is: ", y_pred)