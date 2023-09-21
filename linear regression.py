import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 4, 6])

mean_x = np.mean(x)
mean_y = np.mean(y)

n = len(x)
b1 = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

y_pred = b0 + b1 * x

plt.scatter(x, y, label="Data")

plt.plot(x, y_pred, color='red', label="Linear Regression")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.show()

print("Slope (Coefficient):", b1)
print("Intercept:", b0)
