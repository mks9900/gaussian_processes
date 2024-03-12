import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

# Generate observations from a sine function
X_train = np.array([-1.5, -1.0, -0.75, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(
    0, 0.1, X_train.shape[0]
)  # Add some noise

# Kernel with explicit noise modeling
kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)

# Generate test points
X_test = np.linspace(-2, 4, 100).reshape(-1, 1)

# Predict mean and standard deviation for test points
y_mean, y_std = gp.predict(X_test, return_std=True)

# Do some plotting:
plt.figure()
plt.scatter(X_train, y_train, c="r", label="Training data")
plt.plot(X_test, y_mean, "b-", label="Prediction")
plt.fill_between(X_test[:, 0], y_mean - y_std, y_mean + y_std, color="blue", alpha=0.2)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gaussian Process Regression")
plt.legend(loc="upper left")
plt.show()

# below defines another kernel, the matern kernel instead of the above:
# First, we need to import it from sklearn:
from sklearn.gaussian_process.kernels import Matern


# Now, below function sums all primes below any integer n:
def sum_primes(n):
    if n < 2:
        return 0
    numbers = list(range(2, n))
    for number in numbers:
        numbers = list(filter(lambda x: x == number or x % number, numbers))
    return sum(numbers)


sum_primes(1000)  # 76127
