import GPy
import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(42)
num_batches = 5
num_points_per_batch = 20
X = np.random.uniform(-5, 5, (num_batches, num_points_per_batch))[:, :, np.newaxis]
Y = np.sin(X) + np.random.normal(0, 0.1, (num_batches, num_points_per_batch, 1))

# Reshape data into a single array
X_flat = X.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)

# Define the Gaussian process regression model
kernel = GPy.kern.RBF(input_dim=1)
model = GPy.models.GPRegression(X_flat, Y_flat, kernel)

# Optimize the model parameters
model.optimize()

# Predict on new test points
X_test = np.linspace(-6, 6, 100)[:, np.newaxis]
Y_pred, Y_var = model.predict(X_test)

# Reshape predictions and variances back into batches
Y_pred_batches = Y_pred.reshape(-1, num_points_per_batch, 1)
Y_var_batches = Y_var.reshape(-1, num_points_per_batch, 1)

# Plot the results for each batch
plt.figure(figsize=(10, 6))
for i in range(num_batches):
    plt.scatter(X[i], Y[i], c='red', label='Observations')
    plt.plot(X_test, Y_pred_batches[i], c='blue', label='Predicted Mean')
    plt.fill_between(X_test.flatten(), (Y_pred_batches[i] - 2 * np.sqrt(Y_var_batches[i])).flatten(),
                     (Y_pred_batches[i] + 2 * np.sqrt(Y_var_batches[i])).flatten(),
                     color='gray', alpha=0.3, label='Predicted Variance')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Process Regression with Batches')
plt.legend()
plt.show()
