#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt
from pim_support import *
# New test data
# Input data
na, nb = 5, 5
th_train, u_train, upast_test, thpast_test = import_data()
X,Y=  make_training_data(th_train,u_train,na,nb)

Y = Y[:,None]
X = X[0:1000]
Y = Y[0:1000]
# Define the kernel
kernel = GPy.kern.RBF(input_dim=5) + GPy.kern.RBF(5)

# Create the NARX model
model = GPy.models.GPRegression(X, Y, kernel=kernel)

# Optimize the model
model.optimize('bfgs')

# New test data
#%%
# Make predictions
Y_pred = model.predict(X)
# model.plot()
# plt.plot(X_test, Y_pred[0], c='red', label='Predicted Mean')
plt.show()
# plt.plot(X)
plt.plot(Y[0:100])
plt.plot(Y_pred[0], c='red', label='Predicted Mean')
plt.show()
# %%
