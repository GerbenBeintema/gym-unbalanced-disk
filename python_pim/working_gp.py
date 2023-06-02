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
# # Define the kernel
# kernel = GPy.kern.RBF(input_dim=5, variance=0.1, lengthscale=0.5) 

# Create the NARX model

kernel = GPy.kern.RBF(input_dim=10, variance=0.1, lengthscale=0.5)
model = GPy.models.GPRegression(X, Y,kernel=kernel)

# Optimize the model
model.optimize('bfgs')

# New test data
#%%
# Make predictions
Y_pred_mean, Y_pred_cov = model.predict(X)
Y_pred_mean = Y_pred_mean[0:100]
Y_pred_cov = Y_pred_cov[0:100]
plt.plot(Y[0:100])
plt.plot(Y_pred_mean[0:100])
#plot the confidence intervals
plt.fill_between(np.arange(len(Y_pred_mean)),Y_pred_mean[:,0]-2*np.sqrt(Y_pred_cov[:,0]),Y_pred_mean[:,0]+2*np.sqrt(Y_pred_cov[:,0]),alpha=0.5)


# %%
