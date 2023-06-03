#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt
from pim_support import *
import pandas as pd
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
# variance_interval = [0.01, 0.1, 1, 10, 100]
# lengthscale_interval = [0.01, 0.1, 1, 10, 100]
variance_interval = [0.1]
lengthscale_interval = [0.1,1]
data = pd.DataFrame(columns=['variance', 'lengthscale', 'rms_mean', 'rms_deg', 'nmrs'])
for variance in variance_interval:
    for lengthscale in lengthscale_interval:
        
        kernel = GPy.kern.RBF(input_dim=10, variance=variance, lengthscale=lengthscale)
        model = GPy.models.GPRegression(X, Y,kernel=kernel)
        model.optimize('bfgs')
        
        Y_pred, Y_pred_cov = model.predict(X)
        rms_mean =  np.mean((Y_pred-Y)**2)**0.5
        rms_deg = np.mean((Y_pred-Y)**2)**0.5/(2*np.pi)*360
        nmrs =  np.mean((Y_pred-Y)**2)**0.5/Y.std()*100
        data = data.append({'variance': variance, 'lengthscale': lengthscale, 'rms_mean': rms_mean, 'rms_deg': rms_deg, 'nmrs': nmrs}, ignore_index=True)
        
print(data)        

        

# Optimize the model


# New test data
#%%
# Make predictions
Y_pred_mean, Y_pred_cov = model.predict(X)
Y_pred_mean = Y_pred_mean[0:300]
Y_pred_cov = Y_pred_cov[0:300]
plt.plot(Y[0:100])
plt.plot(Y_pred_mean[0:100])
# plot the confidence intervals
plt.fill_between(np.arange(len(Y_pred_mean)),Y_pred_mean[:,0]-2*np.sqrt(Y_pred_cov[:,0]),Y_pred_mean[:,0]+2*np.sqrt(Y_pred_cov[:,0]),alpha=0.5)


# %%
