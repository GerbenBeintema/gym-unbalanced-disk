#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt
from pim_support import *
import pandas as pd
import os
import time
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

#%%
# Create the NARX model
# variance_interval = np.arange(0,1,0.2)
variance_interval = [1]
lengthscale_interval = [1]
# lengthscale_interval = np.arange(0,1,0.2)
noise_variance_interval = [1]
data = pd.DataFrame(columns=['variance', 'lengthscale', 'rms_mean', 'rms_deg', 'nmrs'])
time_start = time.time()

num_iterations = len(variance_interval)*len(lengthscale_interval)*len(noise_variance_interval)
iteration = 0
for variance in variance_interval:
    for lengthscale in lengthscale_interval:
        for noise_variance in noise_variance_interval:
            print(f'Iteration {iteration+1}/{num_iterations} ', end='', flush=False)
            iteration_start = time.time()           
            iteration += 1
            kernel = GPy.kern.RBF(input_dim=10, variance=variance, lengthscale=lengthscale) + GPy.kern.White(10, variance=noise_variance)
            model = GPy.models.GPRegression(X, Y,kernel=kernel)
            model.optimize('bfgs')
            
            Y_pred, Y_pred_cov = model.predict(X)
            rms_mean =  np.mean((Y_pred-Y)**2)**0.5
            rms_deg = np.mean((Y_pred-Y)**2)**0.5/(2*np.pi)*360
            nmrs =  np.mean((Y_pred-Y)**2)**0.5/Y.std()*100
            data = data.append({'variance': variance, 'lengthscale': lengthscale, 'rms_mean': rms_mean, 'nmrs': nmrs}, ignore_index=True)
            iteration_end = time.time()
           

print('')
print(data)    

#%%
# save the data to a csv file with name data{highest number}.csv
os.chdir(os.path.dirname(os.path.realpath(__file__)))
filenames = os.listdir('data')
maxnum=0
for i in range(len(filenames)):
    num = int(filenames[i][-5])
    if num > maxnum:
        maxnum = num
data.to_csv(f'data/data{maxnum+1}.csv', index=False)
# New test data
#%%
# Make predictions
# Y_pred_mean, Y_pred_cov = model.predict(X)

# make a 3d plot of the variance lengthscale and rms_mean
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(z=data['rms_mean'], x=data['variance'], y=data['lengthscale'], mode='markers')])
fig.add_annotation(x=0.1, y=0.1, text="min", showarrow=True, arrowhead=1)
fig.show()


# Y_pred_mean = Y_pred_mean[0:300]
# Y_pred_cov = Y_pred_cov[0:300]
# plt.plot(Y[0:100])
# plt.plot(Y_pred_mean[0:100])
# # plot the confidence intervals
# plt.fill_between(np.arange(len(Y_pred_mean)),Y_pred_mean[:,0]-2*np.sqrt(Y_pred_cov[:,0]),Y_pred_mean[:,0]+2*np.sqrt(Y_pred_cov[:,0]),alpha=0.5)


# %%
