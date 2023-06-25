#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt
from functions import use_NARX_model_in_simulation, import_data, make_training_data
import pandas as pd
import os

# Input data
na, nb = 5, 5
u_train, th_train, u_pred, th_pred, u_sim, th_sim = import_data()
X_train, Y_train, X_val, Y_val =  make_training_data(u_train, th_train, na, nb, train_amount=0.8)

NUM_TRAINING = 5000
X_train = X_train[0:NUM_TRAINING]
Y_train = Y_train[0:NUM_TRAINING]

data = pd.DataFrame(columns=['noise_variance', 'kern_variance', 'lengthscale', 'rms_mean', 'rms_deg', 'nrms'])

# Define the kernel
# kernel = GPy.kern.RBF(input_dim=5, variance=0.1, lengthscale=0.5) 

#%% Sparse GP regression

model_sparse = GPy.models.SparseGPRegression(X_train, Y_train, num_inducing=1000)
# Z = np.random.rand(1000,na+nb)*2.5
# model_sparse = GPy.models.SparseGPRegression(X_train, Y_train, Z=Z) 
model_sparse.randomize()
model_sparse.Z.unconstrain()
model_sparse.optimize('bfgs')

#%% PREDICTION on VALIDATION data
# Inference
Y_pred_sparse, Y_pred_cov_sparse = model_sparse.predict(X_val)

# Compute metrics:
rms_mean_sparse =  np.mean((Y_pred_sparse-Y_val)**2)**0.5
rms_deg_sparse = np.mean((Y_pred_sparse-Y_val)**2)**0.5/(2*np.pi)*360
nrms_sparse =  np.mean((Y_pred_sparse-Y_val)**2)**0.5/Y_val.std()*100
data = data.append({'noise_variance': model_sparse.Gaussian_noise.variance.item(),  'kern_variance': model_sparse.kern.variance.item(), 'lengthscale': model_sparse.kern.lengthscale.item(), 'rms_mean': rms_mean_sparse, 'rms_deg': rms_deg_sparse, 'nrms': nrms_sparse, 'average_output_variance': np.mean(Y_pred_cov_sparse)}, ignore_index=True)
display(data)
#%%
# Plot the results in prediction for first N datapoints
N=150
Y_pred = Y_pred_sparse[:N,0]
Y_pred_cov = Y_pred_cov_sparse[:N,0]
plt.subplot(1,1,1)
# plt.errorbar(X_val[:N,0], Y_pred, yerr=2*Y_pred_cov,fmt='.r') #a)
# plt.scatter(X_val[:N,0], Y_val[:N], label='Y_gt')
plt.plot(Y_val[:N], label='Y_gt')
plt.plot(Y_pred, label='Y_pred')
plt.plot(Y_val[:N,0]-Y_pred, label='residual')
plt.fill_between(np.arange(N), Y_pred-2*Y_pred_cov,y2=Y_pred+2*Y_pred_cov,alpha=0.25,label='post std function')
plt.legend()
plt.title(f'Sparse GP on {N} datapoints on validation set')


#%% PREDICTION on SUBMISSION file

model = model_sparse # sparse, use model_full for full
X_pred = np.concatenate([u_pred[:,-nb-1:-1], th_pred[:,-na-1:-1]], axis=1)
Y_gt = th_pred[:,-1]
Y_pred_sub, Y_pred_sub_cov = model.predict(X_pred)

# Compute metrics:
rms_mean_pred =  np.mean((Y_pred_sub[:,0]-Y_gt)**2)**0.5
rms_deg_pred = np.mean((Y_pred_sub[:,0]-Y_gt)**2)**0.5/(2*np.pi)*360
nrms_pred =  np.mean((Y_pred_sub[:,0]-Y_gt)**2)**0.5/Y_gt.std()*100
prediction_set = pd.DataFrame()
prediction_set = prediction_set.append({'rms_mean': rms_mean_pred, 'rms_deg': rms_deg_pred, 'nrms': nrms_pred, 'average_output_variance': np.mean(Y_pred_sub_cov)}, ignore_index=True)
display(prediction_set)

# Plot the results in prediction for first N datapoints
N=1500
Y_pred = Y_pred_sub[:N,0]
Y_pred_cov = Y_pred_sub_cov[:N,0]
plt.subplot(1,1,1)
# plt.errorbar(X_val[:N,0], Y_pred, yerr=2*Y_pred_cov,fmt='.r') #a)
# plt.scatter(X_val[:N,0], Y_val[:N], label='Y_gt')
plt.plot(Y_gt[:N], label='Y_gt')
plt.plot(Y_pred, label='Y_pred')
plt.plot(Y_gt[:N]-Y_pred, label='residual')
# plt.fill_between(np.arange(N), Y_pred-2*Y_pred_cov,y2=Y_pred+2*Y_pred_cov,alpha=0.25,label='post std function')
plt.legend()
plt.title(f'GP prediction on {N} datapoints from the submission file')



#%% SIMULATION on VALIDATION data
model = model_sparse # sparse, use model_full for full
fmodel = lambda u, y: model.predict(np.concatenate([u,y])[None,:])

u_val = list(u_train[-len(Y_val):])
th_val = list(th_train[-len(Y_val):])
Y_sim, Y_sim_cov = use_NARX_model_in_simulation(u_val, th_val, fmodel, na, nb) # simulation for submission file

# Compute metrics:
rms_mean_pred =  np.mean((Y_sim[50:]-th_val[50:])**2)**0.5
rms_deg_pred = np.mean((Y_sim[50:]-th_val[50:])**2)**0.5/(2*np.pi)*360
nrms_pred =  np.mean((Y_sim[50:]-th_val[50:])**2)**0.5/np.array(th_val[50:]).std()*100
simulation_val_set = pd.DataFrame()
simulation_val_set = prediction_set.append({'rms_mean': rms_mean_pred, 'rms_deg': rms_deg_pred, 'nrms': nrms_pred, 'average_output_variance': np.mean(Y_pred_sub_cov)}, ignore_index=True)
display(prediction_set)

# Plot simulation
N=500
plt.subplot(1,1,1)
plt.plot(th_val[:N], label='Y_gt', color='blue')
plt.plot(Y_sim[:N], label='Y_pred', color='orange')
plt.plot(th_val[:N]-Y_sim[:N], label='residual', color='green')
plt.fill_between(np.arange(N), Y_sim[:N]-2*Y_sim_cov[:N],y2=Y_sim[:N]+2*Y_sim_cov[:N],alpha=0.25,label='post std function')
plt.legend()
# plt.ylim([-10,10])
plt.title(f'GP in simulation on validation data')


#%% SIMULATION on SUBMISSION file
model = model_sparse # sparse, use model_full for full
fmodel = lambda u, y: model.predict(np.concatenate([u,y])[None,:])

Y_sim, Y_sim_cov = use_NARX_model_in_simulation(list(u_sim), list(th_sim), fmodel, na, nb) # simulation for submission file

# Plot simulation
N=500
plt.subplot(1,1,1)
plt.plot(Y_sim[:N], label='Y_pred', color='orange')
plt.plot(th_sim[:50], label='Y_gt', color='blue')
plt.fill_between(np.arange(N), Y_sim[:N]-2*Y_sim_cov[:N],y2=Y_sim[:N]+2*Y_sim_cov[:N],alpha=0.25,label='post std function')
plt.legend()
plt.title(f'GP in simulation on submission file data')


#%% Full GP regression

model_full = GPy.models.GPRegression(X_train, Y_train)
model_full.optimize('bfgs')

Y_pred_full, Y_pred_cov_full = model_full.predict(X_val)

rms_mean_full =  np.mean((Y_pred_full-Y_val)**2)**0.5
rms_deg_full = np.mean((Y_pred_full-Y_val)**2)**0.5/(2*np.pi)*360
nrms_full =  np.mean((Y_pred_full-Y_val)**2)**0.5/Y_val.std()*100

data = data.append({'noise_variance': model_full.Gaussian_noise.variance.item(), 'kern_variance': model_full.kern.variance.item(), 'lengthscale': model_full.kern.lengthscale.item(), 'rms_mean': rms_mean_full, 'rms_deg': rms_deg_full, 'nrms': nrms_full, 'average_output_variance': np.mean(Y_pred_cov_full)}, ignore_index=True)
display(data)

# Plot the results in prediction for first N datapoints
N=500
Y_pred = Y_pred_full[:N,0]
Y_pred_cov = Y_pred_cov_full[:N,0]
plt.subplot(1,1,1)
plt.plot(Y_val[:N], label='Y_gt')
plt.plot(Y_pred, label='Y_pred')
plt.fill_between(np.arange(N), Y_pred-2*Y_pred_cov,y2=Y_pred+2*Y_pred_cov,alpha=0.25,label='post std function')
plt.legend()
plt.title(f'Full GP on {N} datapoints')


#%% 

# save the data to a csv file with name data{highest number}.csv
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
# filenames = os.listdir('data')
# maxnum=0
# for i in range(len(filenames)):
#     num = int(filenames[i][-5])
#     if num > maxnum:
#         maxnum = num
# data.to_csv(f'data/data{maxnum+1}.csv', index=False)



# Optimize the model


# New test data
#%%
# Make predictions on test sets


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
# plt.fill_between(np.arange(len(Y_pred_mean)),Y_pred_mean[:,0]-2*np.sqrt(Y_pred_cov[:,0]),Y_pred_mean[:,0]+2*np.sqrt(Y_pred_cov[:,0]),alpha=0.25)


# %%
