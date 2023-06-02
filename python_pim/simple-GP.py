#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from pim_support import *
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel


th_train, u_train, upast_test, thpast_test = import_data()
th_train = th_train[:10000]
def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

na = 5
nb = 5
Xtrain, Ytrain = create_IO_data(u_train, th_train, na, nb)

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) 
reg = gaussian_process.GaussianProcessRegressor(kernel=kernel)
reg.fit(Xtrain,Ytrain)
#%%
Ytrain_pred = reg.predict(Xtrain)
print('train prediction errors:')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5,'radians')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/Ytrain.std()*100,'%')

 #only select the ones that are used in the example
Xtest = np.concatenate([upast_test[:,15-nb:], thpast_test[:,15-na:]],axis=1)

Ypredict = reg.predict(Xtest)

assert len(Ypredict)==len(upast_test), 'number of samples changed!!'
plt.plot(Ytrain_pred, c='r',alpha=0.5)
plt.scatter(np.arange(len(Ytrain)),Ytrain,alpha=1, s=0.1)

plt.plot()

# np.savez('test-prediction-example-submission-file.npz', upast=upast_test, thpast=thpast_test, thnow=Ypredict)
# %%
