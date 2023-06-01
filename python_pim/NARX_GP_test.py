#%%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
import os

#%%
def import_data():
    
    # go the parent directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.getcwd()
    print(path)
    out = np.load('disc-benchmark-files/training-data.npz')
    th_train = out['th'] #th[0],th[1],th[2],th[3],...
    u_train = out['u'] #u[0],u[1],u[2],u[3],...

    # data = np.load('test-prediction-submission-file.npz')
    data = np.load('disc-benchmark-files/test-prediction-submission-file.npz')
    upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
    thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
    # thpred = data['thnow'] #all zeros
    return th_train, u_train, upast_test, thpast_test
def f(upast,ypast):
    ukm2, ukm1 = upast
    ykm2, ykm1 = ypast
    ystar = (0.8 - 0.5 * np.exp(-ykm1 ** 2)) * ykm1 - (0.3 + 0.9 * np.exp(-ykm1 ** 2)) * ykm2 \
           + ukm1 + 0.2 * ukm2 + 0.1 * ukm1 * ukm2
    return ystar + np.random.normal(scale=0.01)

def use_NARX_model_in_simulation(ulist, f, na, nb):
    #init upast and ypast as lists.
    upast = [0]*nb 
    ypast = [0]*na 
    
    ylist = []
    for unow in ulist:
        #compute the current y given by f
        ynow = f(upast,ypast) 
        
        #update past arrays
        upast.append(unow)
        upast.pop(0)
        ypast.append(ynow)
        ypast.pop(0)
        
        #save result
        ylist.append(ynow)
    return np.array(ylist) #return result


def make_training_data(ulist,ylist,na,nb):
    #Xdata = (Nsamples,Nfeatures)
    #Ydata = (Nsamples)
    Xdata = []
    Ydata = []
    #for loop over the data:
    for k in range(max(na,nb),len(ulist)): #skip the first few indexes 
        Xdata.append(np.concatenate([ulist[k-nb:k],ylist[k-na:k]])) 
        Ydata.append(ylist[k]) 
    return np.array(Xdata), np.array(Ydata)
def batch_data(batch=0,batch_size=None, split=0.8):
    th_train_batch = th_train[i*batch_size:i*batch_size+batch_size].copy()
    u_train_batch = u_train[i*batch_size:i*batch_size+batch_size].copy()
    split_index = int(len(th_train_batch)*split) 
    Xtrain, Ytrain = make_training_data(th_train_batch[:split_index],u_train_batch[:split_index], na, nb) 
    Xval,   Yval   = make_training_data(th_train_batch[split_index:],u_train_batch[split_index:], na, nb)    
    return Xtrain, Ytrain, Xval, Yval
#%%
if __name__ == '__main__':
    
    batches= 4
    
    th_train, u_train, upast_test, thpast_test = import_data()
    batch_size = len(u_train)//batches
    ker = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)  
    reg = GaussianProcessRegressor(ker, n_restarts_optimizer=0)  
    na, nb = 10, 10
    np.random.seed(42)
    N = 500
    split = 0.8
    
    
    for i in range(batches):
        Xtrain, Ytrain, Xval, Yval = batch_data(i,batch_size,split)
        
        print(f"batch {i}/{batches}")
        reg.fit(Xtrain,Ytrain)  
        y_val_pred, y_val_pred_std = reg.predict(Xval,return_std=True)
        print(f'Validation NRMS= {np.mean((y_val_pred-Yval)**2)**0.5/np.std(y_val_pred)}')
        break


    #residual calculations and plotting
#%%
if __name__ == '__main__':
    Ytrain_pred, Ytrain_pred_std = reg.predict(Xtrain,return_std=True)  
    plt.figure(figsize=(12,5))  
    plt.plot(Ytrain)  
    plt.title('prediction on the training set')
    Ytrain_pred, Ytrain_pred_std = reg.predict(Xtrain,return_std=True)  
    plt.errorbar(np.arange(len(Xtrain)), (Ytrain_pred), yerr=2*Ytrain_pred_std,fmt='.r')  
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred']) 
    plt.show()  

    plt.figure(figsize=(12,5))  
    plt.title('prediction on the validation set')
    plt.plot(Yval)  
    Yval_pred, Yval_pred_std = reg.predict(Xval,return_std=True)  
    plt.errorbar(np.arange(len(Xval)), (Yval_pred), yerr=2*Yval_pred_std,fmt='.r')  
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred'])  
    plt.show()  

    print(f'Validation NRMS= {np.mean((Yval_pred-Yval)**2)**0.5/np.std(Yval)}') 



# %%
