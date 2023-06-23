import numpy as np
from matplotlib import pyplot as plt
import os

def compute_rms(Y_pred, Y_gt):
    assert Y_pred.shape == Y_gt.shape, f'Y_pred and Y_gt should have the same shape, but got shapes {Y_pred.shape} and {Y_gt.shape}'
    rms_rad =  np.mean((Y_pred-Y_gt)**2)**0.5
    rms_deg = np.mean((Y_pred-Y_gt)**2)**0.5/np.pi*180
    nrms =  np.mean((Y_pred-Y_gt)**2)**0.5/Y_gt.std()*100
    return rms_rad, rms_deg, nrms
    
    
def import_data():
    
    # go the parent directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.getcwd()
    print(path)
    out = np.load('disc-benchmark-files/training-data.npz')
    th_train = out['th'] #th[0],th[1],th[2],th[3],...
    u_train = out['u'] #u[0],u[1],u[2],u[3],...

    data = np.load('disc-benchmark-files/test-prediction-submission-file.npz')
    u_pred = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
    th_pred = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
    # thpred = data['thnow'] #all zeros
    
    sim = np.load('disc-benchmark-files/test-simulation-submission-file.npz')
    u_sim = sim['u'] 
    th_sim = sim['th']
    
    return u_train, th_train, u_pred, th_pred, u_sim, th_sim

# def batch_data(th_train,u_train batch=0,batch_size=None, split=1):
    # th_train_batch = th_train[i*batch_size:i*batch_size+batch_size].copy()
    # u_train_batch = u_train[i*batch_size:i*batch_size+batch_size].copy()
    # split_index = int(len(th_train_batch)*split) 
    # Xtrain, Ytrain = make_training_data(th_train_batch[:split_index],u_train_batch[:split_index], na, nb) 
    # Xval,   Yval   = make_training_data(th_train_batch[split_index:],u_train_batch[split_index:], na, nb)    
    # return Xtrain, Ytrain, Xval, Yval
    
def f(upast,ypast):
    ukm2, ukm1 = upast
    ykm2, ykm1 = ypast
    ystar = (0.8 - 0.5 * np.exp(-ykm1 ** 2)) * ykm1 - (0.3 + 0.9 * np.exp(-ykm1 ** 2)) * ykm2 \
           + ukm1 + 0.2 * ukm2 + 0.1 * ukm1 * ukm2
    return ystar + np.random.normal(scale=0.01)


def use_NARX_model_in_simulation(ulist, ylist, f, na, nb): # only first 50 values of thlist are considered
    ylist = ylist[:50]
    y_cov_list = [0]*50
    
    #init upast and ypast as lists.
    upast = ulist[50-nb:50] # u[k-nb]...u[k-1] 
    ypast = ylist[-na:] 
    
    for unow in ulist[50:]:
        #compute the current y given by f
        out = f(upast,ypast) 
        ynow, y_cov = out[0].item(), out[1].item()
                
        #update past arrays
        upast.append(unow)
        upast.pop(0)
        ypast.append(ynow)
        ypast.pop(0)
        
        #save result
        ylist.append(ynow)
        y_cov_list.append(y_cov)
    return np.array(ylist), np.array(y_cov_list) #return result


def make_training_data(ulist,ylist,na,nb, train_amount=1, present_input=False):
    X_train = []
    Y_train = []
    for k in range(max(na+present_input,nb),round(train_amount*len(ulist))): #skip the first few indexes 
        X_train.append(np.concatenate([ulist[k-nb:k],ylist[k-na-present_input:k-present_input]]))
        Y_train.append(ylist[k]) 
    if train_amount < 1:    
        X_val = []
        Y_val = []
        for k in range(round(train_amount*len(ulist)),len(ulist)):  
            X_val.append(np.concatenate([ulist[k-nb:k],ylist[k-na-present_input:k-present_input]])) 
            Y_val.append(ylist[k]) 
        return np.array(X_train), np.array(Y_train)[:,None], np.array(X_val), np.array(Y_val)
    else:
        return np.array(X_train), np.array(Y_train)[:,None]
    