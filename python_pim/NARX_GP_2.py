#%%
import GPy
import numpy as np
from pim_support import *
np.random.seed(101)

N = 50
noise_var = 0.05
th_train, u_train, upast_test, thpast_test = import_data()
na, nb = 5, 5

k = GPy.kern.RBF(1)
batches= 128
batch_size = len(u_train)//batches

th_train, u_train, upast_test, thpast_test = import_data()

na, nb = 5,5 


for i in range(batches):
    make_training_data(th_train,u_train,na,nb)

    Xtrain = th_train[i*batch_size:i*batch_size+batch_size].copy()
    Ytrain =u_train[i*batch_size:i*batch_size+batch_size].copy()
    Xdata, Ydata = make_training_data(Xtrain,Ytrain, na, nb)
    print(f"shape of Xdata: {Xdata.shape} , shape of Ydata: {Ydata.shape}")
    
    print(f"batch {i}/{batches}")
    m_full = GPy.models.GPRegression(Xdata,Ydata[:,None])
    m_full.optimize('bfgs') 
 
    break


#%%
Ytrain_pred = m_full.predict(Xdata)  
plt.figure(figsize=(12,5))  
plt.plot(Ytrain)  
plt.title('prediction on the training set')

plt.plot(Ytrain_pred[0],Ytrain_pred[1],'.r',marker='o')
plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred']) 
plt.show()  
# m_full.plot()
print (m_full)
# %%
