import numpy as np

out = np.load('./Measurement-data.npz')

u = out['u']
th = out['th']

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

na = 5
nb = 5
X = []
Y = []
for i in range(max(na,nb),len(u)):
    X.append(np.concatenate([u[i-nb:i],th[i-nb:i]]))
    Y.append(th[i])
Y = np.array(Y)
X = np.array(X)
reg.fit(X,Y)

out = np.loadtxt('Simulation-task-submission-file.csv',skiprows=1,delimiter=',')
U = out[:,0]
Y = out[:,1] #only the first 20 are filled

k0 = 20 #max(na, nb)
Upast = U[k0-nb:k0]
Ypast = Y[k0-na:k0]

for k in range(k0,len(Y)):
    yhat = reg.predict(np.concatenate([Upast,Ypast],axis=0)[None])[0]
    Upast = np.append(Upast[1:], U[k])
    Ypast = np.append(Ypast[1:], yhat)
    Y[k] = yhat


out = np.stack([U,Y],axis=-1)

with open('Simulation-task-submission-file.csv') as f:
    header = f.readline()[:-1]
np.savetxt('Simulation-task-submission-file-linear.csv', out, delimiter=',', header=header)