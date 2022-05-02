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

out = np.loadtxt('Prediction-task-submission-file.csv',skiprows=1,delimiter=',')
Upast = out[:,:10]
Ypast = out[:,10:20]
Yempty = out[:,20] #empty

Unb = Upast[:,-nb:]
Yna = Ypast[:,-na:]

Ypred = reg.predict(np.concatenate([Unb,Yna],axis=1))

out = np.concatenate([Upast,Ypast,Ypred[:,None]],axis=-1)

with open('Prediction-task-submission-file.csv') as f:
    header = f.readline()[:-1]
np.savetxt('Prediction-task-submission-file-linear.csv', out, delimiter=',', header=header)