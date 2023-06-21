# import plotly as py
import numpy as np
import plotly.express as px
#navigate to disc_benchmark-files
import os
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

import numpy as np
import plotly.graph_objects as go

# Example data


# Convert angles to Cartesian coordinates and rotate by 90 degrees
x = np.cos(th_train - np.pi/2)
y = np.sin(th_train - np.pi/2)
marker_size = np.interp(u_train, (u_train.min(), u_train.max()), (5, 20))
#make a list of random integers from 0 to max of u_train
# random_integers = np.random.randint(0, high=8000, size=100)
random_integers = np.arange(0,80000,1)
print(random_integers)
x = x[random_integers]
y = y[random_integers]
marker_size = marker_size[random_integers]

# Create a scatter plot with associated values
fig = go.Figure(data=go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=np.abs(marker_size), color=u_train,  colorscale='Viridis',showscale=True)
))

# Customize the plot and add layout settings
fig.update_layout(
    title="Distribution of Values on a Disk",
    xaxis=dict(range=[-1.1, 1.1], zeroline=False),
    yaxis=dict(range=[-1.1, 1.1], zeroline=False),
    width=500,
    height=500
)

fig.show()