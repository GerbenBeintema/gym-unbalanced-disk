#%%
import pandas as pd
import numpy as np
import plotly.express as px


#import csv
data = pd.read_csv('data/data5.csv')

#plot data based on varaince and lengthscale


#log scale
# px.scatter(data, y='rms_mean', x='lengthscale', color='nmrs', size='nmrs', hover_data=['rms_mean', 'rms_deg', 'nmrs'], log_x=True, log_y=True)
# px.scatter(data, y='rms_mean', x='variance', color='rms_deg', size='rms_deg', hover_data=['rms_mean', 'rms_deg', 'nmrs'], log_x=True, log_y=True)

#%%
df = px.data.iris()
fig = px.scatter_3d(data, x='lengthscale', y='variance', z='nmrs',
              color='rms_mean')
fig.show()

# %%
