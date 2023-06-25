#%% libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#%% HEATMAP NA NB

def format_annotation(value):
    if 10<= value <= 10000:
        return f"{int(value):.0f}"
    else:
        return f"{value:.2g}"

for split in ['val_pred', 'val_sim']:
    # Read the CSV file
    file_path = '/home/dlehman/5SC28-ML4SC-gym-unbalanced-disk/Gaussian_Processes/results/sparse_GP_varying_na_nb/'+split+'.csv'
    data = pd.read_csv(file_path)

    # Replace 'na' and 'nb' with NaN in the nrms column
    data['nrms'] = (data['nrms']).replace(['na', 'nb'], np.nan)

    # Pivot the data to create a matrix for the heatmap
    heatmap_data = data.pivot('na', 'nb', 'nrms')

    # Set up the heatmap figure size
    plt.figure(figsize=(10, 8))

    # Create the heatmap
    ax = sns.heatmap(heatmap_data, vmin=0.7, vmax=200, cmap=sns.cm.rocket_r,annot=True, linewidths=.5, fmt=".4g", norm=LogNorm())

    # Set the labels for x-axis and y-axis
    ax.set(xlabel='nb', ylabel='na')

    # Set the tick labels for the x-axis
    ax.set_xticklabels(np.arange(1, 16))
    ax.set_yticklabels(np.arange(0, 16))

    # Set the position of x-axis ticks and label
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    for i, annotation in enumerate(ax.texts):
        ax.texts[i].set_text(format_annotation(float(annotation.get_text())))
        


    # Display the heatmap
    plt.show()




# %%  PLOTS OF NA and NB

file_path = '/home/dlehman/5SC28-ML4SC-gym-unbalanced-disk/Gaussian_Processes/results/sparse_GP_varying_na_nb/val_pred.csv'
data = pd.read_csv(file_path)

nrms = data['nrms']


sns.set(rc={'figure.figsize': (8, 5)})
x = np.arange(1, 16)
sns.lineplot(x=x, y=nrms[0, :], label='na=0')
sns.lineplot(x=x, y=nrms[1, :], label='na=1')
sns.lineplot(x=x, y=nrms[2, :], label='na=2', marker='o')
sns.lineplot(x=x, y=nrms[3, :], label='na=3', marker='o')
sns.lineplot(x=x, y=nrms[4, :], label='na=4')
sns.lineplot(x=x, y=nrms[5, :], label='na=5')
sns.lineplot(x=x, y=nrms[6, :], label='na=6')
sns.lineplot(x=x, y=nrms[7, :], label='na=7')
sns.lineplot(x=x, y=nrms[8, :], label='na=8')
sns.lineplot(x=x, y=nrms[9, :], label='na=9')
sns.lineplot(x=x, y=nrms[10, :], label='na=10')
sns.lineplot(x=x, y=nrms[11, :], label='na=11')
sns.lineplot(x=x, y=nrms[12, :], label='na=12')
sns.lineplot(x=x, y=nrms[13, :], label='na=13')
sns.lineplot(x=x, y=nrms[14, :], label='na=14')
sns.lineplot(x=x, y=nrms[15, :], label='na=15')
plt.yscale('log')
plt.xlabel('nb')
plt.ylabel('Loss')
plt.title('Loss for different na and nb values')
plt.legend(bbox_to_anchor=(1, 1))
plt.xticks(np.arange(1, 16))
plt.xlim(0.5, 15.5)
plt.show()

#%% HEATMAP inducing points/training points

for split in ['val_pred', 'val_sim']:
    # Read the CSV file into a pandas DataFrame
    csv_path = '/home/dlehman/5SC28-ML4SC-gym-unbalanced-disk/Gaussian_Processes/results/sparse_GP_varying_inducing_points/'+split+'.csv'
    data = pd.read_csv(csv_path)

    data['nrms'] = (data['nrms'])
    data['N_training_points'] = data['N_training_points'].astype('int')
    data['N_inducing_points'] = data['N_inducing_points'].astype('int')

    # Pivot the DataFrame to create a matrix-like structure for the heatmap
    pivot_data = data.pivot('N_inducing_points', 'N_training_points', 'nrms')

    # Plot the heatmap using seaborn
    plt.figure(figsize=(10, 8))  # Set the figure size
    # sns.set_style('ticks')  # Remove the gridlines
    sns.heatmap(pivot_data, vmin=-0.39, vmax=8 ,annot=True, cmap=sns.cm.rocket_r, fmt=".3g", cbar=True, norm = LogNorm(), linewidths=0.5)  # Customize the heatmap

    # Set the axis labels and title
    plt.xlabel('Number of training points')
    plt.ylabel('Number of inducing points')

    # Display the plot
    plt.show()
    

#%% SPARSE vs FULL

for split in ['val_pred', 'val_sim']:
    path_full = '/home/dlehman/5SC28-ML4SC-gym-unbalanced-disk/Gaussian_Processes/results/full_GP_varying_training_points/'+split+'.csv'
    path_sparse = '/home/dlehman/5SC28-ML4SC-gym-unbalanced-disk/Gaussian_Processes/results/sparse_GP_varying_inducing_points/'+split+'.csv'

    full = pd.read_csv(path_full)
    sparse = pd.read_csv(path_sparse)

    x = full['N_training_points']
    nrms_full = full['nrms']

    nrms_sparse=[]
    for tp in x:
        nrms_sparse.append(np.min(sparse[sparse['N_training_points']==tp]['nrms']))


    sns.set(rc={'figure.figsize': (8, 5)})

    # Plot the lines
    sns.lineplot(x=x, y=nrms_full, label='full')
    sns.lineplot(x=x, y=nrms_sparse, label='sparse')

    # Set logarithmic scale
    plt.yscale('log')
    plt.xscale('log')

    # Add labels and title
    plt.xlabel('Number of training points')
    plt.ylabel('NRMS (log scale)')
    # plt.title('Loss for different na and nb values')

    # Add legend and plot markers for the last two points
    plt.legend(bbox_to_anchor=(0.7, 0.7))

    # Get the last two points of each plot
    last_points_full = nrms_full[:]
    last_points_sparse = nrms_sparse[:]

    # Add annotations for the last two points of each plot
    for i, (x_val, y_val) in enumerate(zip(x[:], last_points_full)):
        plt.annotate(f'{y_val:.2f}', (x_val, y_val), xytext=(5, -15), textcoords='offset points', color='blue')

    for i, (x_val, y_val) in enumerate(zip(x[:], last_points_sparse)):
        plt.annotate(f'{y_val:.2f}', (x_val, y_val), xytext=(5, 10), textcoords='offset points', color='orange')

    plt.show()

# %%
