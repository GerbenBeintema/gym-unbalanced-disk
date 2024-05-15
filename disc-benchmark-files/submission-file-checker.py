import sys
import numpy as np
from scipy.io import loadmat

from matplotlib import pyplot as plt

submission_file = sys.argv[1] #first argument is input file
solution_file = sys.argv[2] #second argument is solution file

def load_task_file(file, target_name):
    if file.endswith('csv'):
        data = np.loadtxt(file, delimiter=',')
        return data[:,-1]
    elif file.endswith('mat'): #target_name is th in simulation and thnow in prediction
        return loadmat(file)[target_name].flatten()
    elif file.endswith('npz'):
        return np.load(file)[target_name].flatten()
    else:
        raise ValueError('The given file={file} does not have a extention of [csv, mat, npz]')

def load_simulation_task_file(file : str):
    return load_task_file(file, 'th')
def load_prediction_task_file(file : str):
    return load_task_file(file, 'thnow')

if 'simulation' in solution_file:
    th = load_simulation_task_file(solution_file)
    th_hat = load_simulation_task_file(submission_file)
    case = 'Simulation'
elif 'prediction' in solution_file:
    th = load_prediction_task_file(solution_file)
    th_hat = load_prediction_task_file(submission_file)
    case = 'Prediction'
else:
    raise ValueError(f'The given Solution file={solution_file} does neither contain "simulation" or "prediction"')

residual = th_hat - th
RMS = (residual**2).mean()**0.5
RMS_deg = RMS/(2*np.pi)*360
NRMS = RMS/th.std()
print(f'################ {case} Result ##################')
print(f'RMS= {RMS:.4f} radians')
print(f'RMS= {RMS_deg:.3f} degrees')
print(f'NRMS= {NRMS:.2%}')

plt.plot(th,label='measured')
plt.plot(residual,label=f'residual {case}')
plt.title(f'{case} NRMS= {NRMS:.2%}')
plt.legend()
plt.xlabel('k')
plt.ylabel('angle (rad)')
plt.show()
