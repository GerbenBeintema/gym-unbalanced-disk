from torch.cuda import is_available, get_device_name
from torch.nn.utils import clip_grad_value_
from torch import device, save, no_grad
from torch.utils.data import DataLoader
from torch.nn import Module, MSELoss
from torch import tensor, float32, mean
from torch.optim import Adam
from pandas import DataFrame
from os.path import join, exists
from sys import stdout
from os import getcwd, makedirs
from tqdm import tqdm
from numpy import arange
from numpy.random import shuffle
import matplotlib.pyplot as plt

class Model_processes():
    """Class to store the model processes"""
    def __init__(self, model, DIR:str):
        self.model_name = model.name
        self.model = model
        self.DIR = DIR

        if not exists(DIR):
            makedirs(DIR)


    def save_model(self, DIR:str|None=None):
        """Save the model"""
        if DIR is not None:
            store_path = join(DIR, self.model_name)
        else:
            store_path = join(self.DIR, self.model_name)

        save(self.model.state_dict(), store_path)

    def load_model(self, model_name:str|None, DIR:str|None=None):
        """Load the model"""
        if DIR is not None:
            store_path = join(DIR, model_name)
        else:
            store_path = join(self.DIR, model_name)
            
        self.model.load_state_dict(store_path)

class Trainer(Model_processes):
    """Trainer class for training a model, works for NL and ANN models (NOE use other trainer)"""
    def __init__(self, model:Module, dl_train:DataLoader, dl_val:DataLoader, dl_test:DataLoader, DIR:str):
        """Initialize the trainer"""
        super().__init__(model, DIR)

        # Check if GPU is available
        self.device = device("cuda" if is_available() else "cpu")
        print(f"The device that will be used in training is {get_device_name(self.device)}")

        self.model = model.to(self.device)
        self.model_name = model.name

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = MSELoss()

        assert self.criterion is not None, "Please define a loss function"
        assert self.optimizer is not None, "Please define an optimizer"
        assert self.model_name is not None, "Please define a model name"

        self.train = dl_train
        self.val = dl_val
        self.test = dl_test

    def train_epoch(self, dl:DataLoader):
        """Train the model for one epoch
        Parameters:
            dl: dataloader
        Returns:
            epoch_metrics: dict"""
        # Put the model in training mode
        self.model.train().float()

    # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss": [],
        }

        # Create a progress bar using TQDM
        stdout.flush()
        with tqdm(total=len(dl), desc=f'Training') as pbar:
            # Iterate over the training dataset
            for inputs, truths in dl:
                # Zero the gradients from the previous step
                self.optimizer.zero_grad()

                # Move the inputs and truths to the target device
                inputs = tensor(inputs, device=self.device, dtype=float32)
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = tensor(truths, device=self.device, dtype=float32)

                # Run model on the inputs
                output = self.model(inputs)

                # Perform backpropagation
                loss = self.criterion(output, truths)
                loss.backward()
                clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.item(),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(1)

                # Add to epoch's metrics
                for k,v in step_metrics.items():
                    epoch_metrics[k].append(v)

        stdout.flush()

        # Return metrics
        return epoch_metrics
    
    def val_epoch(self, dl:DataLoader):
        """Validate the model for one epoch
        Parameters:
            dl: dataloader
        Returns:
            epoch_metrics: dict"""
        # Put the model in evaluation mode
        self.model.eval().float()

        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss = 0

        # Create a progress bar using TQDM
        stdout.flush()
        with no_grad(), tqdm(dl, desc=f'Validation') as pbar:
            # Iterate over the validation dataloader
            for inputs, truths in dl:
                 # Move the inputs and truths to the target device
                inputs = tensor(inputs, device=self.device, dtype=float32 )
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = tensor(truths, device=self.device, dtype=float32)

                # Run model on the inputs
                output = self.model(inputs)
                loss = self.criterion(output, truths)

                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.item(),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(1)

                amount += 1
                total_loss += step_metrics["loss"]

        stdout.flush()

        # Print mean of metrics
        total_loss /= amount
        print(f'Validation loss is {total_loss/amount}')

        # Return mean loss and accuracy
        return {
            "loss": [total_loss],
        }

    def fit(self, epochs:int, batch_size:int):
        """Fit the model
        Parameters:
        epochs: int = The amount of epochs to train the model for
        batch_size: int = The batch size to use for training
        """
        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        dl_train = self.train
        dl_val = self.val

        # Store metrics of the training process (plot this to gain insight)
        df_train = DataFrame()
        df_val = DataFrame()

        # Train the model for the provided amount of epochs
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            metrics_train = self.train_epoch(dl_train)
            df_train = df_train.append(DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

            metrics_val = self.val_epoch(dl_val)
            df_val = df_val.append(DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)

        # Save the model data
        df_train.to_csv(f'{self.DIR}\\train_{self.model_name}.csv')
        df_val.to_csv(f'{self.DIR}\\val_{self.model_name}.csv')
        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.

class NOE_Trainer(Model_processes):
    """NOE Trainer class, for other models use the other trainer class"""
    def __init__(self, model, Xtrain:tensor, Ytrain:tensor, Xval:tensor, Yval:tensor, Xtest:tensor=None, Ytest:tensor=None, DIR:str=None):
        """Initialize the trainer
        Parameters:
            model: model
            Xtrain: training input
            Ytrain: training output
            Xval: validation input
            Yval: validation output
            Xtest: test input
            Ytest: test output
            DIR: directory to store the model data
        """
        super().__init__(model, DIR)

        self.device = 'cuda' if is_available() else 'cpu'

        # Set the device to use for training
        self.model = model.to(self.device)
        self.model_name = model.name

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = MSELoss()

        assert self.criterion is not None, "Please define a loss function"
        assert self.optimizer is not None, "Please define an optimizer"
        assert self.model_name is not None, "Please define a model name"

        self.Xtrain = Xtrain.to(self.device)
        self.Ytrain = Ytrain.to(self.device)
        self.Xval = Xval.to(self.device)
        self.Yval = Yval.to(self.device)

        self.Xtest = Xtest.to(self.device) if Xtest is not None else None
        self.Ytest = Ytest.to(self.device) if Ytest is not None else None

    def fit(self, epochs:int, batch_size:int, n_burn:int, plot:bool=True):
        """Fit the model
        Parameters:
            epochs: int = The amount of epochs to train the model for
            batch_size: int = The batch size to use for training
            n_burn: int = The amount of burn-in samples to use for the loss function
            plot: bool = Whether to plot the training process
        """
        ids = arange(len(self.Xtrain),dtype=int) 

        for epoch in range(epochs):
            shuffle(ids) #inspace shuffle of the ids of the trainin set to select a random subset 
            for i in range(0,len(self.Xtrain),batch_size):
                ids_now = ids[i:i+batch_size] #the ids of the current batch
                Uin = self.Xtrain[ids_now].to(self.device) #d)
                Y_real = self.Ytrain[ids_now].to(self.device) #d)

                Y_predict = self.model(inputs=Uin) #d)
                Loss = self.criterion(Y_predict, Y_real) #d)
                
                self.optimizer.zero_grad()  #d)
                Loss.backward()  #d)
                self.optimizer.step()  #d)
            
            with no_grad(): #monitor
                self.model.eval()
                Loss_val = self.criterion(self.model(inputs=self.Xval)[:,n_burn:], self.Yval[:,n_burn:])
                Loss_train = self.criterion(self.model(inputs=self.Xtrain)[:,n_burn:], self.Ytrain[:,n_burn:])
                print(f'epoch={epoch}, Validation NRMS={Loss_val.item():.2%}, Train NRMS={Loss_train.item():.2%}')
                self.model.train()
        
        if plot:
            self.plot_results()
    
    def plot_results(self):
        with no_grad():
            self.model.eval()
            plt.plot(self.Yval[0].cpu().numpy())
            plt.plot(self.model(inputs=self.Xval)[0].cpu().numpy(),'--')
            plt.xlabel('k')
            plt.ylabel('y')
            plt.xlim(0,100)
            plt.legend(['real','predicted'])
            plt.show()
            plt.plot(mean((self.Ytrain-self.model(inputs=self.Xtrain))**2,axis=0).cpu().numpy()**0.5) #average over the error in batch
            plt.title('batch averaged time-dependent error')
            plt.ylabel('error')
            plt.xlabel('i')
            plt.grid()
            plt.show()
            self.model.train()
