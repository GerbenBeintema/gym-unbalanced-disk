from os import getcwd, path
from pandas import read_csv, DataFrame, Series
from numpy import array, concatenate
from torch.utils.data import Dataset, random_split
from torch import tensor, float64

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        """Initilize the dataset
        Parameters:
            X: input data
            Y: output data
        Return:
            None
        """

        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def split_data(self, split_ratio:list=[0.6,0.2,0.2]):
        """Split the data into train and validation data"""
        return random_split(self, split_ratio)
        

class DATA():
    def __init__(self, na:int=15, nb:int=15, UseOE:bool=False, nf:int=100, NOE_Val:int=8000):
        """
        Initialize the data
        Parameters:
            na: number of past y values used
            nb: number of past u values used
            UseOE: use the output error model
            nf: number of features
            NOE_Val: number of steps used for validation
        returns:
            all in this class
        """


        self._get_data()

        self.train = self._rename_df1(read_csv(self.trainfilepath))
        self.testsim = self._rename_df1(read_csv(self.testsimfilepath))
        self.testsub = self._rename_df2(read_csv(self.testsubfilepath))

        if not UseOE:
            self.Xtrain, self.Ytrain = self.make_training_data(self.train.u, self.train.th, na, nb)
            self.Xval, self.Yval = self.make_training_data(self.testsim.u, self.testsim.th, na, nb)
        else:
            # Assert lenght for validation split
            assert NOE_Val < len(self.train), "NOE_Val must be smaller than the length of the training data"
            assert NOE_Val >= 0, "NOE_Val must be positive"
            val_split = int(len(self.train)-NOE_Val)


            convert = lambda x: [tensor(xi,dtype=float64) for xi in x]
            self.Xtrain, self.Ytrain = convert(self.make_OE_data(self.train.u, self.train.th, nf))
            if NOE_Val != 0:
                self.Xtrain, self.Xval, self.Ytrain, self.Yval = self.Xtrain[val_split:], self.Xtrain[:val_split], self.Ytrain[val_split:], self.Ytrain[:val_split]
            else:
                self.Xval, self.Yval = convert(self.make_OE_data(self.testsim.u, self.testsim.th, nf))
            

        #self.testsub_data = self.make_training_data(self.testsub.u, self.testsub.th, na, nb)



    def _get_data(self):
        """Get the data from the disc-benchmark-files folder"""

        DATA_DIR = path.join(getcwd(),'disc-benchmark-files\\')
    
        self.trainfilepath   = path.join(DATA_DIR,'training-data.csv')
        self.testsubfilepath = path.join(DATA_DIR,'test-prediction-submission-file.csv')
        self.testsimfilepath = path.join(DATA_DIR,'test-simulation-submission-file.csv')
    
    def _get_data_transform(self, na, nb):
        """Transforms the data to the correct format for the model f(u[k-15]-u[k-1],y[k-15]-y[k-1]) and returns X and Y data"""
        self.Xtrain, self.Ytrain = self.make_training_data(self.train)
        self.Xtestsim, self.Ytestsim = self.make_training_data(self.testsim)
        self.Xtestsub, self.Ytestsub = self.testsub.iloc[:, 0:-1] , self.testsub[:,-1]


    def make_training_data(self, ulist:list, ylist:list, na:int, nb:int):
        """Training Data Function (only needs to be called for training data and testsim)"""
        Xdata = []
        Ydata = []

        #create column names
        column_names=[f'u[k-{nb-i}]' for i in range(nb)]
        column_names.extend([f'y[k-{na-i}]' for i in range(na)])

        #for loop over the data:
        for k in range(max(na,nb),len(ulist)): #skip the first few indexes
            Xdata.append(concatenate([ulist[k-nb:k],ylist[k-na:k]])) 
            Ydata.append(ylist[k]) 

        return DataFrame(Xdata,columns=column_names), DataFrame(Ydata, columns=['y[k]'])


    def _rename_df1(self, df):
        """Rename the dataframe colums to something more callable"""
        return df.rename(columns={'# u':'u', ' th':'th'})
    
    def _rename_df2(self, df): # Does not work yet
        """Rename the dataframe colums to something more callable"""
        # Acceptable letters
        accept_letter = ["u", "y", "[", "]", "k", "-","1","2","3","4","5","6","7","8","9","0"]

        # Dict to rename the columns
        dict_name_change = {}

        # Loop over the columns names
        for _, item in enumerate(df.keys()):
            string = ""
            # Loop over the letters in the column name
            for j in item: 

                # If the letter is acceptable add it to the string
                if j in accept_letter:
                    string+=j

            # Add the new name to the dict        
            dict_name_change[item] = string
        
        # Return the renamed dataframe
        return df.rename(columns=dict_name_change)

    def make_OE_data(self, udata, ydata, nf=100):
        U = [] 
        Y = [] 
        for k in range(nf, len(udata)+1):
            U.append(udata[k-nf:k])
            Y.append(ydata[k-nf:k])
        return array(U), array(Y)
    
    def testsubmissionfiles1(self, na, nb):
        """Returns the test submission files"""
        if na<1 or nb<1:
            raise ValueError("na and nb must be positive integers")
        if na>15 or nb>15:
            raise ValueError("na and nb must be smaller than 15")
        
        testsub1 = DataFrame()
        for i in range(1,nb+1):
            testsub1[f"u[k-{i}]"] = self.testsub[f"u[k-{i}]"]
        for i in range(1,na+1):
            testsub1[f"y[k-{i}]"] = self.testsub[f"y[k-{i}]"]
        testsub1["y[k]"] = self.testsub["y[k-0]"]
        
        return testsub1
    
    def testsubmissionfiles2(self, na, nb):
        """Returns the test submission files"""
        if na<1 or nb<1:
            raise ValueError("na and nb must be positive integers")
        if na>15 or nb>15:
            raise ValueError("na and nb must be smaller than 15")
        
        testsub2 = DataFrame()
        mnanb = max(na, nb)
        for i in range(1, nb+1):
            testsub2[f"u[k-{i-1}]"] = self.testsim.iloc[mnanb-i+1:-i]["u"]
        for j in range(1, na+1):
            testsub2[f"y[k-{j}]"] = self.testsim.iloc[mnanb-j:-j]["th"]
        testsub2["y[k]"] = self.testsim.iloc[mnanb:]["u"]
        
        return testsub2